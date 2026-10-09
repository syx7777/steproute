"""每日工作简报：项目进展 + 飞书更新。

* **项目进展**：长期记忆里的项目状态 + 最近决策（传入 ``summary`` 时改为 LLM 归纳的一段摘要）；
* **飞书更新**：自上次生成简报以来，``data/sync_state.json`` 中 ``updated_at`` 变新的文档。

待办不在这里渲染（已在页面的待办卡片独立展示）。简报落盘到 ``data/briefs/<日期>.md``，
并记录时间水位 ``data/memory/last_brief.json``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..corpus.model import Corpus
from ..logging_utils import get_logger
from .calendar import week_range
from .memory import MemoryStore
from .summarize import BriefSummary
from .todo import KB_DOC_PREFIX, MAX_ITEMS, OPEN, TodoStore, merged_todos

logger = get_logger(__name__)

BRIEFS_DIRNAME = "briefs"
LAST_BRIEF_FILENAME = "last_brief.json"


@dataclass
class Brief:
    """一份每日工作简报。"""

    day: str
    #: 待办仍在 Brief 中保留（供调用方复用），但不再渲染进 Markdown。
    todos: list[str] = field(default_factory=list)
    #: 规则模式下是「项目状态行 + 决策/经验条数概览」的短列表；LLM 归纳模式下是一段连贯摘要（字符串）。
    progress: list[str] | str = field(default_factory=list)
    feishu_updates: list[dict[str, str]] = field(default_factory=list)
    markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "todos": self.todos,
            "progress": self.progress,
            "feishu_updates": self.feishu_updates,
            "markdown": self.markdown,
        }


def _last_brief_path(data_dir: Path) -> Path:
    return Path(data_dir) / "memory" / LAST_BRIEF_FILENAME


def _write_last_brief(data_dir: Path, moment: datetime) -> None:
    path = _last_brief_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_brief_at": moment.isoformat(timespec="seconds")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _title_from_file(path: str) -> str:
    """旧版同步状态没存标题时，退化到读落盘文件的 YAML 头部 ``title:``。"""
    if not path:
        return ""
    try:
        head = Path(path).read_text(encoding="utf-8").splitlines()[:12]
    except (OSError, UnicodeDecodeError):
        return ""
    for line in head[1:]:
        if line.strip() == "---":
            break
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
    return ""


def _doc_id_from_path(data_dir: Path, path: str) -> str:
    """由落盘路径反推语料内的 ``doc_id``（如 ``notes/<space_id>/<stem>``）。

    ``sync_state.json`` 的键是飞书文档 token，**不能**用来做知识库过滤；
    真正带知识库信息的是这里记录的 ``path``。
    """
    if not path:
        return ""
    try:
        relative = Path(path).resolve().relative_to((data_dir / "sources").resolve())
    except (ValueError, OSError):
        return ""
    return "/".join(relative.with_suffix("").parts)


def week_feishu_updates(
    data_dir: Path | str,
    start: date,
    end: date,
    *,
    space_id: str = "",
    limit: int = 30,
) -> list[dict[str, str]]:
    """``[start, end]`` 区间内被更新过的飞书文档（默认用于「本周」）。

    数据来自 ``data/sync_state.json``（``steproute sync`` 的增量水位），
    因此只反映**最后一次同步时看到的更新时间**——远端有新改动时，
    需先「拉取更新」（或 ``steproute sync``）再生成简报/周报。
    """
    prefix = f"{KB_DOC_PREFIX}/{space_id}/" if space_id else ""
    data_root = Path(data_dir)
    path = data_root / "sync_state.json"
    if not path.is_file():
        return []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    updates: list[dict[str, str]] = []
    for source in (state.get("sources") or {}).values():
        for target, docs in (source.get("targets") or {}).items():
            for key, raw_meta in (docs or {}).items():
                meta = raw_meta or {}
                local_path = str(meta.get("path") or "")
                doc_id = _doc_id_from_path(data_root, local_path) or str(key)
                if prefix and not doc_id.startswith(prefix):
                    continue
                updated_at = str((meta or {}).get("updated_at", ""))
                try:
                    updated_day = datetime.fromisoformat(updated_at).date()
                except ValueError:
                    continue
                if not start <= updated_day <= end:
                    continue
                updates.append(
                    {
                        "doc_id": str(doc_id),
                        "title": (
                            str((meta or {}).get("title") or "")
                            or _title_from_file(str((meta or {}).get("path") or ""))
                            or str(doc_id)
                        ),
                        "url": str((meta or {}).get("url") or ""),
                        "target": target,
                        "updated_at": updated_at,
                    }
                )
    updates.sort(key=lambda item: item["updated_at"], reverse=True)
    return updates[:limit]


def _rules_progress(store: MemoryStore, *, space_id: str, space_label: str = "") -> list[str]:
    """规则模式的「项目进展」：结构化状态行 + 决策/经验条数概览。

    刻意**不拼接决策原文**，避免简报里出现半句碎片（连贯摘要交给 LLM 归纳）。
    ``space_id`` 给定时只统计该知识库（+ 无归属）的条目，避免跨知识库串味；
    ``space_label``（知识库中文名）优先于 ``space_id`` 用于文案标注。
    """
    scope = space_label or space_id or "未指定"
    lines: list[str] = []
    for state in list(store.projects(space_id=space_id).values())[:8]:
        parts = [p for p in (state.status, state.stage, state.goal) if p]
        lines.append(f"{state.project}：{'；'.join(parts) if parts else '（未填写）'}")

    decisions = len(store.decisions(space_id=space_id))
    lessons = len(store.lessons(space_id=space_id))
    if decisions or lessons:
        lines.append(
            f"当前知识库「{scope}」：最近 {decisions} 条决策 / {lessons} 条实验经验（如需连贯摘要请配置 LLM 后重新归纳）"
        )
    if not lines:
        lines.append(f"该知识库「{scope}」暂无长期记忆，可先导入组会纪要或运行 memory distill")
    return lines


def render_brief(brief: Brief) -> str:
    lines = [f"# 每日工作简报 · {brief.day}", ""]

    lines.append("## 项目进展")
    if isinstance(brief.progress, str):
        lines.append(brief.progress.strip() or "（暂无进展摘要，可先运行 memory distill 或导入组会纪要）")
    elif brief.progress:
        lines += [f"- {item}" for item in brief.progress]
    else:
        lines.append("- （长期记忆为空，可先用 `steproute memory state` 录入）")

    try:
        week_start, week_end = week_range(date.fromisoformat(brief.day))
        lines += ["", f"## 飞书更新（本周 {week_start.isoformat()} ~ {week_end.isoformat()}）"]
    except ValueError:
        lines += ["", "## 飞书更新"]
    if brief.feishu_updates:
        lines += [
            f"- {item.get('title') or item['doc_id']}（{item['updated_at']}）" for item in brief.feishu_updates
        ]
    else:
        lines.append("- （本周暂无更新；远端有改动时请先「拉取更新」）")

    return "\n".join(lines) + "\n"


def build_brief(
    corpus: Corpus,
    *,
    data_dir: Path | str,
    memory: MemoryStore | None = None,
    todos: TodoStore | None = None,
    day: date | None = None,
    space_id: str = "",
    space_label: str = "",
    max_todos: int = MAX_ITEMS,
    summary: BriefSummary | None = None,
    save: bool = True,
) -> Brief:
    """生成（并默认落盘）一份每日简报。

    ``space_id`` 给定时，飞书更新只保留该知识库同步来的文档，「项目进展」也只统计该知识库
    （长期记忆中 ``space_id`` 匹配 + 无归属）的条目，避免跨知识库串味；
    ``space_label``（知识库中文名）仅用于规则回退进展文案的展示（缺省时退回 ``space_id``）。

    传入 ``summary``（LLM 归纳结果）时，「项目进展」用它的一段摘要文本；
    不传时保持规则行为（项目状态行 + 决策 / 经验条数概览）。
    """
    data_root = Path(data_dir)
    target_day = day or date.today()
    store = memory or MemoryStore(data_root)
    todo_store = todos or TodoStore(data_root)

    # 手动待办在最前，且不受条数限制（按知识库隔离）。
    manual: list[str] = []
    for todo in todo_store.list(status=OPEN):
        if space_id and todo.space_id not in ("", space_id):
            continue
        text = todo.text.strip()
        if text and text not in manual:
            manual.append(text)

    if summary is not None:
        todos = list(manual)
        for text in summary.todos:
            text = text.strip()
            if text and text not in todos:
                todos.append(text)
        progress: list[str] | str = summary.progress.strip() or "（暂无进展摘要）"
    else:
        todos = []
        for item in merged_todos(todo_store, corpus, space_id=space_id, max_items=max_todos, day=target_day):
            if item["status"] != "open":
                continue
            todos.append(item["text"])
            if len(todos) >= max_todos:
                break

        progress = _rules_progress(store, space_id=space_id, space_label=space_label)

    week_start, week_end = week_range(target_day)
    brief = Brief(
        day=target_day.isoformat(),
        todos=todos,
        progress=progress,
        feishu_updates=week_feishu_updates(data_root, week_start, week_end, space_id=space_id),
    )
    brief.markdown = render_brief(brief)

    if save:
        out = data_root / BRIEFS_DIRNAME
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{target_day.isoformat()}.md").write_text(brief.markdown, encoding="utf-8")
        _write_last_brief(data_root, datetime.now())
        logger.info("简报已写入 %s", out / f"{target_day.isoformat()}.md")

    return brief