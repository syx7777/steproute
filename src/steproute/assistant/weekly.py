"""周报生成：汇总一周工作 → 渲染 → 写入飞书知识库。

命名与周次口径统一由 :mod:`steproute.assistant.calendar` 提供
（``{学年}-{学期}-week{周次}``）。

「按模板生成」的实现方式：读取模板文档的**标题层级**作为周报结构骨架，
再把本周内容按小节语义填入（见 :func:`render_weekly` 的 ``sections`` 参数）；
找到不模板时回退到内置小节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from ..corpus.model import Corpus
from ..logging_utils import get_logger
from .brief import week_feishu_updates
from .calendar import WeekLabel, label_for
from .memory import Entry, MemoryStore

logger = get_logger(__name__)

DEFAULT_SECTIONS = ("本周进展", "实验与结论", "下周计划", "风险与待协助")

#: 小节标题关键词 → 内容类别。
_SECTION_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("进展", "工作", "总结", "完成", "本周"), "progress"),
    (("实验", "结果", "结论", "分析", "发现"), "lessons"),
    (("计划", "下周", "安排", "待办", "任务"), "todos"),
    (("风险", "问题", "困难", "协助", "阻塞"), "risks"),
)


@dataclass
class WeekDigest:
    """一周的素材汇总。"""

    label: WeekLabel
    decisions: list[Entry] = field(default_factory=list)
    lessons: list[Entry] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    feishu_updates: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.to_dict(),
            "decisions": [entry.to_dict() for entry in self.decisions],
            "lessons": [entry.to_dict() for entry in self.lessons],
            "todos": self.todos,
            "feishu_updates": self.feishu_updates,
        }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def _week_feishu_updates(data_dir: Path, start: date, end: date, *, limit: int = 20) -> list[dict[str, str]]:
    """本周飞书更新（与每日简报共用同一口径）。"""
    return week_feishu_updates(data_dir, start, end, limit=limit)


def collect_week(
    *,
    data_dir: Path | str,
    memory: MemoryStore | None = None,
    corpus: Corpus | None = None,
    day: date | None = None,
    max_todos: int = 20,
) -> WeekDigest:
    """汇总某一周（默认含今天的自然周）的决策 / 经验 / 待办 / 飞书更新。"""
    data_root = Path(data_dir)
    store = memory or MemoryStore(data_root)
    label = label_for(day)
    decisions, lessons = store.between(label.start, label.end)

    todos: list[str] = []
    if corpus is not None:
        seen: set[str] = set()
        for item in corpus.scan_todos():
            if item.status != "open" or item.text.lower() in seen:
                continue
            seen.add(item.text.lower())
            todos.append(item.text)
            if len(todos) >= max_todos:
                break

    return WeekDigest(
        label=label,
        decisions=decisions,
        lessons=lessons,
        todos=todos,
        feishu_updates=_week_feishu_updates(data_root, label.start, label.end),
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #

def _bullets(items: list[str], empty: str = "- （本项暂无内容）") -> list[str]:
    return [f"- {item}" for item in items] if items else [empty]


def _category_of(section_title: str) -> str:
    for keywords, category in _SECTION_HINTS:
        if any(keyword in section_title for keyword in keywords):
            return category
    return ""


def render_weekly(digest: WeekDigest, *, sections: list[str] | None = None) -> str:
    """把一周素材渲染成 Markdown 周报。

    ``sections`` 给定时按其顺序作为小节标题（模板骨架），并按语义填入内容；
    未给定时使用内置小节 :data:`DEFAULT_SECTIONS`。
    """
    progress = [f"[{entry.date}] {entry.text}" for entry in digest.decisions]
    lessons = [f"[{entry.date}] {entry.text}" for entry in digest.lessons]
    todos = list(digest.todos)
    risks: list[str] = []

    if digest.feishu_updates:
        lessons = lessons + [f"文档更新：{item['doc_id']}（{item['updated_at']}）" for item in digest.feishu_updates]

    buckets = {"progress": progress, "lessons": lessons, "todos": todos, "risks": risks}

    lines = [f"# {digest.label.title}", ""]
    lines += [f"> 统计区间：{digest.label.start.isoformat()} ~ {digest.label.end.isoformat()}", ""]

    titles = list(sections) if sections else list(DEFAULT_SECTIONS)
    used: set[str] = set()
    for index, title in enumerate(titles):
        lines.append(f"## {title}")
        category = _category_of(title)
        if category and category not in used:
            used.add(category)
            lines += _bullets(buckets[category])
        elif index == 0:
            # 模板首节若无法归类，装入所有尚未消费的内容，避免丢信息
            leftovers: list[str] = []
            for key, values in buckets.items():
                if key not in used:
                    leftovers += values
            lines += _bullets(leftovers)
        else:
            lines.append("- （待补充）")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# 写入飞书
# --------------------------------------------------------------------------- #

class _WeeklyClient(Protocol):
    def find_template_token(self, space_id: str, keyword: str) -> str: ...
    def read_document_text(self, document_id: str) -> str: ...
    def create_space_node(self, space_id: str, *, title: str, parent_node_token: str | None = ...) -> dict[str, Any]: ...
    def append_markdown(self, document_id: str, markdown: str) -> int: ...


@dataclass
class PublishResult:
    """周报发布结果。"""

    title: str
    document_id: str
    url: str
    blocks: int
    template_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "document_id": self.document_id,
            "url": self.url,
            "blocks": self.blocks,
            "template_token": self.template_token,
        }


def template_sections(template_text: str) -> list[str]:
    """从模板正文里提取小节标题（``##``/``###`` 行）。"""
    sections: list[str] = []
    for line in template_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            title = stripped.lstrip("#").strip()
            if title and title not in sections:
                sections.append(title)
    return sections


def publish_weekly(
    client: _WeeklyClient,
    *,
    space_id: str,
    markdown: str,
    title: str,
    template_token: str = "",
    template_keyword: str = "模板",
    parent_node_token: str | None = None,
) -> PublishResult:
    """在知识库 ``space_id`` 下新建一篇以 ``title`` 命名的周报并写入内容。"""
    if not template_token and template_keyword:
        template_token = client.find_template_token(space_id, template_keyword) or ""

    sections: list[str] = []
    if template_token:
        try:
            sections = template_sections(client.read_document_text(template_token))
        except Exception as exc:  # 模板不可读时回退内置结构，不阻断发布
            logger.warning("读取模板失败，改用内置小节：%s", exc)

    node = client.create_space_node(space_id, title=title, parent_node_token=parent_node_token)
    document_id = str(node.get("obj_token") or node.get("node_token") or "")
    if not document_id:
        raise RuntimeError(f"创建知识库节点失败，返回：{node}")

    body = markdown
    if sections:
        body = _reframe_with_sections(markdown, sections)
    blocks = client.append_markdown(document_id, body)

    node_token = str(node.get("node_token") or "")
    url = f"https://feishu.cn/wiki/{node_token}" if node_token else ""
    return PublishResult(title=title, document_id=document_id, url=url, blocks=blocks, template_token=template_token)


def _reframe_with_sections(markdown: str, sections: list[str]) -> str:
    """按模板小节顺序重排正文（保留内容，替换标题层级）。"""
    titles = template_sections(markdown)
    if not titles:
        return markdown
    lines = markdown.splitlines()
    # 用模板小节标题按序替换正文里的 ## 小节标题；多出的模板小节补空
    out: list[str] = []
    replaced = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if replaced < len(sections):
                out.append(f"## {sections[replaced]}")
                replaced += 1
            else:
                out.append(line)
        else:
            out.append(line)
    while replaced < len(sections):
        out += [f"## {sections[replaced]}", "- （待补充）", ""]
        replaced += 1
    return "\n".join(out).rstrip() + "\n"