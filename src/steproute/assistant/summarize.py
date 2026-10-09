"""把「本周更新的文档正文 + 长期记忆」归纳成一份可用的每日简报摘要。

两种模式
--------
* **LLM 模式**（默认）：调用模型池 ``balanced`` 档，**直读本周更新的文档正文**（外加长期记忆摘要），
  归纳成结构化 JSON（``todos`` + ``progress``）——不再依赖「下一步：」这类固定话术的正则候选；
* **规则模式**（LLM 不可用或本周无更新时自动回退）：正则候选按文档日期倒序取前若干条，
  进展取长期记忆摘要。

结果按「输入指纹」缓存在 ``data/brief_summary.json``，**按知识库（``space_id``）分开保存**：
本周文档与记忆摘要都没变时直接复用缓存、不重复调用模型；切换知识库后再切回来，
仍能读回该知识库各自的归纳结果，不会因提炼别的知识库而被顶掉。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import ModelSpecConfig
from ..corpus.model import Corpus, Document
from ..llm.backends import Backend
from ..logging_utils import get_logger
from .calendar import week_range
from .memory import MemoryStore
from .todo import KB_DOC_PREFIX, TodoStore, doc_day, merged_todos

logger = get_logger(__name__)

#: 归纳结果的缓存文件（位于 data/ 根目录）。
SUMMARY_FILENAME = "brief_summary.json"

#: 规则回退用的候选条数上限。
DEFAULT_CANDIDATE_LIMIT = 60

#: 交给模型的素材字符上限。
MAX_MATERIAL_CHARS = 8000

#: 采集「本周文档」的默认参数。
DEFAULT_WEEK_DOCS = 8
DEFAULT_DOC_CHARS = 1200

_SYSTEM_PROMPT = """你是科研项目助理。请阅读给定的「本周更新的文档正文」与「长期记忆摘要」，归纳成一份简洁的每日工作摘要。

只输出一个 JSON 对象（不要代码围栏、不要解释），字段如下：
{
  "todos": ["可执行的待办，每条一句话，最多 10 条"],
  "progress": "3~5 句连贯的项目进展摘要"
}

要求：
- todos：**从文档内容中识别真正需要跟进的事项**，改写成简短可执行的待办（动词开头、一句话、不超过 20 字）；
  **禁止照抄原文片段**；合并语义重复项；按重要性排序；最多 10 条；文档里没有可执行事项时返回空数组。
- progress：**必须改写为 3~5 句连贯的项目进展总结**（在做什么 / 卡在哪 / 下一步方向），
  **禁止逐条罗列原文**，禁止出现 `#` 标题或 `- [日期]` 之类原始格式。
- 材料不足的字段留空数组或空字符串，不要编造。"""


@dataclass
class BriefSummary:
    """一次简报归纳的结果。"""

    todos: list[str] = field(default_factory=list)
    progress: str = ""
    method: str = "rules"  # llm / rules
    note: str = ""
    candidates: int = 0  # 本次归纳前有多少条原始候选（规则回退用）
    docs: int = 0  # 本次归纳读到的本周文档数（LLM 直读用）
    generated_at: str = ""
    fingerprint: str = ""  # 输入指纹，用于缓存复用
    space_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "todos": self.todos,
            "progress": self.progress,
            "method": self.method,
            "note": self.note,
            "candidates": self.candidates,
            "docs": self.docs,
            "generated_at": self.generated_at,
            "fingerprint": self.fingerprint,
            "space_id": self.space_id,
        }


# --------------------------------------------------------------------------- #
# 素材收集
# --------------------------------------------------------------------------- #

def collect_candidates(
    data_dir: Path | str,
    corpus: Corpus,
    *,
    space_id: str = "",
    limit: int | None = None,
    day: date | None = None,
) -> list[dict[str, Any]]:
    """取「文档候选」待办（``open`` 且 ``source != 'manual'``），按文档日期倒序。

    ``limit`` 为 ``None`` 时返回全部候选（用于统计总数），否则只取前 ``limit`` 条。
    ``day`` 给定时只取**该日期所在自然周内更新过**的文档（与「飞书更新」同一口径）。
    """
    store = TodoStore(data_dir)
    max_items = 10**9 if limit is None else limit
    items = merged_todos(store, corpus, space_id=space_id, max_items=max_items, day=day)
    candidates = [
        item
        for item in items
        if item.get("source") != "manual" and item.get("status") == "open"
    ]
    candidates.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
    return candidates


def collect_week_documents(
    corpus: Corpus,
    *,
    space_id: str = "",
    day: date | None = None,
    max_docs: int = DEFAULT_WEEK_DOCS,
    doc_chars: int = DEFAULT_DOC_CHARS,
    max_chars: int = MAX_MATERIAL_CHARS,
) -> list[Document]:
    """取「本周更新的文档正文」，供 LLM 直读归纳。

    只保留 ``doc_id`` 以 ``notes/<space_id>/`` 开头的文档（``space_id`` 为空则不限知识库）；
    ``day`` 给定时只取**该日期所在自然周**（:func:`~steproute.assistant.calendar.week_range`）
    内更新过的文档（判断口径见 :func:`~steproute.assistant.todo.doc_day`），无日期的文档一律排除；
    按文档日期倒序取前 ``max_docs`` 篇，每篇正文截断到 ``doc_chars``，累计不超过 ``max_chars``
    （返回的 :class:`~steproute.corpus.model.Document` 正文即已截断，便于直接拼素材 / 算指纹）。
    """
    prefix = f"{KB_DOC_PREFIX}/{space_id}/" if space_id else ""
    week = week_range(day) if day is not None else None

    picked: list[tuple[date, Document]] = []
    for doc in corpus.documents.values():
        if prefix and not doc.doc_id.startswith(prefix):
            continue
        updated = doc_day(corpus, doc.doc_id)
        if updated is None:
            continue
        if week is not None and not week[0] <= updated <= week[1]:
            continue
        picked.append((updated, doc))
    picked.sort(key=lambda item: item[0], reverse=True)

    docs: list[Document] = []
    used = 0
    for _, doc in picked:
        if len(docs) >= max(0, max_docs):
            break
        text = doc.text[:doc_chars]
        if used + len(text) > max_chars:
            break
        docs.append(replace(doc, text=text))
        used += len(text)
    return docs


def _fingerprint(corpus: Corpus, week_docs: list[Document], memory_text: str) -> str:
    """输入指纹 = 本周文档（标题 + 日期 + 截断正文）+ 记忆摘要的 sha1 前 12 位。"""
    blocks: list[str] = []
    for doc in week_docs:
        updated = doc_day(corpus, doc.doc_id)
        blocks.append(f"{doc.title}|{updated.isoformat() if updated else ''}|{doc.text}")
    joined = "\n".join(blocks)
    basis = f"{joined}\n::\n{memory_text or ''}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _materials(week_docs: list[Document], corpus: Corpus, memory_text: str) -> str:
    """拼出交给模型的素材：本周文档正文（+ 长期记忆摘要）。"""
    lines: list[str] = ["### 本周更新的文档正文"]
    for doc in week_docs:
        updated = doc_day(corpus, doc.doc_id)
        when = updated.isoformat() if updated else ""
        lines.append(f"### 文档：{doc.title}（{when}）\n{doc.text}")
    text = (memory_text or "").strip()
    if text and text != "（长期记忆为空）":
        lines.append(f"### 长期记忆摘要\n{text}")
    return "\n".join(lines)[:MAX_MATERIAL_CHARS]


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any, *, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    picked: list[str] = []
    for item in value:
        text = str(item).strip()
        if len(text) >= 4 and text not in picked:
            picked.append(text)
    return picked[:limit]


def _rules_progress(space_id: str, *, space_label: str = "", decisions: int, lessons: int, projects: int) -> str:
    """规则回退的「项目进展」：只给一句按知识库作用域生成的短文本。

    刻意**不回填** ``memory.summary()`` 原文，避免简报里出现 ``###`` 标题与整句碎片；
    连贯摘要交给 LLM 归纳。``space_label``（知识库中文名）优先于 ``space_id`` 用于文案标注。
    """
    if projects == 0 and decisions == 0 and lessons == 0:
        return "该知识库暂无长期记忆，可先同步知识库并运行 memory distill。"
    scope = space_label or space_id or "未指定"
    return (
        f"当前知识库「{scope}」：{decisions} 条最近决策 / {lessons} 条实验经验；"
        "如需连贯摘要请配置 LLM 后重新归纳。"
    )


def _rules_todos(candidates: list[dict[str, Any]], *, max_todos: int) -> list[str]:
    todos: list[str] = []
    for item in candidates:
        text = str(item.get("text", "")).strip()
        if not text or text in todos:
            continue
        todos.append(text)
        if len(todos) >= max_todos:
            break
    return todos


# --------------------------------------------------------------------------- #
# 缓存（按知识库分开保存）
# --------------------------------------------------------------------------- #

def _summary_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / SUMMARY_FILENAME


def _load_summaries(data_dir: Path | str) -> dict[str, dict[str, Any]]:
    """读取 ``data/brief_summary.json`` 中全部知识库的归纳结果（按 ``space_id`` 索引）。

    兼容旧版「整份文件只存一个 space 结果」的格式：把顶层那条就势归入其 ``space_id`` 键，
    下次写入时落成新版 ``{"version": 2, "summaries": {...}}``。
    """
    path = _summary_path(data_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("简报摘要缓存损坏，忽略：%s", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        return {str(key): value for key, value in summaries.items() if isinstance(value, dict)}
    if any(key in payload for key in ("todos", "progress", "space_id")):
        return {str(payload.get("space_id") or ""): payload}
    return {}


def save_summary(data_dir: Path | str, summary: BriefSummary) -> Path:
    """把归纳结果按 ``space_id`` 写入 ``data/brief_summary.json``（不影响其它知识库的缓存）。"""
    path = _summary_path(data_dir)
    summaries = _load_summaries(data_dir)
    summaries[summary.space_id] = summary.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "summaries": summaries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_summary(data_dir: Path | str, *, space_id: str = "") -> BriefSummary | None:
    """读取指定 ``space_id`` 的缓存；无对应条目（或文件缺失 / 损坏）时返回 ``None``。"""
    payload = _load_summaries(data_dir).get(space_id)
    if not isinstance(payload, dict):
        return None
    return BriefSummary(
        todos=[str(x) for x in (payload.get("todos") or [])],
        progress=str(payload.get("progress") or ""),
        method=str(payload.get("method") or "rules"),
        note=str(payload.get("note") or ""),
        candidates=int(payload.get("candidates") or 0),
        docs=int(payload.get("docs") or 0),
        generated_at=str(payload.get("generated_at") or ""),
        fingerprint=str(payload.get("fingerprint") or ""),
        space_id=space_id,
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def summarize_brief(
    *,
    data_dir: Path | str,
    corpus: Corpus,
    memory: MemoryStore | None = None,
    space_id: str = "",
    space_label: str = "",
    day: date | None = None,
    backend: Backend | None = None,
    spec: ModelSpecConfig | None = None,
    use_llm: bool = True,
    max_todos: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    candidate_todos: list[dict[str, Any]] | None = None,
) -> BriefSummary:
    """归纳简报：把**本周更新的文档正文**与长期记忆变成「可执行待办 + 项目进展」。

    ``space_id`` 给定时，长期记忆只取该知识库（``space_id`` 匹配 + 无归属）的条目，
    避免跨知识库串味；``space_label`` 为知识库中文名，仅用于规则回退进展文案的展示
    （缺省时退回 ``space_id``）。
    LLM 模式直读 :func:`collect_week_documents` 采集的本周文档正文；``candidate_todos`` 可注入
    正则候选（离线测试 / 规则回退用），默认从 :func:`steproute.assistant.todo.merged_todos` 取文档候选。
    ``use_llm=False``、未注入 backend/spec、本周无更新文档，或模型调用 / 解析失败时回退到规则模式。
    """
    data_root = Path(data_dir)
    store = memory or MemoryStore(data_root)
    memory_text = store.summary(space_id=space_id)
    n_decisions = len(store.decisions(space_id=space_id))
    n_lessons = len(store.lessons(space_id=space_id))
    n_projects = len(store.projects(space_id=space_id))

    if candidate_todos is None:
        candidates = collect_candidates(data_root, corpus, space_id=space_id, limit=candidate_limit, day=day)
    else:
        candidates = [dict(item) for item in candidate_todos]
    candidates.sort(key=lambda item: str(item.get("date") or ""), reverse=True)

    week_docs = collect_week_documents(corpus, space_id=space_id, day=day)

    summary = BriefSummary(
        candidates=len(candidates),
        docs=len(week_docs),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        fingerprint=_fingerprint(corpus, week_docs, memory_text),
        space_id=space_id,
    )

    payload: dict[str, Any] = {}
    if use_llm and backend is not None and spec is not None:
        if not week_docs:
            # 本周该知识库没有更新的文档：素材为空，不必调用模型（省一次无意义调用）。
            summary.note = "本周该知识库没有更新的文档，按规则回退"
            logger.info("简报归纳跳过模型：本周无更新文档 space_id=%s", space_id)
        else:
            cached = load_summary(data_root, space_id=space_id)
            # 只在缓存本来就是 LLM 结果、且输入指纹一致时复用；
            # 若上次是规则回退（缺 Key），配好 Key 后应重新尝试模型。
            if cached is not None and cached.method == "llm" and cached.fingerprint == summary.fingerprint:
                cached.note = cached.note or "输入未变化，复用缓存"
                logger.info("简报归纳复用缓存：fingerprint=%s", cached.fingerprint)
                return cached
            try:
                chat = backend.complete(
                    spec,
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _materials(week_docs, corpus, memory_text)},
                    ],
                    json_mode=True,
                    temperature=0.2,
                    # 归纳是抽取型任务，不需要高推理：balanced 档默认 reasoning_effort=high，
                    # 会把 max_tokens 全用在 reasoning 上、content 为空（表现为"静默回退规则"）。
                    reasoning_effort="low",
                    max_tokens=spec.max_tokens,
                    label="summarize",
                )
                payload = _parse_json(chat.content)
            except Exception as exc:  # 缺 Key / 网络异常等都回退到规则模式
                summary.note = f"LLM 归纳不可用，已回退规则：{exc}"
                logger.warning("简报归纳回退规则模式：%s", exc)
    elif not use_llm:
        # 用户显式关闭模型，而非漏传后端——note 要能区分这两种情况。
        summary.note = "未启用 LLM（use_llm=False）"
    else:
        summary.note = "未提供模型后端，按规则回退"

    if payload:
        todos = _as_list(payload.get("todos"), limit=max_todos)
        progress = str(payload.get("progress") or "").strip()
        if todos or progress:
            summary.method = "llm"
            summary.todos = todos
            summary.progress = progress or _rules_progress(
                space_id, space_label=space_label, decisions=n_decisions, lessons=n_lessons, projects=n_projects
            )
            logger.info("简报归纳完成：method=llm 文档 %d → 待办 %d", summary.docs, len(todos))
            return summary
        summary.note = summary.note or "模型返回中未包含可用内容，已回退规则"
    elif not summary.note:
        # 调用了模型但拿不到 JSON：最常见原因是 reasoning 占满了 max_tokens。
        summary.note = "模型返回内容无法解析为 JSON（常见原因：reasoning 占满 max_tokens），已回退规则"

    summary.method = "rules"
    summary.todos = _rules_todos(candidates, max_todos=max_todos)
    summary.progress = _rules_progress(
        space_id, space_label=space_label, decisions=n_decisions, lessons=n_lessons, projects=n_projects
    )
    logger.info("简报归纳完成：method=rules 候选 %d → 待办 %d", summary.candidates, len(summary.todos))
    return summary