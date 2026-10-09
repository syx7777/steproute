"""组会纪要导入：把 Markdown / 纯文本解析成「决策」与「待办」，并落盘进语料。

* 解析为**确定性规则**（关键词 + 列表项），离线可测、结果可复现；
* 落盘到 ``data/sources/meeting_minutes/<日期>_<标题>.md``，自动带 YAML 头部，
  与现有 :func:`steproute.corpus.loaders.load_corpus` 无缝衔接；
* 可同时把抽取到的「决策」写入长期记忆（:class:`~steproute.assistant.memory.MemoryStore`）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from .memory import MemoryStore

logger = get_logger(__name__)

MINUTES_DIRNAME = "meeting_minutes"

#: 「决策」线索词。
DECISION_MARKERS = (
    "决定",
    "确定",
    "结论",
    "达成一致",
    "一致同意",
    "采用",
    "定为",
    "共识",
    "拍板",
    "通过",
)
#: 「待办」线索词。
TODO_MARKERS = (
    "待办",
    "TODO",
    "Todo",
    "todo",
    "下一步",
    "需要",
    "待补充",
    "待验证",
    "待确认",
    "跟进",
    "安排",
    "补充",
)

_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_CHECKBOX = re.compile(r"^\s*[-*+]\s*\[( |x|X)\]\s*(?P<body>.+)$")
_TODO_RE = re.compile(
    r"(?:" + "|".join(re.escape(m) for m in TODO_MARKERS) + r")\s*[:：]?\s*(?P<body>.+)$"
)
_DECISION_RE = re.compile(
    r"(?:" + "|".join(re.escape(m) for m in DECISION_MARKERS) + r")[^\n]*"
)
_DATE_HINT = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})")
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff._-]+")


@dataclass
class MinutesParse:
    """一篇纪要的解析结果。"""

    title: str
    day: str
    decisions: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "day": self.day,
            "decisions": self.decisions,
            "todos": self.todos,
            "word_count": self.word_count,
        }


@dataclass
class ImportResult:
    """导入结果。"""

    doc_id: str
    path: str
    parse: MinutesParse

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "path": self.path, **self.parse.to_dict()}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -•*\t")


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _detect_date(text: str, fallback: date) -> date:
    match = _DATE_HINT.search(text)
    if not match:
        return fallback
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return fallback


def parse_minutes(text: str, *, day: date | None = None, title: str = "") -> MinutesParse:
    """把纪要正文解析成决策与待办（去重，保持出现顺序）。"""
    fallback = day or date.today()
    resolved_day = _detect_date(text, fallback) if day is None else day
    resolved_title = title.strip() or _first_heading(text) or f"组会纪要 {resolved_day.isoformat()}"

    decisions: list[str] = []
    todos: list[str] = []
    seen_decisions: set[str] = set()
    seen_todos: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        checkbox = _CHECKBOX.match(line)
        if checkbox:
            body = _clean(checkbox.group("body"))
            if len(body) >= 4 and body not in seen_todos:
                seen_todos.add(body)
                todos.append(body)
            continue

        body_line = _LIST_PREFIX.sub("", line)
        todo_match = _TODO_RE.search(body_line)
        if todo_match:
            body = _clean(todo_match.group("body"))
            if len(body) >= 4 and body not in seen_todos:
                seen_todos.add(body)
                todos.append(body)

        decision_match = _DECISION_RE.search(body_line)
        if decision_match:
            body = _clean(decision_match.group(0))
            if len(body) >= 4 and body not in seen_decisions:
                seen_decisions.add(body)
                decisions.append(body)

    return MinutesParse(
        title=resolved_title,
        day=resolved_day.isoformat(),
        decisions=decisions,
        todos=todos,
        word_count=len(text),
    )


def _slug(value: str) -> str:
    cleaned = _SLUG_UNSAFE.sub("-", value.strip()).strip("-._")
    return cleaned or "minutes"


def import_minutes(
    data_dir: Path | str,
    text: str,
    *,
    day: date | None = None,
    title: str = "",
    author: str = "",
    memory: MemoryStore | None = None,
    remember_decisions: bool = True,
) -> ImportResult:
    """解析并落盘一篇纪要；随后可被 ``corpus`` 加载、被简报/周报引用。"""
    parsed = parse_minutes(text, day=day, title=title)

    target_dir = Path(data_dir) / "sources" / MINUTES_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{parsed.day}_{_slug(parsed.title)}"
    path = target_dir / f"{stem}.md"

    meta_lines = ["---", f"title: {parsed.title}", f"date: {parsed.day}"]
    if author:
        meta_lines.append(f"author: {author}")
    meta_lines += ["source: minutes-import", "---", ""]
    path.write_text("\n".join(meta_lines) + text.strip() + "\n", encoding="utf-8")
    logger.info("纪要已导入：%s（决策 %d / 待办 %d）", path, len(parsed.decisions), len(parsed.todos))

    if memory is not None and remember_decisions:
        for decision in parsed.decisions:
            memory.add_decision(decision, date_str=parsed.day, source=stem)

    return ImportResult(doc_id=f"{MINUTES_DIRNAME}/{stem}", path=str(path), parse=parsed)