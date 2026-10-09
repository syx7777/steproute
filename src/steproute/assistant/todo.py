"""待办清单：手动待办 + 文档待办，**全部可增 / 改 / 删 / 勾选完成**。

两类来源
--------
* **手动待办**（``data/todos.json`` 的 ``todos``）—— 用户直接新增的条目；
* **文档待办**（``corpus.scan_todos``）—— 从周报 / 纪要 / 实验记录里抽取的「下一步」，
  不修改源 Markdown（否则会破坏语料加载），而是把「改写 / 完成 / 删除」记为
  ``data/todos.json`` 里的 ``overrides``，展示时叠加生效。

展示规则：手动待办在前，文档待办按文档日期倒序，总数默认不超过 :data:`MAX_ITEMS` 条。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..corpus.model import Corpus
from ..logging_utils import get_logger
from .calendar import week_range

logger = get_logger(__name__)

TODOS_FILENAME = "todos.json"

OPEN = "open"
DONE = "done"

#: 页面默认最多展示多少条待办。
MAX_ITEMS = 10

#: 从某个知识库同步而来的文档所在目录前缀（用于按知识库过滤）。
KB_DOC_PREFIX = "notes"


def corpus_key(doc_id: str, text: str) -> str:
    """文档待办的稳定标识（用于记录 override）。"""
    digest = hashlib.sha1(f"{doc_id}|{text}".encode()).hexdigest()[:10]
    return f"c{digest}"


@dataclass
class Todo:
    """一条手动待办。"""

    id: str
    text: str
    status: str = OPEN
    owner: str = ""
    due: str = ""
    #: 归属的知识库 space_id（空 = 无归属，历史 / 手写条目，任何知识库下都可见）。
    space_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TodoStore:
    """待办与 override 的读写入口（JSON 落盘，整体覆盖写）。"""

    def __init__(self, data_dir: Path | str) -> None:
        self.path = Path(data_dir) / TODOS_FILENAME

    # -- 手动待办 ---------------------------------------------------------- #
    def list(self, *, status: str | None = None) -> list[Todo]:
        items, _ = self._read()
        if status is not None:
            items = [item for item in items if item.status == status]
        return items

    def add(self, text: str, *, owner: str = "", due: str = "", space_id: str = "") -> Todo:
        now = datetime.now().isoformat(timespec="seconds")
        todo = Todo(
            id=f"t{uuid.uuid4().hex[:8]}",
            text=text.strip(),
            owner=owner.strip(),
            due=due.strip(),
            space_id=space_id.strip(),
            created_at=now,
            updated_at=now,
        )
        items, overrides = self._read()
        items.append(todo)
        self._write(items, overrides)
        return todo

    def update(
        self,
        todo_id: str,
        *,
        text: str | None = None,
        status: str | None = None,
        owner: str | None = None,
        due: str | None = None,
    ) -> Todo | None:
        items, overrides = self._read()
        for todo in items:
            if todo.id != todo_id:
                continue
            if text is not None:
                todo.text = text.strip()
            if status is not None:
                todo.status = DONE if status == DONE else OPEN
            if owner is not None:
                todo.owner = owner.strip()
            if due is not None:
                todo.due = due.strip()
            todo.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(items, overrides)
            return todo
        return None

    def toggle(self, todo_id: str) -> Todo | None:
        for todo in self._read()[0]:
            if todo.id == todo_id:
                return self.update(todo_id, status=OPEN if todo.status == DONE else DONE)
        return None

    def delete(self, todo_id: str) -> bool:
        items, overrides = self._read()
        remaining = [item for item in items if item.id != todo_id]
        if len(remaining) == len(items):
            return False
        self._write(remaining, overrides)
        return True

    # -- 文档待办的 override ----------------------------------------------- #
    @property
    def overrides(self) -> dict[str, dict[str, Any]]:
        return self._read()[1]

    def set_override(self, key: str, **fields: Any) -> dict[str, Any]:
        """记录文档待办的改写（``text`` / ``owner`` / ``status`` / ``deleted``）。"""
        items, overrides = self._read()
        merged = dict(overrides.get(key) or {})
        for name, value in fields.items():
            if name == "deleted":
                merged["deleted"] = bool(value)
            elif name == "status":
                merged["status"] = DONE if value == DONE else OPEN
            elif name in {"text", "owner", "due"} and value is not None:
                merged[name] = str(value).strip()
        overrides[key] = merged
        self._write(items, overrides)
        return merged

    def clear_override(self, key: str) -> bool:
        items, overrides = self._read()
        if key not in overrides:
            return False
        overrides.pop(key)
        self._write(items, overrides)
        return True

    # -- 统计 / 内部 ------------------------------------------------------- #
    def stats(self) -> dict[str, int]:
        items = self._read()[0]
        done = sum(1 for item in items if item.status == DONE)
        return {"total": len(items), "open": len(items) - done, "done": done}

    def _read(self) -> tuple[list[Todo], dict[str, dict[str, Any]]]:
        if not self.path.is_file():
            return [], {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("待办文件损坏，按空清单处理：%s", self.path)
            return [], {}
        if not isinstance(payload, dict):
            return [], {}

        items: list[Todo] = []
        for entry in payload.get("todos") or []:
            if isinstance(entry, dict) and entry.get("text"):
                items.append(Todo(**{k: v for k, v in entry.items() if k in Todo.__dataclass_fields__}))
        overrides = {
            str(key): dict(value)
            for key, value in (payload.get("overrides") or {}).items()
            if isinstance(value, dict)
        }
        return items, overrides

    def _write(self, items: list[Todo], overrides: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "todos": [item.to_dict() for item in items],
            "overrides": overrides,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_day(corpus: Corpus, doc_id: str) -> date | None:
    """文档的「更新日期」：优先 front matter 的 ``updated_at``，回退 ``date``。

    同步落盘时 ``date`` 就是从 ``updated_at`` 派生的，因此与「飞书更新」口径一致。
    """
    doc = corpus.get(doc_id)
    if doc is None:
        return None
    for key in ("updated_at", "date"):
        raw = str(doc.meta.get(key) or "").strip()
        if not raw:
            continue
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            continue
    return None


def merged_todos(
    store: TodoStore,
    corpus: Corpus | None = None,
    *,
    space_id: str = "",
    max_items: int = MAX_ITEMS,
    day: date | None = None,
    llm_todos: list[str] | None = None,
) -> list[dict[str, Any]]:
    """合并手动待办与文档待办，并叠加 override。

    ``space_id`` 给定时，只保留该知识库同步来的文档待办（``notes/<space_id>/``），
    以及归属该知识库或无归属（``space_id == ""``）的手动待办；**不显示**其他知识库的
    手动待办。``day`` 给定时，文档待办只取**该日期所在自然周内更新过**的文档
    （手动待办不受周限制）。总数不超过 ``max_items``（手动待办优先，不会被截断）。

    ``llm_todos`` 给定时（LLM 归纳结果），用它**替换**文档待办：清单 = 手动待办 + 归纳待办，
    归纳项以只读形式展示（``source="llm"``，无 id / 不可勾选改删）。
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for todo in store.list():
        if space_id and todo.space_id not in ("", space_id):
            continue
        seen.add(todo.text.strip().lower())
        items.append(
            {
                "id": todo.id,
                "key": todo.id,
                "text": todo.text,
                "status": todo.status,
                "owner": todo.owner,
                "due": todo.due,
                "date": "",
                "source": "manual",
                "doc_id": "",
            }
        )

    if llm_todos is not None:
        extra: list[dict[str, Any]] = []
        for index, raw in enumerate(llm_todos):
            text = str(raw).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            extra.append(
                {
                    "id": f"llm{index}",
                    "key": f"llm{index}",
                    "text": text,
                    "status": OPEN,
                    "owner": "",
                    "due": "",
                    "date": "",
                    "source": "llm",
                    "doc_id": "",
                }
            )
        items.extend(extra[: max(0, max_items - len(items))])
        return items

    corpus_items: list[dict[str, Any]] = []
    if corpus is not None:
        prefix = f"{KB_DOC_PREFIX}/{space_id}/" if space_id else ""
        week = week_range(day) if day is not None else None
        overrides = store.overrides
        for item in corpus.scan_todos():
            if prefix and not item.doc_id.startswith(prefix):
                continue
            if week is not None:
                updated = doc_day(corpus, item.doc_id)
                if updated is None or not week[0] <= updated <= week[1]:
                    continue
            text = item.text.strip()
            if text.lower() in seen:
                continue
            key = corpus_key(item.doc_id, text)
            override = overrides.get(key) or {}
            if override.get("deleted"):
                continue
            seen.add(text.lower())
            corpus_items.append(
                {
                    "id": key,
                    "key": key,
                    "text": override.get("text") or text,
                    "status": override.get("status") or item.status,
                    "owner": override.get("owner") or item.owner,
                    "due": override.get("due") or "",
                    "date": item.date,
                    "source": "corpus",
                    "doc_id": item.doc_id,
                }
            )

    # 只按文档日期倒序（排序稳定，保持语料内的原始顺序）；
    # 注意：绝不能用「改写后的文字」参与排序，否则编辑后条目会莫名跳到别处甚至掉出窗口。
    corpus_items.sort(key=lambda entry: entry["date"], reverse=True)
    remaining = max(0, max_items - len(items))
    items.extend(corpus_items[:remaining])
    return items