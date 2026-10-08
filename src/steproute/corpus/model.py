"""语料数据模型：文档、片段、实验记录与 TODO。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceKind(str, Enum):
    """科研材料的来源类型。"""

    WEEKLY = "weekly_report"      # 周报 / 周报评论（语雀、钉钉）
    MINUTES = "meeting_minutes"   # 组会会议纪要
    EXPERIMENT = "experiment_log"  # 实验记录 / 实验日志
    REPO = "repo"                 # 代码仓库中的配置、结果文件
    NOTE = "note"                 # 其它零散笔记

    @classmethod
    def from_dir(cls, name: str) -> SourceKind:
        mapping = {
            "weekly_reports": cls.WEEKLY,
            "meeting_minutes": cls.MINUTES,
            "experiment_logs": cls.EXPERIMENT,
            "repo": cls.REPO,
            "notes": cls.NOTE,
        }
        return mapping.get(name.lower(), cls.NOTE)

    @property
    def label(self) -> str:
        return {
            SourceKind.WEEKLY: "周报/评论",
            SourceKind.MINUTES: "组会纪要",
            SourceKind.EXPERIMENT: "实验记录",
            SourceKind.REPO: "代码仓库",
            SourceKind.NOTE: "笔记",
        }[self]


@dataclass
class Document:
    """一篇来源文档。"""

    doc_id: str
    title: str
    kind: SourceKind
    text: str
    path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, max_chars: int | None = None) -> dict[str, Any]:
        text = self.text if max_chars is None else self.text[:max_chars]
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "kind": self.kind.value,
            "text": text,
            "meta": self.meta,
        }


@dataclass
class Chunk:
    """文档切分后的检索单元。"""

    chunk_id: str
    doc_id: str
    title: str
    kind: SourceKind
    text: str
    index: int


@dataclass
class TodoItem:
    """从材料中抽取出的待办事项。"""

    text: str
    doc_id: str
    date: str = ""
    owner: str = ""
    status: str = "open"

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "doc_id": self.doc_id, "date": self.date, "owner": self.owner, "status": self.status}


_TODO_PATTERNS = (
    re.compile(r"(?:^|\n)\s*[-*]\s*\[( |x|X)\]\s*(?P<body>.+?)(?:\n|$)"),
    re.compile(r"(?:TODO|Todo|todo|待办|下一步|需要|待补充|待验证|补一个|补充)\s*[:：]?\s*(?P<body>.+?)(?:\n|$)"),
)


class Corpus:
    """内存态语料库：文档 + 片段 + 实验索引 + TODO。"""

    def __init__(self, documents: list[Document], chunks: list[Chunk]) -> None:
        self.documents: dict[str, Document] = {doc.doc_id: doc for doc in documents}
        self.chunks: list[Chunk] = chunks
        self._experiments: dict[str, Document] = {
            str(doc.meta.get("experiment_id")): doc
            for doc in documents
            if doc.kind is SourceKind.EXPERIMENT and doc.meta.get("experiment_id")
        }

    # -- 查询 -------------------------------------------------------------- #
    def get(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def by_kind(self, kind: SourceKind) -> list[Document]:
        return [doc for doc in self.documents.values() if doc.kind is kind]

    def experiment(self, experiment_id: str) -> Document | None:
        return self._experiments.get(str(experiment_id))

    @property
    def experiment_ids(self) -> list[str]:
        return list(self._experiments.keys())

    def list_experiments(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for exp_id, doc in self._experiments.items():
            metrics = doc.meta.get("metrics") or {}
            rows.append(
                {
                    "experiment_id": exp_id,
                    "name": doc.meta.get("name", doc.title),
                    "date": doc.meta.get("date", ""),
                    "status": doc.meta.get("status", ""),
                    "config": doc.meta.get("config") or {},
                    "metrics": metrics,
                    "doc_id": doc.doc_id,
                }
            )
        return rows

    def scan_todos(self) -> list[TodoItem]:
        """扫描所有文档中的 TODO / 待办线索。"""
        items: list[TodoItem] = []
        seen: set[str] = set()
        for doc in self.documents.values():
            for line in doc.text.splitlines():
                for pattern in _TODO_PATTERNS:
                    match = pattern.search(line)
                    if not match:
                        continue
                    body = _clean(match.group("body"))
                    if len(body) < 4:
                        continue
                    key = body.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(
                        TodoItem(
                            text=body,
                            doc_id=doc.doc_id,
                            date=str(doc.meta.get("date", "")),
                            owner=str(doc.meta.get("author", "")),
                            status=_todo_status(line),
                        )
                    )
        return items

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for doc in self.documents.values():
            by_kind[doc.kind.value] = by_kind.get(doc.kind.value, 0) + 1
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "experiments": len(self._experiments),
            "by_kind": by_kind,
            "todos": len(self.scan_todos()),
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -•*\t")


def _todo_status(line: str) -> str:
    if re.search(r"\[[xX]\]", line) or re.search(r"(已完成|done|完成)", line):
        return "done"
    return "open"


# --------------------------------------------------------------------------- #
# 实验比较的通用语义（工具层与评估层共用，避免口径不一致）
# --------------------------------------------------------------------------- #

#: 指标方向线索：越大越好 / 越小越好。
HIGHER_BETTER_HINTS = ("acc", "f1", "score", "precision", "recall", "iou", "auc", "map")
LOWER_BETTER_HINTS = ("latency", "time", "ms", "cost", "flops", "params", "memory")

#: 决定"实验是否可比"的关键配置项。只有这些项不一致才算对比不公平；
#: 像 ``sampling`` / ``fusion`` 这类差异是方法本身的设计，属于预期内。
COMPARABILITY_KEYS = frozenset(
    {"backbone", "frames", "resolution", "eval_split", "seed", "epochs", "batch_size", "train_data"}
)


def metric_direction(name: str) -> int:
    """返回指标方向：``+1`` 越大越好，``-1`` 越小越好，``0`` 未知。"""
    lowered = name.lower()
    if any(hint in lowered for hint in HIGHER_BETTER_HINTS):
        return 1
    if any(hint in lowered for hint in LOWER_BETTER_HINTS):
        return -1
    return 0


def experiment_role(doc: Document) -> str:
    """判断实验角色：``baseline``（基线）还是 ``method``（本方法）。"""
    text = f"{doc.meta.get('name', '')} {doc.meta.get('experiment_id', '')}".lower()
    return "baseline" if ("baseline" in text or "基线" in text) else "method"