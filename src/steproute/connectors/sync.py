"""外部来源 → 本地语料的「落盘式」同步。

核心取舍：远端内容只落地成 ``<data_dir>/sources/<target>/*.md``，
下游的检索、TODO 抽取、路由调度**零改动**复用现有加载器
（:func:`steproute.corpus.loaders.load_corpus`）。

增量：``<data_dir>/sync_state.json`` 记录每个来源每篇文档的 ``updated_at`` 水位，
未变化的文档不重复拉取正文；远端已删除的文档会同步清理本地副本。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from .base import RemoteDoc, RemoteSource

logger = get_logger(__name__)

SYNC_STATE_FILENAME = "sync_state.json"

#: 可接收纯文本语料的来源目录（``experiment_logs`` 需要结构化 JSON，不在此列）。
TEXT_TARGETS = frozenset({"weekly_reports", "meeting_minutes", "repo", "notes"})

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff._-]+")
_DATE_PREFIX = re.compile(r"(\d{4}-\d{2}-\d{2})")


class SyncTargetError(ValueError):
    """目标目录非法。"""


@dataclass
class SyncResult:
    """一次同步的结果摘要。"""

    source: str
    target: str
    listed: int
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    state_path: str = ""
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "listed": self.listed,
            "written": len(self.written),
            "skipped": len(self.skipped),
            "removed": len(self.removed),
            "state_path": self.state_path,
            "dry_run": self.dry_run,
        }


# --------------------------------------------------------------------------- #
# 状态水位
# --------------------------------------------------------------------------- #

def load_state(data_dir: Path | str) -> dict[str, Any]:
    path = Path(data_dir) / SYNC_STATE_FILENAME
    if not path.is_file():
        return {"version": 1, "sources": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("同步状态文件损坏，按空状态处理：%s", path)
        return {"version": 1, "sources": {}}
    return payload if isinstance(payload, dict) else {"version": 1, "sources": {}}


def save_state(data_dir: Path | str, state: dict[str, Any]) -> Path:
    path = Path(data_dir) / SYNC_STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _date_of(iso: str) -> str:
    match = _DATE_PREFIX.search(iso or "")
    return match.group(1) if match else ""


def _sanitize(value: str) -> str:
    return _SLUG_UNSAFE.sub("-", (value or "").strip()).strip("-._")


def assign_stems(docs: list[RemoteDoc]) -> dict[str, str]:
    """为每篇文档分配稳定的文件名主干（按标题，重名时追加 token 前缀）。"""
    base = {doc.doc_key: _sanitize(doc.title) or _sanitize(doc.external_id) or "doc" for doc in docs}
    counts = Counter(base.values())
    stems: dict[str, str] = {}
    for doc in docs:
        stem = base[doc.doc_key]
        if counts[stem] > 1:
            suffix = _sanitize(doc.external_id)[:8]
            stem = f"{stem}-{suffix}" if suffix else stem
        stems[doc.doc_key] = stem
    return stems


def render_document(doc: RemoteDoc, *, source_name: str) -> str:
    """把远端文档渲染成带 YAML 头部的 Markdown（头部字段可被加载器识别）。"""
    meta_lines: list[str] = []

    def add(key: str, value: Any) -> None:
        cleaned = _clean_line(value)
        if cleaned:
            meta_lines.append(f"{key}: {cleaned}")

    add("title", doc.title or doc.external_id)
    add("date", _date_of(doc.updated_at))
    add("author", doc.author)
    add("source", source_name)
    add("source_id", doc.external_id)
    add("source_container", doc.container)
    add("source_url", doc.url)
    add("updated_at", doc.updated_at)

    body = (doc.body or "").strip() or f"# {_clean_line(doc.title) or doc.external_id}"
    if doc.comments:
        body += "\n\n## 评论\n\n" + "\n".join(f"- {comment}" for comment in doc.comments)
    return "---\n" + "\n".join(meta_lines) + "\n---\n\n" + body + "\n"


# --------------------------------------------------------------------------- #
# 同步
# --------------------------------------------------------------------------- #

def sync_source(
    source: RemoteSource,
    *,
    data_dir: Path | str,
    target: str,
    subdir: str = "",
    full: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> SyncResult:
    """把一个外部来源同步到 ``<data_dir>/sources/<target>/<subdir>``。

    * ``subdir``  —— 可选二级目录（如按知识库 space_id 分目录存放）；
    * ``full``    —— 忽略水位，强制重新拉取全部正文；
    * ``limit``   —— 只处理前 N 篇（预览/调试；同时禁用删除清理）；
    * ``dry_run`` —— 只列出将要写入的文档，不落盘、不更新水位。
    """
    if target not in TEXT_TARGETS:
        raise SyncTargetError(
            f"不支持的目标目录：{target}（可选 {'/'.join(sorted(TEXT_TARGETS))}）"
        )

    data_dir = Path(data_dir)
    root = data_dir / "sources" / target
    prefix = target
    if subdir:
        root = root / subdir
        prefix = f"{target}/{subdir}"

    state = load_state(data_dir)
    sources = state.setdefault("sources", {})
    bucket = sources.setdefault(source.name, {})
    targets = bucket.setdefault("targets", {})
    seen_docs = targets.setdefault(prefix, {})

    listed = source.list_documents()
    if limit is not None:
        listed = listed[:limit]
    stems = assign_stems(listed)

    result = SyncResult(source=source.name, target=target, listed=len(listed), dry_run=dry_run)
    current_keys: set[str] = set()

    for remote in listed:
        doc_key = remote.doc_key
        stem = stems[doc_key]
        doc_id = f"{prefix}/{stem}"
        path = root / f"{stem}.md"
        current_keys.add(doc_key)

        previous = seen_docs.get(doc_key)
        unchanged = (
            not full
            and isinstance(previous, dict)
            and previous.get("updated_at") == remote.updated_at
            and path.is_file()
        )
        if unchanged:
            result.skipped.append(doc_id)
            continue

        if dry_run:
            result.written.append(doc_id)
            continue

        fetched = source.fetch_document(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_document(fetched, source_name=source.name), encoding="utf-8")
        seen_docs[doc_key] = {
            "updated_at": remote.updated_at,
            "path": str(path),
            "title": fetched.title or remote.title,
            "url": fetched.url or remote.url,
        }
        result.written.append(doc_id)

    # 清理远端已删除的文档（仅在完整列举、且非预览时进行）
    if not dry_run and limit is None:
        for doc_key in list(seen_docs.keys()):
            if doc_key in current_keys:
                continue
            stale = Path(str(seen_docs[doc_key].get("path", "")))
            if stale.is_file():
                stale.unlink()
            seen_docs.pop(doc_key, None)
            result.removed.append(doc_key)

    if not dry_run:
        bucket["last_sync"] = _now()
        bucket["container"] = getattr(source, "space_id", None) or getattr(source, "folder_token", None) or ""
        result.state_path = str(save_state(data_dir, state))

    logger.info(
        "同步完成：source=%s target=%s listed=%d written=%d skipped=%d removed=%d",
        source.name,
        target,
        result.listed,
        len(result.written),
        len(result.skipped),
        len(result.removed),
    )
    return result