"""语料加载：把 ``data/sources`` 下的多来源材料读成统一的 :class:`Corpus`。

支持格式
--------
* ``.md`` / ``.txt`` —— 可选 YAML 头部（``---`` 包裹的 ``key: value``），正文按标题/空行切分；
* ``.json`` —— ``experiment_logs`` 下按"结构化实验记录"解析（含 ``config`` / ``metrics``），
  其它目录按通用 JSON 解析。

来源类型由**一级子目录名**决定（``weekly_reports`` / ``meeting_minutes`` /
``experiment_logs`` / ``repo`` / ``notes``）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..logging_utils import get_logger
from .model import Chunk, Corpus, Document, SourceKind

logger = get_logger(__name__)

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_JSON_SUFFIXES = {".json", ".jsonl"}
_MAX_CHUNK_CHARS = 700


# --------------------------------------------------------------------------- #
# 头部解析
# --------------------------------------------------------------------------- #

def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """解析可选的 ``---`` YAML 头部（只支持简单的 ``key: value``）。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n")
    meta: dict[str, object] = {}
    for index in range(1, len(parts)):
        line = parts[index].strip()
        if line == "---":
            return meta, "\n".join(parts[index + 1 :]).lstrip("\n")
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return {}, text  # 头部未闭合则视为普通正文


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #

def chunk_text(text: str, *, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """按标题与空行切分，并按 ``max_chars`` 合并 / 拆散，保证片段语义完整。"""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        elif not line.strip() and current:
            blocks.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    chunks: list[str] = []
    buffer = ""
    for block in blocks:
        if not block:
            continue
        if len(block) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for start in range(0, len(block), max_chars):
                chunks.append(block[start : start + max_chars])
        elif len(buffer) + len(block) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{block}".strip()
        else:
            chunks.append(buffer)
            buffer = block
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# 文档构造
# --------------------------------------------------------------------------- #

def _doc_id_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return "/".join(relative.parts)


def _make_text_document(path: Path, root: Path, kind: SourceKind) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(raw)
    title = str(meta.get("title") or _first_heading(body) or path.stem)
    return Document(
        doc_id=_doc_id_from_path(path, root),
        title=title,
        kind=kind,
        text=body.strip(),
        path=str(path),
        meta=meta,
    )


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _make_experiment_document(path: Path, root: Path) -> Document:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = str(payload.get("experiment_id") or path.stem)
    config: dict[str, object] = payload.get("config") or {}
    metrics: dict[str, object] = payload.get("metrics") or {}
    notes = str(payload.get("notes") or "").strip()

    lines = [f"# 实验 {experiment_id}: {payload.get('name', experiment_id)}", ""]
    lines.append(f"- 日期: {payload.get('date', '')}")
    lines.append(f"- 状态: {payload.get('status', '')}")
    lines.append("")
    lines.append("## 配置")
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 指标")
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if notes:
        lines += ["", "## 备注", notes]

    return Document(
        doc_id=_doc_id_from_path(path, root),
        title=f"实验 {experiment_id}: {payload.get('name', experiment_id)}",
        kind=SourceKind.EXPERIMENT,
        text="\n".join(lines),
        path=str(path),
        meta={
            "experiment_id": experiment_id,
            "name": payload.get("name", experiment_id),
            "date": payload.get("date", ""),
            "status": payload.get("status", ""),
            "config": config,
            "metrics": metrics,
            "notes": notes,
            "author": payload.get("author", ""),
        },
    )


def _make_generic_json_document(path: Path, root: Path, kind: SourceKind) -> Document:
    payload = json.loads(path.read_text(encoding="utf-8"))
    title = str(payload.get("title") or path.stem) if isinstance(payload, dict) else path.stem
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        text = payload["content"]
    else:
        text = "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"
    meta = {k: v for k, v in payload.items() if k not in {"content"}} if isinstance(payload, dict) else {}
    return Document(
        doc_id=_doc_id_from_path(path, root),
        title=title,
        kind=kind,
        text=text,
        path=str(path),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def load_corpus(data_dir: Path | str) -> Corpus:
    """加载 ``<data_dir>/sources`` 下的全部材料。"""
    root = Path(data_dir) / "sources"
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        relative = path.relative_to(root)
        kind = SourceKind.from_dir(relative.parts[0] if len(relative.parts) > 1 else "notes")
        try:
            if suffix in _JSON_SUFFIXES:
                if kind is SourceKind.EXPERIMENT:
                    documents.append(_make_experiment_document(path, root))
                else:
                    documents.append(_make_generic_json_document(path, root, kind))
            elif suffix in _TEXT_SUFFIXES:
                documents.append(_make_text_document(path, root, kind))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("跳过无法解析的文件 %s：%s", path, exc)

    chunks: list[Chunk] = []
    for doc in documents:
        for index, piece in enumerate(chunk_text(doc.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    kind=doc.kind,
                    text=piece,
                    index=index,
                )
            )

    corpus = Corpus(documents, chunks)
    logger.info("语料加载完成：%s", corpus.stats())
    return corpus