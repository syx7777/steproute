"""语料加载、切分与检索测试。"""

from __future__ import annotations

from pathlib import Path

from steproute.corpus.loaders import chunk_text, load_corpus
from steproute.corpus.model import SourceKind
from steproute.corpus.retriever import Retriever, tokenize


def test_load_corpus_reads_all_sources(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    stats = corpus.stats()
    assert stats["documents"] >= 10
    assert stats["experiments"] == 4
    assert stats["by_kind"]["weekly_report"] >= 3
    assert stats["by_kind"]["meeting_minutes"] >= 2
    assert stats["todos"] >= 5


def test_experiments_are_structured(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    doc = corpus.experiment("EXP-001")
    assert doc is not None
    assert doc.kind is SourceKind.EXPERIMENT
    assert doc.meta["config"]["frames"] == 32
    assert doc.meta["metrics"]["vqa_acc"] == 73.8


def test_front_matter_parsed(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    doc = corpus.get("weekly_reports/2025-09-12_weekly")
    assert doc is not None
    assert doc.meta["date"] == "2025-09-12"
    assert doc.title.startswith("周报")


def test_scan_todos_finds_open_items(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    texts = " ".join(item.text for item in corpus.scan_todos())
    assert "公平" in texts or "对齐" in texts


def test_load_corpus_missing_dir_raises(tmp_path: Path) -> None:
    try:
        load_corpus(tmp_path / "nope")
    except FileNotFoundError:
        return
    raise AssertionError("应当抛出 FileNotFoundError")


def test_chunk_text_splits_long_block() -> None:
    text = "# A\n\n" + ("x" * 1500)
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) >= 5
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_tokenize_handles_cjk_bigrams() -> None:
    tokens = tokenize("实验对比 baseline")
    assert "baseline" in tokens
    assert any(len(token) == 2 and "\u4e00" <= token[0] <= "\u9fff" for token in tokens)


def test_retriever_finds_fairness_document(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    retriever = Retriever(corpus)
    hits = retriever.search("为什么没有超过 baseline 对比不公平", top_k=5)
    assert hits
    doc_ids = [hit.chunk.doc_id for hit in hits]
    assert any("2025-09-24" in doc_id or "2025-09-12" in doc_id for doc_id in doc_ids)


def test_retriever_kind_filter(project_root: Path) -> None:
    corpus = load_corpus(project_root / "data")
    retriever = Retriever(corpus)
    hits = retriever.search("实验", kind=SourceKind.EXPERIMENT, top_k=3)
    assert hits
    assert all(hit.chunk.kind is SourceKind.EXPERIMENT for hit in hits)