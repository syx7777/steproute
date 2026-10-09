"""语料加载、切分与检索测试。"""

from __future__ import annotations

from pathlib import Path

from steproute.corpus.loaders import chunk_text, load_corpus
from steproute.corpus.model import SourceKind
from steproute.corpus.retriever import Retriever, tokenize


def test_load_corpus_reads_all_sources(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    stats = corpus.stats()
    assert stats["documents"] >= 10
    assert stats["experiments"] == 4
    assert stats["by_kind"]["weekly_report"] >= 3
    assert stats["by_kind"]["meeting_minutes"] >= 2
    assert stats["todos"] >= 5


def test_experiments_are_structured(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    doc = corpus.experiment("EXP-001")
    assert doc is not None
    assert doc.kind is SourceKind.EXPERIMENT
    assert doc.meta["config"]["frames"] == 32
    assert doc.meta["metrics"]["vqa_acc"] == 73.8


def test_front_matter_parsed(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    doc = corpus.get("weekly_reports/2025-09-12_weekly")
    assert doc is not None
    assert doc.meta["date"] == "2025-09-12"
    assert doc.title.startswith("周报")


def test_scan_todos_finds_open_items(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    items = corpus.scan_todos()
    texts = " ".join(item.text for item in items)
    # 列表项 / 显式前缀行仍被抽出
    assert "公平" in texts or "对齐" in texts
    assert any(item.text.startswith("在 32 帧") for item in items)  # "- 下一步：在 32 帧…"
    # 正文中段的半句话（以前的正则会抠出来）不再当成待办
    assert not any(item.text.startswith("先对齐配置") for item in items)
    assert not any(item.text.startswith("补一个消融实验") for item in items)


def test_scan_todos_ignores_mid_sentence_fragments(tmp_path: Path) -> None:
    """收紧后只从「列表项 / 复选框 / 显式前缀行」抽取。"""
    notes = tmp_path / "sources" / "notes"
    notes.mkdir(parents=True)
    (notes / "note.md").write_text(
        "---\ntitle: 笔记\ndate: 2026-10-05\n---\n\n"
        "这个结果还需要再核对一下，我需要在明天给出结论。\n"
        "我们会下一步统一抽帧数再重跑。\n"
        "- 下一步：统一抽帧数后重跑\n"
        "- [ ] 补充 seed=42 的复现记录\n",
        encoding="utf-8",
    )
    texts = [item.text for item in load_corpus(tmp_path).scan_todos()]
    assert not any("再核对" in text for text in texts)  # 正文中段的「需要」不再抽取
    assert not any("我们会下一步" in text for text in texts)  # 前缀不在行首 → 跳过
    assert any("统一抽帧数" in text for text in texts)  # 列表项仍抽取
    assert any("seed=42" in text for text in texts)  # 复选框仍抽取


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


def test_retriever_finds_fairness_document(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    retriever = Retriever(corpus)
    hits = retriever.search("为什么没有超过 baseline 对比不公平", top_k=5)
    assert hits
    doc_ids = [hit.chunk.doc_id for hit in hits]
    assert any("2025-09-24" in doc_id or "2025-09-12" in doc_id for doc_id in doc_ids)


def test_retriever_kind_filter(sample_corpus: Path) -> None:
    corpus = load_corpus(sample_corpus)
    retriever = Retriever(corpus)
    hits = retriever.search("实验", kind=SourceKind.EXPERIMENT, top_k=3)
    assert hits
    assert all(hit.chunk.kind is SourceKind.EXPERIMENT for hit in hits)