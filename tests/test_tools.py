"""工具执行测试（含公平性检测）。"""

from __future__ import annotations

from steproute.agent.state import AgentState
from steproute.agent.tools import TOOL_SPECS, ToolExecutor


def test_tool_catalog_contains_core_tools() -> None:
    names = {spec.name for spec in TOOL_SPECS}
    assert {"search_corpus", "read_document", "compare_experiments", "finalize"} <= names


def test_search_then_read(bundle) -> None:
    state = AgentState(task="整理进展")
    search = bundle.executor.execute(state, "search_corpus", {"query": "baseline 对比"})
    assert search.ok and search.progressed
    assert state.retrieved_ids

    read = bundle.executor.execute(state, "read_document", {"doc_id": state.retrieved_ids[0]})
    assert read.ok
    assert state.read_ids


def test_read_unknown_document_fails() -> None:
    state = AgentState(task="t")
    # 无 bundle 也可构造一个最小执行器
    from steproute.corpus.loaders import load_corpus
    from steproute.corpus.retriever import Retriever

    corpus = load_corpus("data")
    executor = ToolExecutor(corpus, Retriever(corpus))
    result = executor.execute(state, "read_document", {"doc_id": "not-exist"})
    assert not result.ok
    assert state.failures == 0  # 失败计数由运行时统一维护
    assert "找不到文档" in result.output


def test_compare_detects_unfair_comparison(bundle) -> None:
    state = AgentState(task="t")
    result = bundle.executor.execute(
        state, "compare_experiments", {"experiment_a": "EXP-001", "experiment_b": "EXP-002"}
    )
    assert result.ok
    assert "配置不一致" in result.output
    assert result.patch["flags"].get("numeric_conflict") is True
    # 结论应当提示先对齐配置
    assert "对齐配置" in result.output


def test_compare_aligned_configs_is_fair(bundle) -> None:
    state = AgentState(task="t")
    result = bundle.executor.execute(
        state, "compare_experiments", {"experiment_a": "EXP-001", "experiment_b": "EXP-003"}
    )
    assert result.ok
    assert "配置一致" in result.output
    assert not result.patch["flags"]


def test_list_experiments_and_metrics(bundle) -> None:
    state = AgentState(task="t")
    listed = bundle.executor.execute(state, "list_experiments", {})
    assert listed.ok
    assert set(state.experiment_ids) >= {"EXP-001", "EXP-002", "EXP-003", "EXP-004"}

    metrics = bundle.executor.execute(state, "get_experiment_metrics", {"experiment_id": "EXP-003"})
    assert metrics.ok
    assert "74.6" in metrics.output


def test_extract_todos_populates_state(bundle) -> None:
    state = AgentState(task="t")
    result = bundle.executor.execute(state, "extract_todos", {})
    assert result.ok
    assert state.todos
    assert result.progressed


def test_record_finding_creates_evidence(bundle) -> None:
    state = AgentState(task="t")
    state.seed_subgoals(["梳理实验记录", "归纳研究进展"])
    result = bundle.executor.execute(
        state, "record_finding", {"claim": "对齐配置后方法超过 baseline", "source_ids": ["experiment_logs/EXP-003"]}
    )
    assert result.ok
    assert state.evidence and state.evidence[0].id == "e1"


def test_finalize_sets_state(bundle) -> None:
    state = AgentState(task="t")
    state.todos = ["统一评测配置"]
    result = bundle.executor.execute(state, "finalize", {"report": "", "next_tasks": []})
    assert result.ok
    assert state.finished
    assert state.next_tasks == ["统一评测配置"]


def test_unknown_tool_is_reported(bundle) -> None:
    state = AgentState(task="t")
    result = bundle.executor.execute(state, "does_not_exist", {})
    assert not result.ok
    assert "未知工具" in result.output