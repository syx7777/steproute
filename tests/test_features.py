"""步级特征抽取测试。"""

from __future__ import annotations

from steproute.agent.state import AgentState, StepRecord, StepType
from steproute.routing.features import extract_features


def _record(index: int, *, progressed: bool, tool_ok: bool = True) -> StepRecord:
    return StepRecord(
        index=index,
        step_type="retrieve",
        required_tier="fast",
        model_id="flash-fast",
        tier="fast",
        tool="search_corpus",
        tool_ok=tool_ok,
        progressed=progressed,
    )


def test_feature_ranges_are_normalized() -> None:
    state = AgentState(task="t", max_steps=10)
    state.seed_subgoals(["a", "b"])
    features = extract_features(state, predicted_step_type=StepType.RETRIEVE, context_len=100000)
    for key, value in features.normalized().items():
        assert 0.0 <= value <= 1.0, key
    assert features.context_len == 1.0  # 超长输入被钳到 1


def test_step_type_comes_from_prediction() -> None:
    state = AgentState(task="t")
    features = extract_features(state, predicted_step_type=StepType.ANALYZE, context_len=100)
    assert features.step_type == "analyze"


def test_evidence_gap_shrinks_after_evidence() -> None:
    state = AgentState(task="t")
    state.seed_subgoals(["梳理实验记录 baseline 对比"])
    before = extract_features(state, predicted_step_type=StepType.RETRIEVE, context_len=100).evidence_gap
    state.add_evidence("梳理实验记录 baseline 对比的结论", ["doc"])
    after = extract_features(state, predicted_step_type=StepType.RETRIEVE, context_len=100).evidence_gap
    assert before > after


def test_failures_and_no_progress_accumulate() -> None:
    state = AgentState(task="t")
    state.record_step(_record(0, progressed=False, tool_ok=False))
    state.failures = 1
    state.record_step(_record(1, progressed=False, tool_ok=False))
    state.failures = 2
    state.no_progress_streak = 2
    features = extract_features(state, predicted_step_type=StepType.EXTRACT, context_len=100)
    assert features.failures > 0
    assert features.no_progress > 0
    assert features.staleness == 1.0


def test_numeric_conflict_flag_surfaces() -> None:
    state = AgentState(task="t")
    state.flags["numeric_conflict"] = True
    features = extract_features(state, predicted_step_type=StepType.COMPARE, context_len=100)
    assert features.numeric_conflict == 1.0