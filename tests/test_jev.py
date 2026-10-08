"""Jev 能力判断测试。"""

from __future__ import annotations

import pytest

from steproute.agent.state import AgentState, StepRecord, StepType
from steproute.config import JevConfig, default_model_pool, tier_rank
from steproute.llm.pool import ModelPool
from steproute.routing.features import extract_features
from steproute.routing.jev import JevRouter


@pytest.fixture()
def router() -> JevRouter:
    pool = ModelPool(default_model_pool())
    return JevRouter(JevConfig(mode="heuristic"), pool, backend=None, use_llm_judge=False)


def _decide(router: JevRouter, state: AgentState, step_type: StepType, context_len: int = 500):
    features = extract_features(state, predicted_step_type=step_type, context_len=context_len)
    return router.route(state, features)


def test_extraction_steps_are_fast(router: JevRouter) -> None:
    state = AgentState(task="抽取 TODO")
    assert _decide(router, state, StepType.EXTRACT).tier == "fast"
    assert _decide(router, state, StepType.RETRIEVE).tier == "fast"


def test_analysis_step_requires_strong(router: JevRouter) -> None:
    state = AgentState(task="为什么没超过 baseline")
    decision = _decide(router, state, StepType.ANALYZE)
    assert decision.tier == "strong"
    assert decision.base_tier == "strong"


def test_step_type_floor_applies_to_compare(router: JevRouter) -> None:
    state = AgentState(task="比较两组实验")
    assert _decide(router, state, StepType.COMPARE).tier in {"balanced", "strong"}


def test_long_context_does_not_force_strong_for_extraction(router: JevRouter) -> None:
    """关键设计：输入很长但步骤很简单时，仍然只需要 fast。"""
    state = AgentState(task="从长纪要里找一个实验名")
    decision = _decide(router, state, StepType.EXTRACT, context_len=12000)
    assert decision.tier == "fast"


def test_numeric_conflict_raises_difficulty(router: JevRouter) -> None:
    state = AgentState(task="对比")
    baseline = _decide(router, state, StepType.COMPARE).difficulty
    state.flags["numeric_conflict"] = True
    state.flags["ambiguity"] = True
    raised = _decide(router, state, StepType.COMPARE).difficulty
    assert raised > baseline


def test_failure_escalates_tier(router: JevRouter) -> None:
    state = AgentState(task="抽取")
    state.seed_subgoals(["抽取 TODO", "归纳研究进展"])
    base = _decide(router, state, StepType.EXTRACT)
    state.consecutive_failures = 2
    state.failures = 2
    escalated = _decide(router, state, StepType.EXTRACT)
    assert tier_rank(escalated.tier) > tier_rank(base.tier)
    assert escalated.base_tier == base.tier
    assert "升级" in escalated.reason


def test_no_progress_escalates_tier(router: JevRouter) -> None:
    state = AgentState(task="抽取")
    state.seed_subgoals(["抽取 TODO", "归纳研究进展"])
    base = _decide(router, state, StepType.EXTRACT)
    state.no_progress_streak = 2
    escalated = _decide(router, state, StepType.EXTRACT)
    assert tier_rank(escalated.tier) > tier_rank(base.tier)
    assert "无进展" in escalated.reason


def test_escalation_can_be_disabled() -> None:
    config = JevConfig(mode="heuristic")
    config.escalation.enabled = False
    router = JevRouter(config, ModelPool(default_model_pool()), backend=None, use_llm_judge=False)
    state = AgentState(task="抽取")
    state.seed_subgoals(["抽取 TODO", "归纳研究进展"])
    state.consecutive_failures = 5
    state.no_progress_streak = 5
    decision = _decide(router, state, StepType.EXTRACT)
    assert decision.tier == decision.base_tier
    assert "升级" not in decision.reason


def test_escalation_never_exceeds_strong(router: JevRouter) -> None:
    state = AgentState(task="分析")
    state.consecutive_failures = 9
    state.no_progress_streak = 9
    assert _decide(router, state, StepType.ANALYZE).tier == "strong"


def test_decision_payload_is_serializable(router: JevRouter) -> None:
    state = AgentState(task="t")
    payload = _decide(router, state, StepType.SYNTHESIZE).to_dict()
    assert payload["tier"] in {"fast", "balanced", "strong"}
    assert "contributions" in payload["features"]


def test_stale_trajectory_raises_difficulty(router: JevRouter) -> None:
    state = AgentState(task="t")
    fresh = _decide(router, state, StepType.EXTRACT).difficulty
    state.record_step(
        StepRecord(index=0, step_type="extract", required_tier="fast", model_id="flash-fast", tier="fast",
                   tool="search_corpus", tool_ok=True, progressed=False)
    )
    assert _decide(router, state, StepType.EXTRACT).difficulty > fresh