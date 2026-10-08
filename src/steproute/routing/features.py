"""步级特征抽取：把 :class:`AgentState` 压成一组 0~1 的难度信号。

这是"路由依据 Agent State 而非 context length"的落地环节：
context_len 只是一个权重很小的特征，真正驱动判断的是
**步骤类型、失败次数、无进展步数、子目标缺口、数值冲突** 等。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..agent.state import AgentState, StepType

#: 各特征的归一化上限（超过即视为"满格"）。
_FAILURE_SATURATION = 3
_NO_PROGRESS_SATURATION = 3
_DEPTH_SATURATION = 14
_CONTEXT_SATURATION = 8000
_SOURCE_SATURATION = 3


@dataclass
class StepFeatures:
    """单个步骤的特征向量。"""

    step_type: str
    context_len: float
    depth: float
    failures: float
    no_progress: float
    evidence_gap: float
    multi_source: float
    numeric_conflict: float
    ambiguity: float
    staleness: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def normalized(self) -> dict[str, float]:
        data = asdict(self)
        data.pop("step_type", None)
        return {k: float(v) for k, v in data.items()}


def extract_features(
    state: AgentState,
    *,
    predicted_step_type: StepType | str | None = None,
    context_len: int | None = None,
) -> StepFeatures:
    """从状态抽取特征。

    ``predicted_step_type``：Harness 对"本轮要做哪种步骤"的预测
    （来自上一步模型声明的 ``next_step_type``，首个步骤默认 ``retrieve``）。
    """
    step_type = StepType.parse(
        predicted_step_type.value if isinstance(predicted_step_type, StepType) else predicted_step_type,
        state.pending_step_type,
    )
    subgoals = state.subgoals or []
    uncovered = [g for g in subgoals if not g.evidence_ids]
    evidence_gap = (len(uncovered) / len(subgoals)) if subgoals else 1.0

    last_progressed = state.history[-1].progressed if state.history else True
    staleness = 0.0 if last_progressed else 1.0

    ambiguity = 1.0 if state.flags.get("ambiguity") else 0.0

    return StepFeatures(
        step_type=step_type.value,
        context_len=min(1.0, (context_len or state.prompt_tokens_last) / _CONTEXT_SATURATION),
        depth=min(1.0, state.step_index / _DEPTH_SATURATION),
        failures=min(1.0, max(state.failures, state.consecutive_failures) / _FAILURE_SATURATION),
        no_progress=min(1.0, state.no_progress_streak / _NO_PROGRESS_SATURATION),
        evidence_gap=evidence_gap,
        multi_source=min(1.0, state.distinct_evidence_sources() / _SOURCE_SATURATION),
        numeric_conflict=1.0 if state.flags.get("numeric_conflict") else 0.0,
        ambiguity=ambiguity,
        staleness=staleness,
    )