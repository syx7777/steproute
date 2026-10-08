"""Jev：判断"当前 Agent 步骤最低需要哪种能力"。

三级模式
--------
* ``heuristic`` —— 纯特征加权打分，零额外调用，最快；
* ``llm``       —— 用最便宜的模型对当前状态做一次结构化判断；
* ``hybrid``    —— 启发式打分先跑，只有难度分落在**阈值临界区**时才调用 LLM 兜底。

约束与升级
----------
* 每个步骤类型有**能力下限**（硬约束），例如"实验分析"最低要 balanced/strong；
* Failure-aware：连续工具失败或无进展达到阈值时自动升级能力档
  （对应"小模型连续出错就换强模型重新分析"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent.state import AgentState
from ..config import JevConfig, tier_rank
from ..llm.backends import Backend, BackendError
from ..llm.json_utils import JSONExtractError, extract_json_block
from ..llm.pool import ModelPool
from ..logging_utils import get_logger
from .features import StepFeatures

logger = get_logger(__name__)

_TIER_BY_RANK = {0: "fast", 1: "balanced", 2: "strong"}

JUDGE_SYSTEM_PROMPT = """你是「模型能力路由器」。给定一个多步科研 Agent 当前步骤的状态，
判断完成这一步**最低**需要哪个能力档的模型：

- fast     ：单一、机械的信息处理（检索、定位、抽取字段、读取日志中的数值）
- balanced ：需要整合多处信息、做常规比较或归纳
- strong   ：需要在信息不完整 / 相互冲突时做推理与因果判断（如"为什么没超过 baseline"）

注意：**输入长度不等于难度**。从很长的纪要里找一个实验名很简单；
而判断两个接近的实验为什么有差异，即使输入很短也可能需要 strong。

只输出一个 JSON 对象，不要输出其它内容：
{"tier": "fast|balanced|strong", "confidence": 0.0~1.0, "reason": "一句话理由"}
"""


@dataclass
class JevDecision:
    """Jev 的能力判断结果。"""

    tier: str
    difficulty: float
    confidence: float
    mode: str
    reason: str
    step_type: str
    base_tier: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    llm_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "base_tier": self.base_tier,
            "difficulty": round(self.difficulty, 4),
            "confidence": round(self.confidence, 3),
            "mode": self.mode,
            "reason": self.reason,
            "step_type": self.step_type,
            "llm_tier": self.llm_tier,
            "features": self.features,
        }


class JevRouter:
    """能力判断器。"""

    def __init__(
        self,
        config: JevConfig,
        pool: ModelPool,
        *,
        backend: Backend | None = None,
        use_llm_judge: bool = True,
    ) -> None:
        self.config = config
        self.pool = pool
        self.backend = backend
        self.use_llm_judge = use_llm_judge
        self.judge_calls = 0
        self.judge_cost_usd = 0.0

    # -- 主入口 ------------------------------------------------------------ #
    def route(self, state: AgentState, features: StepFeatures) -> JevDecision:
        difficulty, contributions = self._score(features)
        tier, confidence, reason = self._tier_from_difficulty(difficulty, features.step_type)

        mode = "heuristic"
        llm_tier: str | None = None

        if self._should_ask_llm(difficulty, tier):
            judged = self._ask_llm(state, features, tier)
            if judged is not None:
                llm_tier = str(judged.get("tier", "")).lower()
                if llm_tier in _TIER_BY_RANK.values():
                    mode = "hybrid-llm" if self.config.mode == "hybrid" else "llm"
                    if tier_rank(llm_tier) > tier_rank(tier):
                        reason = f"LLM 判定需要 {llm_tier}（{judged.get('reason', '')}）；启发式判定 {tier}"
                        tier = llm_tier
                        confidence = float(judged.get("confidence", confidence) or confidence)
                    else:
                        reason = f"{reason}；LLM 复核确认 {llm_tier}，不高于启发式判断"
                else:
                    logger.warning("Jev LLM 返回了非法 tier: %r", llm_tier)

        # 能力下限（硬约束）
        floor = self.config.step_classifier.min_tier.get(features.step_type, "fast")
        if tier_rank(tier) < tier_rank(floor):
            reason = f"{reason}；步骤类型 {features.step_type} 的能力下限为 {floor}"
            tier = floor

        # Failure-aware 升级
        base_tier = tier
        tier, escalated_reason = self._apply_escalation(tier, state)
        if escalated_reason:
            reason = f"{reason}；{escalated_reason}"

        return JevDecision(
            tier=tier,
            base_tier=base_tier,
            difficulty=difficulty,
            confidence=confidence,
            mode=mode,
            reason=reason,
            step_type=features.step_type,
            features={**features.to_dict(), "contributions": contributions},
            llm_tier=llm_tier,
        )

    # -- 启发式打分 -------------------------------------------------------- #
    def _score(self, features: StepFeatures) -> tuple[float, dict[str, float]]:
        base = self.config.step_classifier.base_difficulty.get(features.step_type, 0.30)
        normalized = features.normalized()
        contributions: dict[str, float] = {}
        total = base
        for key, value in normalized.items():
            weight = self.config.feature_weights.get(key, 0.0)
            contribution = weight * value
            contributions[key] = round(contribution, 4)
            total += contribution
        contributions["_base"] = round(base, 4)
        return max(0.0, min(1.0, total)), contributions

    def _tier_from_difficulty(self, difficulty: float, step_type: str) -> tuple[str, float, str]:
        t0, t1 = self.config.tier_thresholds
        if difficulty < t0:
            tier, nearest = "fast", t0
        elif difficulty < t1:
            tier, nearest = "balanced", (t0 if difficulty - t0 < t1 - difficulty else t1)
        else:
            tier, nearest = "strong", t1
        distance = abs(difficulty - nearest)
        confidence = max(0.05, min(1.0, distance / 0.2))
        reason = f"{step_type} 步骤难度={difficulty:.2f} → {tier}"
        return tier, confidence, reason

    def _should_ask_llm(self, difficulty: float, heuristic_tier: str) -> bool:
        if not self.use_llm_judge or self.backend is None:
            return False
        if self.config.mode == "llm":
            return True
        if self.config.mode == "heuristic":
            return False
        t0, t1 = self.config.tier_thresholds
        band = self.config.hybrid_band
        near_boundary = abs(difficulty - t0) <= band or abs(difficulty - t1) <= band
        # 启发式落在最高档时无需再问
        return near_boundary and heuristic_tier != "strong"

    def _apply_escalation(self, tier: str, state: AgentState) -> tuple[str, str]:
        cfg = self.config.escalation
        if not cfg.enabled:
            return tier, ""
        reasons: list[str] = []
        if state.consecutive_failures >= cfg.consecutive_failures and state.consecutive_failures > 0:
            reasons.append(f"连续失败 {state.consecutive_failures} 次")
        if state.no_progress_streak >= cfg.no_progress_streak and state.no_progress_streak > 0:
            reasons.append(f"连续 {state.no_progress_streak} 步无进展")
        if not reasons:
            return tier, ""
        bumped = min(2, tier_rank(tier) + max(1, cfg.max_bump))
        new_tier = _TIER_BY_RANK[bumped]
        if new_tier == tier:
            return tier, "；".join(reasons) + "（已在最高档）"
        return new_tier, "；".join(reasons) + f" → 升级到 {new_tier}"

    # -- LLM 兜底 ---------------------------------------------------------- #
    def _ask_llm(self, state: AgentState, features: StepFeatures, heuristic_tier: str) -> dict[str, Any] | None:
        spec = self._judge_spec()
        runtime = self.pool.runtime(spec.id)
        summary = {
            "task": state.task,
            "predicted_step_type": features.step_type,
            "step_index": state.step_index,
            "open_subgoals": [g.text for g in state.open_subgoals],
            "evidence_count": len(state.evidence),
            "distinct_sources": state.distinct_evidence_sources(),
            "consecutive_failures": state.consecutive_failures,
            "no_progress_streak": state.no_progress_streak,
            "flags": state.flags,
            "heuristic_tier": heuristic_tier,
        }
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"当前步骤状态：\n{summary}\n\n请给出最低所需能力档。"},
        ]
        try:
            from .scheduler import RuntimeScheduler  # 延迟导入避免循环

            RuntimeScheduler.acquire(runtime)
            try:
                result = self.backend.complete(  # type: ignore[union-attr]
                    spec,
                    messages,
                    max_tokens=self.config.llm_judge_max_tokens,
                    json_mode=True,
                    label="jev-judge",
                )
            finally:
                RuntimeScheduler.release(runtime)
            runtime.observe(result.usage, result.latency_s, ok=True)
            self.judge_calls += 1
            self.judge_cost_usd += runtime.cost_of(result.usage)
            payload = extract_json_block(result.content)
            return payload if isinstance(payload, dict) else None
        except (BackendError, JSONExtractError) as exc:
            logger.warning("Jev LLM 兜底失败，回退启发式：%s", exc)
            return None

    def _judge_spec(self):
        options = self.pool.by_tier(self.config.llm_judge_tier)
        if options:
            return options[0].spec
        return min(self.pool.all, key=lambda rt: rt.spec.price_prompt_per_m + rt.spec.price_completion_per_m).spec

    # -- 指标 -------------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        return {"judge_calls": self.judge_calls, "judge_cost_usd": round(self.judge_cost_usd, 6)}