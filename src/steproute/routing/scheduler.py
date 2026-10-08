"""Runtime Scheduler：在满足能力下限的候选模型中选"运行时最合适"的那个。

与 Jev 的分工
-------------
* Jev 只回答"这一步**最低**需要什么能力"（语义层面的能力下限）；
* Scheduler 承认"语义上最合适的模型不一定运行时最合适"：
  强模型可能正在排队，而稍弱的档位已能满足要求。
  因此它在候选集合内综合 **预估延迟、GPU 队列（在途请求）、前缀缓存命中、成本** 打分量：

      score = w_latency·latency + w_cost·cost + w_queue·queue − w_cache·cache_hit

  分数最低者即为本次实际调用的模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SchedulerConfig, tier_rank
from ..llm.pool import ModelPool, ModelRuntime
from ..llm.schema import Usage
from .cache import PrefixCacheTracker


@dataclass
class CandidateScore:
    """候选模型的打分明细（可观测性 / 可解释性）。"""

    model_id: str
    tier: str
    est_latency_s: float
    est_cost_usd: float
    queue: int
    cache_hit_ratio: float
    score: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model_id": self.model_id,
            "tier": self.tier,
            "est_latency_s": round(self.est_latency_s, 3),
            "est_cost_usd": round(self.est_cost_usd, 6),
            "queue": self.queue,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "score": round(self.score, 4),
        }


@dataclass
class ScheduleDecision:
    """一次调度决策。"""

    model_id: str
    tier: str
    reason: str
    required_tier: str
    est_latency_s: float
    est_cost_usd: float
    score: float
    inflight: int
    cache_hit_ratio: float
    candidates: list[CandidateScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "tier": self.tier,
            "required_tier": self.required_tier,
            "reason": self.reason,
            "inflight": self.inflight,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class RuntimeScheduler:
    """运行时调度器。"""

    def __init__(self, config: SchedulerConfig, pool: ModelPool, cache: PrefixCacheTracker) -> None:
        self.config = config
        self.pool = pool
        self.cache = cache

    # -- 打分 -------------------------------------------------------------- #
    def _estimate(self, runtime: ModelRuntime, *, prompt_tokens: int, completion_tokens: int, cache_hit: float) -> CandidateScore:
        cfg = self.config
        latency = runtime.est_latency_s(prompt_tokens, completion_tokens)
        latency += cfg.queue_penalty_s * runtime.inflight
        latency -= cfg.cache_bonus_s * cache_hit
        latency = max(0.05, latency)
        cost = runtime.cost_of(Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
        score = (
            cfg.w_latency * latency
            + cfg.w_cost * cfg.cost_scale * cost
            + cfg.w_queue * runtime.inflight
            - cfg.w_cache * cache_hit
        )
        return CandidateScore(
            model_id=runtime.id,
            tier=runtime.tier,
            est_latency_s=latency,
            est_cost_usd=cost,
            queue=runtime.inflight,
            cache_hit_ratio=cache_hit,
            score=score,
        )

    def schedule(
        self,
        required_tier: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        prefix_key: str = "",
    ) -> ScheduleDecision:
        """为一次调用选择模型。"""
        candidates = self.pool.candidates(required_tier)
        cfg = self.config
        if not cfg.allow_stronger:
            exact = [rt for rt in candidates if rt.tier == required_tier]
            candidates = exact or candidates
        elif cfg.freeze_strong_unless_required and tier_rank(required_tier) < tier_rank("strong"):
            # 只有确实需要 strong 时才允许把 strong 纳入候选，避免无谓升级
            trimmed = [rt for rt in candidates if rt.tier != "strong"]
            candidates = trimmed or candidates

        if not candidates:  # pragma: no cover - 池至少一个模型
            candidates = self.pool.all

        scored: list[CandidateScore] = []
        for runtime in candidates:
            hit = 0.0
            if runtime.spec.supports_cache and prefix_key:
                hit = self.cache.ratio(prefix_key, runtime.id)
            scored.append(
                self._estimate(runtime, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cache_hit=hit)
            )

        best = min(scored, key=lambda item: item.score)
        chosen = self.pool.runtime(best.model_id)
        reason = self._explain(best, scored, required_tier)
        return ScheduleDecision(
            model_id=best.model_id,
            tier=chosen.tier,
            reason=reason,
            required_tier=required_tier,
            est_latency_s=best.est_latency_s,
            est_cost_usd=best.est_cost_usd,
            score=best.score,
            inflight=best.queue,
            cache_hit_ratio=best.cache_hit_ratio,
            candidates=scored,
        )

    @staticmethod
    def _explain(best: CandidateScore, scored: list[CandidateScore], required_tier: str) -> str:
        parts: list[str] = []
        if best.cache_hit_ratio > 0.05:
            parts.append(f"前缀缓存命中 {best.cache_hit_ratio:.0%}")
        if best.queue > 0:
            parts.append(f"该模型在途 {best.queue} 请求")
        if len(scored) > 1:
            second = sorted(scored, key=lambda item: item.score)[1]
            parts.append(f"优于次优候选 {second.model_id}（score {second.score:.2f}→{best.score:.2f}）")
        parts.append(f"能力下限={required_tier}")
        return "；".join(parts)

    # -- 在途请求 ---------------------------------------------------------- #
    @staticmethod
    def acquire(runtime: ModelRuntime) -> None:
        runtime.inflight += 1

    @staticmethod
    def release(runtime: ModelRuntime) -> None:
        runtime.inflight = max(0, runtime.inflight - 1)