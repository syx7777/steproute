"""模型池：把「能力档」映射到具体可调用的模型，并维护运行时画像。

职责
----
* 持有各模型的静态画像（端点、成本、延迟斜率、并发上限、仿真能力值）；
* 记录运行时统计（在途请求数、EMA 延迟、累计成本、缓存命中 token），
  供 :class:`steproute.routing.scheduler.RuntimeScheduler` 打分使用。

设计要点：**能力档（fast/balanced/strong）与具体模型解耦**。
换成 vLLM 本地部署时，只要在 ``configs/models.yaml`` 里改 ``endpoint``/``model``，
路由与调度逻辑无需改动。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelPoolConfig, ModelSpecConfig, tier_at_least, tier_rank
from .schema import Usage


@dataclass
class ModelRuntime:
    """单个模型的运行时状态与画像。"""

    spec: ModelSpecConfig
    inflight: int = 0
    total_calls: int = 0
    total_latency_s: float = 0.0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    failures: int = 0
    ema_latency_s: float = 0.0

    # -- 画像 -------------------------------------------------------------- #
    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def tier(self) -> str:
        return self.spec.tier

    def cost_of(self, usage: Usage) -> float:
        """按 token 用量估算本次调用成本（美元）。"""
        return (
            usage.prompt_tokens / 1_000_000 * self.spec.price_prompt_per_m
            + usage.completion_tokens / 1_000_000 * self.spec.price_completion_per_m
        )

    def est_latency_s(self, prompt_tokens: int, completion_tokens: int) -> float:
        """估算一次调用的端到端延迟（不含队列与缓存）。

        * 空载基线 + 随 prompt/completion 规模增长的线性项；
        * 若该模型已积累观测延迟，则用 EMA 与理论值取加权，贴近真实表现。
        """
        theoretical = (
            self.spec.base_latency_s
            + prompt_tokens / 1000.0 * self.spec.latency_per_1k_prompt_s
            + completion_tokens / 1000.0 * self.spec.latency_per_1k_completion_s
        )
        if self.total_calls == 0:
            return theoretical
        # 观测值与理论值各占一半，避免冷启动偏差
        return 0.5 * theoretical + 0.5 * self.ema_latency_s

    def queue_factor(self) -> int:
        """当前在途请求数（用于队列惩罚）。"""
        return self.inflight

    def cache_hit_ratio(self) -> float:
        """历史前缀缓存命中率（0~1），供 Cache-aware 调度参考。"""
        if self.total_prompt_tokens <= 0:
            return 0.0
        return min(1.0, self.total_cached_tokens / self.total_prompt_tokens)

    # -- 观测 -------------------------------------------------------------- #
    def observe(self, usage: Usage, latency_s: float, *, ok: bool = True, cost: float | None = None) -> None:
        self.total_calls += 1
        self.total_latency_s += latency_s
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_cached_tokens += usage.cached_tokens
        self.total_cost_usd += self.cost_of(usage) if cost is None else cost
        alpha = 0.35
        self.ema_latency_s = latency_s if self.total_calls == 1 else (1 - alpha) * self.ema_latency_s + alpha * latency_s
        if not ok:
            self.failures += 1

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "id": self.id,
            "tier": self.tier,
            "model": self.spec.model,
            "calls": self.total_calls,
            "failures": self.failures,
            "inflight": self.inflight,
            "avg_latency_s": round(self.total_latency_s / self.total_calls, 3) if self.total_calls else 0.0,
            "ema_latency_s": round(self.ema_latency_s, 3),
            "cost_usd": round(self.total_cost_usd, 6),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cached_tokens": self.total_cached_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio(), 4),
        }


class ModelPool:
    """模型池：按能力档索引模型，并集中维护运行时画像。"""

    def __init__(self, config: ModelPoolConfig) -> None:
        self.config = config
        self._runtimes: dict[str, ModelRuntime] = {m.id: ModelRuntime(spec=m) for m in config.models}

    # -- 查询 -------------------------------------------------------------- #
    @property
    def all(self) -> list[ModelRuntime]:
        return list(self._runtimes.values())

    def runtime(self, model_id: str) -> ModelRuntime:
        if model_id not in self._runtimes:
            raise KeyError(f"模型池中不存在 id={model_id}")
        return self._runtimes[model_id]

    def by_tier(self, tier: str) -> list[ModelRuntime]:
        return [rt for rt in self._runtimes.values() if rt.tier == tier]

    def candidates(self, min_tier: str) -> list[ModelRuntime]:
        """返回能力不低于 ``min_tier`` 的所有模型（保持池中定义顺序）。"""
        return [rt for rt in self._runtimes.values() if tier_at_least(rt.tier, min_tier)]

    def strongest(self) -> ModelRuntime:
        return max(self._runtimes.values(), key=lambda rt: tier_rank(rt.tier))

    def tier_of(self, model_id: str) -> str:
        return self.runtime(model_id).tier

    # -- 指标 -------------------------------------------------------------- #
    def total_cost_usd(self) -> float:
        return sum(rt.total_cost_usd for rt in self._runtimes.values())

    def total_calls(self) -> int:
        return sum(rt.total_calls for rt in self._runtimes.values())

    def tier_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for rt in self._runtimes.values():
            dist[rt.tier] = dist.get(rt.tier, 0) + rt.total_calls
        return dist

    def snapshot(self) -> dict[str, object]:
        return {
            "models": [rt.snapshot() for rt in self._runtimes.values()],
            "total_calls": self.total_calls(),
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "tier_distribution": self.tier_distribution(),
        }