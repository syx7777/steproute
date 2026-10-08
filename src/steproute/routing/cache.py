"""前缀缓存追踪（Cache-aware 调度的事实来源）。

科研 Agent 在一条 trajectory 里会反复使用相同的**系统 Prompt + 工具定义 + 实验背景**。
如果某个模型已经缓存了这段前缀，继续用它就能显著降低 TTFT。
本模块按 ``(前缀指纹, 模型)`` 维度记录观测到的缓存命中情况，
为 :class:`~steproute.routing.scheduler.RuntimeScheduler` 提供打分依据。

真实数据来源：OpenAI 兼容响应里的 ``usage.prompt_tokens_details.cached_tokens``
（DeepSeek 会返回 ``prompt_cache_hit_tokens``）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def prefix_fingerprint(stable_prefix: str) -> str:
    """计算稳定前缀的短指纹。"""
    return hashlib.sha1(stable_prefix.encode("utf-8")).hexdigest()[:12]


@dataclass
class CacheStat:
    """某个模型在某段前缀上的缓存观测。"""

    calls: int = 0
    hits: int = 0
    observed_prompt_tokens: int = 0
    observed_cached_tokens: int = 0
    ema_ratio: float = 0.0

    def observe(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.calls += 1
        self.observed_prompt_tokens = prompt_tokens
        self.observed_cached_tokens = cached_tokens
        if cached_tokens > 0:
            self.hits += 1
        ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
        ratio = min(1.0, max(0.0, ratio))
        self.ema_ratio = ratio if self.calls == 1 else 0.6 * self.ema_ratio + 0.4 * ratio


@dataclass
class PrefixCacheTracker:
    """按 ``(前缀指纹, 模型 id)`` 追踪缓存命中率。"""

    _stats: dict[tuple[str, str], CacheStat] = field(default_factory=dict)

    def observe(self, prefix_key: str, model_id: str, *, prompt_tokens: int, cached_tokens: int) -> None:
        stat = self._stats.setdefault((prefix_key, model_id), CacheStat())
        stat.observe(prompt_tokens, cached_tokens)

    def ratio(self, prefix_key: str, model_id: str) -> float:
        """估计该模型继续承接这段前缀时的缓存命中率（0~1）。"""
        stat = self._stats.get((prefix_key, model_id))
        return stat.ema_ratio if stat else 0.0

    def cached_tokens_estimate(self, prefix_key: str, model_id: str) -> int:
        stat = self._stats.get((prefix_key, model_id))
        return stat.observed_cached_tokens if stat else 0

    def warm_models(self, prefix_key: str) -> list[str]:
        """已经在这段前缀上产生过缓存命中的模型。"""
        return [model_id for (key, model_id), stat in self._stats.items() if key == prefix_key and stat.hits > 0]

    def snapshot(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (prefix_key, model_id), stat in sorted(self._stats.items()):
            rows.append(
                {
                    "prefix": prefix_key,
                    "model_id": model_id,
                    "calls": stat.calls,
                    "hits": stat.hits,
                    "cache_hit_ratio": round(stat.ema_ratio, 4),
                    "cached_tokens": stat.observed_cached_tokens,
                }
            )
        return rows