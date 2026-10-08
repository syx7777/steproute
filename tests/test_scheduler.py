"""Runtime Scheduler 与缓存追踪测试。"""

from __future__ import annotations

from steproute.config import SchedulerConfig, default_model_pool, tier_at_least
from steproute.llm.pool import ModelPool
from steproute.routing.cache import PrefixCacheTracker, prefix_fingerprint
from steproute.routing.scheduler import RuntimeScheduler

PREFIX = "stable-system-prompt"


def _scheduler(config: SchedulerConfig | None = None) -> tuple[RuntimeScheduler, PrefixCacheTracker]:
    pool = ModelPool(default_model_pool())
    cache = PrefixCacheTracker()
    return RuntimeScheduler(config or SchedulerConfig(), pool, cache), cache


def test_scheduled_model_satisfies_required_tier() -> None:
    scheduler, _ = _scheduler()
    for required in ("fast", "balanced", "strong"):
        decision = scheduler.schedule(required, prompt_tokens=1200, completion_tokens=400, prefix_key=PREFIX)
        assert tier_at_least(decision.tier, required)


def test_strong_is_frozen_unless_required() -> None:
    scheduler, _ = _scheduler()
    decision = scheduler.schedule("fast", prompt_tokens=800, completion_tokens=300, prefix_key=PREFIX)
    assert decision.tier != "strong"


def test_cache_aware_prefers_warm_model() -> None:
    scheduler, cache = _scheduler()
    cache.observe(PREFIX, "flash-balanced", prompt_tokens=3000, cached_tokens=3000)
    decision = scheduler.schedule("fast", prompt_tokens=3000, completion_tokens=512, prefix_key=PREFIX)
    assert decision.model_id == "flash-balanced"
    assert decision.cache_hit_ratio > 0.5


def test_queue_penalty_avoids_busy_model() -> None:
    scheduler, _ = _scheduler()
    runtime = scheduler.pool.runtime("flash-fast")
    runtime.inflight = 6
    decision = scheduler.schedule("fast", prompt_tokens=1000, completion_tokens=400, prefix_key=PREFIX)
    assert decision.model_id != "flash-fast"
    runtime.inflight = 0


def test_acquire_release_tracks_inflight() -> None:
    scheduler, _ = _scheduler()
    runtime = scheduler.pool.runtime("flash-fast")
    scheduler.acquire(runtime)
    scheduler.acquire(runtime)
    assert runtime.inflight == 2
    scheduler.release(runtime)
    scheduler.release(runtime)
    scheduler.release(runtime)
    assert runtime.inflight == 0


def test_cost_scoring_prefers_cheaper_model_when_latency_close() -> None:
    config = SchedulerConfig(w_latency=0.05, w_cost=5.0, w_cache=0.0, w_queue=0.0)
    scheduler, _ = _scheduler(config)
    decision = scheduler.schedule("balanced", prompt_tokens=4000, completion_tokens=800, prefix_key="")
    # strong 被冻结，balanced/fast 中 balanced 恰好满足下限
    assert decision.tier == "balanced"


def test_decisions_are_explainable() -> None:
    scheduler, cache = _scheduler()
    cache.observe(PREFIX, "flash-fast", prompt_tokens=2000, cached_tokens=1800)
    decision = scheduler.schedule("fast", prompt_tokens=2000, completion_tokens=400, prefix_key=PREFIX)
    assert decision.candidates
    assert decision.reason
    assert decision.to_dict()["candidates"]


def test_prefix_fingerprint_is_stable() -> None:
    assert prefix_fingerprint("abc") == prefix_fingerprint("abc")
    assert prefix_fingerprint("abc") != prefix_fingerprint("abd")


def test_cache_tracker_reports_warm_models() -> None:
    cache = PrefixCacheTracker()
    cache.observe(PREFIX, "flash-fast", prompt_tokens=1000, cached_tokens=900)
    assert cache.warm_models(PREFIX) == ["flash-fast"]
    assert cache.cached_tokens_estimate(PREFIX, "flash-fast") == 900
    assert cache.ratio(PREFIX, "pro-strong") == 0.0