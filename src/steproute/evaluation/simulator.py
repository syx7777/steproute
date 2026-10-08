"""路由性能仿真器：量化"步级路由"相对"统一强模型 / Query 级路由"的收益。

为什么需要仿真？
----------------
真实调用（尤其强模型）又慢又贵，想跑几十条轨迹 × 多种策略 × 多次消融并不现实。
因此这里用一个**显式的难度-能力模型**来复现真实系统里观察到的现象：

* 一条 trajectory 里各步难度差异巨大（检索/抽取很轻，实验分析很重）；
* 低能力模型遇到超纲步骤时，要么**直接失败**（触发重试、抬高成本），
  要么**悄悄地把结果做差**（质量下降但不报错）——后者正是统一用小模型的隐患；
* 保持使用同一模型可以吃到**前缀缓存**，频繁切换模型则会丢失缓存、抬高 TTFT；
* 更强的模型单价更高、队列更长，因此"语义上最合适"不一定"运行时最合适"。

在这个模型下对比四种策略与三项消融，结论具备可复现性与可解释性。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..config import EvalConfig, ModelPoolConfig, ModelSpecConfig, tier_rank

#: 各能力档"够用"所需的模型能力值。
REQUIRED_CAPABILITY = {"fast": 0.40, "balanced": 0.66, "strong": 0.90}

#: 生成轨迹时的步骤类型分布（越靠后越可能遇到高难度步骤）。
_STEP_TYPE_POOL = (
    ["retrieve"] * 3 + ["read"] * 2 + ["extract"] * 4 + ["compare"] * 2 + ["analyze"] * 2 + ["synthesize"] * 1
)

#: 步骤类型 → 其难度分布（多数情形 / 偶尔升档）。
_STEP_TIER_POOL = {
    "retrieve": ["fast"] * 9 + ["balanced"],
    "read": ["fast"] * 9 + ["balanced"],
    "extract": ["fast"] * 7 + ["balanced"] * 3,
    "compare": ["balanced"] * 7 + ["strong"] * 3,
    "analyze": ["strong"] * 6 + ["balanced"] * 4,
    "synthesize": ["balanced"] * 6 + ["strong"] * 4,
}


@dataclass
class SimStep:
    """仿真中的单个步骤。"""

    index: int
    step_type: str
    required_tier: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def required_capability(self) -> float:
        return REQUIRED_CAPABILITY[self.required_tier]


@dataclass
class SimTask:
    """一条仿真轨迹。"""

    task_id: str
    task: str
    steps: list[SimStep]
    anomaly_after: int = -1  # 第几步开始出现"突然变难"（Query 级路由的盲区）

    @property
    def query_tier(self) -> str:
        """Query 级路由只能看到的最初难度（第一步的表观难度）。"""
        return self.steps[0].required_tier if self.steps else "fast"


@dataclass
class StrategyOutcome:
    """某策略在一条轨迹上的结果。"""

    task_id: str
    cost_usd: float
    latency_s: float
    quality: float
    success: bool
    retries: int
    cache_hits: int
    tier_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "quality": round(self.quality, 4),
            "success": self.success,
            "retries": self.retries,
            "cache_hits": self.cache_hits,
            "tier_counts": self.tier_counts,
        }


@dataclass
class SimConfig:
    """仿真参数（含消融开关）。"""

    failure_prob_when_underpowered: float = 0.75
    retry_cost_multiplier: float = 1.6
    max_retries: int = 3
    cache_bonus_s: float = 0.9
    quality_success_threshold: float = 0.85
    jev_accuracy: float = 0.90
    # 消融开关
    use_step_type: bool = True
    use_cache: bool = True
    use_failure_escalation: bool = True


# --------------------------------------------------------------------------- #
# 轨迹生成
# --------------------------------------------------------------------------- #

def generate_tasks(eval_config: EvalConfig, *, seed: int | None = None) -> list[SimTask]:
    """生成一组带真实难度标注的仿真轨迹。"""
    rng = random.Random(seed if seed is not None else eval_config.seed)
    tasks: list[SimTask] = []
    for index in range(eval_config.num_tasks):
        num_steps = rng.randint(eval_config.min_steps, eval_config.max_steps)
        steps: list[SimStep] = []
        for step_index in range(num_steps):
            step_type = rng.choice(_STEP_TYPE_POOL)
            # 后半段更容易突然出现高难度步骤（模拟"执行中才发现异常"）
            late = step_index >= num_steps // 2
            tier_pool = _STEP_TIER_POOL[step_type]
            required = rng.choice(tier_pool)
            if late and step_type in {"compare", "analyze", "synthesize"}:
                required = rng.choice([required, "strong"])
            steps.append(
                SimStep(
                    index=step_index,
                    step_type=step_type,
                    required_tier=required,
                    prompt_tokens=rng.randint(600, 6000),
                    completion_tokens=rng.randint(160, 900),
                )
            )
        anomaly_after = next(
            (s.index for s in steps if s.index > 0 and tier_rank(s.required_tier) > tier_rank(steps[0].required_tier)),
            -1,
        )
        tasks.append(
            SimTask(
                task_id=f"T{index + 1:03d}",
                task="整理本周科研进展并判断方法为何未超过 baseline",
                steps=steps,
                anomaly_after=anomaly_after,
            )
        )
    return tasks


# --------------------------------------------------------------------------- #
# 模型选择
# --------------------------------------------------------------------------- #

def _models_of_tier(pool: ModelPoolConfig, tier: str) -> list[ModelSpecConfig]:
    models = pool.by_tier(tier)
    if models:
        return models
    # 该档位没有模型时，退到最接近的档位
    ranked = sorted(pool.models, key=lambda m: abs(tier_rank(m.tier) - tier_rank(tier)))
    return ranked[:1]


def _pick_model(pool: ModelPoolConfig, tier: str, warm: set[str], cfg: SimConfig) -> ModelSpecConfig:
    candidates = _models_of_tier(pool, tier)
    if cfg.use_cache:
        for model in candidates:
            if model.id in warm:
                return model
    return candidates[0]


def _tier_for_length(prompt_tokens: int) -> str:
    """只用 context length 判难度（消融：不用 step type）。"""
    if prompt_tokens < 1500:
        return "fast"
    if prompt_tokens < 4000:
        return "balanced"
    return "strong"


def _choose_tier(
    task: SimTask,
    step: SimStep,
    strategy: str,
    cfg: SimConfig,
    rng: random.Random,
) -> str:
    if strategy == "always_strong":
        return "strong"
    if strategy == "always_fast":
        return "fast"
    if strategy == "query_level":
        return task.query_tier

    # ---- step_level（本文方案）----
    if not cfg.use_step_type:
        return _tier_for_length(step.prompt_tokens)
    # Jev 并非完美：以小概率把难度判低一档
    if rng.random() > cfg.jev_accuracy:
        degraded = max(0, tier_rank(step.required_tier) - 1)
        return {0: "fast", 1: "balanced", 2: "strong"}[degraded]
    return step.required_tier


# --------------------------------------------------------------------------- #
# 单条轨迹仿真
# --------------------------------------------------------------------------- #

def simulate_task(
    task: SimTask,
    strategy: str,
    pool: ModelPoolConfig,
    cfg: SimConfig,
    rng: random.Random,
) -> StrategyOutcome:
    """在给定策略下仿真一条轨迹。"""
    total_cost = 0.0
    total_latency = 0.0
    quality_sum = 0.0
    retries = 0
    cache_hits = 0
    tier_counts: dict[str, int] = {}
    warm: set[str] = set()
    aborted = False

    for step in task.steps:
        tier = _choose_tier(task, step, strategy, cfg, rng)
        attempt = 0
        step_quality = 0.0
        step_ok = False

        while attempt <= cfg.max_retries:
            model = _pick_model(pool, tier, warm, cfg)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            # 成本
            cost = (
                step.prompt_tokens / 1_000_000 * model.price_prompt_per_m
                + step.completion_tokens / 1_000_000 * model.price_completion_per_m
            )
            if attempt > 0:
                cost *= cfg.retry_cost_multiplier
            total_cost += cost

            # 延迟（含缓存抵扣）
            latency = (
                model.base_latency_s
                + step.prompt_tokens / 1000 * model.latency_per_1k_prompt_s
                + step.completion_tokens / 1000 * model.latency_per_1k_completion_s
            )
            cached = cfg.use_cache and model.id in warm
            if cached:
                latency = max(0.05, latency - cfg.cache_bonus_s)
                cache_hits += 1
            total_latency += latency
            warm.add(model.id)

            # 能力是否够用
            shortfall = max(0.0, step.required_capability - model.capability)
            if shortfall > 0:
                fail_prob = min(0.95, shortfall * 2.2)
                failed = rng.random() < fail_prob
                step_quality = min(1.0, model.capability / step.required_capability)
            else:
                failed = False
                step_quality = 1.0

            if not failed:
                step_ok = True
                break

            # 失败：重试并从失败中升级能力档（Failure-aware）
            retries += 1
            attempt += 1
            if cfg.use_failure_escalation:
                bumped = min(2, tier_rank(tier) + 1)
                tier = {0: "fast", 1: "balanced", 2: "strong"}[bumped]

        if not step_ok:
            aborted = True
            quality_sum += 0.0
            break
        quality_sum += step_quality

    steps_done = len(task.steps) if not aborted else max(1, len(task.steps) - 1)
    quality = quality_sum / steps_done if steps_done else 0.0
    success = (not aborted) and quality >= cfg.quality_success_threshold
    return StrategyOutcome(
        task_id=task.task_id,
        cost_usd=total_cost,
        latency_s=total_latency,
        quality=min(1.0, quality),
        success=success,
        retries=retries,
        cache_hits=cache_hits,
        tier_counts=tier_counts,
    )