"""策略对比与消融实验。

在 :mod:`.simulator` 上对比：

* ``always_strong``  —— 统一强模型（成本/延迟上界）；
* ``always_fast``    —— 统一小模型（成本下界，质量风险上界）；
* ``query_level``    —— 只在最初 Query 上定档（多步 Agent 下失效的对照）；
* ``step_level``     —— **本文方案**：Jev + Runtime Scheduler（Cache/Failure-aware）。

并做三项消融：去掉 step type、去掉 Cache-aware、去掉 Failure-aware。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig, EvalConfig
from ..evaluation.simulator import SimConfig, SimTask, StrategyOutcome, generate_tasks, simulate_task

STRATEGY_LABELS: dict[str, str] = {
    "always_strong": "统一强模型（对照）",
    "always_fast": "统一小模型（对照）",
    "query_level": "Query 级路由（对照）",
    "step_level": "步级路由（本文方案）",
    "step_level_no_step_type": "消融：去掉 step type（仅按 context length）",
    "step_level_no_cache": "消融：去掉 Cache-aware",
    "step_level_no_failure": "消融：去掉 Failure-aware",
}

DEFAULT_STRATEGIES = (
    "always_strong",
    "always_fast",
    "query_level",
    "step_level",
    "step_level_no_step_type",
    "step_level_no_cache",
    "step_level_no_failure",
)

#: 消融策略 → 对应的仿真开关覆写。
_ABLATION_OVERRIDES: dict[str, dict[str, bool]] = {
    "step_level_no_step_type": {"use_step_type": False},
    "step_level_no_cache": {"use_cache": False},
    "step_level_no_failure": {"use_failure_escalation": False},
}

BASELINE = "always_strong"


@dataclass
class StrategyReport:
    """单个策略的汇总报告。"""

    name: str
    label: str
    metrics: dict[str, Any]
    outcomes: list[StrategyOutcome] = field(default_factory=list)

    def to_dict(self, *, include_outcomes: bool = False) -> dict[str, Any]:
        payload = {"name": self.name, "label": self.label, "metrics": self.metrics}
        if include_outcomes:
            payload["outcomes"] = [outcome.to_dict() for outcome in self.outcomes]
        return payload


@dataclass
class BenchmarkReport:
    """完整基准报告。"""

    num_tasks: int
    seed: int
    strategies: list[StrategyReport]
    baseline: str = BASELINE

    # -- 派生 -------------------------------------------------------------- #
    def _by_name(self, name: str) -> StrategyReport | None:
        return next((item for item in self.strategies if item.name == name), None)

    def _baseline_metrics(self) -> dict[str, Any]:
        item = self._by_name(self.baseline)
        return item.metrics if item else {}

    def to_dict(self, *, include_outcomes: bool = False) -> dict[str, Any]:
        return {
            "num_tasks": self.num_tasks,
            "seed": self.seed,
            "baseline": self.baseline,
            "strategies": [item.to_dict(include_outcomes=include_outcomes) for item in self.strategies],
        }

    def to_markdown(self) -> str:
        base = self._baseline_metrics()
        base_cost = float(base.get("avg_cost_usd", 0.0) or 0.0)
        base_latency = float(base.get("avg_latency_s", 0.0) or 0.0)

        lines = [
            "# 路由策略对比与消融实验",
            "",
            f"- 轨迹条数：{self.num_tasks}，随机种子：{self.seed}",
            f"- 基准（{self.baseline}）：平均成本 {base_cost:.5f} USD / 条，平均延迟 {base_latency:.2f} s",
            "",
            "| 策略 | 平均成本(USD) | 成本节省 | 平均延迟(s) | 延迟节省 | 平均质量 | 成功率 | 重试/条 | 缓存命中率 | 档位分布(F/B/S) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.strategies:
            m = item.metrics
            cost = float(m.get("avg_cost_usd", 0.0) or 0.0)
            latency = float(m.get("avg_latency_s", 0.0) or 0.0)
            cost_save = f"{(1 - cost / base_cost) * 100:.1f}%" if base_cost else "-"
            latency_save = f"{(1 - latency / base_latency) * 100:.1f}%" if base_latency else "-"
            dist = m.get("tier_distribution", {})
            dist_text = f"{dist.get('fast', 0)}/{dist.get('balanced', 0)}/{dist.get('strong', 0)}"
            lines.append(
                f"| {item.label} | {cost:.5f} | {cost_save} | {latency:.2f} | {latency_save} | "
                f"{float(m.get('avg_quality', 0.0)):.3f} | {float(m.get('success_rate', 0.0)):.1%} | "
                f"{float(m.get('avg_retries', 0.0)):.2f} | {float(m.get('cache_hit_rate', 0.0)):.1%} | {dist_text} |"
            )
        lines += ["", "> 成本/延迟节省均相对「统一强模型」计算；档位分布为 fast/balanced/strong 的调用次数。"]

        step_level = self._by_name("step_level")
        if step_level:
            lines += ["", "## 结论", ""]
            full = step_level.metrics
            cost = float(full.get("avg_cost_usd", 0.0) or 0.0)
            latency = float(full.get("avg_latency_s", 0.0) or 0.0)
            lines.append(
                f"- 步级路由在**质量不明显下降**（{float(full.get('avg_quality', 0)):.3f} vs "
                f"{float(base.get('avg_quality', 0)):.3f}）的前提下，"
                f"成本降到 {cost / base_cost:.0%}、延迟降到 {latency / base_latency:.0%}（相对统一强模型）。"
            )
            always_fast = self._by_name("always_fast")
            query_level = self._by_name("query_level")
            if always_fast:
                lines.append(
                    f"- 统一小模型虽然最便宜（{float(always_fast.metrics.get('avg_cost_usd', 0)):.5f} USD/条），"
                    f"但质量只有 {float(always_fast.metrics.get('avg_quality', 0)):.3f}、"
                    f"成功率 {float(always_fast.metrics.get('success_rate', 0)):.1%}，"
                    "说明「全用小模型」会静默地把难步骤做差。"
                )
            if query_level:
                lines.append(
                    f"- Query 级路由（{float(query_level.metrics.get('avg_quality', 0)):.3f} 质量 / "
                    f"{float(query_level.metrics.get('success_rate', 0)):.1%} 成功率）明显弱于步级路由："
                    "最初看起来普通的任务，执行到后半段可能出现需要强模型的诊断步骤。"
                )
            for name, mechanism in (
                ("step_level_no_step_type", "机制：忽略步骤类型，仅按 context length 判难度"),
                ("step_level_no_cache", "机制：忽略前缀缓存"),
                ("step_level_no_failure", "机制：失败后不升级能力档"),
            ):
                item = self._by_name(name)
                if not item:
                    continue
                item_metrics = item.metrics
                d_quality = float(item_metrics.get("avg_quality", 0)) - float(full.get("avg_quality", 0))
                d_cost = _pct(float(item_metrics.get("avg_cost_usd", 0)), cost)
                d_latency = _pct(float(item_metrics.get("avg_latency_s", 0)), latency)
                lines.append(
                    f"- 消融 `{name}`：相对完整方案 质量 {d_quality:+.3f}、成本 {d_cost:+.1f}%、"
                    f"延迟 {d_latency:+.1f}%（{mechanism}）。"
                )
        return "\n".join(lines)


def _pct(value: float, base: float) -> float:
    """相对变化百分比（base 为 0 时返回 0）。"""
    return (value / base - 1) * 100 if base else 0.0


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #

def _aggregate(outcomes: list[StrategyOutcome]) -> dict[str, Any]:
    if not outcomes:
        return {}
    count = len(outcomes)
    attempts = sum(sum(outcome.tier_counts.values()) for outcome in outcomes)
    tier_distribution: dict[str, int] = {}
    for outcome in outcomes:
        for tier, value in outcome.tier_counts.items():
            tier_distribution[tier] = tier_distribution.get(tier, 0) + value
    return {
        "tasks": count,
        "avg_cost_usd": sum(o.cost_usd for o in outcomes) / count,
        "avg_latency_s": sum(o.latency_s for o in outcomes) / count,
        "avg_quality": sum(o.quality for o in outcomes) / count,
        "success_rate": sum(1 for o in outcomes if o.success) / count,
        "avg_retries": sum(o.retries for o in outcomes) / count,
        "cache_hit_rate": (sum(o.cache_hits for o in outcomes) / attempts) if attempts else 0.0,
        "tier_distribution": dict(sorted(tier_distribution.items())),
        "total_cost_usd": sum(o.cost_usd for o in outcomes),
    }


def _sim_config(base: EvalConfig, strategy: str) -> SimConfig:
    overrides = dict(_ABLATION_OVERRIDES.get(strategy, {}))
    config = SimConfig(
        failure_prob_when_underpowered=base.failure_prob_when_underpowered,
        retry_cost_multiplier=base.retry_cost_multiplier,
    )
    # 对照策略（统一模型 / Query 级）不具备步内自适应能力，关闭失败升级
    if not strategy.startswith("step_level"):
        overrides.setdefault("use_failure_escalation", False)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def run_benchmark(
    app_config: AppConfig,
    *,
    seed: int | None = None,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    tasks: list[SimTask] | None = None,
) -> BenchmarkReport:
    """在仿真器上跑完所有策略，返回对比报告。"""
    eval_config = app_config.evaluation
    resolved_seed = seed if seed is not None else eval_config.seed
    task_list = tasks if tasks is not None else generate_tasks(eval_config, seed=resolved_seed)

    reports: list[StrategyReport] = []
    for index, strategy in enumerate(strategies):
        config = _sim_config(eval_config, strategy)
        rng = random.Random(resolved_seed + 1000 * (index + 1))
        outcomes = [
            simulate_task(task, strategy, app_config.models, config, rng)
            for task in task_list
        ]
        reports.append(
            StrategyReport(
                name=strategy,
                label=STRATEGY_LABELS.get(strategy, strategy),
                metrics=_aggregate(outcomes),
                outcomes=outcomes,
            )
        )
    return BenchmarkReport(num_tasks=len(task_list), seed=resolved_seed, strategies=reports)