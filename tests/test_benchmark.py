"""策略对比与消融实验测试。"""

from __future__ import annotations

import pytest

from steproute.config import load_app_config
from steproute.evaluation.benchmark import run_benchmark


@pytest.fixture(scope="module")
def small_config(project_root):
    config = load_app_config(project_root / "configs")
    return config.model_copy(update={"evaluation": config.evaluation.model_copy(update={"num_tasks": 16})})


@pytest.fixture(scope="module")
def report(small_config):
    return run_benchmark(small_config, seed=7)


def _metrics(report, name: str) -> dict:
    return next(item.metrics for item in report.strategies if item.name == name)


def test_all_strategies_present(report) -> None:
    names = {item.name for item in report.strategies}
    assert {"always_strong", "always_fast", "query_level", "step_level"} <= names
    assert "step_level_no_cache" in names


def test_step_level_costs_less_than_always_strong(report) -> None:
    step = _metrics(report, "step_level")
    strong = _metrics(report, "always_strong")
    assert step["avg_cost_usd"] < strong["avg_cost_usd"]
    assert step["avg_latency_s"] < strong["avg_latency_s"]


def test_step_level_quality_beats_always_fast(report) -> None:
    step = _metrics(report, "step_level")
    fast = _metrics(report, "always_fast")
    assert step["avg_quality"] > fast["avg_quality"]
    assert step["success_rate"] >= fast["success_rate"]


def test_step_level_beats_query_level(report) -> None:
    step = _metrics(report, "step_level")
    query = _metrics(report, "query_level")
    assert step["avg_quality"] >= query["avg_quality"]
    assert step["success_rate"] >= query["success_rate"]


def test_cache_ablation_increases_latency(report) -> None:
    step = _metrics(report, "step_level")
    no_cache = _metrics(report, "step_level_no_cache")
    assert no_cache["avg_latency_s"] > step["avg_latency_s"]
    assert no_cache["cache_hit_rate"] == 0.0


def test_step_type_ablation_shifts_tier_distribution(report) -> None:
    step = _metrics(report, "step_level")
    no_step_type = _metrics(report, "step_level_no_step_type")
    # 仅按长度判难度会让"档位分布"明显更偏向强模型
    assert no_step_type["tier_distribution"].get("strong", 0) > step["tier_distribution"].get("strong", 0)


def test_benchmark_deterministic_with_same_seed(small_config) -> None:
    first = run_benchmark(small_config, seed=11)
    second = run_benchmark(small_config, seed=11)
    assert _metrics(first, "step_level")["avg_cost_usd"] == _metrics(second, "step_level")["avg_cost_usd"]
    assert _metrics(first, "step_level")["avg_quality"] == _metrics(second, "step_level")["avg_quality"]


def test_markdown_report_contains_conclusions(report) -> None:
    markdown = report.to_markdown()
    assert "步级路由" in markdown
    assert "消融" in markdown
    assert "成本节省" in markdown


def test_report_serializable(report) -> None:
    import json

    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "always_strong" in payload