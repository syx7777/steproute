"""Agent 运行时端到端测试（离线仿真后端）。"""

from __future__ import annotations

from steproute.agent.runtime import STRATEGIES, parse_action

TASK = "整理当前科研进展，并判断我们的方法为什么没有超过 baseline，下一步应优先做什么。"


def test_sim_run_finishes_with_report(bundle) -> None:
    result = bundle.runtime("step_level").run(TASK)
    assert result.success
    assert result.steps >= 5
    assert result.report
    assert result.next_tasks
    assert "配置" in result.report or "公平" in result.report


def test_trajectory_records_routing_fields(bundle) -> None:
    result = bundle.runtime("step_level").run(TASK)
    first = result.trajectory[0]
    for key in ("step_type", "required_tier", "model_id", "tier", "tool", "latency_s", "cost_usd"):
        assert key in first
    assert all(record["tier"] for record in result.trajectory)


def test_step_level_mixes_tiers(bundle) -> None:
    result = bundle.runtime("step_level").run(TASK)
    tiers = set(result.tier_usage)
    assert "fast" in tiers
    assert "strong" in tiers, "比较/分析步骤应当用到强模型"


def test_always_strong_uses_only_strong(bundle) -> None:
    result = bundle.runtime("always_strong").run(TASK)
    assert set(result.tier_usage) == {"strong"}


def test_always_fast_uses_only_fast(bundle) -> None:
    result = bundle.runtime("always_fast").run(TASK)
    assert set(result.tier_usage) == {"fast"}
    assert result.tier_usage["fast"] > 0


def test_always_fast_costs_less_than_always_strong(bundle) -> None:
    fast = bundle.runtime("always_fast").run(TASK)
    strong = bundle.runtime("always_strong").run(TASK)
    assert fast.cost_usd < strong.cost_usd


def test_query_level_locks_single_tier(bundle) -> None:
    result = bundle.runtime("query_level").run(TASK)
    # 任务文本包含"为什么"，Query 级路由会锁定为 strong
    assert set(result.tier_usage) == {"strong"}


def test_metrics_snapshot_is_present(bundle) -> None:
    result = bundle.runtime("step_level").run(TASK)
    assert "pool" in result.metrics
    assert "cache" in result.metrics
    assert "jev" in result.metrics
    assert result.metrics["subgoals_total"] > 0


def test_result_serializable(bundle) -> None:
    import json

    result = bundle.runtime("step_level").run(TASK)
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "report" in payload


def test_unknown_strategy_rejected(bundle) -> None:
    try:
        bundle.runtime("nope")
    except ValueError as exc:
        assert "未知策略" in str(exc)
    else:
        raise AssertionError("应当拒绝未知策略")


def test_strategies_constant() -> None:
    assert "step_level" in STRATEGIES


# --------------------------------------------------------------------------- #
# 动作解析
# --------------------------------------------------------------------------- #

def test_parse_action_plain_json() -> None:
    action, error = parse_action('{"step_type":"extract","tool":"list_experiments","args":{}}')
    assert action is not None and action["tool"] == "list_experiments"
    assert error == ""


def test_parse_action_fenced_json_with_prose() -> None:
    content = '好的，下一步：\n```json\n{"tool": "search_corpus", "args": {"query": "baseline"}}\n```'
    action, _ = parse_action(content)
    assert action is not None
    assert action["args"]["query"] == "baseline"


def test_parse_action_tolerates_report_only() -> None:
    action, _ = parse_action('{"report": "## 进展", "next_tasks": ["a"]}')
    assert action is not None
    assert action["tool"] == "finalize"


def test_parse_action_rejects_garbage() -> None:
    action, error = parse_action("这不是 JSON")
    assert action is None
    assert error


def test_parse_action_rejects_missing_tool() -> None:
    action, error = parse_action('{"step_type": "extract"}')
    assert action is None
    assert "tool" in error