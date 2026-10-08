"""Web UI（可选，依赖 ``gradio``）。

启动：``uv run steproute ui``（需先 ``uv sync --extra ui``）。

界面刻意保持"研究面板"风格：左侧配置任务与策略，右侧实时展示
**路由与调度轨迹**（每步用了哪个档位、为什么、缓存命中、成本与延迟），
让"动态模型 Harness"的决策过程可见、可对照。
"""

from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .harness import build_harness
from .logging_utils import get_logger

logger = get_logger(__name__)

_TRAJECTORY_HEADERS = ["step", "步骤类型", "Jev下限", "模型", "档位", "工具", "延迟(s)", "成本(USD)", "缓存命中", "升级"]
_TIER_OPTIONS = ["balanced", "fast", "strong"]
_STRATEGY_OPTIONS = ["step_level", "query_level", "always_strong", "always_fast"]
_BACKEND_OPTIONS = ["sim", "api"]

DEFAULT_TASK = "整理本项目当前的科研进展，并判断我们的方法为什么没有超过 baseline，下一步应优先做什么。"


def _rows(result: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in result.get("trajectory", []):
        rows.append(
            [
                record.get("index"),
                record.get("step_type"),
                record.get("required_tier"),
                record.get("model_id"),
                record.get("tier"),
                record.get("tool") or "-",
                round(float(record.get("latency_s", 0.0)), 2),
                round(float(record.get("cost_usd", 0.0)), 6),
                f"{float(record.get('cache_hit_ratio', 0.0)):.0%}",
                "↑" if record.get("escalated") else "",
            ]
        )
    return rows


def run_task(settings: Settings, task: str, strategy: str, backend: str, max_steps: int) -> tuple[Any, ...]:
    """UI 回调：跑一次 Agent，返回可渲染的各个字段。"""
    if not task or not task.strip():
        return "请先填写任务。", "", "", "", [], "", ""
    harness = build_harness(settings=settings, backend_name=backend)
    try:
        config = harness.config.model_copy(
            update={"agent": harness.config.agent.model_copy(update={"max_steps": int(max_steps)})}
        )
        harness.config = config
        result = harness.runtime(strategy).run(task.strip())
        payload = result.to_dict()
        summary = {
            "策略": result.strategy,
            "后端": backend,
            "步数": result.steps,
            "耗时(s)": round(result.elapsed_s, 2),
            "成本(USD)": round(result.cost_usd, 6),
            "档位分布": result.tier_usage,
            "成功": result.success,
        }
        tasks_md = "\n".join(f"{index}. {item}" for index, item in enumerate(result.next_tasks, 1))
        metrics = json.dumps(payload.get("metrics", {}), ensure_ascii=False, indent=2)
        return (
            summary,
            result.report or "（无报告）",
            f"### 下一步任务\n{tasks_md or '（无）'}",
            json.dumps(payload, ensure_ascii=False, indent=2),
            _rows(payload),
            metrics,
        )
    except Exception as exc:  # UI 不应因异常白屏
        logger.exception("UI 运行失败")
        return {"错误": str(exc)}, "", "", "", [], ""


def build_demo(settings: Settings):  # pragma: no cover - 依赖 gradio
    import gradio as gr

    with gr.Blocks(title="科研工作流 Agent Harness") as demo:
        gr.Markdown(
            "# 科研工作流 Agent Harness\n"
            "把散落在**周报评论 / 组会纪要 / 实验记录 / 代码仓库**里的信息串起来，"
            "自动整理研究进展与下一步任务；并展示**步级动态模型路由**的决策过程。"
        )
        with gr.Row():
            with gr.Column(scale=2):
                task = gr.Textbox(label="科研任务", value=DEFAULT_TASK, lines=3)
                with gr.Row():
                    strategy = gr.Dropdown(_STRATEGY_OPTIONS, value="step_level", label="路由策略")
                    backend = gr.Dropdown(_BACKEND_OPTIONS, value=settings.backend, label="后端")
                    max_steps = gr.Slider(4, 20, value=12, step=1, label="最大步数")
                submit = gr.Button("运行", variant="primary")
                summary = gr.JSON(label="运行摘要")
            with gr.Column(scale=3):
                report = gr.Markdown(label="研究进度报告")
                tasks_md = gr.Markdown(label="下一步任务")

        trajectory = gr.Dataframe(headers=_TRAJECTORY_HEADERS, label="路由与调度轨迹", wrap=True)
        with gr.Accordion("完整结果 JSON / 指标明细", open=False):
            payload_json = gr.Code(label="结果 JSON", language="json")
            metrics_json = gr.Code(label="指标", language="json")

        submit.click(
            fn=lambda t, s, b, n: run_task(settings, t, s, b, n),
            inputs=[task, strategy, backend, max_steps],
            outputs=[summary, report, tasks_md, payload_json, trajectory, metrics_json],
        )
    return demo


def launch(settings: Settings, *, host: str = "127.0.0.1", port: int = 7860) -> None:  # pragma: no cover
    demo = build_demo(settings)
    demo.launch(server_name=host, server_port=port)