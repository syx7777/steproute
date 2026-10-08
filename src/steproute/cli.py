"""命令行入口。

主要命令
--------
* ``run``        —— 在真实/仿真后端上跑一次科研工作流 Agent，输出进度报告与下一步任务；
* ``evaluate``   —— 真实运行 + rubric 质量评分；
* ``benchmark``  —— 路由策略对比与消融实验（仿真，秒级完成，结果落盘）；
* ``ingest`` / ``search`` —— 查看语料与检索效果；
* ``secrets``    —— 安全地写入 / 查看 / 删除 LLM API Key；
* ``ui``         —— 启动 Web UI。
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent.runtime import STRATEGIES, RunResult
from .config import Settings, load_app_config
from .evaluation.benchmark import run_benchmark
from .evaluation.rubric import score_report
from .harness import build_harness
from .logging_utils import setup_logging
from .secrets import (
    BackendUnavailableError,
    SecretError,
    default_manager,
    mask_secret,
)

app = typer.Typer(add_completion=False, help="科研工作流 Agent Harness：动态模型路由与调度")
secrets_app = typer.Typer(help="LLM API Key 的安全管理（不落明文）")
app.add_typer(secrets_app, name="secrets")

console = Console()

DEFAULT_TASK = "整理本项目当前的科研进展，并判断我们的方法为什么没有超过 baseline，下一步应优先做什么。"


# --------------------------------------------------------------------------- #
# 公共
# --------------------------------------------------------------------------- #

def _settings(
    proxy: str | None,
    backend: str | None,
    data_dir: Path | None,
    config_dir: Path | None,
    log_level: str | None,
) -> Settings:
    settings = Settings()
    updates: dict[str, object] = {}
    if data_dir is not None:
        updates["data_dir"] = data_dir
    if config_dir is not None:
        updates["config_dir"] = config_dir
    if backend is not None:
        updates["backend"] = backend
    if log_level is not None:
        updates["log_level"] = log_level
    if updates:
        settings = settings.model_copy(update=updates)
    return settings.with_proxy(proxy)


def _print_result(result: RunResult, *, show_trajectory: bool = True) -> None:
    console.print()
    console.rule(f"[bold]运行结果 · 策略={result.strategy}")
    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="cyan")
    summary.add_column()
    summary.add_row("步数", str(result.steps))
    summary.add_row("耗时", f"{result.elapsed_s:.2f} s")
    summary.add_row("成本", f"${result.cost_usd:.6f}")
    summary.add_row("档位分布", json.dumps(result.tier_usage, ensure_ascii=False))
    summary.add_row("成功", "是" if result.success else "否")
    if result.error:
        summary.add_row("错误", result.error)
    console.print(summary)

    if show_trajectory and result.trajectory:
        table = Table(title="路由与调度轨迹", show_lines=False)
        for column in ("step", "预测类型", "Jev下限", "模型", "档位", "工具", "延迟(s)", "成本(USD)", "缓存", "升级"):
            table.add_column(column, overflow="fold")
        for record in result.trajectory:
            table.add_row(
                str(record["index"]),
                record["step_type"],
                record["required_tier"],
                record["model_id"],
                record["tier"],
                record["tool"] or "-",
                f"{record['latency_s']:.2f}",
                f"{record['cost_usd']:.6f}",
                f"{record['cache_hit_ratio']:.0%}",
                "↑" if record["escalated"] else "",
            )
        console.print(table)

    console.print(Panel(result.report or "（无报告）", title="研究进度报告", border_style="green"))
    tasks = "\n".join(f"{index}. {task}" for index, task in enumerate(result.next_tasks, 1))
    console.print(Panel(tasks or "（无）", title="下一步任务", border_style="magenta"))


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

@app.command()
def run(
    task: str = typer.Option(DEFAULT_TASK, "--task", "-t", help="要处理的科研任务"),
    strategy: str = typer.Option("step_level", "--strategy", "-s", help=f"路由策略，可选 {STRATEGIES}"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="后端：api（真实调用）或 sim（离线仿真）"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="最大步数"),
    proxy: str | None = typer.Option(None, "--proxy", help="代理，如 http://127.0.0.1:7892；none 表示直连"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    config_dir: Path | None = typer.Option(None, "--config-dir"),
    out: Path | None = typer.Option(None, "--out", help="把完整结果写入 JSON"),
    quiet: bool = typer.Option(False, "--quiet", help="只输出最终报告"),
) -> None:
    """跑一次科研工作流 Agent。"""
    settings = _settings(proxy, backend, data_dir, config_dir, "WARNING" if quiet else None)
    bundle = build_harness(settings=settings)
    try:
        if max_steps is not None:
            bundle.config = bundle.config.model_copy(
                update={"agent": bundle.config.agent.model_copy(update={"max_steps": max_steps})}
            )
        runtime = bundle.runtime(strategy)
        result = runtime.run(task)
        _print_result(result, show_trajectory=not quiet)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            console.print(f"[dim]结果已写入 {out}[/dim]")
    finally:
        bundle.close()


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

@app.command()
def evaluate(
    task: str = typer.Option(DEFAULT_TASK, "--task", "-t"),
    strategy: str = typer.Option("step_level", "--strategy", "-s"),
    backend: str | None = typer.Option(None, "--backend", "-b"),
    proxy: str | None = typer.Option(None, "--proxy"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    config_dir: Path | None = typer.Option(None, "--config-dir"),
) -> None:
    """真实运行 + rubric 质量评分（结论是否覆盖了应有事实）。"""
    settings = _settings(proxy, backend, data_dir, config_dir, None)
    bundle = build_harness(settings=settings)
    try:
        runtime = bundle.runtime(strategy)
        result = runtime.run(task)
        rubric = score_report(
            result.report,
            result.next_tasks,
            bundle.reference,
            evidence_count=int(result.metrics.get("evidence", 0)),
        )
        _print_result(result)
        table = Table(title="Rubric 质量评分")
        table.add_column("检查项")
        table.add_column("通过", justify="center")
        for key, passed in rubric.checks.items():
            table.add_row(key, "✅" if passed else "❌")
        console.print(table)
        console.print(f"[bold]质量得分：{rubric.score:.3f}[/bold]（基准事实：{rubric.detail}）")
    finally:
        bundle.close()


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #

@app.command()
def benchmark(
    tasks: int | None = typer.Option(None, "--tasks", help="仿真轨迹条数"),
    seed: int | None = typer.Option(None, "--seed"),
    out_dir: Path = typer.Option(Path("artifacts"), "--out-dir"),
    config_dir: Path | None = typer.Option(None, "--config-dir"),
    include_outcomes: bool = typer.Option(False, "--include-outcomes", help="JSON 中是否包含逐轨迹明细"),
) -> None:
    """路由策略对比与消融实验（仿真）。"""
    settings = Settings()
    if config_dir is not None:
        settings = settings.model_copy(update={"config_dir": config_dir})
    setup_logging(settings.log_level)
    app_config = load_app_config(settings.config_dir)
    if tasks is not None:
        app_config = app_config.model_copy(
            update={"evaluation": app_config.evaluation.model_copy(update={"num_tasks": tasks})}
        )

    report = run_benchmark(app_config, seed=seed)
    markdown = report.to_markdown()
    console.print(markdown)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "routing_benchmark.md").write_text(markdown + "\n", encoding="utf-8")
    (out_dir / "routing_benchmark.json").write_text(
        json.dumps(report.to_dict(include_outcomes=include_outcomes), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]已写入 {out_dir / 'routing_benchmark.md'} 与 routing_benchmark.json[/dim]")


# --------------------------------------------------------------------------- #
# ingest / search
# --------------------------------------------------------------------------- #

@app.command()
def ingest(
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """查看语料加载结果。"""
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    setup_logging(settings.log_level)
    bundle = build_harness(settings=settings, backend_name="sim")
    try:
        console.print(json.dumps(bundle.corpus.stats(), ensure_ascii=False, indent=2))
        table = Table(title="文档列表")
        table.add_column("doc_id")
        table.add_column("来源")
        table.add_column("标题")
        for doc in bundle.corpus.documents.values():
            table.add_row(doc.doc_id, doc.kind.label, doc.title)
        console.print(table)
        todos = bundle.corpus.scan_todos()
        console.print(Panel("\n".join(f"- [{t.status}] {t.text} ({t.doc_id})" for t in todos), title="抽取到的待办"))
    finally:
        bundle.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="检索词"),
    top_k: int = typer.Option(5, "--top-k"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """快速验证 BM25 检索效果。"""
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    setup_logging(settings.log_level)
    bundle = build_harness(settings=settings, backend_name="sim")
    try:
        hits = bundle.retriever.search(query, top_k=top_k)
        if not hits:
            console.print("[yellow]无命中[/yellow]")
        for hit in hits:
            console.print(f"[bold]{hit.score:.3f}[/bold] {hit.chunk.doc_id} · {hit.chunk.title}")
            console.print(f"    {hit.chunk.text[:200].replace(chr(10), ' ')}")
    finally:
        bundle.close()


# --------------------------------------------------------------------------- #
# secrets
# --------------------------------------------------------------------------- #

@secrets_app.command("set")
def secrets_set(
    value: str | None = typer.Option(None, "--value", help="密钥值（不推荐，会进入命令历史）；留空则交互输入"),
    backend: str = typer.Option("keyring", "--backend", help="keyring（系统凭据管理器）或 vault（本地加密库）"),
) -> None:
    """安全写入 LLM API Key。"""
    manager = default_manager()
    if value is None:
        value = typer.prompt("请输入 LLM API Key", hide_input=True)
    try:
        used = manager.set_llm_api_key(value, backend=backend)
    except (SecretError, BackendUnavailableError) as exc:
        console.print(f"[red]写入失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]已写入 {used}（{mask_secret(value)}）[/green]")


@secrets_app.command("status")
def secrets_status() -> None:
    """查看密钥解析来源（只显示掩码）。"""
    console.print(json.dumps(default_manager().status(), ensure_ascii=False, indent=2))


@secrets_app.command("delete")
def secrets_delete() -> None:
    """从所有可用后端删除密钥。"""
    removed = default_manager().delete_llm_api_key()
    console.print(f"[green]已清理：{removed or '（无）'}[/green]")


# --------------------------------------------------------------------------- #
# ui
# --------------------------------------------------------------------------- #

@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="默认后端：api 或 sim"),
    proxy: str | None = typer.Option(None, "--proxy"),
) -> None:
    """启动 Web UI（需要 gradio）。"""
    try:
        from .webui import launch
    except ImportError as exc:  # pragma: no cover - 取决于是否安装可选依赖
        console.print("[red]未安装 gradio。请运行：uv sync --extra ui[/red]")
        raise typer.Exit(code=1) from exc
    settings = _settings(proxy, backend, None, None, None)
    launch(settings, host=host, port=port)


@app.command()
def version() -> None:
    """显示版本。"""
    console.print(f"steproute {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()