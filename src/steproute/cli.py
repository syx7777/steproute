"""命令行入口。

主要命令
--------
* ``run``        —— 在真实/仿真后端上跑一次科研工作流 Agent，输出进度报告与下一步任务；
* ``evaluate``   —— 真实运行 + rubric 质量评分；
* ``ingest`` / ``search`` —— 查看语料与检索效果；
* ``sync``       —— 从外部知识库（飞书）同步文档与评论到本地语料；
* ``brief``      —— 生成每日工作简报（项目进展 / 飞书更新）；
* ``minutes``    —— 导入组会纪要，抽取决策与待办；
* ``memory``     —— 项目长期记忆：状态 / 决策 / 实验经验；
* ``weekly``     —— 周报预览与写入飞书知识库；
* ``serve``      —— 启动可视化网站（需 ``--extra server``）；
* ``secrets``    —— 安全地写入 / 查看 / 删除密钥（不落明文）；
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
from .assistant import (
    build_brief,
    collect_week,
    import_minutes,
    publish_weekly,
    render_weekly,
)
from .assistant.calendar import as_date
from .assistant.distill import build_distill_backend, distill_memory
from .assistant.memory import MemoryStore
from .config import Settings
from .connectors.feishu import FeishuClient, FeishuError
from .connectors.sync import SyncTargetError, sync_source
from .corpus.loaders import load_corpus
from .evaluation.rubric import score_report
from .harness import build_harness
from .logging_utils import setup_logging
from .secrets import (
    BackendUnavailableError,
    SecretError,
    manager_for,
    mask_secret,
)

app = typer.Typer(add_completion=False, help="科研工作流 Agent Harness：动态模型路由与调度")
secrets_app = typer.Typer(help="密钥的安全管理（不落明文）")
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
# sync（外部来源）
# --------------------------------------------------------------------------- #

@app.command()
def sync(
    wiki_space: str | None = typer.Option(
        None, "--wiki-space", help="飞书知识库 space_id（“我的文档库”可填 my_library）"
    ),
    folder: str | None = typer.Option(
        None, "--folder", help='飞书云空间文件夹 token；传空串 "" 表示个人云空间根目录'
    ),
    into: str = typer.Option("meeting_minutes", "--into", help="写入 data/sources 下的目标目录"),
    source: str = typer.Option("feishu", "--source", help="外部来源类型（目前支持 feishu）"),
    app_id: str | None = typer.Option(None, "--app-id", help="飞书应用 App ID（默认读 STEPROUTE_FEISHU_APP_ID）"),
    no_comments: bool = typer.Option(False, "--no-comments", help="不同步文档评论"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    proxy: str | None = typer.Option(None, "--proxy", help="代理，如 http://127.0.0.1:7892；none 表示直连"),
    full: bool = typer.Option(False, "--full", help="忽略增量水位，强制全量重新拉取"),
    limit: int | None = typer.Option(None, "--limit", help="最多同步多少篇（预览/调试用）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只列出将要写入的文档，不落盘"),
) -> None:
    """从飞书知识库 / 云空间同步文档（含评论）到本地语料。"""
    if source.lower() != "feishu":
        console.print(f"[red]暂不支持来源：{source}（目前支持 feishu）[/red]")
        raise typer.Exit(code=1)
    if wiki_space is None and folder is None:
        console.print("[red]请指定数据入口：--wiki-space <space_id> 或 --folder <folder_token>[/red]")
        raise typer.Exit(code=1)

    settings = _settings(proxy, None, data_dir, None, None)
    setup_logging(settings.log_level)

    client = FeishuClient(
        space_id=wiki_space,
        folder_token=folder,
        app_id=app_id,
        base_url=settings.feishu_base_url,
        site_url=settings.feishu_site_url,
        settings=settings,
        include_comments=not no_comments,
    )
    entrance = f"wiki:{wiki_space}" if wiki_space is not None else f"folder:{folder or 'root'}"
    try:
        result = sync_source(
            client, data_dir=settings.data_dir, target=into, full=full, limit=limit, dry_run=dry_run
        )
    except (FeishuError, SecretError, SyncTargetError) as exc:
        console.print(f"[red]同步失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="cyan")
    summary.add_column()
    summary.add_row("来源", result.source)
    summary.add_row("入口", entrance)
    summary.add_row("目标目录", f"data/sources/{result.target}")
    summary.add_row("远端文档", str(result.listed))
    summary.add_row("写入", str(len(result.written)))
    summary.add_row("跳过（未变化）", str(len(result.skipped)))
    summary.add_row("清理（远端已删）", str(len(result.removed)))
    if dry_run:
        summary.add_row("模式", "[yellow]dry-run（未落盘）[/yellow]")
    console.print(Panel(summary, title="同步结果 · 飞书", border_style="green"))

    if result.written:
        table = Table(title="写入的文档（doc_id）")
        table.add_column("doc_id")
        for doc_id in result.written:
            table.add_row(doc_id)
        console.print(table)
    if not dry_run and result.state_path:
        console.print(f"[dim]增量水位已写入 {result.state_path}[/dim]")


# --------------------------------------------------------------------------- #
# secrets
# --------------------------------------------------------------------------- #

@secrets_app.command("set")
def secrets_set(
    value: str | None = typer.Option(None, "--value", help="密钥值（不推荐，会进入命令历史）；留空则交互输入"),
    backend: str = typer.Option("keyring", "--backend", help="keyring（系统凭据管理器）或 vault（本地加密库）"),
    name: str = typer.Option("llm_api_key", "--name", help="密钥名：llm_api_key（默认）或 feishu_app_secret"),
) -> None:
    """安全写入密钥（不落明文）。"""
    manager = manager_for(name)
    if value is None:
        value = typer.prompt(f"请输入 {name}", hide_input=True)
    try:
        used = manager.set_secret(value, backend=backend)
    except (SecretError, BackendUnavailableError) as exc:
        console.print(f"[red]写入失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]已写入 {used}（{name}={mask_secret(value)}）[/green]")


@secrets_app.command("status")
def secrets_status(
    name: str = typer.Option("llm_api_key", "--name", help="密钥名：llm_api_key（默认）或 feishu_app_secret"),
) -> None:
    """查看密钥解析来源（只显示掩码）。"""
    console.print(json.dumps(manager_for(name).status(), ensure_ascii=False, indent=2))


@secrets_app.command("delete")
def secrets_delete(
    name: str = typer.Option("llm_api_key", "--name", help="密钥名：llm_api_key（默认）或 feishu_app_secret"),
) -> None:
    """从所有可用后端删除密钥。"""
    removed = manager_for(name).delete_secret()
    console.print(f"[green]已清理：{removed or '（无）'}[/green]")


# --------------------------------------------------------------------------- #
# assistant：brief / minutes / memory / weekly
# --------------------------------------------------------------------------- #

memory_app = typer.Typer(help="项目长期记忆：状态 / 决策 / 实验经验")
weekly_app = typer.Typer(help="周报：预览与写入飞书知识库")
app.add_typer(memory_app, name="memory")
app.add_typer(weekly_app, name="weekly")


def _assistant_settings(data_dir: Path | None) -> Settings:
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    setup_logging(settings.log_level)
    return settings


@app.command()
def brief(
    day: str | None = typer.Option(None, "--date", help="日期 YYYY-MM-DD，默认今天"),
    space: str = typer.Option("", "--space", help="只统计该知识库同步来的文档待办（默认全部）"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    out: Path | None = typer.Option(None, "--out", help="额外写入指定文件"),
    no_save: bool = typer.Option(False, "--no-save", help="只看结果，不落盘"),
) -> None:
    """生成每日工作简报（项目进展 + 飞书更新）。"""
    settings = _assistant_settings(data_dir)
    corpus = load_corpus(settings.data_dir)
    result = build_brief(
        corpus, data_dir=settings.data_dir, day=as_date(day), space_id=space, save=not no_save
    )
    console.print(Panel(result.markdown, title="每日工作简报", border_style="green"))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.markdown, encoding="utf-8")
        console.print(f"[dim]已写入 {out}[/dim]")


@app.command()
def minutes(
    text: str | None = typer.Option(None, "--text", help="直接传入纪要正文"),
    file: Path | None = typer.Option(None, "--file", help="从文件读取纪要（.md/.txt）"),
    day: str | None = typer.Option(None, "--date", help="纪要日期 YYYY-MM-DD，默认今天或从正文识别"),
    title: str = typer.Option("", "--title", help="纪要标题，默认取首个标题"),
    author: str = typer.Option("", "--author"),
    remember: bool = typer.Option(True, "--remember/--no-remember", help="是否把抽取到的决策写入长期记忆"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """导入组会纪要：解析决策与待办并落盘进语料。"""
    if file is None and not text:
        console.print("[red]请用 --file 或 --text 提供纪要内容[/red]")
        raise typer.Exit(code=1)
    content = file.read_text(encoding="utf-8") if file is not None else str(text)
    settings = _assistant_settings(data_dir)
    result = import_minutes(
        settings.data_dir,
        content,
        day=as_date(day) if day else None,
        title=title,
        author=author,
        memory=MemoryStore(settings.data_dir),
        remember_decisions=remember,
    )
    console.print(f"[green]已导入：{result.doc_id}[/green]")
    decisions = "\n".join(f"- {item}" for item in result.parse.decisions) or "（未识别到决策）"
    todos = "\n".join(f"- {item}" for item in result.parse.todos) or "（未识别到待办）"
    console.print(Panel(decisions, title="决策", border_style="cyan"))
    console.print(Panel(todos, title="待办", border_style="magenta"))


@memory_app.command("show")
def memory_show(data_dir: Path | None = typer.Option(None, "--data-dir")) -> None:
    """查看长期记忆摘要与统计。"""
    settings = _assistant_settings(data_dir)
    store = MemoryStore(settings.data_dir)
    console.print(json.dumps(store.stats(), ensure_ascii=False))
    console.print(Panel(store.summary(), title="长期记忆", border_style="cyan"))


@memory_app.command("state")
def memory_state(
    project: str = typer.Option(..., "--project", help="项目名"),
    status: str = typer.Option("", "--status", help="状态，如 进行中 / 已完成"),
    stage: str = typer.Option("", "--stage", help="当前阶段"),
    goal: str = typer.Option("", "--goal", help="目标"),
    notes: str = typer.Option("", "--notes", help="备注"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """新增 / 更新一个项目的长期状态。"""
    settings = _assistant_settings(data_dir)
    state = MemoryStore(settings.data_dir).set_project(project, status=status, stage=stage, goal=goal, notes=notes)
    console.print(f"[green]已更新：{state.project}（{state.updated_at}）[/green]")


@memory_app.command("decision")
def memory_decision(
    text: str = typer.Argument(..., help="决策内容"),
    day: str | None = typer.Option(None, "--date"),
    project: str = typer.Option("", "--project"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """追加一条历史决策。"""
    settings = _assistant_settings(data_dir)
    entry = MemoryStore(settings.data_dir).add_decision(text, date_str=day or "", project=project)
    console.print(f"[green]已记录决策（{entry.date}）：{entry.text}[/green]")


@memory_app.command("lesson")
def memory_lesson(
    text: str = typer.Argument(..., help="实验经验内容"),
    day: str | None = typer.Option(None, "--date"),
    project: str = typer.Option("", "--project"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """追加一条实验经验。"""
    settings = _assistant_settings(data_dir)
    entry = MemoryStore(settings.data_dir).add_lesson(text, date_str=day or "", project=project)
    console.print(f"[green]已记录经验（{entry.date}）：{entry.text}[/green]")


@memory_app.command("distill")
def memory_distill(
    space: str = typer.Option("", "--space", help="只提炼该知识库同步来的文档（默认全部文档）"),
    project: str = typer.Option("", "--project", help="项目名，留空则由模型从材料里取名"),
    no_llm: bool = typer.Option(False, "--no-llm", help="只用规则抽取，不调用 LLM"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
) -> None:
    """从文档提炼「项目状态 / 决策 / 经验」并写入长期记忆。"""
    settings = _assistant_settings(data_dir)
    backend, spec = build_distill_backend(settings)
    try:
        result = distill_memory(
            data_dir=settings.data_dir,
            corpus=load_corpus(settings.data_dir),
            memory=MemoryStore(settings.data_dir),
            space_id=space,
            project=project,
            backend=backend,
            spec=spec,
            use_llm=not no_llm,
        )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()
    console.print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@weekly_app.command("preview")
def weekly_preview(
    day: str | None = typer.Option(None, "--date", help="该周内任意日期，默认今天"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """本地预览周报（不回写飞书）。"""
    settings = _assistant_settings(data_dir)
    corpus = load_corpus(settings.data_dir)
    digest = collect_week(data_dir=settings.data_dir, corpus=corpus, day=as_date(day))
    markdown = render_weekly(digest)
    console.print(f"[bold]周报命名：{digest.label.title}[/bold]（{digest.label.start} ~ {digest.label.end}）")
    console.print(Panel(markdown, title="周报预览", border_style="green"))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[dim]已写入 {out}[/dim]")


@weekly_app.command("publish")
def weekly_publish(
    day: str | None = typer.Option(None, "--date"),
    space: str | None = typer.Option(None, "--space", help="知识库 space_id（默认读 STEPROUTE_FEISHU_WEEKLY_SPACE_ID）"),
    template: str | None = typer.Option(None, "--template", help="模板文档 token，优先于关键词匹配"),
    template_keyword: str | None = typer.Option(None, "--template-keyword", help="模板标题关键词，默认「模板」"),
    parent: str | None = typer.Option(None, "--parent", help="父节点 token，默认挂在空间根目录"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    proxy: str | None = typer.Option(None, "--proxy"),
) -> None:
    """生成周报并写入飞书知识库（在知识库下新建一篇 docx）。"""
    settings = _settings(proxy, None, data_dir, None, None)
    setup_logging(settings.log_level)
    space_id = space or settings.feishu_weekly_space_id
    if not space_id:
        console.print("[red]未指定知识库：用 --space 或设置 STEPROUTE_FEISHU_WEEKLY_SPACE_ID[/red]")
        raise typer.Exit(code=1)

    corpus = load_corpus(settings.data_dir)
    digest = collect_week(data_dir=settings.data_dir, corpus=corpus, day=as_date(day))
    markdown = render_weekly(digest)

    client = FeishuClient(
        space_id=space_id,
        base_url=settings.feishu_base_url,
        site_url=settings.feishu_site_url,
        settings=settings,
    )
    try:
        result = publish_weekly(
            client,
            space_id=space_id,
            markdown=markdown,
            title=digest.label.title,
            template_token=template or settings.feishu_weekly_template,
            template_keyword=template_keyword or settings.feishu_weekly_template_keyword,
            parent_node_token=parent,
        )
    except (FeishuError, SecretError) as exc:
        console.print(f"[red]发布失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="cyan")
    summary.add_column()
    summary.add_row("标题", result.title)
    summary.add_row("文档", result.document_id)
    summary.add_row("链接", result.url or "（见知识库）")
    summary.add_row("写入块数", str(result.blocks))
    summary.add_row("模板", result.template_token or "（未使用）")
    console.print(Panel(summary, title="周报已发布", border_style="green"))


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    proxy: str | None = typer.Option(None, "--proxy"),
) -> None:
    """启动可视化网站（每日简报 / 待办 / 纪要导入 / 周报）。"""
    try:
        from .server.app import create_app
    except ImportError as exc:  # pragma: no cover - 取决于是否安装可选依赖
        console.print("[red]未安装 server 依赖。请运行：uv sync --extra server[/red]")
        raise typer.Exit(code=1) from exc

    import uvicorn

    settings = _settings(proxy, None, data_dir, None, None)
    setup_logging(settings.log_level)
    console.print(f"[green]可视化网站已启动：http://{host}:{port}[/green]")
    uvicorn.run(create_app(settings), host=host, port=port)


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