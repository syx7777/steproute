"""可视化网站：独立前后端（FastAPI 后端 + 单页静态前端）。

启动：``uv run steproute serve``（需 ``uv sync --extra server``）。

页面结构（两个页签）：

1. **工作台** —— 知识库选择（可顺带提炼长期记忆）+ 每日工作简报 + 待办清单 + 项目长期记忆；
2. **周报** —— 预览并一键写入固定知识库。

知识库选项与周报目标来自 ``configs/knowledge_bases.yaml``。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..assistant import (
    build_brief,
    collect_candidates,
    load_summary,
    load_weekly_summary,
    publish_weekly,
    save_summary,
    save_weekly_summary,
    summarize_brief,
    summarize_weekly,
)
from ..assistant.calendar import as_date, label_for
from ..assistant.distill import build_distill_backend, distill_memory
from ..assistant.memory import MemoryStore
from ..assistant.todo import TodoStore, merged_todos
from ..config import KnowledgeBaseConfig, Settings, load_knowledge_bases
from ..connectors.feishu import FeishuClient, FeishuError
from ..connectors.sync import SyncTargetError, sync_source
from ..corpus.loaders import load_corpus
from ..logging_utils import get_logger
from ..secrets import SecretError

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: 从某个知识库拉取文档时写入语料的位置（二级目录按 space_id 区分）。
SYNC_TARGET = "notes"


class TodoCreatePayload(BaseModel):
    text: str = Field(..., min_length=1)
    owner: str = ""
    due: str = ""
    #: 归属的知识库（前端创建时带上当前选中的知识库 id）。
    space_id: str = ""


class TodoUpdatePayload(BaseModel):
    text: str | None = None
    status: str | None = None
    owner: str | None = None
    due: str | None = None


class RefreshPayload(BaseModel):
    space_id: str = Field(..., min_length=1)
    day: str = ""
    distill: bool = True


class DistillPayload(BaseModel):
    space_id: str = ""
    project: str = ""
    day: str = ""
    use_llm: bool = True


class SummarizePayload(BaseModel):
    space_id: str = ""
    day: str = ""
    use_llm: bool = True


class WeeklySummaryPayload(BaseModel):
    #: 汇总范围：跟随工作台选中的知识库（空 = 全部知识库）。
    space_id: str = ""
    day: str = ""
    use_llm: bool = True


class WeeklyPublishPayload(BaseModel):
    space_id: str = ""
    date: str = ""


def _llm_todos(summary: Any) -> list[str] | None:
    """归纳成功（method=llm）时用其待办替换文档待办；否则返回 ``None``（走规则合并）。"""
    if summary is not None and getattr(summary, "method", "") == "llm":
        return list(getattr(summary, "todos", []) or [])
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。"""
    runtime = settings or Settings()
    knowledge: KnowledgeBaseConfig = load_knowledge_bases(runtime.config_dir)
    app = FastAPI(title="steproute 科研助手", version="1.0.0")

    def _corpus():
        return load_corpus(runtime.data_dir)

    def _memory() -> MemoryStore:
        return MemoryStore(runtime.data_dir)

    def _todos() -> TodoStore:
        return TodoStore(runtime.data_dir)

    def _todo_stats(items: list[dict[str, Any]]) -> dict[str, int]:
        done = sum(1 for item in items if item["status"] == "done")
        return {"total": len(items), "open": len(items) - done, "done": done}

    def _dashboard(day: str = "", space_id: str = "") -> dict[str, Any]:
        target = as_date(day or None)
        corpus = _corpus()
        memory = _memory()
        todo_store = _todos()
        # dashboard 只读缓存，绝不主动调用 LLM；归纳由 /api/brief/summarize 或刷新时触发。
        summary = load_summary(runtime.data_dir, space_id=space_id)
        brief = build_brief(
            corpus,
            data_dir=runtime.data_dir,
            memory=memory,
            todos=todo_store,
            day=target,
            space_id=space_id,
            space_label=knowledge.name_of(space_id),
            summary=summary,
            save=False,
        )
        todos = merged_todos(
            todo_store,
            corpus,
            space_id=space_id,
            day=target,
            llm_todos=_llm_todos(summary),
        )
        candidates = collect_candidates(runtime.data_dir, corpus, space_id=space_id, day=target)
        return {
            "date": target.isoformat(),
            "space_id": space_id,
            "brief": brief.to_dict(),
            "todos": todos,
            "todo_stats": _todo_stats(todos),
            "todo_candidates": len(candidates),
            "candidate_todos": candidates,
            "summary": summary.to_dict() if summary is not None else None,
            "memory": {
                # 记忆条目的 project 是「LLM 归纳出的项目名」，与知识库无关；
                # 这里按条目的「知识库归属」（space_id）过滤，无归属条目（手写 / 旧数据）各库都可见。
                "stats": memory.stats(space_id=space_id),
                "summary": memory.summary(space_id=space_id),
                "projects": [state.to_dict() for state in memory.projects(space_id=space_id).values()],
                "decisions": [entry.to_dict() for entry in memory.decisions(limit=20, space_id=space_id)],
                "lessons": [entry.to_dict() for entry in memory.lessons(limit=20, space_id=space_id)],
            },
            "weekly": _weekly_payload(target, space_id),
            "knowledge_bases": knowledge.to_list(),
            "weekly_space_id": knowledge.weekly_space_id,
        }

    def _run_summarize(*, space_id: str, day: str = "", use_llm: bool = True) -> Any:
        """归纳简报摘要（LLM 默认、失败回退规则）并把结果落缓存。

        只要 ``use_llm`` 为真就**必然**构造后端并传给 :func:`summarize_brief`
        （缺 Key 由后端在调用时报错、回退规则），避免"漏传后端"被误判成"未启用 LLM"。
        """
        backend = spec = None
        if use_llm:
            backend, spec = build_distill_backend(runtime)
        try:
            summary = summarize_brief(
                data_dir=runtime.data_dir,
                corpus=_corpus(),
                memory=_memory(),
                space_id=space_id,
                space_label=knowledge.name_of(space_id),
                day=as_date(day or None),
                backend=backend,
                spec=spec,
                use_llm=use_llm,
            )
        finally:
            if backend is not None:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    closer()
        save_summary(runtime.data_dir, summary)
        return summary

    def _weekly_preview(target: date, space_id: str) -> Any:
        """周报预览：有「同一周 + 同一知识库」的缓存就直接用，否则只做规则回退（不调模型）。"""
        label = label_for(target)
        cached = load_weekly_summary(runtime.data_dir, space_id=space_id)
        if cached is not None and cached.week == label.title:
            return cached
        return summarize_weekly(
            data_dir=runtime.data_dir,
            corpus=_corpus(),
            space_id=space_id,
            day=target,
            use_llm=False,
        )

    def _weekly_payload(target: date, space_id: str) -> dict[str, Any]:
        summary = _weekly_preview(target, space_id)
        return {
            "label": label_for(target).to_dict(),
            "markdown": summary.markdown,
            "method": summary.method,
            "note": summary.note,
            "docs": summary.docs,
            "space_id": space_id,
        }

    def _run_weekly_summarize(*, space_id: str, day: str = "", use_llm: bool = True) -> Any:
        """按「本周 + 知识库」归纳周报（LLM 默认、失败回退规则）并落缓存。"""
        backend = spec = None
        if use_llm:
            backend, spec = build_distill_backend(runtime)
        try:
            summary = summarize_weekly(
                data_dir=runtime.data_dir,
                corpus=_corpus(),
                space_id=space_id,
                day=as_date(day or None),
                backend=backend,
                spec=spec,
                use_llm=use_llm,
            )
        finally:
            if backend is not None:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    closer()
        save_weekly_summary(runtime.data_dir, summary)
        return summary

    # -- 概览 -------------------------------------------------------------- #
    @app.get("/api/dashboard")
    def dashboard(day: str = "", space: str = "") -> dict[str, Any]:
        return _dashboard(day, space)

    @app.get("/api/knowledge-bases")
    def knowledge_bases() -> dict[str, Any]:
        return {
            "knowledge_bases": knowledge.to_list(),
            "weekly_space_id": knowledge.weekly_space_id,
        }

    @app.post("/api/brief/refresh")
    def refresh_brief(payload: RefreshPayload) -> dict[str, Any]:
        """从选定知识库增量拉取文档（可选接着提炼长期记忆），再刷新整页数据。"""
        client = FeishuClient(
            space_id=payload.space_id,
            base_url=runtime.feishu_base_url,
            site_url=runtime.feishu_site_url,
            settings=runtime,
        )
        try:
            result = sync_source(
                client, data_dir=runtime.data_dir, target=SYNC_TARGET, subdir=payload.space_id
            )
        except (FeishuError, SecretError, SyncTargetError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            client.close()

        out = _dashboard(payload.day, payload.space_id)
        out["sync"] = result.to_dict()
        out["space_name"] = knowledge.name_of(payload.space_id)

        if payload.distill:
            backend, spec = build_distill_backend(runtime)
            try:
                distilled = distill_memory(
                    data_dir=runtime.data_dir,
                    corpus=_corpus(),
                    memory=_memory(),
                    space_id=payload.space_id,
                    backend=backend,
                    spec=spec,
                )
            finally:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    closer()
            # 提炼完长期记忆后，再归纳一次简报（待办 + 项目进展）。
            summary = _run_summarize(space_id=payload.space_id, day=payload.day)
            out["distill"] = distilled.to_dict()
            out["summary"] = summary.to_dict()
            out = {**out, **_dashboard(payload.day, payload.space_id), "sync": out["sync"],
                   "distill": out["distill"], "summary": out["summary"], "space_name": out["space_name"]}
        return out

    @app.post("/api/brief/summarize")
    def post_summarize(payload: SummarizePayload) -> dict[str, Any]:
        """用 LLM 归纳简报待办与项目进展（未配置 LLM 时回退规则），并写入缓存。"""
        summary = _run_summarize(space_id=payload.space_id, day=payload.day, use_llm=payload.use_llm)
        return {**summary.to_dict(), "dashboard": _dashboard(payload.day, payload.space_id)}

    # -- 长期记忆 ---------------------------------------------------------- #
    @app.get("/api/memory")
    def get_memory(space: str = "") -> dict[str, Any]:
        """长期记忆；``space`` 给定时只返回该知识库 + 无归属的条目，默认不过滤。"""
        memory = _memory()
        return {
            "space_id": space,
            "stats": memory.stats(space_id=space),
            "summary": memory.summary(space_id=space),
            "projects": [state.to_dict() for state in memory.projects(space_id=space).values()],
            "decisions": [entry.to_dict() for entry in memory.decisions(space_id=space)],
            "lessons": [entry.to_dict() for entry in memory.lessons(space_id=space)],
        }

    @app.post("/api/memory/distill")
    def post_distill(payload: DistillPayload) -> dict[str, Any]:
        """手动触发一次「从文档提炼长期记忆」。"""
        backend, spec = build_distill_backend(runtime)
        try:
            result = distill_memory(
                data_dir=runtime.data_dir,
                corpus=_corpus(),
                memory=_memory(),
                space_id=payload.space_id,
                project=payload.project,
                backend=backend,
                spec=spec,
                use_llm=payload.use_llm,
            )
        finally:
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()
        return {**result.to_dict(), "dashboard": _dashboard(payload.day, payload.space_id)}

    # -- 待办 CRUD --------------------------------------------------------- #
    @app.get("/api/todos")
    def list_todos(space: str = "", day: str = "") -> dict[str, Any]:
        corpus = _corpus()
        target = as_date(day or None)
        summary = load_summary(runtime.data_dir, space_id=space)
        items = merged_todos(
            _todos(), corpus, space_id=space, day=target, llm_todos=_llm_todos(summary)
        )
        candidates = collect_candidates(runtime.data_dir, corpus, space_id=space, day=target)
        return {
            "todos": items,
            "stats": _todo_stats(items),
            "candidates": len(candidates),
            "candidate_todos": candidates,
        }

    @app.post("/api/todos")
    def create_todo(payload: TodoCreatePayload) -> dict[str, Any]:
        return _todos().add(
            payload.text, owner=payload.owner, due=payload.due, space_id=payload.space_id
        ).to_dict()

    @app.put("/api/todos/manual/{todo_id}")
    def update_manual_todo(todo_id: str, payload: TodoUpdatePayload) -> dict[str, Any]:
        todo = _todos().update(
            todo_id, text=payload.text, status=payload.status, owner=payload.owner, due=payload.due
        )
        if todo is None:
            raise HTTPException(status_code=404, detail=f"待办不存在：{todo_id}")
        return todo.to_dict()

    @app.delete("/api/todos/manual/{todo_id}")
    def delete_manual_todo(todo_id: str) -> dict[str, Any]:
        if not _todos().delete(todo_id):
            raise HTTPException(status_code=404, detail=f"待办不存在：{todo_id}")
        return {"ok": True, "id": todo_id}

    @app.put("/api/todos/corpus/{key}")
    def override_corpus_todo(key: str, payload: TodoUpdatePayload) -> dict[str, Any]:
        """文档待办的「改写 / 完成」——只记录覆盖，不改源 Markdown。"""
        return {
            "key": key,
            "override": _todos().set_override(
                key, text=payload.text, status=payload.status, owner=payload.owner, due=payload.due
            ),
        }

    @app.delete("/api/todos/corpus/{key}")
    def hide_corpus_todo(key: str) -> dict[str, Any]:
        """「删除」文档待办 —— 记录隐藏标记（源文档不动）。"""
        _todos().set_override(key, deleted=True)
        return {"ok": True, "key": key}

    # -- 周报 -------------------------------------------------------------- #
    @app.get("/api/weekly")
    def get_weekly(day: str = "", space: str = "") -> dict[str, Any]:
        """周报预览：跟随工作台选中的知识库（``space``）；只读缓存 / 规则回退，不调模型。"""
        return {**_weekly_payload(as_date(day or None), space), "weekly_space_id": knowledge.weekly_space_id}

    @app.post("/api/weekly/summarize")
    def post_weekly_summarize(payload: WeeklySummaryPayload) -> dict[str, Any]:
        """按「本周该知识库的更新文档正文」用 LLM 归纳周报，并写入缓存。"""
        summary = _run_weekly_summarize(space_id=payload.space_id, day=payload.day, use_llm=payload.use_llm)
        return {**summary.to_dict(), "label": label_for(as_date(payload.day or None)).to_dict()}

    @app.post("/api/weekly/publish")
    def post_weekly(payload: WeeklyPublishPayload) -> dict[str, Any]:
        target_space = knowledge.weekly_space_id
        if not target_space:
            raise HTTPException(status_code=500, detail="未配置周报写入的知识库（configs/knowledge_bases.yaml）")

        # 发布当前预览（已生成则用 LLM 结果，否则规则回退）；生成请走 /api/weekly/summarize。
        summary = _weekly_preview(as_date(payload.date or None), payload.space_id)
        client = FeishuClient(
            space_id=target_space,
            base_url=runtime.feishu_base_url,
            site_url=runtime.feishu_site_url,
            settings=runtime,
        )
        try:
            result = publish_weekly(
                client,
                space_id=target_space,
                markdown=summary.markdown,
                title=summary.week,
                template_token=runtime.feishu_weekly_template,
                template_keyword=runtime.feishu_weekly_template_keyword,
            )
        except (FeishuError, SecretError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            client.close()
        return result.to_dict()

    # -- 前端 -------------------------------------------------------------- #
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    logger.info("可视化网站已装配：data_dir=%s，知识库 %d 个", runtime.data_dir, len(knowledge.options))
    return app