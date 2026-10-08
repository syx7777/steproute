"""Agent 工具集。

工具是 Agent 与科研材料之间的唯一接口：检索、读取、抽取、比较、登记证据、收尾。
每个工具返回 :class:`ToolResult`，其中的 ``patch`` 会被运行时合并回
:class:`~steproute.agent.state.AgentState`，从而把"工具副作用"变成
下一轮路由可用的状态信号（例如 ``numeric_conflict`` 标记会显著抬高难度判断）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..corpus.model import (
    COMPARABILITY_KEYS,
    Corpus,
    SourceKind,
    experiment_role,
    metric_direction,
)
from ..corpus.retriever import Retriever
from .state import AgentState


@dataclass
class ToolResult:
    """工具执行结果。"""

    tool: str
    output: str
    ok: bool = True
    progressed: bool = False
    patch: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "progressed": self.progressed, "output": self.output}


@dataclass
class ToolSpec:
    """工具元信息（用于生成 prompt 中的工具目录）。"""

    name: str
    description: str
    args: dict[str, str]
    step_type: str

    def render(self) -> str:
        args = ", ".join(f"{key}: {value}" for key, value in self.args.items()) or "无"
        return f"- `{self.name}`（{self.step_type}）：{self.description}\n    参数：{args}"


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("search_corpus", "在周报/纪要/实验记录/仓库中做 BM25 检索，返回最相关片段与 doc_id。",
             {"query": "检索词", "top_k": "返回条数，默认 5", "kind": "可选，限定来源类型"}, "retrieve"),
    ToolSpec("read_document", "通读指定 doc_id 的文档正文，用于提取结论、实验编号与待办。",
             {"doc_id": "文档 id（来自 search_corpus 结果）"}, "read"),
    ToolSpec("list_experiments", "列出全部实验记录（实验编号、状态、关键指标、配置）。", {}, "extract"),
    ToolSpec("get_experiment_metrics", "获取单个实验的完整指标与配置。", {"experiment_id": "实验编号"}, "extract"),
    ToolSpec("compare_experiments", "对比两组实验的指标差异，并检查**配置是否公平**（会检出不一致项）。",
             {"experiment_a": "实验编号 A", "experiment_b": "实验编号 B", "metric": "可选，指定单一指标"}, "compare"),
    ToolSpec("extract_todos", "从周报评论与会议纪要中抽取尚未完成的待办事项。", {}, "extract"),
    ToolSpec("record_finding", "把一条关键发现登记为证据（可被最终报告引用）。",
             {"claim": "发现结论", "source_ids": "来源 doc_id 列表", "kind": "finding/metric/analysis"}, "synthesize"),
    ToolSpec("finalize", "收尾：输出研究进度报告与下一步任务列表。",
             {"report": "Markdown 报告正文", "next_tasks": "下一步任务列表"}, "final"),
]

TOOL_INDEX: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def render_tool_catalog() -> str:
    return "\n".join(spec.render() for spec in TOOL_SPECS)


# --------------------------------------------------------------------------- #
# 执行器
# --------------------------------------------------------------------------- #

class ToolExecutor:
    """在给定语料上执行工具调用。"""

    def __init__(self, corpus: Corpus, retriever: Retriever, *, max_output_chars: int = 3000) -> None:
        self.corpus = corpus
        self.retriever = retriever
        self.max_output_chars = max_output_chars
        self._handlers: dict[str, Callable[[AgentState, dict[str, Any]], ToolResult]] = {
            "search_corpus": self._search_corpus,
            "read_document": self._read_document,
            "list_experiments": self._list_experiments,
            "get_experiment_metrics": self._get_experiment_metrics,
            "compare_experiments": self._compare_experiments,
            "extract_todos": self._extract_todos,
            "record_finding": self._record_finding,
            "finalize": self._finalize,
        }

    def execute(self, state: AgentState, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        args = args or {}
        state.tool_calls += 1
        state.last_tool = tool
        handler = self._handlers.get(tool)
        if handler is None:
            result = ToolResult(
                tool=tool,
                ok=False,
                output=f"未知工具 `{tool}`。可用工具：{', '.join(self._handlers)}",
                patch={"last_error": f"unknown tool: {tool}"},
            )
        else:
            try:
                result = handler(state, args)
            except Exception as exc:  # 工具异常也要变成可观测状态，而不是崩溃
                result = ToolResult(
                    tool=tool,
                    ok=False,
                    output=f"工具 `{tool}` 执行失败：{exc}",
                    patch={"last_error": f"{tool}: {exc}"},
                )
        result.output = result.output[: self.max_output_chars]
        if result.ok:
            result.patch.setdefault("clear_error", True)
            state.mark_tool_done(tool)
        # 副作用统一在这里合并回状态（工具是状态的唯一写入方）
        state.apply_patch(result.patch)
        return result

    # -- 工具实现 ---------------------------------------------------------- #
    def _search_corpus(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or state.task)
        top_k = int(args.get("top_k") or 5)
        kind_raw = args.get("kind")
        kind = None
        if kind_raw:
            try:
                kind = SourceKind(str(kind_raw))
            except ValueError:
                kind = None
        hits = self.retriever.search(query, kind=kind, top_k=max(1, top_k))
        if not hits:
            return ToolResult("search_corpus", f"没有检索到与「{query}」相关的材料。", ok=True, progressed=False)
        new_ids = [hit.chunk.doc_id for hit in hits if hit.chunk.doc_id not in state.retrieved_ids]
        lines = [f"检索「{query}」命中 {len(hits)} 段：", ""]
        for rank, hit in enumerate(hits, 1):
            lines.append(f"[{rank}] {hit.chunk.title} ({hit.chunk.doc_id}, {hit.chunk.kind.label}, score={hit.score:.2f})")
            lines.append("    " + hit.chunk.text[:220].replace("\n", " "))
        return ToolResult(
            "search_corpus",
            "\n".join(lines),
            progressed=bool(new_ids),
            patch={"retrieved_ids": [hit.chunk.doc_id for hit in hits]},
        )

    def _read_document(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        doc_id = str(args.get("doc_id") or "")
        doc = self.corpus.get(doc_id)
        if doc is None:
            return ToolResult("read_document", f"找不到文档 {doc_id}。请先用 search_corpus 获取有效 doc_id。", ok=False,
                              patch={"last_error": f"doc not found: {doc_id}"})
        experiment_ids: list[str] = []
        if doc.kind is SourceKind.EXPERIMENT and doc.meta.get("experiment_id"):
            experiment_ids.append(str(doc.meta["experiment_id"]))
        if state.flags.get("_pending_doc_flag"):
            state.flags.pop("_pending_doc_flag", None)
        progressed = doc_id not in state.read_ids
        body = doc.text
        header = f"# {doc.title}\n（doc_id={doc.doc_id}, 来源={doc.kind.label}）\n\n"
        return ToolResult(
            "read_document",
            header + body,
            progressed=progressed,
            patch={"read_ids": [doc_id], "experiment_ids": experiment_ids},
        )

    def _list_experiments(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        rows = self.corpus.list_experiments()
        if not rows:
            return ToolResult("list_experiments", "未找到任何实验记录。", ok=False,
                              patch={"last_error": "no experiments"})
        lines = ["| 实验编号 | 名称 | 状态 | 关键指标 |", "| --- | --- | --- | --- |"]
        for row in rows:
            metrics = ", ".join(f"{k}={v}" for k, v in list(row["metrics"].items())[:3])
            lines.append(f"| {row['experiment_id']} | {row['name']} | {row['status']} | {metrics} |")
        return ToolResult(
            "list_experiments",
            "\n".join(lines),
            progressed=not state.experiment_ids,
            patch={"experiment_ids": [row["experiment_id"] for row in rows]},
        )

    def _get_experiment_metrics(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        experiment_id = str(args.get("experiment_id") or "")
        doc = self.corpus.experiment(experiment_id)
        if doc is None:
            return ToolResult("get_experiment_metrics", f"找不到实验 {experiment_id}。", ok=False,
                              patch={"last_error": f"experiment not found: {experiment_id}"})
        payload = {
            "experiment_id": experiment_id,
            "name": doc.meta.get("name", doc.title),
            "status": doc.meta.get("status", ""),
            "config": doc.meta.get("config") or {},
            "metrics": doc.meta.get("metrics") or {},
            "notes": doc.meta.get("notes", ""),
        }
        return ToolResult(
            "get_experiment_metrics",
            json.dumps(payload, ensure_ascii=False, indent=2),
            progressed=True,
            patch={"experiment_ids": [experiment_id]},
        )

    def _compare_experiments(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        a_id = str(args.get("experiment_a") or "")
        b_id = str(args.get("experiment_b") or "")
        doc_a = self.corpus.experiment(a_id)
        doc_b = self.corpus.experiment(b_id)
        if doc_a is None or doc_b is None:
            return ToolResult("compare_experiments", f"实验编号无效：{a_id} / {b_id}", ok=False,
                              patch={"last_error": "invalid experiment ids"})
        return self._compare_docs(state, doc_a, doc_b, metric=args.get("metric"))

    def _compare_docs(self, state: AgentState, doc_a, doc_b, *, metric: str | None) -> ToolResult:
        id_a = str(doc_a.meta.get("experiment_id"))
        id_b = str(doc_b.meta.get("experiment_id"))
        config_a: dict[str, Any] = doc_a.meta.get("config") or {}
        config_b: dict[str, Any] = doc_b.meta.get("config") or {}

        # 区分"方法"与"baseline"，让结论直接用"是否超过 baseline"表述
        if experiment_role(doc_a) == "baseline" and experiment_role(doc_b) != "baseline":
            base, method = doc_a, doc_b
        elif experiment_role(doc_b) == "baseline" and experiment_role(doc_a) != "baseline":
            base, method = doc_b, doc_a
        else:
            base, method = doc_a, doc_b
        base_id = str(base.meta.get("experiment_id"))
        method_id = str(method.meta.get("experiment_id"))
        base_metrics: dict[str, Any] = base.meta.get("metrics") or {}
        method_metrics: dict[str, Any] = method.meta.get("metrics") or {}

        common = [name for name in method_metrics if name in base_metrics]
        if metric:
            common = [name for name in common if name == metric] or common

        lines = [f"对比：方法 {method_id} vs baseline {base_id}", ""]
        worse_primary: list[str] = []
        better_primary: list[str] = []
        for name in common:
            try:
                method_value = float(method_metrics[name])
                base_value = float(base_metrics[name])
            except (TypeError, ValueError):
                continue
            delta = method_value - base_value
            direction = metric_direction(name)
            if abs(delta) < 1e-9:
                verdict = "持平"
            elif direction == 0:
                verdict = "有差异"
            elif (delta > 0) == (direction > 0):
                verdict = "更好"
            else:
                verdict = "更差"
            if direction > 0:
                if verdict == "更好":
                    better_primary.append(name)
                elif verdict == "更差":
                    worse_primary.append(name)
            lines.append(f"- {name}: 方法={method_value:g}  baseline={base_value:g}  Δ={delta:+g}  判定={verdict}")

        # 公平性检查：只有"可比较性关键项"不一致才会让指标失去意义；
        # sampling / fusion 这类差异是方法本身带来的，属于预期内。
        all_keys = set(config_a) | set(config_b)
        critical_mismatch = sorted(
            key for key in all_keys if key in COMPARABILITY_KEYS and config_a.get(key) != config_b.get(key)
        )
        method_diffs = sorted(
            key for key in all_keys if key not in COMPARABILITY_KEYS and config_a.get(key) != config_b.get(key)
        )
        lines.append("")
        if critical_mismatch:
            lines.append("⚠️ 可比较性关键配置不一致（指标对比不公平）：")
            for key in critical_mismatch:
                lines.append(f"  - {key}: {id_a}={config_a.get(key)}  {id_b}={config_b.get(key)}")
        else:
            lines.append("✅ 可比较性关键配置一致（骨干/帧数/分辨率/评测切分/种子/训练超参），指标对比公平。")
        if method_diffs:
            lines.append("方法相关差异（预期内，不构成不公平）：")
            for key in method_diffs:
                lines.append(f"  - {key}: {id_a}={config_a.get(key)}  {id_b}={config_b.get(key)}")

        flags: dict[str, bool] = {}
        lines.append("")
        if critical_mismatch and worse_primary:
            flags["numeric_conflict"] = True
            lines.append(
                f"结论提示：方法在 {', '.join(worse_primary)} 上更差，但关键配置与 baseline 不一致——"
                "**当前对比不公平，先对齐配置（抽帧数/分辨率/评测切分/随机种子）再判断方法优劣**。"
            )
        elif critical_mismatch:
            lines.append(
                "结论提示：关键配置与 baseline 不一致，指标差异不能作为结论，"
                "需要先对齐配置（抽帧数/分辨率/评测切分/随机种子）。"
            )
        elif worse_primary:
            lines.append(
                f"结论提示：关键配置一致，但方法在 {', '.join(worse_primary)} 上更差，需要从方法本身排查原因。"
            )
        else:
            lines.append(
                "结论提示：关键配置一致，方法在主要指标上不落后于 baseline"
                f"（领先项：{', '.join(better_primary) or '无'}）。"
            )

        patch: dict[str, Any] = {"flags": flags, "experiment_ids": [method_id, base_id]}
        return ToolResult("compare_experiments", "\n".join(lines), progressed=True, patch=patch)

    def _extract_todos(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        todos = self.corpus.scan_todos()
        if not todos:
            return ToolResult("extract_todos", "没有抽取到待办事项。", progressed=False)
        lines = [f"共抽取到 {len(todos)} 条待办：", ""]
        for todo in todos:
            mark = "x" if todo.status == "done" else " "
            lines.append(f"- [{mark}] {todo.text}  (来源: {todo.doc_id})")
        open_todos = [t.text for t in todos if t.status == "open"]
        return ToolResult(
            "extract_todos",
            "\n".join(lines),
            progressed=bool(open_todos) and not state.todos,
            patch={"todos": open_todos},
        )

    def _record_finding(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        claim = str(args.get("claim") or "").strip()
        if not claim:
            return ToolResult("record_finding", "claim 不能为空。", ok=False, patch={"last_error": "empty claim"})
        source_ids = args.get("source_ids") or []
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        kind = str(args.get("kind") or "finding")
        evidence = state.add_evidence(claim, [str(s) for s in source_ids], kind=kind, detail=str(args.get("detail") or ""))
        return ToolResult(
            "record_finding",
            f"已登记证据 {evidence.id}: {evidence.claim}",
            progressed=True,
            patch={"evidence_added": evidence.id},
        )

    def _finalize(self, state: AgentState, args: dict[str, Any]) -> ToolResult:
        report = str(args.get("report") or "").strip()
        raw_tasks = args.get("next_tasks") or []
        if isinstance(raw_tasks, str):
            raw_tasks = [line for line in raw_tasks.splitlines() if line.strip()]
        tasks = [str(t).strip(" -*\t") for t in raw_tasks if str(t).strip()]
        if not tasks:
            tasks = list(state.todos[:6]) or ["统一评测配置后重新对比 baseline"]
        if not report:
            report = "## 当前研究进展\n- " + ("\n- ".join(e.claim for e in state.evidence) or "证据不足")
        return ToolResult(
            "finalize",
            "已生成研究进度报告与下一步任务。",
            progressed=True,
            patch={"finished": True, "final_report": report, "next_tasks": tasks},
        )