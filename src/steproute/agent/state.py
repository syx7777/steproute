"""Agent 状态：路由所需的全部"当前处境"信号都在这里。

设计动机（来自项目背景）
------------------------
仅凭 query 或 context length 判断难度并不可靠——
「从长纪要里找一个实验名」很简单，而「判断两个接近的实验为什么有差异」即使输入很短也很难。
因此状态里显式携带：**步骤类型、轨迹深度、工具失败、无进展步数、子目标缺口、
已有多来源证据、数值冲突 / 歧义标记** 等 Agent State 信号。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..llm.schema import Usage


class StepType(str, Enum):
    """Agent 步骤类型（对应背景里提到的信息抽取 / 检索 / 结果比较 / 实验分析等）。"""

    RETRIEVE = "retrieve"        # 检索：从语料里找相关材料
    READ = "read"                # 读取：通读某篇文档
    EXTRACT = "extract"          # 信息抽取：抽取 TODO / 指标 / 实验编号
    COMPARE = "compare"          # 结果比较：对比两组实验
    ANALYZE = "analyze"          # 实验分析：解释差异、判断优先级
    SYNTHESIZE = "synthesize"    # 综合：把发现登记为证据、归纳进度
    FINAL = "final"              # 收尾：产出进度报告与下一步任务

    @classmethod
    def parse(cls, value: str | None, default: StepType = None) -> StepType:  # type: ignore[assignment]
        if value:
            try:
                return cls(str(value).strip().lower())
            except ValueError:
                pass
        return default or cls.RETRIEVE


@dataclass
class Subgoal:
    """一条待完成的子目标。"""

    id: str
    text: str
    status: str = "open"  # open / done
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "status": self.status, "evidence_ids": list(self.evidence_ids)}


@dataclass
class Evidence:
    """一条可引用的证据（发现 / 指标 / 结论）。"""

    id: str
    claim: str
    source_ids: list[str] = field(default_factory=list)
    kind: str = "finding"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "claim": self.claim, "source": ",".join(self.source_ids), "kind": self.kind}


@dataclass
class StepRecord:
    """单个 Agent Step 的执行记录（路由与调度的可观测性来源）。"""

    index: int
    step_type: str
    required_tier: str          # Jev 判断出的能力下限
    model_id: str               # Scheduler 实际选中的模型
    tier: str
    tool: str = ""
    tool_ok: bool = True
    progressed: bool = False
    escalated: bool = False
    jev_mode: str = ""
    jev_reason: str = ""
    jev_confidence: float = 0.0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    inflight_at_pick: int = 0
    cache_hit_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step_type": self.step_type,
            "required_tier": self.required_tier,
            "model_id": self.model_id,
            "tier": self.tier,
            "tool": self.tool,
            "tool_ok": self.tool_ok,
            "progressed": self.progressed,
            "escalated": self.escalated,
            "jev_mode": self.jev_mode,
            "jev_reason": self.jev_reason,
            "jev_confidence": round(self.jev_confidence, 3),
            "latency_s": round(self.latency_s, 3),
            "cost_usd": round(self.cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "inflight_at_pick": self.inflight_at_pick,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
        }


@dataclass
class AgentState:
    """一次运行中不断演化的完整状态。"""

    task: str
    max_steps: int = 14
    step_index: int = 0

    subgoals: list[Subgoal] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    retrieved_ids: list[str] = field(default_factory=list)
    read_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    completed_tools: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)

    history: list[StepRecord] = field(default_factory=list)
    tool_calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    no_progress_streak: int = 0
    prompt_tokens_last: int = 0

    pending_step_type: StepType = StepType.RETRIEVE
    last_tool: str = ""
    last_error: str = ""
    flags: dict[str, bool] = field(default_factory=dict)

    finished: bool = False
    final_report: str = ""
    next_tasks: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    # -- 子目标 ------------------------------------------------------------ #
    def seed_subgoals(self, texts: list[str]) -> None:
        self.subgoals = [Subgoal(id=f"g{i + 1}", text=text) for i, text in enumerate(texts)]

    @property
    def open_subgoals(self) -> list[Subgoal]:
        return [g for g in self.subgoals if g.status != "done"]

    @property
    def covered_subgoals(self) -> list[Subgoal]:
        return [g for g in self.subgoals if g.status == "done"]

    # -- 证据 -------------------------------------------------------------- #
    def next_evidence_id(self) -> str:
        return f"e{len(self.evidence) + 1}"

    def add_evidence(self, claim: str, source_ids: list[str], *, kind: str = "finding", detail: str = "") -> Evidence:
        item = Evidence(id=self.next_evidence_id(), claim=claim.strip(), source_ids=list(source_ids), kind=kind, detail=detail)
        self.evidence.append(item)
        # 依据证据来源，把相关子目标标记为已覆盖
        for goal in self.open_subgoals:
            if _goal_matches(goal.text, claim):
                goal.status = "done"
                goal.evidence_ids.append(item.id)
        return item

    def distinct_evidence_sources(self) -> int:
        sources: set[str] = set()
        for item in self.evidence:
            sources.update(item.source_ids)
        return len(sources)

    # -- 进度 -------------------------------------------------------------- #
    def record_step(self, record: StepRecord) -> None:
        self.history.append(record)
        self.step_index = record.index + 1
        if record.progressed:
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
        if record.tool_ok:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def mark_tool_done(self, tool: str) -> None:
        if tool and tool not in self.completed_tools:
            self.completed_tools.append(tool)

    # -- 工具副作用合并 ---------------------------------------------------- #
    def apply_patch(self, patch: dict[str, Any]) -> None:
        """把 :class:`~steproute.agent.tools.ToolResult` 的副作用合并回状态。

        工具是状态的唯一写入方，这样下一轮路由能立即看到"数值冲突""新增证据"
        "实验编号"等信号。
        """
        for key, value in patch.items():
            if key in {"retrieved_ids", "read_ids", "experiment_ids"}:
                bucket: list[str] = getattr(self, key)
                for item in value:
                    if item and item not in bucket:
                        bucket.append(item)
            elif key == "todos":
                for item in value:
                    if item and item not in self.todos:
                        self.todos.append(item)
            elif key == "flags":
                for flag, flag_value in value.items():
                    self.flags[flag] = bool(flag_value)
            elif key == "finished":
                self.finished = bool(value)
            elif key == "final_report":
                self.final_report = str(value)
            elif key == "next_tasks":
                self.next_tasks = [str(item) for item in value]
            elif key == "last_error":
                self.last_error = str(value)
            elif key == "clear_error":
                if value:
                    self.last_error = ""
            # 其它键（如 evidence_added）仅作观测，无需回写

    # -- 路由特征输入 ------------------------------------------------------ #
    def prompt_state(self) -> dict[str, Any]:
        """内嵌到 prompt 的**结构化状态**（同时被离线仿真后端解析使用）。"""
        return {
            "task": self.task,
            "step_index": self.step_index,
            "max_steps": self.max_steps,
            "pending_step_type": self.pending_step_type.value,
            "subgoals": [g.to_dict() for g in self.subgoals],
            "evidence": [e.to_dict() for e in self.evidence],
            "retrieved_ids": list(self.retrieved_ids),
            "read_ids": list(self.read_ids),
            "experiment_ids": list(self.experiment_ids),
            "completed_tools": list(self.completed_tools),
            "todos": list(self.todos),
            "tool_calls": self.tool_calls,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "no_progress_streak": self.no_progress_streak,
            "last_tool": self.last_tool,
            "last_error": self.last_error,
            "flags": dict(self.flags),
        }

    def state_json(self) -> str:
        return json.dumps(self.prompt_state(), ensure_ascii=False, indent=2)

    def cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.history)

    def tier_usage(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for record in self.history:
            usage[record.tier] = usage.get(record.tier, 0) + 1
        return usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": len(self.history),
            "finished": self.finished,
            "subgoals": [g.to_dict() for g in self.subgoals],
            "evidence": [e.to_dict() for e in self.evidence],
            "next_tasks": list(self.next_tasks),
            "flags": dict(self.flags),
            "trajectory": [r.to_dict() for r in self.history],
        }


def _goal_matches(goal_text: str, claim: str) -> bool:
    """粗糙的"子目标是否被该证据覆盖"判断：关键词重合度。"""
    from ..corpus.retriever import tokenize

    goal_tokens = set(tokenize(goal_text))
    if not goal_tokens:
        return False
    claim_tokens = set(tokenize(claim))
    overlap = len(goal_tokens & claim_tokens)
    return overlap >= max(2, int(len(goal_tokens) * 0.34))