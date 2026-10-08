"""Agent 运行时：**逐步骤**运行的 Harness 主循环。

每一步的执行顺序（这就是"Harness"本身）：

1. 构造当前状态（:class:`AgentState`）→ 抽取步级特征（:mod:`.routing.features`）；
2. **Jev** 判断该步最低需要的能力档；
3. **RuntimeScheduler** 结合队列 / 延迟 / 前缀缓存 / 成本选出实际模型；
4. 调用模型 → 解析动作 JSON → 执行工具 → 把副作用合并回状态；
5. 记录本步的路由/调度/成本/延迟，进入下一步，直到 ``finalize`` 或步数耗尽。

同一套循环通过 ``strategy`` 参数即可切换对照策略
（``always_strong`` / ``query_level`` / ``step_level``），用于对比实验。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig, Settings, tier_rank
from ..corpus.model import Corpus
from ..corpus.retriever import Retriever
from ..llm.backends import Backend, BackendError
from ..llm.json_utils import JSONExtractError, extract_json_block
from ..llm.pool import ModelPool
from ..logging_utils import get_logger
from ..routing.cache import PrefixCacheTracker
from ..routing.features import extract_features
from ..routing.jev import JevRouter
from ..routing.scheduler import RuntimeScheduler
from .prompts import SYSTEM_PROMPT, build_messages, seed_subgoals
from .state import AgentState, StepRecord, StepType
from .tools import ToolExecutor, ToolResult

logger = get_logger(__name__)

#: 对照策略。
STRATEGIES = ("step_level", "always_strong", "always_fast", "query_level")


@dataclass
class RunResult:
    """一次运行的完整产物。"""

    task: str
    strategy: str
    report: str
    next_tasks: list[str]
    steps: int
    elapsed_s: float
    cost_usd: float
    tier_usage: dict[str, int]
    success: bool
    state: dict[str, Any]
    trajectory: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "strategy": self.strategy,
            "report": self.report,
            "next_tasks": self.next_tasks,
            "steps": self.steps,
            "elapsed_s": round(self.elapsed_s, 3),
            "cost_usd": round(self.cost_usd, 6),
            "tier_usage": self.tier_usage,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "state": self.state,
            "trajectory": self.trajectory,
        }


class HarnessRuntime:
    """多步 Agent 的运行时（Step-level Harness）。"""

    def __init__(
        self,
        *,
        config: AppConfig,
        settings: Settings,
        pool: ModelPool,
        backend: Backend,
        corpus: Corpus,
        retriever: Retriever,
        jev: JevRouter,
        scheduler: RuntimeScheduler,
        cache: PrefixCacheTracker,
        executor: ToolExecutor,
        strategy: str = "step_level",
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略 {strategy}，可选：{STRATEGIES}")
        self.config = config
        self.settings = settings
        self.pool = pool
        self.backend = backend
        self.corpus = corpus
        self.jev = jev
        self.scheduler = scheduler
        self.cache = cache
        self.executor = executor
        self.strategy = strategy
        #: 固定前缀（系统 Prompt）的指纹，用于 Cache-aware 打分。
        self.prefix_key = self._prefix_key()

    def _prefix_key(self) -> str:
        from ..routing.cache import prefix_fingerprint

        return prefix_fingerprint(SYSTEM_PROMPT)

    # -- 主循环 ------------------------------------------------------------ #
    def run(self, task: str) -> RunResult:
        state = AgentState(task=task, max_steps=self.config.agent.max_steps)
        state.seed_subgoals(seed_subgoals(task))
        state.pending_step_type = StepType.RETRIEVE
        predicted = StepType.RETRIEVE

        locked_tier: str | None = None
        if self.strategy == "query_level":
            locked_tier = self._query_level_tier(task)
            logger.info("[query_level] 初始 Query 判档为 %s，全程锁定", locked_tier)

        started = time.perf_counter()
        error = ""

        while state.step_index < state.max_steps:
            messages = build_messages(
                state,
                history_window=self.config.agent.history_window,
                max_tool_chars=self.config.agent.max_tool_chars,
            )
            context_len = estimate_prompt_tokens(messages)
            features = extract_features(state, predicted_step_type=predicted, context_len=context_len)

            # ---- 1) Jev 能力判断 ----
            decision = self.jev.route(state, features)
            required = decision.tier
            reason = decision.reason
            mode = decision.mode

            # ---- 2) 对照策略覆写 ----
            required, override_reason = self._apply_strategy(required, locked_tier)
            if override_reason:
                reason = f"{reason}；策略覆写：{override_reason}"
                mode = f"{mode}+{self.strategy}"

            # ---- 3) Runtime Scheduler 选模型 ----
            prompt_tokens = context_len
            completion_tokens = 512
            sched = self.scheduler.schedule(
                required,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefix_key=self.prefix_key,
            )
            runtime = self.pool.runtime(sched.model_id)
            spec = runtime.spec

            # ---- 4) 调用模型 ----
            self.scheduler.acquire(runtime)
            try:
                result = self.backend.complete(spec, messages, json_mode=True, label=f"step{state.step_index}-{sched.model_id}")
                usage = result.usage
                latency = result.latency_s
                content = result.content
                call_ok = True
            except BackendError as exc:
                usage = None
                latency = 0.0
                content = ""
                call_ok = False
                error = str(exc)
                logger.error("步骤 %d 调用失败：%s", state.step_index, exc)
            finally:
                self.scheduler.release(runtime)

            if not call_ok or usage is None:
                state.failures += 1
                state.last_error = error
                record = StepRecord(
                    index=state.step_index,
                    step_type=features.step_type,
                    required_tier=required,
                    model_id=sched.model_id,
                    tier=runtime.tier,
                    tool="",
                    tool_ok=False,
                    progressed=False,
                    jev_mode=mode,
                    jev_reason=reason,
                    jev_confidence=decision.confidence,
                    latency_s=latency,
                    inflight_at_pick=sched.inflight,
                    cache_hit_ratio=sched.cache_hit_ratio,
                )
                state.record_step(record)
                if state.consecutive_failures >= 3:
                    break
                continue

            # 观测：成本 / 缓存
            runtime.observe(usage, latency, ok=True)
            self.cache.observe(
                self.prefix_key,
                sched.model_id,
                prompt_tokens=usage.prompt_tokens,
                cached_tokens=usage.cached_tokens,
            )
            state.prompt_tokens_last = usage.prompt_tokens
            state.usage = state.usage + usage
            cost = runtime.cost_of(usage)

            # ---- 5) 解析动作并执行工具 ----
            action, parse_error = parse_action(content)
            tool = ""
            tool_result = ToolResult(tool="", output="", ok=False)
            if action is None:
                state.failures += 1
                state.last_error = parse_error
            else:
                tool = str(action.get("tool") or "")
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                # 工具会在执行时把副作用合并回 state（见 ToolExecutor.execute）
                tool_result = self.executor.execute(state, tool, args)
                if not tool_result.ok:
                    state.failures += 1

            record = StepRecord(
                index=state.step_index,
                step_type=StepType.parse(str(action.get("step_type")) if action else None, features.step_type).value,
                required_tier=required,
                model_id=sched.model_id,
                tier=runtime.tier,
                tool=tool,
                tool_ok=tool_result.ok and action is not None,
                progressed=tool_result.progressed,
                escalated=tier_rank(required) > tier_rank(decision.base_tier or required),
                jev_mode=mode,
                jev_reason=reason,
                jev_confidence=decision.confidence,
                latency_s=latency,
                cost_usd=cost,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                inflight_at_pick=sched.inflight,
                cache_hit_ratio=sched.cache_hit_ratio,
            )
            state.record_step(record)

            # ---- 6) 预测下一步（供下一轮路由） ----
            state.pending_step_type = StepType.parse(
                str(action.get("step_type")) if action else None, predicted
            )
            predicted = StepType.parse(str(action.get("next_step_type")) if action else None, state.pending_step_type)

            if state.finished:
                break

        elapsed = time.perf_counter() - started
        if not state.finished:
            self._fallback_finalize(state)

        return RunResult(
            task=task,
            strategy=self.strategy,
            report=state.final_report,
            next_tasks=state.next_tasks,
            steps=len(state.history),
            elapsed_s=elapsed,
            cost_usd=state.cost_usd(),
            tier_usage=state.tier_usage(),
            success=state.finished and bool(state.evidence),
            state=state.to_dict(),
            trajectory=[record.to_dict() for record in state.history],
            metrics={
                "pool": self.pool.snapshot(),
                "cache": self.cache.snapshot(),
                "jev": self.jev.stats(),
                "evidence": len(state.evidence),
                "distinct_sources": state.distinct_evidence_sources(),
                "subgoals_done": len(state.covered_subgoals),
                "subgoals_total": len(state.subgoals),
                "todos": len(state.todos),
                "failures": state.failures,
            },
            error=error,
        )

    # -- 策略 -------------------------------------------------------------- #
    def _apply_strategy(self, required: str, locked_tier: str | None) -> tuple[str, str]:
        if self.strategy == "always_strong":
            return "strong", "始终使用 strong"
        if self.strategy == "always_fast":
            return "fast", "始终使用 fast"
        if self.strategy == "query_level" and locked_tier:
            return locked_tier, f"Query 级路由锁定 {locked_tier}"
        return required, ""

    @staticmethod
    def _query_level_tier(task: str) -> str:
        """Query-level Routing：只看用户最初的问题就一次性定档（对照组）。"""
        lowered = task.lower()
        hard = ("为什么", "原因", "没超过", "未超过", "落后", "分析", "诊断", "对比", "比较", "权衡", "判断")
        easy = ("列出", "抽取", "提取", "整理", "汇总", "找")
        if any(keyword in lowered for keyword in hard):
            return "strong"
        if any(keyword in lowered for keyword in easy):
            return "fast"
        return "balanced"

    # -- 兜底收尾 ---------------------------------------------------------- #
    def _fallback_finalize(self, state: AgentState) -> None:
        """步数耗尽时的兜底收尾：基于已收集证据与 TODO 生成报告。"""
        lines = ["## 当前研究进展", ""]
        if state.evidence:
            lines += [f"- {item.claim}（证据 {item.id}）" for item in state.evidence]
        else:
            lines.append("- 未能在预算内收集到充分证据。")
        lines += ["", "## 下一步任务", ""]
        tasks = state.next_tasks or state.todos or ["重新运行并提高步数预算"]
        lines += [f"1. {task}" for task in tasks]
        state.final_report = state.final_report or "\n".join(lines)
        state.next_tasks = tasks
        state.finished = True
        state.flags["fallback_finalize"] = True


def parse_action(content: str) -> tuple[dict[str, Any] | None, str]:
    """解析模型输出的动作 JSON。"""
    if not content.strip():
        return None, "模型未返回内容"
    try:
        payload = extract_json_block(content)
    except JSONExtractError as exc:
        return None, f"动作 JSON 解析失败：{exc}"
    if not isinstance(payload, dict):
        return None, f"动作必须是 JSON 对象，实际为 {type(payload).__name__}"
    if "tool" not in payload:
        # 容忍直接给出 report 的收尾形式
        if "report" in payload:
            return {"step_type": "final", "tool": "finalize", "args": {"report": payload.get("report"), "next_tasks": payload.get("next_tasks", [])}}, ""
        return None, "动作 JSON 缺少 tool 字段"
    return payload, ""


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """粗略估算 prompt token 数（中文按 1.5 字符/token 计）。"""
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(64, int(chars / 1.6))