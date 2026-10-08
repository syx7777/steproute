"""Prompt 构造。

关键约束（服务 Cache-aware 机制）：**系统提示是稳定前缀**，
系统 Prompt + 工具定义 + 固定背景在同一 trajectory 内不变，
这样重复调用同一模型时可命中服务端前缀缓存，降低 TTFT。
动态内容（任务、当前状态、最近步骤）一律放在最后一条 user 消息里。
"""

from __future__ import annotations

from typing import Any

from .state import AgentState
from .tools import render_tool_catalog

SYSTEM_PROMPT = f"""你是「科研工作流 Agent」的执行核心。你的工作是把散落在周报评论、组会纪要、
实验记录和代码仓库里的零散信息串起来，整理出当前研究进度与下一步任务。

# 可用工具
{render_tool_catalog()}

# 输出协议（必须严格遵守）
每一步只输出**一个 JSON 对象**，不要输出任何额外文字：
{{
  "step_type": "retrieve|read|extract|compare|analyze|synthesize|final",
  "thought": "一句话说明你为什么这样做",
  "next_step_type": "你预计下一步属于哪种类型（供 Harness 预测下一轮难度）",
  "tool": "要调用的工具名",
  "args": {{ ... }}
}}

# 行为准则
1. 每一步只调用一个工具，拿到结果后再决定下一步。
2. 抽取类工作（找 TODO、读指标、列实验）尽量高效，不要重复检索同一内容。
3. 涉及"为什么没超过 baseline / 下一步优先做什么"时，必须先比较实验配置与指标，
   确认对比是否公平，再下结论。
4. 结论必须来自工具返回的材料，并在 finalize 时通过 record_finding 登记证据。
5. 当证据与待办已经足够时，用 `finalize` 收尾，报告用 Markdown，
   next_tasks 要是可执行的优先级列表。
"""


def build_messages(state: AgentState, *, history_window: int = 6, max_tool_chars: int = 800) -> list[dict[str, str]]:
    """构造本轮调用消息：稳定 system + 动态 user。"""
    recent = state.history[-history_window:]
    history_lines: list[str] = []
    for record in recent:
        status = "ok" if record.tool_ok else "FAILED"
        history_lines.append(f"- step {record.index} [{record.step_type}] {record.tool} -> {status}")
    if state.last_error:
        history_lines.append(f"- 最近一次错误：{state.last_error}")

    user = f"""任务：{state.task}

## 当前状态
```json
{state.state_json()}
```

## 最近步骤
{chr(10).join(history_lines) if history_lines else "- （尚无历史步骤）"}

请输出下一步动作的 JSON 对象。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# 子目标模板（确定性规划，零模型开销）
# --------------------------------------------------------------------------- #

_INTENT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("对比", "比较", "baseline", "基线", "公平"), "公平对比方法与其 baseline 的实验配置与指标差异"),
    (("为什么", "原因", "没超过", "未超过", "落后", "不如"), "定位方法未超过 baseline 的根本原因"),
    (("消融", "ablation"), "确认消融实验的完成情况与缺失项"),
    (("复现", "复跑", "reproduce"), "核对实验是否可复现（随机种子、评测切分、配置）"),
    (("下一步", "优先", "计划", "安排"), "给出下一步任务的优先级排序"),
    (("进展", "进度", "总结", "整理", "汇总"), "归纳当前研究进展"),
]

_BASE_SUBGOALS = (
    "梳理与该任务相关的周报评论、会议纪要与实验记录",
    "提取关键实验的指标与配置，判断已有结论是否成立",
    "归纳当前研究进展与已完成的实验",
    "给出下一步优先任务清单",
)


def seed_subgoals(task: str) -> list[str]:
    """根据任务文本确定性地生成子目标（避免额外一次规划调用）。"""
    goals = list(_BASE_SUBGOALS)
    lowered = task.lower()
    for keywords, goal in _INTENT_RULES:
        if any(keyword.lower() in lowered for keyword in keywords):
            if goal not in goals:
                goals.insert(1, goal)
    return goals


def summarize_tool_output(text: str, *, max_chars: int = 400) -> str:
    """把工具输出压缩成一行摘要（用于历史记录）。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[:max_chars] + "…"


def action_schema_hint() -> dict[str, Any]:
    """给 LLM 的字段提示（同时供 Jev 的 judge prompt 复用）。"""
    return {"step_type": "str", "thought": "str", "next_step_type": "str", "tool": "str", "args": "object"}