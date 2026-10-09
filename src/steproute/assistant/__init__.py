"""科研助手业务层：把散落材料变成可用的日常产出。

* :mod:`~steproute.assistant.calendar` —— 学期 / 周次与周报命名；
* :mod:`~steproute.assistant.memory`   —— 项目长期记忆（状态 / 决策 / 经验）；
* :mod:`~steproute.assistant.brief`    —— 每日工作简报；
* :mod:`~steproute.assistant.weekly`   —— 周报汇总 / 渲染 / 写入飞书。
"""

from .brief import Brief, build_brief, render_brief, week_feishu_updates
from .calendar import WeekLabel, label_for, report_title, semester_of, week_index
from .distill import DistillResult, build_distill_backend, distill_memory
from .memory import Entry, MemoryStore, ProjectState
from .summarize import (
    BriefSummary,
    collect_candidates,
    load_summary,
    save_summary,
    summarize_brief,
)
from .todo import MAX_ITEMS, Todo, TodoStore, merged_todos
from .weekly import (
    DEFAULT_SECTIONS,
    PublishResult,
    WeeklySummary,
    load_weekly_summary,
    publish_weekly,
    save_weekly_summary,
    summarize_weekly,
    template_sections,
)

__all__ = [
    "Brief",
    "BriefSummary",
    "DEFAULT_SECTIONS",
    "DistillResult",
    "Entry",
    "MAX_ITEMS",
    "MemoryStore",
    "ProjectState",
    "PublishResult",
    "Todo",
    "TodoStore",
    "WeekLabel",
    "WeeklySummary",
    "build_brief",
    "build_distill_backend",
    "collect_candidates",
    "distill_memory",
    "label_for",
    "load_summary",
    "load_weekly_summary",
    "merged_todos",
    "publish_weekly",
    "render_brief",
    "report_title",
    "save_summary",
    "save_weekly_summary",
    "semester_of",
    "summarize_brief",
    "summarize_weekly",
    "template_sections",
    "week_feishu_updates",
    "week_index",
]