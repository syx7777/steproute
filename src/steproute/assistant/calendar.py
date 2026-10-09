"""学期 / 周次计算：周报命名的唯一口径来源。

命名格式：``{学年}-{学期}-week{周次}``

* **学期编号**：春季（2–8 月）→ ``1``；秋冬（9 月–次年 1 月）→ ``2``；
* **学年**：取**学期起始年**——春季取当年 2 月；秋冬取当年 9 月，
  因此 1 月的周报归属**上一年**（2026-01 属于 2025 年秋冬学期）；
* **周次**：以学期起始月 1 号**当天或之后**的第一个周一为第 1 周起点
  （保证首周一定落在起始月内），按自然周递增；早于首周的日期计为第 1 周。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

SPRING = 1
AUTUMN_WINTER = 2

SPRING_FIRST_MONTH = 2
AUTUMN_FIRST_MONTH = 9


def semester_of(day: date) -> int:
    """判断学期编号：2–8 月为春季（1），其余为秋冬（2）。"""
    return SPRING if SPRING_FIRST_MONTH <= day.month <= 8 else AUTUMN_WINTER


def semester_start_year(day: date) -> int:
    """学期起始年（秋季取当年，1 月归上一年）。"""
    if day.month == 1:
        return day.year - 1
    return day.year


def semester_start_month(day: date) -> int:
    return SPRING_FIRST_MONTH if semester_of(day) == SPRING else AUTUMN_FIRST_MONTH


def first_week_monday(day: date) -> date:
    """学期第 1 周的周一（起始月 1 号当天或之后的第一个周一）。"""
    first = date(semester_start_year(day), semester_start_month(day), 1)
    offset = (7 - first.weekday()) % 7  # 周一为 0，则为 0
    return first + timedelta(days=offset)


def week_index(day: date) -> int:
    """该日期在学期内的第几周（第 1 周起算；早于首周按第 1 周计）。"""
    return max(1, (day - first_week_monday(day)).days // 7 + 1)


def week_range(day: date) -> tuple[date, date]:
    """该日期所在自然周的周一 ~ 周日。"""
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def report_title(day: date) -> str:
    """周报标题：``{学年}-{学期}-week{周次}``。"""
    return f"{semester_start_year(day)}-{semester_of(day)}-week{week_index(day)}"


def as_date(value: date | datetime | str | None) -> date:
    """把 ``date`` / ``datetime`` / ``YYYY-MM-DD`` 统一成 :class:`date`；空值取今天。"""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class WeekLabel:
    """一次周报的命名与时间范围。"""

    title: str
    year: int
    semester: int
    week: int
    start: date
    end: date

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "year": self.year,
            "semester": self.semester,
            "week": self.week,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


def label_for(day: date | datetime | str | None = None) -> WeekLabel:
    """返回该日期的周报命名与所属自然周范围。"""
    target = as_date(day)
    start, end = week_range(target)
    return WeekLabel(
        title=report_title(target),
        year=semester_start_year(target),
        semester=semester_of(target),
        week=week_index(target),
        start=start,
        end=end,
    )