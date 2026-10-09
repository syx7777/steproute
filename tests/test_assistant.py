"""科研助手业务层测试：周次命名 / 长期记忆 / 简报 / 周报（全部离线）。"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from typer.testing import CliRunner

from steproute.assistant.brief import build_brief, week_feishu_updates
from steproute.assistant.calendar import first_week_monday, label_for, report_title, week_index, week_range
from steproute.assistant.distill import build_distill_backend, distill_memory
from steproute.assistant.memory import MemoryStore
from steproute.assistant.summarize import (
    BriefSummary,
    collect_week_documents,
    load_summary,
    save_summary,
    summarize_brief,
)
from steproute.assistant.todo import TodoStore, merged_todos
from steproute.assistant.weekly import (
    WeeklySummary,
    load_weekly_summary,
    publish_weekly,
    save_weekly_summary,
    summarize_weekly,
    template_sections,
)
from steproute.cli import app
from steproute.config import Settings
from steproute.connectors.feishu import markdown_to_blocks
from steproute.corpus.loaders import load_corpus
from steproute.llm.schema import ChatResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    minutes_dir = data / "sources" / "meeting_minutes"
    minutes_dir.mkdir(parents=True)
    (minutes_dir / "2026-10-06_group.md").write_text(
        "---\ntitle: 组会纪要\ndate: 2026-10-06\n---\n\n# 组会\n\n- 下一步：统一抽帧数后重跑\n",
        encoding="utf-8",
    )
    return data


#: 测试用的知识库 space_id（文档落在 ``sources/notes/<space_id>/`` 下）。
NOTES_SPACE = "S1"


def _write_note(data: Path, *, space_id: str, name: str, title: str, date_line: str, body: str) -> None:
    notes_dir = data / "sources" / "notes" / space_id
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / f"{name}.md").write_text(
        f"---\ntitle: {title}\n{date_line}---\n\n{body}\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# 周次 / 命名
# --------------------------------------------------------------------------- #

def test_semester_labels() -> None:
    assert report_title(date(2026, 6, 1)).startswith("2026-1-week")
    assert report_title(date(2026, 10, 1)).startswith("2026-2-week")
    # 1 月归属上一年秋冬学期
    assert report_title(date(2026, 1, 15)).startswith("2025-2-week")


def test_first_week_monday_and_week_index() -> None:
    day = date(2026, 3, 9)
    start = first_week_monday(day)
    assert start.weekday() == 0
    # 首周周一必须落在学期起始月（2 月）内
    assert date(2026, 2, 1) <= start < date(2026, 2, 8)
    assert week_index(start) == 1
    assert week_index(start + timedelta(days=7)) == 2
    assert week_index(day) == 6


def test_week_range_is_monday_to_sunday() -> None:
    start, end = week_range(date(2026, 10, 7))
    assert start.weekday() == 0 and end.weekday() == 6
    assert (end - start).days == 6


def test_label_for_matches_report_title() -> None:
    label = label_for("2026-10-01")
    assert label.title == report_title(date(2026, 10, 1))
    assert label.semester == 2


# --------------------------------------------------------------------------- #
# 长期记忆
# --------------------------------------------------------------------------- #

def test_memory_round_trip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "data")
    store.set_project("视频理解", status="进行中", stage="对齐配置", goal="本周跑通公平对比")
    store.add_decision("统一 16 帧后重跑对比", date_str="2026-10-01")
    store.add_lesson("tiny ASR 代理受限，改用本地模型", date_str="2026-10-02")

    assert store.stats() == {"projects": 1, "decisions": 1, "lessons": 1}
    assert store.projects()["视频理解"].stage == "对齐配置"

    # 重载（从磁盘读回）
    reloaded = MemoryStore(tmp_path / "data")
    assert reloaded.stats()["decisions"] == 1
    decisions, lessons = reloaded.between(date(2026, 10, 1), date(2026, 10, 3))
    assert len(decisions) == 1 and len(lessons) == 1
    assert "视频理解" in reloaded.summary()


def test_memory_project_filter_isolates_knowledge_bases(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "data")
    store.set_project("实习", status="进行中")
    store.set_project("风控项目", status="阻塞")
    store.add_decision("实习的决策", project="实习")
    store.add_decision("风控的决策", project="风控项目")
    store.add_decision("无归属的手写决策")  # project 为空
    store.add_lesson("实习的经验", project="实习")
    store.add_lesson("风控的经验", project="风控项目")

    assert set(store.projects("风控项目")) == {"风控项目"}

    scoped_decisions = [entry.text for entry in store.decisions(project="风控项目")]
    assert "风控的决策" in scoped_decisions
    assert "无归属的手写决策" in scoped_decisions  # 空 project 的手写条目保留
    assert "实习的决策" not in scoped_decisions  # 不串其他知识库

    scoped_lessons = [entry.text for entry in store.lessons(project="风控项目")]
    assert scoped_lessons == ["风控的经验"]

    scoped_summary = store.summary(project="风控项目")
    assert "风控的决策" in scoped_summary and "实习的决策" not in scoped_summary

    # 默认不过滤：全部条目都返回
    assert len(store.decisions()) == 3
    assert len(store.lessons()) == 2
    assert set(store.projects()) == {"实习", "风控项目"}


def test_memory_space_filter_isolates_knowledge_bases(tmp_path: Path) -> None:
    """知识库归属（space_id）过滤：只返回该库 + 无归属，且与 project 过滤相互独立。"""
    store = MemoryStore(tmp_path / "data")
    store.set_project("项目A", status="进行中", space_id="A")
    store.set_project("项目B", status="阻塞", space_id="B")
    store.add_decision("A 的决策", project="p1", space_id="A")
    store.add_decision("B 的决策", project="p2", space_id="B")
    store.add_decision("无归属的手写决策")  # project / space_id 均为空
    store.add_lesson("A 的经验", space_id="A")
    store.add_lesson("B 的经验", space_id="B")
    store.add_lesson("无归属的经验")

    scoped_decisions = [entry.text for entry in store.decisions(space_id="A")]
    assert "A 的决策" in scoped_decisions
    assert "无归属的手写决策" in scoped_decisions  # 无归属条目各库都可见
    assert "B 的决策" not in scoped_decisions  # 不串其他知识库

    assert [entry.text for entry in store.lessons(space_id="A")] == ["A 的经验", "无归属的经验"]
    assert set(store.projects(space_id="A")) == {"项目A"}
    assert store.stats(space_id="A") == {"projects": 1, "decisions": 2, "lessons": 2}

    scoped_summary = store.summary(space_id="A")
    assert "A 的决策" in scoped_summary and "B 的决策" not in scoped_summary

    # 无归属条目在任意知识库下都可见
    assert "无归属的手写决策" in [entry.text for entry in store.decisions(space_id="B")]

    # 同时按 project + space_id 过滤（两者独立生效）
    both = [entry.text for entry in store.decisions(project="p1", space_id="A")]
    assert "A 的决策" in both and "B 的决策" not in both

    # 不传 space_id：行为不变（默认不过滤）
    assert len(store.decisions()) == 3
    assert len(store.lessons()) == 3
    assert set(store.projects()) == {"项目A", "项目B"}
    assert store.stats() == {"projects": 2, "decisions": 3, "lessons": 3}


def test_memory_reads_legacy_entries_without_space_id(tmp_path: Path) -> None:
    """旧格式数据（无 space_id 字段）能正常读出，且视为「无归属」。"""
    memory_dir = tmp_path / "data" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "decisions.jsonl").write_text(
        json.dumps({"text": "旧格式决策", "date": "2026-10-01", "source": "", "project": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (memory_dir / "lessons.jsonl").write_text(
        json.dumps({"text": "旧格式经验", "date": "2026-10-02", "project": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (memory_dir / "project_state.json").write_text(
        json.dumps({"旧项目": {"status": "进行中", "stage": "", "goal": "", "updated_at": "", "notes": ""}},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    store = MemoryStore(tmp_path / "data")
    decisions = store.decisions()
    assert len(decisions) == 1 and decisions[0].text == "旧格式决策" and decisions[0].space_id == ""
    # 无归属 → 任意知识库都可见
    assert [entry.text for entry in store.decisions(space_id="ANY")] == ["旧格式决策"]
    assert [entry.text for entry in store.lessons(space_id="ANY")] == ["旧格式经验"]
    assert store.projects()["旧项目"].space_id == ""


def test_adopt_orphans_claims_matching_entries(tmp_path: Path) -> None:
    """就地认领：命中且无归属的条目改写归属，保持顺序与总数，且幂等。"""
    store = MemoryStore(tmp_path / "data")
    store.add_decision("认领我")
    store.add_lesson("认领经验")
    store.add_decision("别认领我", space_id="SP_B")

    assert store.adopt_orphans("SP_A", ["认领我", "认领经验"]) == 2
    assert [entry.space_id for entry in store.decisions()] == ["SP_A", "SP_B"]  # 顺序不变
    assert store.lessons()[0].space_id == "SP_A"
    # 认领后从其它知识库视图消失
    assert [entry.text for entry in store.decisions(space_id="SP_A")] == ["认领我"]
    assert "认领我" not in [entry.text for entry in store.decisions(space_id="SP_B")]
    # 再次认领无新增（幂等）
    assert store.adopt_orphans("SP_A", ["认领我"]) == 0
    # 无 space_id 时不做任何事
    assert store.adopt_orphans("", ["认领我"]) == 0


def test_adopt_orphans_no_match_returns_zero(tmp_path: Path) -> None:
    """文本不命中时返回 0，且不改动文件。"""
    store = MemoryStore(tmp_path / "data")
    store.add_decision("已有决策")  # 无归属
    before = store.decisions_path.read_text(encoding="utf-8")

    assert store.adopt_orphans("SP_A", ["不存在的文本"]) == 0
    assert store.decisions_path.read_text(encoding="utf-8") == before  # 文件未被改写
    assert store.decisions()[0].space_id == ""


# --------------------------------------------------------------------------- #
# 待办清单
# --------------------------------------------------------------------------- #

def test_todo_store_crud(tmp_path: Path) -> None:
    store = TodoStore(tmp_path / "data")
    todo = store.add("写周报", owner="我")
    assert store.stats() == {"total": 1, "open": 1, "done": 0}

    store.update(todo.id, text="写周报（含实验）", status="done")
    reloaded = TodoStore(tmp_path / "data").list(status="done")
    assert reloaded[0].text == "写周报（含实验）"

    assert store.toggle(todo.id).status == "open"
    assert store.delete(todo.id) is True
    assert store.delete(todo.id) is False
    assert store.stats()["total"] == 0


def test_merged_todos_marks_source_and_dedupes(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    store = TodoStore(data)
    store.add("统一抽帧数后重跑")  # 与文档待办同文本，应只保留手动那条
    merged = merged_todos(store, load_corpus(data))

    manual = [item for item in merged if item["source"] == "manual"]
    corpus_items = [item for item in merged if item["source"] == "corpus"]
    assert len(manual) == 1
    assert all("统一抽帧数" not in item["text"] for item in corpus_items)


def test_merged_todos_caps_and_scopes(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SPACE1"
    kb_dir.mkdir(parents=True)
    (kb_dir / "note.md").write_text(
        "---\ntitle: KB 笔记\ndate: 2026-10-04\n---\n\n- 下一步：KB 里的待办\n", encoding="utf-8"
    )
    store = TodoStore(data)
    corpus = load_corpus(data)

    scoped = merged_todos(store, corpus, space_id="SPACE1")
    assert [item["text"] for item in scoped] == ["KB 里的待办"]

    assert len(merged_todos(store, corpus, max_items=1)) == 1


def test_manual_todos_are_scoped_by_space(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    store = TodoStore(data)
    store.add("A 库的手动待办", space_id="A")
    store.add("B 库的手动待办", space_id="B")
    store.add("无归属的手动待办")  # space_id 为空

    scoped = [item["text"] for item in merged_todos(store, load_corpus(data), space_id="A")]
    assert "A 库的手动待办" in scoped
    assert "无归属的手动待办" in scoped
    assert "B 库的手动待办" not in scoped  # 不显示其他知识库的手动待办

    # 不指定 space_id 时全部可见
    assert "B 库的手动待办" in [item["text"] for item in merged_todos(store, load_corpus(data))]


def test_corpus_todo_override(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    store = TodoStore(data)
    corpus = load_corpus(data)
    key = next(item for item in merged_todos(store, corpus) if item["source"] == "corpus")["key"]

    store.set_override(key, status="done")
    assert next(i for i in merged_todos(store, corpus) if i["key"] == key)["status"] == "done"

    store.set_override(key, text="改写过")
    assert next(i for i in merged_todos(store, corpus) if i["key"] == key)["text"] == "改写过"

    store.set_override(key, deleted=True)
    assert all(i["key"] != key for i in merged_todos(store, corpus))


def test_edited_corpus_todo_stays_in_window(tmp_path: Path) -> None:
    """回归：改写文档待办后，不能因排序键变化而掉出 10 条窗口。"""
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SP1"
    kb_dir.mkdir(parents=True)
    lines = "\n".join(f"- 下一步：占位待办 {index}" for index in range(15))
    (kb_dir / "note.md").write_text(
        f"---\ntitle: 笔记\ndate: 2026-10-04\n---\n\n{lines}\n", encoding="utf-8"
    )

    store = TodoStore(data)
    corpus = load_corpus(data)
    key = merged_todos(store, corpus, space_id="SP1", max_items=10)[0]["key"]

    store.set_override(key, text="!aaa 改成小写字母开头也应留在列表里")
    after = merged_todos(store, corpus, space_id="SP1", max_items=10)
    assert after[0]["key"] == key  # 排序稳定，位置不变
    assert after[0]["text"].startswith("!aaa")


# --------------------------------------------------------------------------- #
# 长期记忆提炼
# --------------------------------------------------------------------------- #

def test_distill_rules_fallback_and_dedupe(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SP1"
    kb_dir.mkdir(parents=True)
    (kb_dir / "note.md").write_text(
        "---\ntitle: 笔记\ndate: 2026-10-04\n---\n\n"
        "结论：EXP-002 与 baseline 配置不一致。\n教训：代理不稳时别用大模型下载。\n",
        encoding="utf-8",
    )
    memory = MemoryStore(data)
    result = distill_memory(
        data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP1", use_llm=False
    )

    assert result.method == "rules"
    assert any("配置不一致" in item for item in result.decisions)
    assert any("代理不稳" in item for item in result.lessons)
    assert memory.stats()["decisions"] == 1
    assert memory.stats()["lessons"] == 1

    # 重复提炼不应产生重复条目
    distill_memory(data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP1", use_llm=False)
    assert memory.stats()["decisions"] == 1
    assert memory.stats()["lessons"] == 1


def test_distill_without_documents(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    result = distill_memory(
        data_dir=data, corpus=load_corpus(data), space_id="NOTHING", use_llm=False
    )
    assert result.method == "none"
    assert "没有可提炼的文档" in result.note


def test_distill_tags_entries_with_space_id(tmp_path: Path) -> None:
    """提炼写入的条目要带上「知识库归属」，因此其它知识库看不到它们。"""
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SP_A"
    kb_dir.mkdir(parents=True)
    (kb_dir / "note.md").write_text(
        "---\ntitle: 笔记\ndate: 2026-10-04\n---\n\n"
        "结论：EXP-002 与 baseline 配置不一致。\n教训：代理不稳时别用大模型下载。\n",
        encoding="utf-8",
    )
    memory = MemoryStore(data)
    distill_memory(data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP_A", use_llm=False)

    assert memory.decisions() and all(entry.space_id == "SP_A" for entry in memory.decisions())
    assert memory.lessons() and all(entry.space_id == "SP_A" for entry in memory.lessons())
    # SP_A 能看到；SP_B 看不到该库的条目
    assert memory.decisions(space_id="SP_A")
    assert memory.decisions(space_id="SP_B") == []


def test_distill_adopts_orphans_without_duplicating(tmp_path: Path) -> None:
    """对某库重新提炼时，就地认领此前「提炼全部」写入的无归属同文本条目，不产生重复。"""
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SP_A"
    kb_dir.mkdir(parents=True)
    (kb_dir / "note.md").write_text(
        "---\ntitle: 笔记\ndate: 2026-10-04\n---\n\n"
        "结论：EXP-002 与 baseline 配置不一致。\n教训：代理不稳时别用大模型下载。\n",
        encoding="utf-8",
    )
    memory = MemoryStore(data)
    # 第一次「提炼全部」：写入无归属条目
    distill_memory(data_dir=data, corpus=load_corpus(data), memory=memory, use_llm=False)
    assert memory.decisions() and all(entry.space_id == "" for entry in memory.decisions())
    assert memory.lessons() and all(entry.space_id == "" for entry in memory.lessons())
    total_decisions = len(memory.decisions())
    total_lessons = len(memory.lessons())

    # 对 SP_A 重新提炼：认领旧条目，不新增重复
    distill_memory(data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP_A", use_llm=False)
    assert len(memory.decisions()) == total_decisions  # 总数不变（无重复）
    assert len(memory.lessons()) == total_lessons
    assert all(entry.space_id == "SP_A" for entry in memory.decisions())  # 已打上归属
    assert all(entry.space_id == "SP_A" for entry in memory.lessons())
    assert len(memory.decisions(space_id="SP_A")) == total_decisions
    # 认领后从其它知识库视图消失（正确隔离）
    assert memory.decisions(space_id="SP_B") == []
    assert memory.lessons(space_id="SP_B") == []


def test_week_feishu_updates_scopes_by_week_and_space(tmp_path: Path) -> None:
    """「飞书更新」必须只取【本周】且只取【当前知识库】的文档。"""
    data = _make_data_dir(tmp_path)
    kb_dir = data / "sources" / "notes" / "SP_A"
    other_dir = data / "sources" / "notes" / "SP_B"
    kb_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    this_week = kb_dir / "本周文档.md"
    this_week.write_text("---\ntitle: 本周文档\ndate: 2026-10-07\n---\n\n正文\n", encoding="utf-8")
    other_kb = other_dir / "别的库.md"
    other_kb.write_text("---\ntitle: 别的库\ndate: 2026-10-07\n---\n\n正文\n", encoding="utf-8")

    # sync_state 的键是飞书 token，知识库信息只在 path 里
    state = {
        "sources": {
            "feishu": {
                "targets": {
                    "notes/SP_A": {
                        "tokThisWeek": {"updated_at": "2026-10-07T10:00:00+00:00", "path": str(this_week)},
                        "tokLastWeek": {"updated_at": "2026-09-28T10:00:00+00:00", "path": str(kb_dir / "旧.md")},
                    },
                    "notes/SP_B": {
                        "tokOtherKb": {"updated_at": "2026-10-07T09:00:00+00:00", "path": str(other_kb)}
                    },
                }
            }
        }
    }
    (data / "sync_state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    updates = week_feishu_updates(data, date(2026, 10, 5), date(2026, 10, 11), space_id="SP_A")
    assert [item["title"] for item in updates] == ["本周文档"]  # 上周的、别的知识库的都被排除
    assert updates[0]["doc_id"].startswith("notes/SP_A/")  # doc_id 由 path 推导，而不是飞书 token

    brief = build_brief(
        load_corpus(data), data_dir=data, day=date(2026, 10, 7), space_id="SP_A", save=False
    )
    assert "## 飞书更新（本周 2026-10-05 ~ 2026-10-11）" in brief.markdown
    assert "本周文档" in brief.markdown
    assert "别的库" not in brief.markdown


def test_document_todos_only_from_current_week(tmp_path: Path) -> None:
    """文档待办只取【当前周更新过】的文档（手动待办不受限）。"""
    data = _make_data_dir(tmp_path)
    kb = data / "sources" / "notes" / "SP_A"
    kb.mkdir(parents=True)
    (kb / "本周.md").write_text(
        "---\ntitle: 本周\ndate: 2026-10-06\n---\n\n- 下一步：本周事项\n", encoding="utf-8"
    )
    (kb / "上周.md").write_text(
        "---\ntitle: 上周\ndate: 2026-09-29\n---\n\n- 下一步：上周事项\n", encoding="utf-8"
    )
    store = TodoStore(data)
    store.add("手动事项")
    corpus = load_corpus(data)

    scoped = merged_todos(store, corpus, space_id="SP_A", day=date(2026, 10, 7))
    texts = [item["text"] for item in scoped]
    assert any("本周事项" in text for text in texts)
    assert all("上周事项" not in text for text in texts)
    assert any("手动事项" in text for text in texts)  # 手动待办不受周限制

    # 不传 day 时不做周过滤
    assert any("上周事项" in item["text"] for item in merged_todos(store, corpus, space_id="SP_A"))


# --------------------------------------------------------------------------- #
# 每日简报
# --------------------------------------------------------------------------- #

def test_build_brief_without_saving(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    brief = build_brief(load_corpus(data), data_dir=data, day=date(2026, 10, 5), save=False)

    assert brief.day == "2026-10-05"
    assert any("统一抽帧数" in item for item in brief.todos)
    # 简报只保留「项目进展」与「飞书更新」，不再渲染「今日待办」（已在待办卡片独立展示）
    assert "## 今日待办" not in brief.markdown
    assert "## 项目进展" in brief.markdown
    assert "## 飞书更新" in brief.markdown
    assert not (data / "briefs").exists()


def test_build_brief_saves_and_records_watermark(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    build_brief(load_corpus(data), data_dir=data, day=date(2026, 10, 5), save=True)
    assert (data / "briefs" / "2026-10-05.md").is_file()
    assert (data / "memory" / "last_brief.json").is_file()


def test_build_brief_with_summary_puts_manual_first(tmp_path: Path) -> None:
    """有归纳摘要时，用归纳后的待办 + 一段进展；手动待办永远在最前。"""
    data = _make_data_dir(tmp_path)
    store = TodoStore(data)
    store.add("手动：回复导师邮件")
    summary = BriefSummary(todos=["归纳后的第一条待办"], progress="项目正在进行公平对比，下周补齐消融实验。")

    brief = build_brief(load_corpus(data), data_dir=data, todos=store, summary=summary, save=False)

    assert brief.todos[0] == "手动：回复导师邮件"
    assert "归纳后的第一条待办" in brief.todos
    assert brief.progress == "项目正在进行公平对比，下周补齐消融实验。"
    assert "项目正在进行公平对比" in brief.markdown
    assert brief.markdown.count("## 项目进展") == 1


def test_build_brief_scopes_progress_to_space(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    memory = MemoryStore(data)
    memory.set_project("风控项目", status="进行中", stage="对齐配置", space_id="SP_RISK")
    memory.set_project("实习", status="收尾", space_id="SP_INTERN")
    memory.add_decision("风控项目的决策", space_id="SP_RISK")
    memory.add_lesson("风控项目的经验", space_id="SP_RISK")
    memory.add_decision("实习知识库的决策", space_id="SP_INTERN")

    brief = build_brief(load_corpus(data), data_dir=data, memory=memory, space_id="SP_RISK", save=False)
    progress = "\n".join(brief.progress) if isinstance(brief.progress, list) else brief.progress
    assert "风控项目" in progress
    assert "最近 1 条决策 / 1 条实验经验" in progress
    # 不包含其他知识库的状态 / 决策文本，也不回填原文长句
    assert "实习" not in progress
    assert "实习知识库的决策" not in brief.markdown

    # 不带 space_id 时不做隔离（全部统计）
    everything = build_brief(load_corpus(data), data_dir=data, memory=memory, save=False)
    full = "\n".join(everything.progress) if isinstance(everything.progress, list) else everything.progress
    assert "最近 2 条决策" in full


def test_build_brief_rules_progress_uses_space_label(tmp_path: Path) -> None:
    """规则回退进展文案优先用知识库中文名（space_label），缺省时退回 space_id。"""
    data = _make_data_dir(tmp_path)
    empty = MemoryStore(data)
    labeled = build_brief(
        load_corpus(data), data_dir=data, memory=empty, space_id="SP_A", space_label="实习", save=False
    )
    assert "实习" in "\n".join(labeled.progress)

    fallback = build_brief(load_corpus(data), data_dir=data, memory=empty, space_id="SP_A", save=False)
    assert "SP_A" in "\n".join(fallback.progress)


# --------------------------------------------------------------------------- #
# 简报 / 待办归纳（LLM + 规则回退）
# --------------------------------------------------------------------------- #

class _FakeLLMBackend:
    """返回固定 JSON 字符串的假后端：离线、不发任何网络请求，并记录调用次数与 messages。"""

    name = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.kwargs: list[dict] = []

    def complete(self, spec, messages, **kwargs) -> ChatResult:
        self.calls += 1
        self.messages.append(list(messages))
        self.kwargs.append(dict(kwargs))
        return ChatResult(content=self.content)


class _RaisingBackend:
    """调用即抛异常的后端，用于验证规则回退。"""

    name = "raising"

    def complete(self, spec, messages, **kwargs) -> ChatResult:
        raise RuntimeError("no api key")


def _balanced_spec():
    settings = Settings(_env_file=None, config_dir=PROJECT_ROOT / "configs", backend="sim", log_level="WARNING")
    _, spec = build_distill_backend(settings)
    return spec


def test_summarize_rules_fallback_when_no_llm(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    result = summarize_brief(data_dir=data, corpus=load_corpus(data), use_llm=False)

    assert result.method == "rules"
    assert 0 < len(result.todos) <= 10
    assert len(result.todos) == len(set(result.todos))  # 已去重
    assert result.progress  # 进展非空（无长期记忆时给提示）
    assert result.candidates >= 1
    assert result.fingerprint


def test_summarize_rules_fallback_on_backend_error(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    result = summarize_brief(
        data_dir=data, corpus=load_corpus(data), backend=_RaisingBackend(), spec=_balanced_spec()
    )
    assert result.method == "rules"
    assert "回退" in result.note


def test_summarize_without_candidates(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    result = summarize_brief(data_dir=data, corpus=load_corpus(data), space_id="NOTHING", use_llm=False)
    assert result.todos == []
    assert result.candidates == 0
    assert result.method == "rules"
    assert "暂无长期记忆" in result.progress


def test_summarize_rules_progress_is_short_and_scoped(tmp_path: Path) -> None:
    """规则回退的 progress 必须是一句按知识库作用域生成的短文本，不得倒原文。"""
    data = _make_data_dir(tmp_path)
    memory = MemoryStore(data)
    long_sentence = "这是一条不应出现在规则回退进展里的超长决策原文，包含整句照搬的碎片内容用于断言不被回填"
    memory.add_decision(long_sentence, space_id="SP_A")
    memory.add_lesson("SP_A 的另一条经验", space_id="SP_A")
    memory.add_decision("风控项目的决策", space_id="SP_B")

    result = summarize_brief(
        data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP_A", use_llm=False
    )
    assert result.method == "rules"
    assert "###" not in result.progress
    assert long_sentence not in result.progress
    assert len(result.progress) < 200
    assert "SP_A" in result.progress
    assert "1 条最近决策" in result.progress and "1 条实验经验" in result.progress


def test_summarize_rules_progress_uses_space_label(tmp_path: Path) -> None:
    """规则回退进展文案优先用知识库中文名（space_label），缺省时退回 space_id。"""
    data = _make_data_dir(tmp_path)
    memory = MemoryStore(data)
    memory.add_decision("某条决策", space_id="SP_A")

    labeled = summarize_brief(
        data_dir=data, corpus=load_corpus(data), memory=memory,
        space_id="SP_A", space_label="实习", use_llm=False,
    )
    assert "实习" in labeled.progress
    assert "SP_A" not in labeled.progress  # 用中文名而非数字 id

    fallback = summarize_brief(
        data_dir=data, corpus=load_corpus(data), memory=memory, space_id="SP_A", use_llm=False
    )
    assert "SP_A" in fallback.progress


def test_summarize_note_distinguishes_disabled_vs_missing_backend(tmp_path: Path) -> None:
    """note 要能区分「显式关闭模型」与「漏传后端」，避免误导。"""
    data = _make_data_dir(tmp_path)
    disabled = summarize_brief(data_dir=data, corpus=load_corpus(data), use_llm=False)
    assert disabled.method == "rules"
    assert "use_llm=False" in disabled.note

    missing = summarize_brief(data_dir=data, corpus=load_corpus(data), use_llm=True)  # 未传 backend/spec
    assert missing.method == "rules"
    assert "未提供模型后端" in missing.note


def test_summarize_with_fake_backend(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    backend = _FakeLLMBackend('{"todos": ["统一抽帧数后重跑", "补齐实习周报模板"], "progress": "项目正在进行公平对比。"}')
    result = summarize_brief(
        data_dir=data, corpus=load_corpus(data), backend=backend, spec=_balanced_spec()
    )
    assert backend.calls == 1
    assert result.method == "llm"
    assert result.todos == ["统一抽帧数后重跑", "补齐实习周报模板"]
    assert result.progress == "项目正在进行公平对比。"


def test_assistant_calls_use_low_reasoning(tmp_path: Path) -> None:
    """归纳 / 提炼都是抽取型任务：必须传 reasoning_effort=low，否则 reasoning 会占满 max_tokens。"""
    data = _make_data_dir(tmp_path)
    spec = _balanced_spec()

    summarize_backend = _FakeLLMBackend('{"todos": ["推进实验"], "progress": "进行中。"}')
    summarize_brief(data_dir=data, corpus=load_corpus(data), backend=summarize_backend, spec=spec)
    assert summarize_backend.kwargs[0]["reasoning_effort"] == "low"

    distill_backend = _FakeLLMBackend('{"project": "P", "status": "进行中", "decisions": [], "lessons": []}')
    distill_memory(data_dir=data, corpus=load_corpus(data), backend=distill_backend, spec=spec)
    assert distill_backend.kwargs[0]["reasoning_effort"] == "low"

    _write_note(data, space_id=NOTES_SPACE, name="w", title="笔记",
                date_line=f"date: {date.today().isoformat()}\n", body="本周正文。")
    weekly_backend = _FakeLLMBackend('{"本周进展": "进展", "实验与结论": "x", "下周计划": "y", "风险与待协助": "z"}')
    summarize_weekly(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                     day=date.today(), backend=weekly_backend, spec=spec)
    assert weekly_backend.kwargs[0]["reasoning_effort"] == "low"


def test_merged_todos_prefers_llm_summary(tmp_path: Path) -> None:
    """给定 LLM 归纳结果时，清单 = 手动待办 + 归纳项（只读 source=llm），文档待办被替换。"""
    data = _make_data_dir(tmp_path)
    _write_note(data, space_id=NOTES_SPACE, name="n", title="笔记",
                date_line="date: 2026-10-06\n", body="- 下一步：文档里抽出的待办\n")
    corpus = load_corpus(data)
    store = TodoStore(data)
    store.add("手动待办", space_id=NOTES_SPACE)

    with_llm = merged_todos(store, corpus, space_id=NOTES_SPACE, llm_todos=["归纳 A", "归纳 B"])
    assert [item["text"] for item in with_llm] == ["手动待办", "归纳 A", "归纳 B"]
    assert [item["source"] for item in with_llm] == ["manual", "llm", "llm"]

    plain = merged_todos(store, corpus, space_id=NOTES_SPACE)
    assert all(item["source"] != "llm" for item in plain)
    assert any(item["source"] == "corpus" for item in plain)


def test_summarize_cache_round_trip_and_fingerprint_reuse(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    corpus = load_corpus(data)
    backend = _FakeLLMBackend('{"todos": ["统一抽帧数后重跑"], "progress": "进度摘要。"}')
    spec = _balanced_spec()

    first = summarize_brief(data_dir=data, corpus=corpus, backend=backend, spec=spec)
    assert backend.calls == 1
    save_summary(data, first)

    loaded = load_summary(data, space_id="")
    assert loaded is not None
    assert loaded.todos == first.todos and loaded.progress == first.progress
    assert load_summary(data, space_id="OTHER") is None  # space_id 不匹配视为无缓存

    again = summarize_brief(data_dir=data, corpus=corpus, backend=backend, spec=spec)
    assert backend.calls == 1  # 指纹相同 → 复用缓存，未再调模型
    assert again.todos == first.todos


def test_summary_cache_is_isolated_per_space(tmp_path: Path) -> None:
    """切换知识库再切回来要能读回各自的归纳结果：保存一个库不能顶掉另一个库的缓存。"""
    data = _make_data_dir(tmp_path)
    save_summary(data, BriefSummary(todos=["A 的待办"], progress="A 的进展", method="llm", space_id="SP_A"))
    save_summary(data, BriefSummary(todos=["B 的待办"], progress="B 的进展", method="llm", space_id="SP_B"))

    a = load_summary(data, space_id="SP_A")
    b = load_summary(data, space_id="SP_B")
    assert a is not None and a.todos == ["A 的待办"] and a.progress == "A 的进展"
    assert b is not None and b.todos == ["B 的待办"] and b.progress == "B 的进展"
    assert load_summary(data, space_id="SP_C") is None


def test_legacy_single_summary_file_is_still_readable(tmp_path: Path) -> None:
    """旧的「单条 summary」文件升级后仍能按 space_id 读回，不丢历史缓存。"""
    data = _make_data_dir(tmp_path)
    (data / "brief_summary.json").write_text(
        json.dumps({"todos": ["旧待办"], "progress": "旧进展", "method": "llm", "space_id": "SP_OLD"}, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = load_summary(data, space_id="SP_OLD")
    assert loaded is not None and loaded.todos == ["旧待办"] and loaded.progress == "旧进展"
    assert load_summary(data, space_id="SP_NEW") is None


# --------------------------------------------------------------------------- #
# 「本周文档」采集 + LLM 直读正文
# --------------------------------------------------------------------------- #

def test_collect_week_documents_filters_by_week_space_and_date(tmp_path: Path) -> None:
    """只取「本周 + 该知识库 + 有日期」的文档；上周 / 别的知识库 / 无日期一律排除。"""
    data = tmp_path / "data"
    today = date.today()
    last_week = today - timedelta(days=9)

    _write_note(data, space_id=NOTES_SPACE, name="this", title="本周文档",
                date_line=f"date: {today.isoformat()}\n", body="本周正文。")
    _write_note(data, space_id=NOTES_SPACE, name="last", title="上周文档",
                date_line=f"date: {last_week.isoformat()}\n", body="上周正文。")
    _write_note(data, space_id=NOTES_SPACE, name="nodate", title="无日期文档",
                date_line="", body="无日期正文。")
    _write_note(data, space_id="OTHER", name="other", title="别的知识库",
                date_line=f"date: {today.isoformat()}\n", body="别的正文。")

    corpus = load_corpus(data)

    scoped = collect_week_documents(corpus, space_id=NOTES_SPACE, day=today)
    assert [doc.title for doc in scoped] == ["本周文档"]

    # space_id 为空 → 不限知识库（本周两篇都在），但上周 / 无日期仍被排除
    unscoped = collect_week_documents(corpus, day=today)
    assert {doc.title for doc in unscoped} == {"本周文档", "别的知识库"}


def test_collect_week_documents_truncates_body(tmp_path: Path) -> None:
    """每篇正文按 doc_chars 截断、累计不超过 max_chars。"""
    data = tmp_path / "data"
    today = date.today()
    _write_note(data, space_id=NOTES_SPACE, name="long", title="长文档",
                date_line=f"date: {today.isoformat()}\n", body="A" * 500)
    _write_note(data, space_id=NOTES_SPACE, name="short", title="短文档",
                date_line=f"date: {today.isoformat()}\n", body="B" * 300)
    corpus = load_corpus(data)

    docs = collect_week_documents(corpus, space_id=NOTES_SPACE, day=today, doc_chars=100, max_chars=150)
    assert len(docs) == 1  # 第二篇会超出 max_chars → 不再纳入
    assert len(docs[0].text) == 100


def test_summarize_llm_reads_week_documents(tmp_path: Path) -> None:
    """LLM 模式直接把本周文档正文喂给模型（不再依赖正则候选话术）。"""
    data = tmp_path / "data"
    body = "本周在跑风控模型的公平对比实验，卡在数据对齐，下一步统一配置后重跑。"
    _write_note(data, space_id=NOTES_SPACE, name="proj", title="风控周记",
                date_line=f"date: {date.today().isoformat()}\n", body=body)
    backend = _FakeLLMBackend('{"todos": ["对齐风控数据配置"], "progress": "本周在跑公平对比。"}')

    result = summarize_brief(
        data_dir=data,
        corpus=load_corpus(data),
        space_id=NOTES_SPACE,
        day=date.today(),
        backend=backend,
        spec=_balanced_spec(),
    )

    assert backend.calls == 1
    assert result.method == "llm"
    assert result.docs == 1
    user_message = backend.messages[0][-1]["content"]
    assert "### 文档：风控周记" in user_message  # 文档正文确实进了 user message
    assert "风控模型的公平对比实验" in user_message


def test_summarize_skips_llm_when_no_week_documents(tmp_path: Path) -> None:
    """本周该知识库没有更新文档 → 不调用模型，直接走规则回退。"""
    data = tmp_path / "data"
    _write_note(data, space_id=NOTES_SPACE, name="old", title="旧文档",
                date_line="date: 2026-10-01\n", body="旧正文。")
    backend = _FakeLLMBackend('{"todos": ["不该被用到"], "progress": "不该被用到。"}')

    result = summarize_brief(
        data_dir=data,
        corpus=load_corpus(data),
        space_id=NOTES_SPACE,
        day=date(2026, 10, 20),  # 与文档日期不在同一自然周
        backend=backend,
        spec=_balanced_spec(),
    )

    assert backend.calls == 0
    assert result.method == "rules"
    assert result.docs == 0
    assert "没有更新的文档" in result.note


def test_summarize_fingerprint_tracks_week_document_changes(tmp_path: Path) -> None:
    """指纹基于本周文档 → 正文改动后重新归纳，不再复用缓存。"""
    data = tmp_path / "data"
    today = date.today()
    _write_note(data, space_id=NOTES_SPACE, name="proj", title="风控周记",
                date_line=f"date: {today.isoformat()}\n", body="初版正文。")
    backend = _FakeLLMBackend('{"todos": ["推进实验"], "progress": "进行中。"}')
    spec = _balanced_spec()

    first = summarize_brief(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                            day=today, backend=backend, spec=spec)
    save_summary(data, first)
    assert backend.calls == 1

    _write_note(data, space_id=NOTES_SPACE, name="proj", title="风控周记",
                date_line=f"date: {today.isoformat()}\n", body="改版后的正文。")
    second = summarize_brief(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                             day=today, backend=backend, spec=spec)

    assert second.fingerprint != first.fingerprint
    assert backend.calls == 2


# --------------------------------------------------------------------------- #
# 周报
# --------------------------------------------------------------------------- #

def test_summarize_weekly_uses_default_sections_with_llm(tmp_path: Path) -> None:
    """周报 = 本周该知识库文档正文 → LLM 按内置小节归纳。"""
    data = _make_data_dir(tmp_path)
    _write_note(data, space_id=NOTES_SPACE, name="proj", title="风控周记",
                date_line="date: 2026-10-06\n", body="本周跑公平对比实验，卡在配置不一致。")
    backend = _FakeLLMBackend(
        '{"本周进展": "对齐配置并重跑。", "实验与结论": "对比不公平。",'
        ' "下周计划": "统一抽帧数。", "风险与待协助": "算力不足。"}'
    )

    summary = summarize_weekly(
        data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
        day=date(2026, 10, 6), backend=backend, spec=_balanced_spec(),
    )

    assert summary.method == "llm"
    assert summary.docs == 1  # 只有 notes/<space>/ 下本周的文档计入
    assert summary.markdown.startswith(f"# {summary.week}")
    for section in ("本周进展", "实验与结论", "下周计划", "风险与待协助"):
        assert f"## {section}" in summary.markdown
    assert "对齐配置并重跑。" in summary.markdown
    assert "风控周记" in backend.messages[0][-1]["content"]  # 文档正文进了素材


def test_summarize_weekly_scopes_space_and_reuses_cache(tmp_path: Path) -> None:
    """只读所选知识库的本周文档；指纹一致时复用缓存不再调模型。"""
    data = _make_data_dir(tmp_path)
    _write_note(data, space_id=NOTES_SPACE, name="a", title="A 库文档",
                date_line="date: 2026-10-06\n", body="A 正文。")
    _write_note(data, space_id="OTHER", name="b", title="B 库文档",
                date_line="date: 2026-10-06\n", body="B 正文。")
    backend = _FakeLLMBackend(
        '{"本周进展": "A 的进展", "实验与结论": "x", "下周计划": "y", "风险与待协助": "z"}'
    )
    spec = _balanced_spec()

    first = summarize_weekly(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                             day=date(2026, 10, 6), backend=backend, spec=spec)
    assert first.docs == 1 and backend.calls == 1
    assert "A 正文" in backend.messages[0][-1]["content"]
    save_weekly_summary(data, first)

    again = summarize_weekly(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                             day=date(2026, 10, 6), backend=backend, spec=spec)
    assert backend.calls == 1  # 指纹一致 → 复用缓存
    assert again.markdown == first.markdown


def test_summarize_weekly_rules_fallback_without_llm(tmp_path: Path) -> None:
    """未启用 LLM 时回退规则：按文档标题 + 首句罗列进首个小节。"""
    data = _make_data_dir(tmp_path)
    _write_note(data, space_id=NOTES_SPACE, name="proj", title="风控周记",
                date_line="date: 2026-10-06\n", body="本周跑公平对比实验。")

    summary = summarize_weekly(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE,
                               day=date(2026, 10, 6), use_llm=False)

    assert summary.method == "rules"
    assert "## 本周进展" in summary.markdown
    assert "风控周记" in summary.markdown


def test_summarize_weekly_without_documents(tmp_path: Path) -> None:
    """本周该知识库没有更新文档 → method=none，不调模型。"""
    data = _make_data_dir(tmp_path)
    summary = summarize_weekly(data_dir=data, corpus=load_corpus(data), space_id=NOTES_SPACE, day=date(2026, 10, 6))
    assert summary.method == "none"
    assert summary.docs == 0
    assert "## 本周进展" in summary.markdown


def test_weekly_summary_cache_is_isolated_per_space(tmp_path: Path) -> None:
    """切换知识库再切回来要能读回各自的周报缓存：保存一个库不能顶掉另一个库。"""
    data = _make_data_dir(tmp_path)
    save_weekly_summary(data, WeeklySummary(markdown="A 周报", method="llm", week="2026-2-week6", space_id="SP_A"))
    save_weekly_summary(data, WeeklySummary(markdown="B 周报", method="llm", week="2026-2-week6", space_id="SP_B"))

    assert load_weekly_summary(data, space_id="SP_A").markdown == "A 周报"
    assert load_weekly_summary(data, space_id="SP_B").markdown == "B 周报"
    assert load_weekly_summary(data, space_id="SP_C") is None


def test_template_sections_extracts_headings() -> None:
    template = "# 周报\n\n## 本周工作总结\n- x\n\n## 实验与结论\n- y\n\n## 下周计划\n- z\n"
    assert template_sections(template) == ["本周工作总结", "实验与结论", "下周计划"]


class _FakeFeishu:
    """鸭子类型客户端：只实现 publish_weekly 需要的四个方法。"""

    def __init__(self, template_text: str = "") -> None:
        self.template_text = template_text
        self.appended: list[tuple[str, str]] = []
        self.created: dict[str, object] = {}

    def find_template_token(self, space_id: str, keyword: str) -> str:
        return "doxTEMPLATE" if self.template_text else ""

    def read_document_text(self, document_id: str) -> str:
        return self.template_text

    def create_space_node(self, space_id: str, *, title: str, parent_node_token: str | None = None):
        self.created = {"space_id": space_id, "title": title, "parent": parent_node_token}
        return {"node_token": "wikNEW", "obj_token": "doxNEW", "obj_type": "docx"}

    def append_markdown(self, document_id: str, markdown: str) -> int:
        self.appended.append((document_id, markdown))
        return 7


def test_publish_weekly_creates_node_and_writes_blocks() -> None:
    client = _FakeFeishu("# 周报模板\n\n## 本周工作总结\n\n## 实验与结论\n\n## 下周计划\n")
    result = publish_weekly(client, space_id="sp1", markdown="# 2026-2-week6\n\n## 本周进展\n- a\n", title="2026-2-week6")

    assert result.document_id == "doxNEW"
    assert result.blocks == 7
    assert result.template_token == "doxTEMPLATE"
    assert client.created["title"] == "2026-2-week6"
    _, written = client.appended[0]
    # 模板小节被套用到正文
    assert "## 本周工作总结" in written


def test_publish_weekly_without_template() -> None:
    client = _FakeFeishu()
    result = publish_weekly(client, space_id="sp1", markdown="# t\n\n## 本周进展\n- a\n", title="t")
    assert result.template_token == ""
    assert result.document_id == "doxNEW"


# --------------------------------------------------------------------------- #
# Markdown → 文档块
# --------------------------------------------------------------------------- #

def test_markdown_to_blocks_maps_common_syntax() -> None:
    blocks = markdown_to_blocks(
        "# 标题\n## 二级\n- 项目\n1. 有序\n- [ ] 待办\n- [x] 完成\n> 引用\n---\n普通文本\n"
    )
    types = [block["block_type"] for block in blocks]
    assert types == [3, 4, 12, 13, 17, 17, 15, 22, 2]
    assert blocks[0]["heading1"]["elements"][0]["text_run"]["content"] == "标题"
    assert blocks[4]["todo"]["style"]["done"] is False
    assert blocks[5]["todo"]["style"]["done"] is True


def test_markdown_to_blocks_handles_code_fence() -> None:
    blocks = markdown_to_blocks("```python\nprint(1)\n```\n")
    assert len(blocks) == 1
    assert blocks[0]["block_type"] == 14
    assert blocks[0]["code"]["style"]["language"] == "python"


# --------------------------------------------------------------------------- #
# CLI 冒烟
# --------------------------------------------------------------------------- #

def test_cli_help_lists_assistant_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("brief", "memory", "weekly", "serve"):
        assert command in result.stdout


def test_cli_brief_and_weekly_preview(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    runner = CliRunner()

    brief_result = runner.invoke(app, ["brief", "--data-dir", str(data), "--no-save"])
    assert brief_result.exit_code == 0, brief_result.stdout
    assert "项目进展" in brief_result.stdout

    weekly_result = runner.invoke(app, ["weekly", "preview", "--data-dir", str(data), "--no-llm"])
    assert weekly_result.exit_code == 0, weekly_result.stdout
    assert "周报命名" in weekly_result.stdout


def test_cli_memory(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    runner = CliRunner()

    state = runner.invoke(app, ["memory", "state", "--data-dir", str(data), "--project", "视频理解", "--status", "进行中"])
    assert state.exit_code == 0, state.stdout

    decision = runner.invoke(app, ["memory", "decision", "--data-dir", str(data), "统一配置后重跑"])
    assert decision.exit_code == 0, decision.stdout