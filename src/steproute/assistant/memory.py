"""项目长期记忆：项目状态 / 历史决策 / 实验经验。

落盘位置（全部为普通 JSON / JSONL，可版本管理、可手工编辑）::

    data/memory/
    ├── project_state.json     # 项目状态（键值，可覆盖更新）
    ├── decisions.jsonl        # 历史决策（追加）
    └── lessons.jsonl          # 实验经验（追加）
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

logger = get_logger(__name__)

MEMORY_DIRNAME = "memory"
STATE_FILENAME = "project_state.json"
DECISIONS_FILENAME = "decisions.jsonl"
LESSONS_FILENAME = "lessons.jsonl"


@dataclass
class Entry:
    """一条长期记忆条目（决策 / 经验）。

    ``project`` 是 LLM 归纳出的**项目名**；``space_id`` 是条目的**知识库归属**，
    两者相互独立——前者用于「项目」维度，后者用于「知识库」维度过滤。
    """

    text: str
    date: str = ""
    source: str = ""
    project: str = ""
    space_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectState:
    """单个项目的长期状态。"""

    project: str
    status: str = ""
    stage: str = ""
    goal: str = ""
    updated_at: str = ""
    notes: str = ""
    space_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """项目长期记忆的读写入口。"""

    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir) / MEMORY_DIRNAME

    # -- 路径 -------------------------------------------------------------- #
    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILENAME

    @property
    def decisions_path(self) -> Path:
        return self.root / DECISIONS_FILENAME

    @property
    def lessons_path(self) -> Path:
        return self.root / LESSONS_FILENAME

    # -- 通用读写 ---------------------------------------------------------- #
    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("记忆文件损坏，按空值处理：%s", path)
            return default

    def _append(self, path: Path, entry: Entry) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def _rewrite(self, path: Path, entries: list[Entry]) -> None:
        """整文件重写 JSONL（一行一条、``ensure_ascii=False``，保持原顺序与字段）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def _read_entries(self, path: Path, limit: int | None = None) -> list[Entry]:
        if not path.is_file():
            return []
        entries: list[Entry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(Entry(**{k: v for k, v in payload.items() if k in Entry.__dataclass_fields__}))
        return entries[-limit:] if limit else entries

    # -- 项目状态 ---------------------------------------------------------- #
    def projects(self, project: str = "", space_id: str = "") -> dict[str, ProjectState]:
        """项目状态。

        ``project`` 给定时只返回该项目（``.project`` 精确匹配）；
        ``space_id`` 给定时只保留该知识库 + 无归属（``space_id == ""``）的状态，不含其他知识库。
        二者可同时生效。
        """
        payload = self._read_json(self.state_path, {})
        if not isinstance(payload, dict):
            return {}
        result: dict[str, ProjectState] = {}
        for name, fields in payload.items():
            if not isinstance(fields, dict):
                continue
            state = ProjectState(
                project=str(name),
                **{k: str(v) for k, v in fields.items() if k in ProjectState.__dataclass_fields__ and k != "project"},
            )
            if project and state.project != project:
                continue
            if space_id and state.space_id not in (space_id, ""):
                continue
            result[str(name)] = state
        return result

    def set_project(self, project: str, *, space_id: str = "", **fields: Any) -> ProjectState:
        """新增或更新一个项目的状态（只覆盖传入的字段）。

        ``space_id`` 非空时把该项目**归属**到该知识库（重新提炼即可回填归属）；
        留空则沿用原有归属，不覆盖。
        """
        projects = self.projects()
        current = projects.get(project, ProjectState(project=project))
        if space_id:
            current.space_id = space_id
        for key, value in fields.items():
            if key in ProjectState.__dataclass_fields__ and key != "project":
                setattr(current, key, str(value))
        current.updated_at = date.today().isoformat()
        projects[project] = current

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({name: state.to_dict() for name, state in projects.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return current

    def delete_project(self, project: str) -> bool:
        projects = self.projects()
        if project not in projects:
            return False
        projects.pop(project)
        self.state_path.write_text(
            json.dumps({name: state.to_dict() for name, state in projects.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True

    # -- 决策 / 经验 ------------------------------------------------------- #
    def add_decision(
        self, text: str, *, date_str: str = "", source: str = "", project: str = "", space_id: str = ""
    ) -> Entry:
        entry = Entry(
            text=text.strip(),
            date=date_str or date.today().isoformat(),
            source=source,
            project=project,
            space_id=space_id,
        )
        self._append(self.decisions_path, entry)
        return entry

    def add_lesson(
        self,
        text: str,
        *,
        date_str: str = "",
        source: str = "",
        project: str = "",
        space_id: str = "",
        tags: list[str] | None = None,
    ) -> Entry:
        entry = Entry(
            text=text.strip(),
            date=date_str or date.today().isoformat(),
            source=source,
            project=project,
            space_id=space_id,
            tags=list(tags or []),
        )
        self._append(self.lessons_path, entry)
        return entry

    @staticmethod
    def _keep(entries: list[Entry], *, project: str = "", space_id: str = "") -> list[Entry]:
        """按「项目」「知识库归属」两个维度过滤。

        两个维度的语义一致：传值时只保留该值 + 无归属（``""``）的条目，不串其他值；
        均不传时不过滤；两者可同时生效。
        """
        if project:
            entries = [entry for entry in entries if entry.project in (project, "")]
        if space_id:
            entries = [entry for entry in entries if entry.space_id in (space_id, "")]
        return entries

    def decisions(self, *, limit: int | None = None, project: str = "", space_id: str = "") -> list[Entry]:
        """历史决策；``project`` / ``space_id`` 给定时只保留匹配项 + 无归属（``""``）的条目。"""
        entries = self._keep(self._read_entries(self.decisions_path), project=project, space_id=space_id)
        return entries[-limit:] if limit else entries

    def lessons(self, *, limit: int | None = None, project: str = "", space_id: str = "") -> list[Entry]:
        """实验经验；``project`` / ``space_id`` 给定时只保留匹配项 + 无归属（``""``）的条目。"""
        entries = self._keep(self._read_entries(self.lessons_path), project=project, space_id=space_id)
        return entries[-limit:] if limit else entries

    def adopt_orphans(self, space_id: str, texts: Iterable[str]) -> int:
        """把无归属（``space_id == ""``）且 ``text`` 命中的条目**就地认领**为当前知识库。

        用于「对某知识库重新提炼」时，认领此前由「提炼全部」写入的无归属同文本条目——
        既避免重复条目，又让这些内容从此只在该知识库可见（从其它知识库视图消失）。
        返回被认领的条数；无 ``space_id``、无命中时不改动任何文件。
        """
        if not space_id:
            return 0
        wanted = {text.strip() for text in texts if text and text.strip()}
        if not wanted:
            return 0
        adopted = 0
        for path in (self.decisions_path, self.lessons_path):
            entries = self._read_entries(path)
            changed = False
            for entry in entries:
                if entry.space_id == "" and entry.text in wanted:
                    entry.space_id = space_id
                    adopted += 1
                    changed = True
            if changed:
                self._rewrite(path, entries)
        return adopted

    def between(self, start: date, end: date, *, space_id: str = "") -> tuple[list[Entry], list[Entry]]:
        """返回 ``[start, end]`` 区间内的决策与经验（``space_id`` 给定时按知识库归属过滤）。"""
        def keep(entries: list[Entry]) -> list[Entry]:
            picked = []
            for entry in entries:
                try:
                    day = date.fromisoformat(entry.date[:10])
                except ValueError:
                    continue
                if start <= day <= end:
                    picked.append(entry)
            return picked

        return keep(self.decisions(space_id=space_id)), keep(self.lessons(space_id=space_id))

    # -- 摘要 -------------------------------------------------------------- #
    def summary(self, *, project: str = "", space_id: str = "", max_projects: int = 8, max_entries: int = 8) -> str:
        """给 Agent / 简报用的紧凑文本摘要。

        ``project`` / ``space_id`` 给定时只汇总匹配项（以及无归属的条目），避免跨知识库串味。
        """
        lines: list[str] = []
        projects = self.projects(project, space_id)
        if projects:
            lines.append("### 项目状态")
            for state in list(projects.values())[:max_projects]:
                parts = [p for p in (state.status, state.stage, state.goal, state.notes) if p]
                lines.append(f"- {state.project}：{'；'.join(parts) if parts else '（未填写）'}")
        decisions = self.decisions(limit=max_entries, project=project, space_id=space_id)
        if decisions:
            lines.append("\n### 历史决策")
            lines += [f"- [{entry.date}] {entry.text}" for entry in decisions]
        lessons = self.lessons(limit=max_entries, project=project, space_id=space_id)
        if lessons:
            lines.append("\n### 实验经验")
            lines += [f"- [{entry.date}] {entry.text}" for entry in lessons]
        return "\n".join(lines) if lines else "（长期记忆为空）"

    def stats(self, *, space_id: str = "", project: str = "") -> dict[str, int]:
        """长期记忆计数；``space_id`` / ``project`` 给定时按对应归属过滤。"""
        return {
            "projects": len(self.projects(project=project, space_id=space_id)),
            "decisions": len(self.decisions(project=project, space_id=space_id)),
            "lessons": len(self.lessons(project=project, space_id=space_id)),
        }