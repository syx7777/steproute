"""从文档里提炼「项目状态 / 历史决策 / 实验经验」，写入长期记忆。

两种模式
--------
* **LLM 模式**（默认）：调用模型池里 ``balanced`` 档，把材料归纳成结构化 JSON；
* **规则模式**（LLM 不可用时自动回退）：按关键词抽取决策与经验，不推断项目状态。

提炼结果会去重后写入 :class:`~steproute.assistant.memory.MemoryStore`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ModelSpecConfig, Settings, load_app_config
from ..corpus.model import Corpus
from ..llm.backends import APIBackend, Backend
from ..logging_utils import get_logger
from .memory import MemoryStore
from .todo import KB_DOC_PREFIX

logger = get_logger(__name__)

#: 「决策」线索词（规则回退时按关键词抽取）。
DECISION_MARKERS = (
    "决定",
    "确定",
    "结论",
    "达成一致",
    "一致同意",
    "采用",
    "定为",
    "共识",
    "拍板",
    "通过",
)

LESSON_MARKERS = ("教训", "踩坑", "经验", "注意", "失败原因", "提醒")
MAX_MATERIAL_CHARS = 6000
DOC_CHARS = 1200

_SYSTEM_PROMPT = """你是科研项目助理。请阅读给定材料，提炼该项目当前的状况与关键结论。

只输出一个 JSON 对象（不要代码围栏、不要解释），字段如下：
{
  "project": "项目名（优先使用材料中出现的名称）",
  "status": "一句话状态，如 进行中 / 阻塞 / 已完成",
  "stage": "当前阶段，如 对齐评测配置",
  "goal": "当前最重要的目标，一句话",
  "decisions": ["已确定的决策，每条一句话，最多 6 条"],
  "lessons": ["实验经验或教训，每条一句话，最多 6 条"]
}
材料不足的字段留空字符串或空数组，不要编造。"""


@dataclass
class DistillResult:
    """一次提炼的结果。"""

    method: str  # llm / rules / none
    project: str = ""
    status: str = ""
    stage: str = ""
    goal: str = ""
    decisions: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    docs: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "project": self.project,
            "status": self.status,
            "stage": self.stage,
            "goal": self.goal,
            "decisions": self.decisions,
            "lessons": self.lessons,
            "docs": self.docs,
            "note": self.note,
        }


def build_distill_backend(settings: Settings) -> tuple[Backend, ModelSpecConfig]:
    """用模型池里 ``balanced`` 档（缺省回退 strong/首个）构造一次提炼所需的后端。"""
    config = load_app_config(settings.config_dir)
    specs = config.models.by_tier("balanced") or config.models.by_tier("strong") or config.models.models[:1]
    return APIBackend(settings), specs[0]


# --------------------------------------------------------------------------- #
# 素材收集
# --------------------------------------------------------------------------- #

def _scoped_docs(corpus: Corpus, space_id: str, *, max_docs: int) -> list[Any]:
    """取待提炼的文档；``space_id`` 给定时只看该知识库同步来的文档。"""
    prefix = f"{KB_DOC_PREFIX}/{space_id}/" if space_id else ""
    docs = [
        doc
        for doc in corpus.documents.values()
        if not prefix or doc.doc_id.startswith(prefix)
    ]
    docs.sort(key=lambda doc: str(doc.meta.get("date") or ""), reverse=True)
    return docs[:max_docs]


def _materials(docs: list[Any], *, max_chars: int = MAX_MATERIAL_CHARS) -> str:
    parts = [f"--- 文档：{doc.title} ---\n{doc.text.strip()[:DOC_CHARS]}" for doc in docs]
    return "\n\n".join(parts)[:max_chars]


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    picked: list[str] = []
    for item in value:
        text = str(item).strip()
        if len(text) >= 4 and text not in picked:
            picked.append(text)
    return picked[:limit]


def _rules_extract(docs: list[Any]) -> tuple[list[str], list[str]]:
    """规则回退：按关键词抽取决策与经验（不推断项目状态）。"""
    decisions: list[str] = []
    lessons: list[str] = []
    for doc in docs:
        for line in doc.text.splitlines():
            body = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line.strip())
            if not body or len(body) < 4:
                continue
            snippet = body if len(body) <= 120 else body[:120]
            if any(marker in body for marker in DECISION_MARKERS):
                if snippet not in decisions:
                    decisions.append(snippet)
            elif any(marker in body for marker in LESSON_MARKERS):
                if snippet not in lessons:
                    lessons.append(snippet)
    return decisions[:6], lessons[:6]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def distill_memory(
    *,
    data_dir: Path | str,
    corpus: Corpus,
    memory: MemoryStore | None = None,
    space_id: str = "",
    project: str = "",
    backend: Backend | None = None,
    spec: ModelSpecConfig | None = None,
    use_llm: bool = True,
    max_docs: int = 12,
) -> DistillResult:
    """提炼并写入长期记忆（决策/经验去重后追加，项目状态覆盖更新）。"""
    store = memory or MemoryStore(data_dir)
    docs = _scoped_docs(corpus, space_id, max_docs=max_docs)
    if not docs:
        return DistillResult(method="none", note="没有可提炼的文档：请先同步知识库")

    result = DistillResult(method="rules", docs=len(docs), project=project)
    payload: dict[str, Any] = {}

    if use_llm and backend is not None and spec is not None:
        try:
            chat = backend.complete(
                spec,
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _materials(docs)},
                ],
                json_mode=True,
                temperature=0.2,
                # 抽取型任务用低推理，避免 reasoning token 占满 max_tokens 导致 content 为空。
                reasoning_effort="low",
                max_tokens=spec.max_tokens,
                label="distill",
            )
            payload = _parse_json(chat.content)
        except Exception as exc:  # 缺 Key / 网络异常等都回退到规则模式
            result.note = f"LLM 提炼不可用，已回退规则抽取：{exc}"
            logger.warning("提炼回退规则模式：%s", exc)

    if payload:
        result.method = "llm"
        result.project = str(payload.get("project") or project or "").strip()
        result.status = str(payload.get("status") or "").strip()
        result.stage = str(payload.get("stage") or "").strip()
        result.goal = str(payload.get("goal") or "").strip()
        result.decisions = _as_list(payload.get("decisions"))
        result.lessons = _as_list(payload.get("lessons"))
    else:
        if not result.note:
            result.note = "未启用 LLM，仅按规则抽取决策与经验"
        result.decisions, result.lessons = _rules_extract(docs)

    # -- 写入长期记忆 ------------------------------------------------------ #
    # ``project`` 是 LLM 归纳出的项目名；``space_id`` 是知识库归属（两者独立）。
    if result.project and (result.status or result.stage or result.goal):
        store.set_project(result.project, space_id=space_id, status=result.status, stage=result.stage, goal=result.goal)

    # 先「就地认领」：把此前由「提炼全部」写入的无归属同文本条目改成当前知识库归属，
    # 避免重新提炼时产生重复条目（认领后这些内容只在当前知识库可见）。
    store.adopt_orphans(space_id, result.decisions)
    store.adopt_orphans(space_id, result.lessons)

    # 去重：不带 space_id（提炼全部）时对全部条目去重；给定 space_id 时只对**该知识库**的条目去重
    # （含刚刚被认领的条目），其余文本才追加。
    known_decisions = {entry.text for entry in store.decisions() if not space_id or entry.space_id == space_id}
    for text in result.decisions:
        if text not in known_decisions:
            store.add_decision(text, source="distill", project=result.project, space_id=space_id)
            known_decisions.add(text)

    known_lessons = {entry.text for entry in store.lessons() if not space_id or entry.space_id == space_id}
    for text in result.lessons:
        if text not in known_lessons:
            store.add_lesson(text, source="distill", project=result.project, space_id=space_id)
            known_lessons.add(text)

    logger.info(
        "提炼完成：method=%s docs=%d 决策 %d / 经验 %d",
        result.method,
        result.docs,
        len(result.decisions),
        len(result.lessons),
    )
    return result