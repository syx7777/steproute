"""周报生成：读「本周更新的飞书文档正文」→ 归纳成简明周报 → 写入飞书知识库。

* **素材**：本周（自然周）内、指定知识库（``notes/<space_id>/``）下更新过的文档正文，
  直读本地语料（``space_id`` 为空则不限知识库）；
* **归纳**：默认调用模型池 ``balanced`` 档，按给定小节骨架生成简明周报；未配置 Key / 调用失败 /
  本周无更新文档时回退规则模式（按文档标题 + 首句罗列）；
* **缓存**：结果按输入指纹缓存在 ``data/weekly_summary.json``，**按知识库（``space_id``）分开保存**，
  切换知识库再切回来仍能读回各自结果；
* **命名与周次**：统一由 :mod:`steproute.assistant.calendar` 提供（``{学年}-{学期}-week{周次}``）。

发布时读取模板文档的 ``##`` 小节作为骨架，把生成的小节标题按序替换（见 :func:`publish_weekly`）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from ..config import ModelSpecConfig
from ..corpus.model import Corpus
from ..llm.backends import Backend
from ..llm.json_utils import extract_json_block
from ..logging_utils import get_logger
from .calendar import WeekLabel, as_date, label_for
from .summarize import collect_week_documents
from .todo import doc_day

logger = get_logger(__name__)

#: 内置小节骨架（模板文档读不到时使用）。
DEFAULT_SECTIONS = ("本周进展", "实验与结论", "下周计划", "风险与待协助")

#: 归纳结果的缓存文件（位于 data/ 根目录，按 space_id 分开保存）。
WEEKLY_SUMMARY_FILENAME = "weekly_summary.json"

#: 采集「本周文档」的默认参数（比简报放宽，周报要覆盖一整周）。
DEFAULT_WEEKLY_DOCS = 20
DEFAULT_WEEKLY_DOC_CHARS = 1500
DEFAULT_WEEKLY_MATERIAL_CHARS = 15000


@dataclass
class WeeklySummary:
    """一次周报归纳的结果。"""

    markdown: str = ""
    method: str = "rules"  # llm / rules / none
    note: str = ""
    docs: int = 0  # 本次归纳读到的本周文档数
    week: str = ""  # 归属周（``{学年}-{学期}-week{N}``），用于校验缓存是否对应当前周
    generated_at: str = ""
    fingerprint: str = ""  # 输入指纹，用于缓存复用
    space_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "method": self.method,
            "note": self.note,
            "docs": self.docs,
            "week": self.week,
            "generated_at": self.generated_at,
            "fingerprint": self.fingerprint,
            "space_id": self.space_id,
        }


# --------------------------------------------------------------------------- #
# 素材 / 提示词
# --------------------------------------------------------------------------- #

def _system_prompt(titles: list[str]) -> str:
    keys = "，".join(f'"{title}"' for title in titles)
    return (
        "你是科研项目助理。请阅读给定的「本周更新的飞书文档正文」，写成一份简明的周报。\n\n"
        f"只输出一个 JSON 对象（不要代码围栏、不要解释），键为下列小节标题：{keys}；值为该小节的正文。\n\n"
        "要求：\n"
        "- 每个小节 2~4 句、连贯成文，禁止逐条罗列原文，禁止出现 `#` 标题或 `- [日期]` 之类的原始格式；\n"
        "- 内容必须来自给定文档，不要编造；某小节材料不足时写「（本周无相关内容）」。"
    )


def _materials(week_docs: list[Any], corpus: Corpus) -> str:
    lines: list[str] = ["### 本周更新的文档正文"]
    for doc in week_docs:
        updated = doc_day(corpus, doc.doc_id)
        when = updated.isoformat() if updated else ""
        lines.append(f"### 文档：{doc.title}（{when}）\n{doc.text}")
    return "\n".join(lines)[:DEFAULT_WEEKLY_MATERIAL_CHARS]


def _fingerprint(week_docs: list[Any], corpus: Corpus, titles: list[str]) -> str:
    blocks: list[str] = []
    for doc in week_docs:
        updated = doc_day(corpus, doc.doc_id)
        blocks.append(f"{doc.title}|{updated.isoformat() if updated else ''}|{doc.text}")
    joined = "\n".join(blocks)
    basis = f"{joined}\n::\n{'|'.join(titles)}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _first_sentence(text: str, *, limit: int = 80) -> str:
    for line in (text or "").splitlines():
        body = line.strip().lstrip("#>-* ").strip()
        if body:
            return body if len(body) <= limit else body[:limit] + "…"
    return ""


def _rules_content(week_docs: list[Any], corpus: Corpus, titles: list[str]) -> dict[str, str]:
    """规则回退：把本周每篇文档按「标题（日期）：首句」罗列进首个小节。"""
    first = titles[0] if titles else DEFAULT_SECTIONS[0]
    lines: list[str] = []
    for doc in week_docs:
        updated = doc_day(corpus, doc.doc_id)
        when = f"（{updated.isoformat()}）" if updated else ""
        snippet = _first_sentence(doc.text)
        lines.append(f"- 《{doc.title}》{when}：{snippet}" if snippet else f"- 《{doc.title}》{when}")
    return {first: "\n".join(lines)}


def _render_markdown(label: WeekLabel, titles: list[str], content: dict[str, str]) -> str:
    lines = [f"# {label.title}", "", f"> 统计区间：{label.start.isoformat()} ~ {label.end.isoformat()}", ""]
    for title in titles:
        body = str(content.get(title) or "").strip()
        lines += [f"## {title}", body or "（本周无相关内容）", ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# 缓存（按知识库分开保存）
# --------------------------------------------------------------------------- #

def _weekly_summary_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / WEEKLY_SUMMARY_FILENAME


def _load_all(data_dir: Path | str) -> dict[str, dict[str, Any]]:
    path = _weekly_summary_path(data_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("周报缓存损坏，忽略：%s", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        return {str(key): value for key, value in summaries.items() if isinstance(value, dict)}
    return {}


def save_weekly_summary(data_dir: Path | str, summary: WeeklySummary) -> Path:
    """把归纳结果按 ``space_id`` 写入 ``data/weekly_summary.json``（不影响其它知识库的缓存）。"""
    path = _weekly_summary_path(data_dir)
    summaries = _load_all(data_dir)
    summaries[summary.space_id] = summary.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_weekly_summary(data_dir: Path | str, *, space_id: str = "") -> WeeklySummary | None:
    """读取指定 ``space_id`` 的缓存；无对应条目（或文件缺失 / 损坏）时返回 ``None``。"""
    payload = _load_all(data_dir).get(space_id)
    if not isinstance(payload, dict):
        return None
    return WeeklySummary(
        markdown=str(payload.get("markdown") or ""),
        method=str(payload.get("method") or "rules"),
        note=str(payload.get("note") or ""),
        docs=int(payload.get("docs") or 0),
        week=str(payload.get("week") or ""),
        generated_at=str(payload.get("generated_at") or ""),
        fingerprint=str(payload.get("fingerprint") or ""),
        space_id=space_id,
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def summarize_weekly(
    *,
    data_dir: Path | str,
    corpus: Corpus,
    space_id: str = "",
    day: date | None = None,
    backend: Backend | None = None,
    spec: ModelSpecConfig | None = None,
    use_llm: bool = True,
    sections: list[str] | None = None,
    max_docs: int = DEFAULT_WEEKLY_DOCS,
    doc_chars: int = DEFAULT_WEEKLY_DOC_CHARS,
    max_chars: int = DEFAULT_WEEKLY_MATERIAL_CHARS,
) -> WeeklySummary:
    """把「本周更新的文档正文」归纳成一份简明周报。

    ``space_id`` 给定时只读该知识库（``notes/<space_id>/``）的文档，避免跨知识库串味；
    ``sections`` 给定时用它作为小节骨架（默认 :data:`DEFAULT_SECTIONS`）。``use_llm=False``、
    未注入 backend/spec、本周无更新文档，或模型调用 / 解析失败时回退规则模式。
    """
    data_root = Path(data_dir)
    label = label_for(day)
    titles = list(sections) if sections else list(DEFAULT_SECTIONS)
    week_docs = collect_week_documents(
        corpus,
        space_id=space_id,
        day=as_date(day),
        max_docs=max_docs,
        doc_chars=doc_chars,
        max_chars=max_chars,
    )

    summary = WeeklySummary(
        docs=len(week_docs),
        week=label.title,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        fingerprint=_fingerprint(week_docs, corpus, titles),
        space_id=space_id,
    )

    if not week_docs:
        summary.method = "none"
        summary.note = "本周该知识库没有更新的文档，先「拉取更新」再生成"
        summary.markdown = _render_markdown(label, titles, {})
        logger.info("周报归纳跳过模型：本周无更新文档 space_id=%s", space_id)
        return summary

    payload: dict[str, Any] = {}
    if use_llm and backend is not None and spec is not None:
        cached = load_weekly_summary(data_root, space_id=space_id)
        # 只在缓存本来就是 LLM 结果、且输入指纹一致时复用（规则回退结果不缓存复用）。
        if cached is not None and cached.method == "llm" and cached.week == summary.week and cached.fingerprint == summary.fingerprint:
            cached.note = cached.note or "输入未变化，复用缓存"
            logger.info("周报归纳复用缓存：fingerprint=%s", cached.fingerprint)
            return cached
        try:
            chat = backend.complete(
                spec,
                [
                    {"role": "system", "content": _system_prompt(titles)},
                    {"role": "user", "content": _materials(week_docs, corpus)},
                ],
                json_mode=True,
                temperature=0.2,
                # 归纳是抽取型任务，用低推理，避免 reasoning token 占满 max_tokens 导致 content 为空。
                reasoning_effort="low",
                max_tokens=spec.max_tokens,
                label="weekly",
            )
            parsed = extract_json_block(chat.content)
            payload = parsed if isinstance(parsed, dict) else {}
        except Exception as exc:  # 缺 Key / 网络异常 / 解析失败等都回退规则
            summary.note = f"LLM 归纳不可用，已回退规则：{exc}"
            logger.warning("周报归纳回退规则模式：%s", exc)
    elif not use_llm:
        summary.note = "未启用 LLM（use_llm=False）"
    else:
        summary.note = "未提供模型后端，按规则回退"

    if payload:
        content = {title: str(payload.get(title) or "").strip() for title in titles}
        if any(content.values()):
            summary.method = "llm"
            summary.markdown = _render_markdown(label, titles, content)
            logger.info("周报归纳完成：method=llm 文档 %d", summary.docs)
            return summary
        summary.note = summary.note or "模型返回中未包含可用内容，已回退规则"
    elif use_llm and backend is not None and spec is not None and not summary.note:
        summary.note = "模型返回无法解析为 JSON（常见原因：reasoning 占满 max_tokens），已回退规则"

    summary.method = "rules"
    summary.markdown = _render_markdown(label, titles, _rules_content(week_docs, corpus, titles))
    logger.info("周报归纳完成：method=rules 文档 %d", summary.docs)
    return summary


# --------------------------------------------------------------------------- #
# 写入飞书
# --------------------------------------------------------------------------- #

class _WeeklyClient(Protocol):
    def find_template_token(self, space_id: str, keyword: str) -> str: ...
    def read_document_text(self, document_id: str) -> str: ...
    def create_space_node(self, space_id: str, *, title: str, parent_node_token: str | None = ...) -> dict[str, Any]: ...
    def append_markdown(self, document_id: str, markdown: str) -> int: ...


@dataclass
class PublishResult:
    """周报发布结果。"""

    title: str
    document_id: str
    url: str
    blocks: int
    template_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "document_id": self.document_id,
            "url": self.url,
            "blocks": self.blocks,
            "template_token": self.template_token,
        }


def template_sections(template_text: str) -> list[str]:
    """从模板正文里提取小节标题（``##``/``###`` 行）。"""
    sections: list[str] = []
    for line in template_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            title = stripped.lstrip("#").strip()
            if title and title not in sections:
                sections.append(title)
    return sections


def publish_weekly(
    client: _WeeklyClient,
    *,
    space_id: str,
    markdown: str,
    title: str,
    template_token: str = "",
    template_keyword: str = "模板",
    parent_node_token: str | None = None,
) -> PublishResult:
    """在知识库 ``space_id`` 下新建一篇以 ``title`` 命名的周报并写入内容。"""
    if not template_token and template_keyword:
        template_token = client.find_template_token(space_id, template_keyword) or ""

    sections: list[str] = []
    if template_token:
        try:
            sections = template_sections(client.read_document_text(template_token))
        except Exception as exc:  # 模板不可读时回退内置结构，不阻断发布
            logger.warning("读取模板失败，改用内置小节：%s", exc)

    node = client.create_space_node(space_id, title=title, parent_node_token=parent_node_token)
    document_id = str(node.get("obj_token") or node.get("node_token") or "")
    if not document_id:
        raise RuntimeError(f"创建知识库节点失败，返回：{node}")

    body = markdown
    if sections:
        body = _reframe_with_sections(markdown, sections)
    blocks = client.append_markdown(document_id, body)

    node_token = str(node.get("node_token") or "")
    url = f"https://feishu.cn/wiki/{node_token}" if node_token else ""
    return PublishResult(title=title, document_id=document_id, url=url, blocks=blocks, template_token=template_token)


def _reframe_with_sections(markdown: str, sections: list[str]) -> str:
    """按模板小节顺序重排正文（保留内容，替换标题层级）。"""
    titles = template_sections(markdown)
    if not titles:
        return markdown
    lines = markdown.splitlines()
    # 用模板小节标题按序替换正文里的 ## 小节标题；多出的模板小节补空
    out: list[str] = []
    replaced = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if replaced < len(sections):
                out.append(f"## {sections[replaced]}")
                replaced += 1
            else:
                out.append(line)
        else:
            out.append(line)
    while replaced < len(sections):
        out += [f"## {sections[replaced]}", "- （待补充）", ""]
        replaced += 1
    return "\n".join(out).rstrip() + "\n"