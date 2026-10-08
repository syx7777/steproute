"""通用文本处理工具（工具层与奖励层共用）。"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_WS = re.compile(r"[ \t\u00a0]+")
_MULTI_NL = re.compile(r"\n{3,}")
_HTML_DROP = ("script", "style", "noscript", "template", "svg", "form", "iframe")
_HTML_BOILERPLATE = ("nav", "footer", "header", "aside")
_WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "by", "as", "at", "it", "its", "this", "that", "from", "we",
    "you", "they", "not", "but", "can", "will", "has", "have", "had", "which", "what",
    "how", "why", "when", "who", "than", "then", "into", "about", "their", "our",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def html_to_text(html: str, *, keep_boilerplate: bool = False) -> str:
    """抽取 HTML 正文文本（去除脚本/样式/导航等噪声）。"""
    try:
        from bs4 import BeautifulSoup
    except Exception:  # pragma: no cover - 依赖缺失时退化为正则清洗
        stripped = re.sub(r"<[^>]+>", " ", html)
        return normalize_whitespace(stripped)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_HTML_DROP):
        tag.decompose()
    if not keep_boilerplate:
        for tag in soup(_HTML_BOILERPLATE):
            tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text(separator="\n")
    return normalize_whitespace(text)


def truncate(text: str, max_chars: int, suffix: str = " …[已截断]") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def tokenize(text: str) -> list[str]:
    """按英文单词 / 中文字切分并过滤停用词。"""
    result: list[str] = []
    for token in _WORD.findall(text or ""):
        token = token.lower()
        if _is_cjk(token):
            result.append(token)
        elif token not in STOPWORDS and len(token) > 1:
            result.append(token)
    return result


def _is_cjk(token: str) -> bool:
    return bool(token) and "\u4e00" <= token[0] <= "\u9fff"


def shingles(text: str, k: int = 5) -> set[str]:
    """构造 k-shingle 集合，用于粗粒度去重（无需 embedding）。"""
    words = tokenize(text)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def novelty_ratio(new_shingles: set[str], seen_shingles: set[str]) -> float:
    """新信息占比 = 1 - 与历史信息的最大重叠度。"""
    if not new_shingles:
        return 0.0
    overlap = 0.0
    if seen_shingles:
        overlap = len(new_shingles & seen_shingles) / len(new_shingles)
    return max(0.0, 1.0 - overlap)


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:  # pragma: no cover
        return ""
    return host[4:] if host.startswith("www.") else host


def fingerprint(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def count_words(text: str) -> int:
    """统计字数：英文按词、中文按字。"""
    return len(_WORD.findall(text or ""))


def char_overlap(text: str, reference: str) -> float:
    """字符集合重合度（对中文友好，无需分词）。

    返回 ``|text ∩ reference| / |reference|``，即 ``reference`` 的字符被 ``text``
    覆盖的比例（containment），取值 ``[0, 1]``。
    """
    text_chars = {char for char in (text or "") if not char.isspace()}
    ref_chars = {char for char in (reference or "") if not char.isspace()}
    if not ref_chars:
        return 0.0
    return len(text_chars & ref_chars) / len(ref_chars)


def containment(text: str, reference: str, k: int = 4) -> float:
    """shingle 级包含度：``reference`` 的 k-shingle 有多大比例出现在 ``text`` 中。

    相比 :func:`char_overlap`，它对「只共享常见汉字但语义无关」的情况更不敏感，
    因此适合用于「关键事实是否被笔记覆盖」这类判定。
    """
    ref_shingles = shingles(reference, k)
    if not ref_shingles:
        return 0.0
    text_shingles = shingles(text, k)
    if not text_shingles:
        return 0.0
    return len(ref_shingles & text_shingles) / len(ref_shingles)