"""检索：面向中英文混排科研材料的 BM25。

不依赖任何外部搜索服务：本地建倒排索引，零成本、可离线、可复现。
中文用**字符二元组（bigram）**切词（无需分词词典），英文/数字按词切分。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .model import Chunk, Corpus, SourceKind

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英文混排分词：ASCII 词元 + 中文 bigram。"""
    text = text.lower()
    tokens: list[str] = _ASCII_WORD.findall(text)

    cjk_run: list[str] = []
    for char in text:
        if _CJK.match(char):
            cjk_run.append(char)
        elif cjk_run:
            tokens.extend(_cjk_bigrams(cjk_run))
            cjk_run = []
    if cjk_run:
        tokens.extend(_cjk_bigrams(cjk_run))
    return tokens


def _cjk_bigrams(chars: list[str]) -> list[str]:
    if len(chars) == 1:
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


@dataclass
class ChunkScore:
    """一个召回片段及其得分。"""

    chunk: Chunk
    score: float

    @property
    def snippet(self) -> str:
        return self.chunk.text

    def to_dict(self, *, max_chars: int = 320) -> dict[str, object]:
        return {
            "doc_id": self.chunk.doc_id,
            "chunk_id": self.chunk.chunk_id,
            "title": self.chunk.title,
            "kind": self.chunk.kind.value,
            "score": round(self.score, 4),
            "snippet": self.chunk.text[:max_chars],
        }


class Retriever:
    """BM25 检索器。"""

    def __init__(self, corpus: Corpus, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = [tokenize(chunk.text) for chunk in corpus.chunks]
        self._doc_len: list[int] = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len: float = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 1.0
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        for tokens in self._doc_tokens:
            counter = Counter(tokens)
            self._tf.append(counter)
            for term in counter:
                self._df[term] += 1

    # -- 检索 -------------------------------------------------------------- #
    def search(
        self,
        query: str,
        *,
        kind: SourceKind | None = None,
        top_k: int = 5,
        doc_ids: set[str] | None = None,
    ) -> list[ChunkScore]:
        """返回与 ``query`` 最相关的片段。"""
        terms = tokenize(query)
        if not terms:
            return []
        total_docs = len(self.corpus.chunks)
        scores: list[ChunkScore] = []
        for index, chunk in enumerate(self.corpus.chunks):
            if kind is not None and chunk.kind is not kind:
                continue
            if doc_ids is not None and chunk.doc_id not in doc_ids:
                continue
            score = self._score(index, terms, total_docs)
            if score > 0:
                scores.append(ChunkScore(chunk=chunk, score=score))
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

    def _score(self, index: int, terms: list[str], total_docs: int) -> float:
        tf = self._tf[index]
        length = self._doc_len[index] or 1
        score = 0.0
        for term in set(terms):
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            score += idf * numerator / denominator
        return score