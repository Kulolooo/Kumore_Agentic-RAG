"""BM25 检索（M0 基线，纯关键词，不上向量）。

刻意手写而不是装 ``rank_bm25``：一共不到 80 行，自己写才能保证
"分数 = 可解释的公式"，后续 M1 加向量时也能明确说出混合的是哪两路分数。

公式（Lucene 变体的 IDF，恒为正，避免高频词出现负分）::

    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.text.tokenize import tokenize


@dataclass
class BM25Stats:
    n_docs: int
    n_terms: int
    avg_doc_len: float
    total_tokens: int


class BM25Index:
    """内存倒排索引。M0 规模（百份文档、数千 chunk）下每次启动重建即可。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError(f"k1 必须为正数，当前 {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b 必须在 [0,1] 之间，当前 {b}")
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._doc_len: list[int] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._total_tokens: int = 0

    # ---- 构建 ----
    def build(self, items: Iterable[tuple[str, str]]) -> "BM25Index":
        self._ids = []
        self._doc_len = []
        self._postings = defaultdict(dict)
        self._df = defaultdict(int)
        total = 0

        for doc_ref, text in items:
            idx = len(self._ids)
            self._ids.append(doc_ref)
            tokens = tokenize(text)
            total += len(tokens)
            self._doc_len.append(len(tokens))
            counts: dict[str, int] = defaultdict(int)
            for tok in tokens:
                counts[tok] += 1
            for term, tf in counts.items():
                self._postings[term][idx] = tf
                self._df[term] += 1

        self._total_tokens = total
        n = len(self._ids)
        self._avgdl = total / n if n else 0.0
        return self

    # ---- 查询 ----
    def _ranked(self, query: str) -> list[tuple[str, float]]:
        """全部得分 > 0 的 (doc_ref, score)，按分数降序。"""
        if not self._ids:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        n = len(self._ids)
        k1, b, avgdl = self.k1, self.b, self._avgdl
        scores: dict[int, float] = defaultdict(float)

        # 同一个词在查询里出现多次不重复加分（BM25 标准做法是不含 qtf，或用 qtf 加权）。
        # M0 用去重版本，行为更容易预测。
        for term in set(q_terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            df = self._df[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in posting.items():
                dl = self._doc_len[idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl) if avgdl else tf
                scores[idx] += idf * tf * (k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(self._ids[idx], score) for idx, score in ranked if score > 0]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序；分数为 0 的不返回。"""
        if top_k <= 0:
            return []
        return self._ranked(query)[:top_k]

    def score_all(self, query: str) -> list[tuple[str, float]]:
        """全量排名，供检索层做"同族去重 + 多版本展开"。

        分层做而不是在 BM25 里做：BM25 只负责打分，怎么用分数是检索层的决定。
        """
        return self._ranked(query)

    def score(self, query: str, doc_ref: str) -> float:
        """单文档得分，评测里用来核对"金标文档是否被检索到"。"""
        for ref, s in self.search(query, top_k=len(self._ids)):
            if ref == doc_ref:
                return s
        return 0.0

    @property
    def stats(self) -> BM25Stats:
        return BM25Stats(
            n_docs=len(self._ids),
            n_terms=len(self._postings),
            avg_doc_len=self._avgdl,
            total_tokens=self._total_tokens,
        )
