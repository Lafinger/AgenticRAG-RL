from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .types import Chunk, RetrievalResult

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:\.[0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    segments = TOKEN_PATTERN.findall(text)
    tokens: list[str] = []
    for segment in segments:
        if re.match(r"[\u4e00-\u9fff]", segment):
            if jieba is None:
                tokens.extend(list(segment))
            else:
                tokens.extend([piece for piece in jieba.lcut(segment) if piece.strip()])
        else:
            tokens.append(segment)
    return [token for token in tokens if token.strip()]


def _score_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


class KeywordRetriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.index = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(scores[idx]),
                    title=chunk.title,
                    text=chunk.text,
                    source="keyword",
                    metadata={"company": chunk.company, **chunk.metadata},
                )
            )
        return results


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self._vectorizer.transform([query])
        values = linear_kernel(query_vector, self._matrix)[0]
        order = np.argsort(values)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(values[int(idx)]),
                    title=chunk.title,
                    text=chunk.text,
                    source="dense",
                    metadata={"company": chunk.company, **chunk.metadata},
                )
            )
        return results


class LightReranker:
    def rerank(self, query: str, results: Sequence[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for item in results:
            overlap = _score_overlap(query, item.text)
            scored.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=0.7 * item.score + 0.3 * overlap,
                    title=item.title,
                    text=item.text,
                    source=f"{item.source}+rerank",
                    metadata=item.metadata,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def rrf_fuse(results_list: Sequence[Sequence[RetrievalResult]], k: int = 60) -> list[RetrievalResult]:
    fused_scores: dict[str, float] = defaultdict(float)
    merged: dict[str, RetrievalResult] = {}
    sources: dict[str, list[str]] = defaultdict(list)

    for results in results_list:
        for rank, item in enumerate(results, start=1):
            fused_scores[item.chunk_id] += 1.0 / (k + rank)
            merged[item.chunk_id] = item
            sources[item.chunk_id].append(item.source)

    fused: list[RetrievalResult] = []
    for chunk_id, score in fused_scores.items():
        item = merged[chunk_id]
        fused.append(
            RetrievalResult(
                chunk_id=item.chunk_id,
                score=score,
                title=item.title,
                text=item.text,
                source="+".join(sorted(set(sources[item.chunk_id]))),
                metadata=item.metadata,
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.keyword = KeywordRetriever(self.chunks)
        self.dense = DenseRetriever(self.chunks)
        self.reranker = LightReranker()

    def keyword_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.keyword.search(query, top_k=top_k)

    def dense_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.dense.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 5, candidate_k: int = 8) -> list[RetrievalResult]:
        keyword_results = self.keyword_search(query, top_k=candidate_k)
        dense_results = self.dense_search(query, top_k=candidate_k)
        fused = rrf_fuse([keyword_results, dense_results])
        return self.reranker.rerank(query, fused[:candidate_k], top_k=top_k)

    def dispatch(self, tool_name: str, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if tool_name == "keyword_search":
            return self.keyword_search(query, top_k=top_k)
        if tool_name == "dense_search":
            return self.dense_search(query, top_k=top_k)
        if tool_name == "hybrid_search":
            return self.hybrid_search(query, top_k=top_k)
        raise ValueError(f"Unsupported tool: {tool_name}")
