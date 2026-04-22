from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .types import Chunk, RetrievalResult

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency fallback
    jieba = None

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency fallback
    faiss = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    CrossEncoder = None
    SentenceTransformer = None

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
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    text_set = set(text_tokens)
    return len(query_set & text_set) / max(len(query_set), 1)


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
    def __init__(self, chunks: Sequence[Chunk], embedding_model_name: str | None = None):
        self.chunks = list(chunks)
        self.embedding_model_name = embedding_model_name
        self._faiss_index = None
        self._sentence_model = None
        self._matrix = None
        self._vectorizer = None
        self._embeddings = None
        texts = [chunk.text for chunk in self.chunks]

        if embedding_model_name and SentenceTransformer is not None:
            self._sentence_model = SentenceTransformer(embedding_model_name)
            embeddings = self._sentence_model.encode(texts, normalize_embeddings=True)
            self._embeddings = np.asarray(embeddings, dtype="float32")
            if faiss is not None:
                self._faiss_index = faiss.IndexFlatIP(self._embeddings.shape[1])
                self._faiss_index.add(self._embeddings)
        else:
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._sentence_model is not None and self._embeddings is not None:
            query_vector = self._sentence_model.encode([query], normalize_embeddings=True)
            query_vector = np.asarray(query_vector, dtype="float32")
            if self._faiss_index is not None:
                scores, indices = self._faiss_index.search(query_vector, top_k)
                order = indices[0]
                values = scores[0]
            else:
                values = np.dot(self._embeddings, query_vector[0])
                order = np.argsort(values)[::-1][:top_k]
                values = values[order]
        else:
            query_vector = self._vectorizer.transform([query])
            values = linear_kernel(query_vector, self._matrix)[0]
            order = np.argsort(values)[::-1][:top_k]
            values = values[order]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(order):
            if int(idx) < 0:
                continue
            chunk = self.chunks[int(idx)]
            score = float(values[rank]) if rank < len(values) else 0.0
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    title=chunk.title,
                    text=chunk.text,
                    source="dense",
                    metadata={"company": chunk.company, **chunk.metadata},
                )
            )
        return results


class LightReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._cross_encoder = CrossEncoder(model_name) if model_name and CrossEncoder is not None else None

    def rerank(self, query: str, results: Sequence[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        candidates = list(results)
        if not candidates:
            return []

        if self._cross_encoder is not None:
            scores = self._cross_encoder.predict([[query, item.text] for item in candidates])
            order = np.argsort(scores)[::-1][:top_k]
            reranked: list[RetrievalResult] = []
            for idx in order:
                item = candidates[int(idx)]
                reranked.append(
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        score=float(scores[int(idx)]),
                        title=item.title,
                        text=item.text,
                        source=f"{item.source}+rerank",
                        metadata=item.metadata,
                    )
                )
            return reranked

        scored = []
        for item in candidates:
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
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
    ):
        self.chunks = list(chunks)
        self.keyword = KeywordRetriever(self.chunks)
        self.dense = DenseRetriever(self.chunks, embedding_model_name=embedding_model_name)
        self.reranker = LightReranker(model_name=reranker_model_name)

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

