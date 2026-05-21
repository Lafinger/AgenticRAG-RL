from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Any
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


def _chunk_metadata(chunk: Chunk) -> dict[str, object]:
    metadata: dict[str, object] = dict(chunk.metadata)
    if chunk.company:
        metadata["company"] = chunk.company
    if chunk.pages:
        metadata["pages"] = chunk.pages
    if chunk.section:
        metadata["section"] = chunk.section
    return metadata


def _record_to_result(record: dict[str, Any], *, score: float, source: str) -> RetrievalResult:
    metadata = dict(record.get("metadata", {}))
    if record.get("pages") is not None:
        metadata["pages"] = record.get("pages", [])
    if record.get("section"):
        metadata["section"] = record["section"]
    return RetrievalResult(
        chunk_id=record["chunk_id"],
        score=float(score),
        title=record["title"],
        text=record["text"],
        source=source,
        metadata=metadata,
    )


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
                    metadata=_chunk_metadata(chunk),
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
                    metadata=_chunk_metadata(chunk),
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


class CrossEncoderReranker:
    def __init__(self, model_path: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, max_length=max_length)

    def rerank(self, query: str, results: Sequence[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        if not results:
            return []
        scores = self.model.predict([[query, item.text] for item in results])
        reranked: list[RetrievalResult] = []
        for item, score in zip(results, scores, strict=False):
            reranked.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=float(score),
                    title=item.title,
                    text=item.text,
                    source=f"{item.source}+rerank",
                    metadata=item.metadata,
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


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

    def semantic_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.dense_search(query, top_k=top_k)

    def graph_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.hybrid_search(query, top_k=top_k)

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
        if tool_name == "semantic_search":
            return self.semantic_search(query, top_k=top_k)
        if tool_name == "graph_search":
            return self.graph_search(query, top_k=top_k)
        if tool_name == "hybrid_search":
            return self.hybrid_search(query, top_k=top_k)
        raise ValueError(f"Unsupported tool: {tool_name}")


class _SentenceTransformerQueryEncoder:
    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32)


class IndexedKeywordRetriever:
    def __init__(self, index_dir: str | Path):
        target = Path(index_dir)
        self.chunk_ids = json.loads((target / "chunk_ids.json").read_text(encoding="utf-8"))
        with (target / "chunk_store.pkl").open("rb") as handle:
            self.chunk_store = pickle.load(handle)
        with (target / "bm25.pkl").open("rb") as handle:
            self.index = pickle.load(handle)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            _record_to_result(self.chunk_store[self.chunk_ids[int(idx)]], score=float(scores[int(idx)]), source="keyword_search")
            for idx in order
        ]


class IndexedSemanticRetriever:
    def __init__(self, index_dir: str | Path, embedding_model: str):
        import faiss

        target = Path(index_dir)
        self.index = faiss.read_index(str(target / "faiss.index"))
        self.chunk_ids = json.loads((target / "chunk_ids.json").read_text(encoding="utf-8"))
        with (target / "chunk_store.pkl").open("rb") as handle:
            self.chunk_store = pickle.load(handle)
        self.encoder = _SentenceTransformerQueryEncoder(embedding_model)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        scores, indices = self.index.search(self.encoder.encode_query(query), top_k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if int(idx) < 0:
                continue
            chunk_id = self.chunk_ids[int(idx)]
            results.append(_record_to_result(self.chunk_store[chunk_id], score=float(score), source="semantic_search"))
        return results


class IndexedGraphRetriever:
    def __init__(self, index_dir: str | Path, embedding_model: str, reranker: LightReranker | CrossEncoderReranker):
        import networkx as nx

        target = Path(index_dir)
        graph_path = target / "knowledge_graph.json"
        embeddings_path = target / "entity_embeddings.pkl"
        if not graph_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError("knowledge_graph.json and entity_embeddings.pkl are required for graph_search.")
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        edge_key = "edges" if "edges" in graph_payload else "links"
        self.graph = nx.node_link_graph(graph_payload, edges=edge_key)
        with embeddings_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.entities: list[str] = payload.get("entities", [])
        self.entity_embeddings = np.asarray(payload.get("embeddings", np.zeros((0, 0))), dtype=np.float32)
        with (target / "chunk_store.pkl").open("rb") as handle:
            self.chunk_store = pickle.load(handle)
        self.encoder = _SentenceTransformerQueryEncoder(embedding_model)
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5, top_entities: int = 5, max_hops: int = 2, candidate_k: int = 20) -> list[RetrievalResult]:
        if not self.entities or self.entity_embeddings.size == 0:
            return []
        query_vector = self.encoder.encode_query(query)[0]
        scores = self.entity_embeddings @ query_vector
        order = np.argsort(scores)[::-1][:top_entities]

        chunk_scores: dict[str, float] = {}
        queue: deque[tuple[str, int, float]] = deque((self.entities[int(idx)], 0, float(scores[int(idx)])) for idx in order)
        seen: set[tuple[str, int]] = set()
        while queue:
            entity, depth, base_score = queue.popleft()
            if (entity, depth) in seen or entity not in self.graph:
                continue
            seen.add((entity, depth))

            for chunk_id in self.graph.nodes[entity].get("mentions", []):
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), base_score / (1 + depth))

            if depth >= max_hops:
                continue
            neighbors = set(self.graph.successors(entity)) | set(self.graph.predecessors(entity))
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(entity, neighbor, default={})
                for edge in edge_data.values() if isinstance(edge_data, dict) else []:
                    chunk_id = edge.get("chunk_id")
                    if chunk_id:
                        chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), base_score / (1 + depth))
                queue.append((neighbor, depth + 1, base_score))

        candidates = [
            _record_to_result(self.chunk_store[chunk_id], score=score, source="graph_search")
            for chunk_id, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
            if chunk_id in self.chunk_store
        ]
        return self.reranker.rerank(query, candidates, top_k=top_k)


class IndexedHybridRetriever:
    def __init__(self, index_dir: str | Path, *, embedding_model: str, reranker_model: str | None = None):
        self.keyword = IndexedKeywordRetriever(index_dir)
        self.semantic = IndexedSemanticRetriever(index_dir, embedding_model)
        self.reranker: LightReranker | CrossEncoderReranker = (
            CrossEncoderReranker(reranker_model) if reranker_model else LightReranker()
        )
        try:
            self.graph = IndexedGraphRetriever(index_dir, embedding_model, self.reranker)
        except FileNotFoundError:
            self.graph = None

    def keyword_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.keyword.search(query, top_k=top_k)

    def dense_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.semantic_search(query, top_k=top_k)

    def semantic_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.semantic.search(query, top_k=top_k)

    def graph_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self.graph is None:
            return []
        return self.graph.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievalResult]:
        keyword_results = self.keyword_search(query, top_k=candidate_k)
        semantic_results = self.semantic_search(query, top_k=candidate_k)
        fused = rrf_fuse([keyword_results, semantic_results])
        return self.reranker.rerank(query, fused[:candidate_k], top_k=top_k)

    def dispatch(self, tool_name: str, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if tool_name == "keyword_search":
            return self.keyword_search(query, top_k=top_k)
        if tool_name == "dense_search":
            return self.dense_search(query, top_k=top_k)
        if tool_name == "semantic_search":
            return self.semantic_search(query, top_k=top_k)
        if tool_name == "graph_search":
            return self.graph_search(query, top_k=top_k)
        if tool_name == "hybrid_search":
            return self.hybrid_search(query, top_k=top_k)
        raise ValueError(f"Unsupported tool: {tool_name}")
