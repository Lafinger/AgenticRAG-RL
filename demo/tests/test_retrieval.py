from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.retrieval import HybridRetriever, rrf_fuse, tokenize


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_tokenize_supports_zh_and_en() -> None:
    tokens = tokenize("孙少平 lived in 双水村")
    assert "孙少平" in tokens or "少平" in tokens
    assert "lived" in tokens
    assert "双水村" in tokens or "双水" in tokens


def test_rrf_fuse_merges_sources() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)
    keyword_results = retriever.keyword_search("双水村 东拉河 哭咽河", top_k=2)
    dense_results = retriever.dense_search("双水村 东拉河 哭咽河", top_k=2)

    fused = rrf_fuse([keyword_results, dense_results])

    assert fused
    assert fused[0].chunk_id == "corpus_chunkids_000002"
    assert "keyword" in fused[0].source
    assert "dense" in fused[0].source


def test_hybrid_search_prefers_novel_chunk() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)

    results = retriever.hybrid_search("孙少安 田润叶 同班", top_k=3)

    assert results
    assert results[0].chunk_id == "corpus_chunkids_000003"


def test_dispatch_supports_semantic_and_graph_tool_names() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)

    semantic_results = retriever.dispatch("semantic_search", "双水村 东拉河", top_k=2)
    graph_results = retriever.dispatch("graph_search", "双水村 东拉河", top_k=2)

    assert semantic_results
    assert graph_results
