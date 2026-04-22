from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.retrieval import HybridRetriever, rrf_fuse, tokenize


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_financial"


def test_tokenize_supports_zh_and_en() -> None:
    tokens = tokenize("永辉超市 revenue 377.79 亿元")
    assert "永辉超市" in tokens or "永辉" in tokens
    assert "revenue" in tokens
    assert "377.79" in tokens


def test_rrf_fuse_merges_sources() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)
    keyword_results = retriever.keyword_search("永辉超市 营业收入", top_k=2)
    dense_results = retriever.dense_search("永辉超市 营业收入", top_k=2)

    fused = rrf_fuse([keyword_results, dense_results])

    assert fused
    assert fused[0].chunk_id == "yh_0002"
    assert "keyword" in fused[0].source
    assert "dense" in fused[0].source


def test_hybrid_search_prefers_financial_chunk() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)

    results = retriever.hybrid_search("永辉超市 营业收入", top_k=3)

    assert results
    assert results[0].chunk_id == "yh_0002"
