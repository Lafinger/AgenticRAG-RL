from __future__ import annotations

from pathlib import Path

from agentic_rag_study.demo_data import load_chunks
from agentic_rag_study.retrieval import HybridRetriever


def test_keyword_search_hits_governance_chunk() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "demo_financial" / "corpus.jsonl"
    chunks = load_chunks(data_path)
    retriever = HybridRetriever(chunks)

    results = retriever.keyword_search("曹世如 对应 公司", top_k=3)

    assert results
    assert results[0].metadata["company"] == "红旗连锁"
    assert any(item.chunk_id == "hq_0001" for item in results)


def test_hybrid_search_keeps_finance_chunk() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "demo_financial" / "corpus.jsonl"
    chunks = load_chunks(data_path)
    retriever = HybridRetriever(chunks)

    results = retriever.hybrid_search("红旗连锁 净利润", top_k=3)

    assert any(item.chunk_id == "hq_0002" for item in results)
