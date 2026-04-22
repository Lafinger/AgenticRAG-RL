from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.agentic import run_agentic_episode
from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.retrieval import HybridRetriever


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_financial"


def test_run_agentic_episode_answers_comparison_question() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)

    result = run_agentic_episode("永辉超市和红旗连锁哪家的营业收入更高？", retriever, max_turns=4)

    assert result["final_answer"] == "永辉超市"
    assert result["tool_calls"] >= 2
    assert len(result["retrieved_chunk_ids"]) >= 2
