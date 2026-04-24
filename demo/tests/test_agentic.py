from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.agentic import run_agentic_episode
from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.retrieval import HybridRetriever


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_run_agentic_episode_answers_novel_question() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever(chunks)

    result = run_agentic_episode("双水村为什么叫双水村？", retriever, max_turns=4)

    assert "东拉河" in result["final_answer"]
    assert "哭咽河" in result["final_answer"]
    assert result["tool_calls"] >= 1
    assert "corpus_chunkids_000002" in result["retrieved_chunk_ids"]
