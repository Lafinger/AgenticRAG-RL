from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.grpo_data import build_grpo_rows
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.retrieval import HybridRetriever
from agentic_rag_rl.traces import build_oracle_traces


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_financial"


def test_smoke_pipeline_contracts_hold_together() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    examples = load_multihop_examples(DATA_DIR / "qa_pairs.jsonl")
    retriever = HybridRetriever(chunks)

    traces = build_oracle_traces(examples, chunks, use_zh=True)
    results = retriever.hybrid_search(examples[0].final_question, top_k=2)
    grpo_rows = build_grpo_rows(examples)

    assert traces[0]["metadata"]["hop_count"] == 2
    assert results
    assert grpo_rows[0]["prompt"][1]["content"] == examples[0].final_question
