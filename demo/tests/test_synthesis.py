from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.synthesis import clean_multihop_examples, generate_seed_questions, synthesize_multihop_examples


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_generate_seed_questions_extracts_novel_facts() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, max_per_chunk=1)

    assert seeds
    assert any("孙少平" in item["question"] or "双水村" in item["question"] for item in seeds)


def test_synthesize_and_clean_multihop_examples() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, max_per_chunk=1)
    examples = synthesize_multihop_examples(seeds, chunks, target_count=2)
    cleaned = clean_multihop_examples(examples, {chunk.chunk_id for chunk in chunks})

    assert cleaned
    assert cleaned[0]["hop_count"] >= 2
    assert all(hop["doc_chunk_id"] for hop in cleaned[0]["hops"])
