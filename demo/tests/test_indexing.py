from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.indexing import build_index_bundle
from agentic_rag_rl.io import load_chunks


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_financial"


def test_build_index_bundle_keeps_chunk_alignment() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    bundle = build_index_bundle(chunks)

    assert bundle["manifest"]["chunk_count"] == len(chunks)
    assert bundle["chunk_ids"][0] == chunks[0].chunk_id
    assert bundle["chunk_store"]["yh_0002"]["title"] == "永辉超市 2025 半年报 财务摘要"
