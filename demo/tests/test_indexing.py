from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.indexing import build_index_bundle
from agentic_rag_rl.io import load_chunks


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_build_index_bundle_keeps_chunk_alignment() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    bundle = build_index_bundle(chunks)

    assert bundle["manifest"]["chunk_count"] == len(chunks)
    assert bundle["chunk_ids"][0] == chunks[0].chunk_id
    assert bundle["chunk_store"]["corpus_chunkids_000002"]["title"] == "平凡的世界 段落 2"
