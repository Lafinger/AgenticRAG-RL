from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .types import Chunk


def _chunk_record(chunk: Chunk) -> dict[str, Any]:
    record = {"title": chunk.title, "text": chunk.text, "metadata": chunk.metadata}
    if chunk.company:
        record["company"] = chunk.company
    return record


def build_index_bundle(chunks: list[Chunk]) -> dict[str, Any]:
    return {
        "manifest": {"chunk_count": len(chunks), "index_type": "bm25+tifidf-light"},
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "chunk_store": {chunk.chunk_id: _chunk_record(chunk) for chunk in chunks},
    }


def save_index_bundle(bundle: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "chunk_ids.json").write_text(json.dumps(bundle["chunk_ids"], ensure_ascii=False, indent=2), encoding="utf-8")
    with (target / "chunk_store.pkl").open("wb") as handle:
        pickle.dump(bundle["chunk_store"], handle)
