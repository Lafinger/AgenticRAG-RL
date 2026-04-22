from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Chunk, Hop, MultiHopExample


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in load_jsonl(path):
        chunks.append(
            Chunk(
                chunk_id=record["chunk_id"],
                title=record["title"],
                text=record["text"],
                company=record.get("company", ""),
                metadata=record.get("metadata", {}),
            )
        )
    return chunks


def load_examples(path: str | Path) -> list[MultiHopExample]:
    examples: list[MultiHopExample] = []
    for record in load_jsonl(path):
        hops = [
            Hop(
                hop_idx=hop["hop_idx"],
                question=hop["question"],
                answer=hop["answer"],
                doc_chunk_id=hop["doc_chunk_id"],
                qa_type=hop["qa_type"],
            )
            for hop in record["hops"]
        ]
        examples.append(
            MultiHopExample(
                final_question=record["final_question"],
                final_answer=record["final_answer"],
                hop_count=record["hop_count"],
                qa_type=record["qa_type"],
                subset=record["subset"],
                hops=hops,
            )
        )
    return examples

