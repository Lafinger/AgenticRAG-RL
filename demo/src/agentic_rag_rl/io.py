from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

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


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            write_jsonl_record(handle, record)


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\r\n")


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
                pages=record.get("pages", []),
                section=record.get("section", record.get("metadata", {}).get("section", "")),
            )
        )
    return chunks


def chunk_map(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def load_multihop_examples(path: str | Path) -> list[MultiHopExample]:
    examples: list[MultiHopExample] = []
    for record in load_jsonl(path):
        hops = [
            Hop(
                hop_idx=hop["hop_idx"],
                question=hop["question"],
                answer=hop["answer"],
                doc_chunk_id=hop["doc_chunk_id"],
                qa_type=hop["qa_type"],
                search_tools=list(hop.get("search_tools") or ["keyword_search"]),
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
                answer_aliases=record.get("answer_aliases", []),
            )
        )
    return examples
