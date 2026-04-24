from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl
from agentic_rag_rl.synthesis import clean_multihop_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean synthesized multi-hop examples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    valid_chunk_ids = set()
    if args.corpus:
        valid_chunk_ids = {chunk.chunk_id for chunk in load_chunks(args.corpus)}
    else:
        for record in records:
            for hop in record.get("hops", []):
                if hop.get("doc_chunk_id"):
                    valid_chunk_ids.add(hop["doc_chunk_id"])

    cleaned = clean_multihop_examples(records, valid_chunk_ids)
    write_jsonl(cleaned, args.output)
    print(f"cleaned_count={len(cleaned)}")


if __name__ == "__main__":
    main()
