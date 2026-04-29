from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl
from agentic_rag_rl.synthesis import clean_seed_qa_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and refine generated seed QA records.")
    parser.add_argument("--input", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "seeds_clean.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--dropped-output", default=str(ROOT / "data" / "novel_eval" / "seeds_dropped.jsonl"))
    args = parser.parse_args()

    records = load_jsonl(args.input)
    chunks = load_chunks(args.corpus)
    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}
    cleaned, dropped = clean_seed_qa_records(records, valid_chunk_ids=valid_chunk_ids)

    write_jsonl(cleaned, args.output)
    write_jsonl(dropped, args.dropped_output)

    reason_counts: Counter[str] = Counter()
    for item in dropped:
        reason_counts.update(item.get("drop_reasons", []))

    print(f"input_count={len(records)}")
    print(f"cleaned_count={len(cleaned)}")
    print(f"dropped_count={len(dropped)}")
    for reason, count in sorted(reason_counts.items()):
        print(f"drop_reason.{reason}={count}")
    print(f"output={args.output}")
    print(f"dropped_output={args.dropped_output}")


if __name__ == "__main__":
    main()
