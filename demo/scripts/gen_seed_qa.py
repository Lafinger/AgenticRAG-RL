from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, write_jsonl
from agentic_rag_rl.synthesis import generate_seed_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed QA from novel corpus chunks.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--max-per-chunk", type=int, default=2)
    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    seeds = generate_seed_questions(chunks, max_per_chunk=args.max_per_chunk)
    write_jsonl(seeds, args.output)
    print(f"seed_count={len(seeds)}")


if __name__ == "__main__":
    main()
