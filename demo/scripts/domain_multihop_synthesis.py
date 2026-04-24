from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl
from agentic_rag_rl.synthesis import synthesize_multihop_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize multi-hop QA examples from seeds.")
    parser.add_argument("--seeds", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "qa_pairs.jsonl"))
    parser.add_argument("--target-count", type=int, default=200)
    args = parser.parse_args()

    seeds = load_jsonl(args.seeds)
    chunks = load_chunks(args.corpus)
    examples = synthesize_multihop_examples(seeds, chunks, target_count=args.target_count)
    write_jsonl(examples, args.output)
    print(f"multihop_count={len(examples)}")


if __name__ == "__main__":
    main()
