from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split multihop dataset into train and test sets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    test_size = min(args.test_size, len(records))
    test_records = records[:test_size]
    train_records = records[test_size:]

    output_dir = Path(args.output_dir)
    write_jsonl(train_records, output_dir / "train.jsonl")
    write_jsonl(test_records, output_dir / "test.jsonl")
    print(f"train_count={len(train_records)}")
    print(f"test_count={len(test_records)}")


if __name__ == "__main__":
    main()
