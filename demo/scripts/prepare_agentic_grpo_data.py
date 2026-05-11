from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.grpo_data import GRPO_DATA_SOURCE, build_grpo_rows
from agentic_rag_rl.io import load_multihop_examples


def _mark_split(rows: list[dict], split: str) -> list[dict]:
    for row in rows:
        extra_info = row.setdefault("extra_info", {})
        extra_info["split"] = split
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Agentic GRPO parquet data.")
    parser.add_argument("--input", default=str(ROOT / "data" / "novel_eval" / "qa_pairs.jsonl"))
    parser.add_argument("--train-output", default=str(ROOT / "data" / "novel_eval" / "grpo_agentic_train.parquet"))
    parser.add_argument("--val-output", default=str(ROOT / "data" / "novel_eval" / "grpo_agentic_val.parquet"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", default=GRPO_DATA_SOURCE)
    args = parser.parse_args()

    examples = load_multihop_examples(args.input)
    rows = build_grpo_rows(examples, data_source=args.data_source)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split_index = max(1, int(len(rows) * (1 - args.val_ratio)))
    train_rows = _mark_split(rows[:split_index], "train")
    val_rows = _mark_split(rows[split_index:], "val")
    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.train_output, index=False)
    pd.DataFrame(val_rows).to_parquet(args.val_output, index=False)
    print(f"train_count={len(train_rows)}")
    print(f"val_count={len(val_rows)}")


if __name__ == "__main__":
    main()
