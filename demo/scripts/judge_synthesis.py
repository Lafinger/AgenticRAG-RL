from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_jsonl, write_jsonl


def _score_record(record: dict) -> dict:
    answer_correctness = 5 if record.get("final_answer") else 1
    multihop_necessity = 5 if record.get("hop_count", 0) >= 2 else 1
    question_clarity = 5 if len(record.get("final_question", "")) >= 8 else 2
    total = answer_correctness + multihop_necessity + question_clarity
    return {
        **record,
        "judge_scores": {
            "answer_correctness": answer_correctness,
            "multihop_necessity": multihop_necessity,
            "question_clarity": question_clarity,
        },
        "judge_total_score": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristically judge synthesized examples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--min-score", type=int, default=9)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.filter_only:
        filtered = [record for record in records if record.get("judge_total_score", 0) >= args.min_score]
        write_jsonl(filtered, args.output)
        print(f"filtered_count={len(filtered)}")
        return

    judged = [_score_record(record) for record in records]
    write_jsonl(judged, args.output)
    print(f"judged_count={len(judged)}")


if __name__ == "__main__":
    main()
