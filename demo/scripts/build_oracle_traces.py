from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, load_multihop_examples, write_jsonl
from agentic_rag_rl.traces import build_oracle_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Build oracle traces from QA pairs and corpus.")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-zh", action="store_true")
    args = parser.parse_args()

    examples = load_multihop_examples(args.qa)
    chunks = load_chunks(args.corpus)
    traces = build_oracle_traces(examples, chunks, use_zh=args.use_zh)
    write_jsonl(traces, args.output)
    print(f"trace_count={len(traces)}")


if __name__ == "__main__":
    main()
