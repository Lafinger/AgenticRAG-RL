from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_jsonl, write_jsonl
from agentic_rag_rl.traces import convert_traces_to_sharegpt, convert_traces_to_sft_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert oracle traces to SFT records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lang", default="zh")
    args = parser.parse_args()

    del args.lang
    traces = load_jsonl(args.input)
    sft_records = convert_traces_to_sft_records(traces)
    sharegpt_records = convert_traces_to_sharegpt(sft_records)
    output_dir = Path(args.output_dir)
    write_jsonl(sft_records, output_dir / "react.jsonl")
    write_jsonl(sharegpt_records, output_dir / "sharegpt.jsonl")
    write_jsonl(sharegpt_records, output_dir / "messages.jsonl")
    print(f"sft_count={len(sft_records)}")


if __name__ == "__main__":
    main()
