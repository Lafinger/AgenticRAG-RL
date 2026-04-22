from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SFT records to LLaMA-Factory sharegpt format.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(input_dir / "sharegpt.jsonl")
    (output_dir / "data.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_info = {
        "financial_agent_zh_react": {
            "file_name": "data.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        }
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sharegpt_count={len(records)}")


if __name__ == "__main__":
    main()
