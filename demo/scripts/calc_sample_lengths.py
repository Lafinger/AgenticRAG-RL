from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag_rl.protocols import render_canonical_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate token lengths for SFT JSONL samples.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--data-path")
    parser.add_argument("--limits", nargs="+", type=int, default=[512, 1024, 1536, 2048, 4096])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def percentile(sorted_values: list[int], p: int) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p / 100)))
    return sorted_values[index]


def render_text(tokenizer: Any, record: dict[str, Any], line_no: int) -> str:
    del tokenizer
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Line {line_no} is missing messages.")
    tools = record.get("tools") if isinstance(record.get("tools"), list) else None
    return render_canonical_chat(messages, tools=tools, add_generation_prompt=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = args.model_name_or_path or config["model_name_or_path"]
    data_path = project_path(args.data_path or config["data_path"])
    max_seq_length = int(config.get("max_seq_length", 2048))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    lengths: list[tuple[int, int]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = render_text(tokenizer, record, line_no)
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            lengths.append((line_no, token_count))

    if not lengths:
        raise SystemExit(f"No samples found: {data_path}")

    values = [token_count for _, token_count in lengths]
    sorted_values = sorted(values)
    print(f"data: {data_path}")
    print(f"model tokenizer: {model_name}")
    print(f"samples: {len(values)}")
    print(f"max_seq_length: {max_seq_length}")
    print(
        "min: {min_len} p50: {p50} avg: {avg} p90: {p90} p95: {p95} p99: {p99} max: {max_len}".format(
            min_len=min(values),
            p50=percentile(sorted_values, 50),
            avg=round(mean(values), 1),
            p90=percentile(sorted_values, 90),
            p95=percentile(sorted_values, 95),
            p99=percentile(sorted_values, 99),
            max_len=max(values),
        )
    )
    for limit in args.limits:
        count = sum(token_count > limit for _, token_count in lengths)
        print(f"> {limit}: {count} ({count / len(values) * 100:.1f}%)")

    if args.top > 0:
        print(f"top {args.top} longest:")
        for line_no, token_count in sorted(lengths, key=lambda item: item[1], reverse=True)[: args.top]:
            print(f"  line {line_no}: {token_count}")


if __name__ == "__main__":
    main()
