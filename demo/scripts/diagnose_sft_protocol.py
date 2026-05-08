from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_rl.protocols import render_canonical_chat
from training.sft_label_mask import IGNORE_INDEX, tokenize_chat_with_assistant_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose canonical ReAct SFT rendering and labels.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--data-path")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--show-rendered", action="store_true")
    parser.add_argument("--show-supervised", action="store_true")
    parser.add_argument("--eval-summary", help="Optional Agent loop summary JSON to include in the report.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
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


def iter_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def assistant_contents(record: dict[str, Any]) -> list[str]:
    return [str(message.get("content", "")) for message in record.get("messages", []) if message.get("role") == "assistant"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_path = project_path(args.data_path or config["data_path"])
    model_name = args.model_name_or_path or config["model_name_or_path"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    records = iter_jsonl(data_path, args.max_samples)
    report = {
        "data": str(data_path),
        "model_tokenizer": model_name,
        "sampled_records": len(records),
        "tool_turns": 0,
        "tool_turns_with_think": 0,
        "answer_turns": 0,
        "rendered_empty_answer_think": 0,
        "tool_response_label_leaks": 0,
        "max_token_length": 0,
        "min_supervised_tokens": None,
    }
    first_rendered = ""
    first_supervised = ""

    for record in records:
        tools = record.get("tools") if isinstance(record.get("tools"), list) else None
        rendered = render_canonical_chat(record["messages"], tools=tools, add_generation_prompt=False)
        sample = tokenize_chat_with_assistant_labels(tokenizer, record["messages"], tools=tools)
        labels = [token_id for token_id, label in zip(sample.input_ids, sample.labels, strict=True) if label != IGNORE_INDEX]
        supervised = tokenizer.decode(labels, skip_special_tokens=False)
        report["max_token_length"] = max(int(report["max_token_length"]), sample.token_length)
        current_min = report["min_supervised_tokens"]
        report["min_supervised_tokens"] = sample.supervised_token_count if current_min is None else min(current_min, sample.supervised_token_count)
        if "<tool_response>" in supervised:
            report["tool_response_label_leaks"] += 1
        if "<think>\n\n</think>\n\n<answer>" in rendered or "<think></think>\n<answer>" in rendered:
            report["rendered_empty_answer_think"] += 1
        for content in assistant_contents(record):
            if "<tool_call>" in content:
                report["tool_turns"] += 1
                if "<think>" in content and "</think>" in content:
                    report["tool_turns_with_think"] += 1
            if "<answer>" in content:
                report["answer_turns"] += 1
        if not first_rendered:
            first_rendered = rendered
            first_supervised = supervised

    if report["tool_turns"]:
        report["tool_turn_think_rate"] = report["tool_turns_with_think"] / report["tool_turns"]
    else:
        report["tool_turn_think_rate"] = None

    if args.eval_summary:
        with project_path(args.eval_summary).open("r", encoding="utf-8") as handle:
            report["eval_summary"] = json.load(handle)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show_rendered and first_rendered:
        print("\n--- rendered sample ---")
        print(first_rendered[:3000])
    if args.show_supervised and first_supervised:
        print("\n--- supervised sample ---")
        print(first_supervised[:1500])


if __name__ == "__main__":
    main()
