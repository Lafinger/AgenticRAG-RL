from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ShareGPT SFT records to Unsloth JSONL format.")
    parser.add_argument("--input-dir", required=True, help="Directory containing sharegpt.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train.jsonl.")
    parser.add_argument("--input-file", default="sharegpt.jsonl")
    parser.add_argument("--output-file", default="train.jsonl")
    return parser.parse_args()


def validate_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {index} does not contain a non-empty messages list.")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Record {index} message {message_index} must be an object.")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Record {index} message {message_index} has unsupported role: {message.get('role')!r}.")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"Record {index} message {message_index} content must be a string.")
    tools = record.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"Record {index} tools must be a list when present.")
    return record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_dir) / args.input_file
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_file

    records = [validate_record(record, index) for index, record in enumerate(load_jsonl(input_path), start=1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_path)
    manifest = {
        "format": "sharegpt_messages_jsonl",
        "source": str(input_path),
        "output": str(output_path),
        "count": len(records),
        "required_field": "messages",
        "trainer": "unsloth",
        "tool_schema": "qwen3",
        "tool_role_preserved": True,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(manifest_text.replace("\n", "\r\n"))
        handle.write("\r\n")
    print(json.dumps({"sft_count": len(records), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
