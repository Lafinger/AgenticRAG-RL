from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REACT_TOOL_TURN_RE = re.compile(r"^\s*<think>(?P<think>.*?)</think>\s*<tool_call>\s*(?P<payload>.*?)\s*</tool_call>\s*$", re.DOTALL)
TOOL_TAG_RE = re.compile(r"</?tool_call\b[^>]*>", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


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


def validate_react_tool_turn(content: str, record_index: int, message_index: int) -> None:
    matched = REACT_TOOL_TURN_RE.fullmatch(content)
    if matched is None:
        raise ValueError(
            f"Record {record_index} message {message_index} assistant tool turn must be exactly "
            "<think>...</think> followed by one JSON <tool_call>...</tool_call>."
        )

    think = matched.group("think").strip()
    if not think:
        raise ValueError(f"Record {record_index} message {message_index} think content must not be empty.")
    if any(fragment in think.lower() for fragment in ("<tool_call", "</tool_call", "<answer", "</answer")):
        raise ValueError(f"Record {record_index} message {message_index} think content must not contain action tags.")
    if "{" in think or "}" in think:
        raise ValueError(f"Record {record_index} message {message_index} think content must not contain JSON.")

    payload_text = matched.group("payload").strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Record {record_index} message {message_index} tool_call JSON is invalid: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Record {record_index} message {message_index} tool_call payload must be an object.")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ValueError(f"Record {record_index} message {message_index} tool_call payload missing name.")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"Record {record_index} message {message_index} tool_call arguments must be an object.")
    if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
        raise ValueError(f"Record {record_index} message {message_index} tool_call arguments missing query.")


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
        content = message["content"]
        if message.get("role") == "assistant" and TOOL_TAG_RE.search(content):
            validate_react_tool_turn(content, index, message_index)
        elif message.get("role") == "assistant" and THINK_TAG_RE.search(content):
            raise ValueError(f"Record {index} message {message_index} think tag is only allowed in tool turns.")
    tools = record.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"Record {index} tools must be a list when present.")
    return record


def sample_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        sample_type = metadata.get("sample_type")
        key = str(sample_type).strip() if isinstance(sample_type, str) and sample_type.strip() else "default"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
        "chat_renderer": "canonical_qwen3_react",
        "tool_role_preserved": True,
        "tool_turn_requires_think": True,
        "message_loss_metadata": "assistant messages with loss=false are context only",
        "sample_type_counts": sample_type_counts(records),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(manifest_text.replace("\n", "\r\n"))
        handle.write("\r\n")
    print(json.dumps({"sft_count": len(records), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
