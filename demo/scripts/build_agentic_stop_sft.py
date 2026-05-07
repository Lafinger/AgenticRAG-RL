from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "novel_eval" / "sft_zh_unsloth"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "novel_eval" / "sft_agentic_stop"
HELDOUT_TEST_PATH = ROOT / "data" / "novel_eval" / "test.jsonl"

AGENTIC_STOP_SYSTEM_PROMPT = (
    "你是一个中文小说阅读问答 Agent。你可以通过检索工具逐步查找证据。"
    "每一轮只能输出一个 <tool_call>{\"name\":\"keyword_search\",\"arguments\":{\"query\":\"...\"}}</tool_call> "
    "或最终 <answer>...</answer>。不要在同一轮同时输出工具调用和最终答案。"
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def reject_heldout_test(path: str | Path) -> None:
    input_path = resolve_project_path(path).resolve()
    if input_path == HELDOUT_TEST_PATH.resolve():
        raise ValueError(f"held-out test file must not be used for agentic-stop SFT data: {path}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL record must be an object: {path}")
                records.append(payload)
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def normalize_tool_call(content: str) -> str:
    match = TOOL_CALL_RE.search(content)
    if not match:
        raise ValueError("assistant tool turn is missing <tool_call>...</tool_call>.")
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant tool turn contains invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool_call payload must be an object.")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool_call payload is missing name.")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call arguments must be an object.")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("tool_call arguments are missing query.")
    normalized = {"name": name.strip(), "arguments": {"query": query.strip()}}
    return f"<tool_call>{json.dumps(normalized, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def normalize_tool_response(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("<tool_response>"):
        return content
    return f"<tool_response>{content}</tool_response>"


def extract_answer(content: str, metadata: dict[str, Any]) -> str:
    matches = ANSWER_RE.findall(content)
    if matches:
        return matches[-1].strip()
    answer = metadata.get("final_answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    raise ValueError("record is missing final <answer> and metadata.final_answer.")


def first_user_question(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and not content.lstrip().startswith("<tool_response>"):
                return content
    raise ValueError("record is missing the final question user message.")


def base_metadata(record: dict[str, Any], index: int, source_path: Path, sample_type: str) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {index} is missing messages.")
    final_question = metadata.get("final_question")
    if not isinstance(final_question, str) or not final_question.strip():
        final_question = first_user_question(messages)
    final_answer = metadata.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                final_answer = extract_answer(str(message["content"]), metadata)
                break
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError(f"Record {index} is missing final_answer.")
    gold_chunks = metadata.get("gold_chunks")
    if not isinstance(gold_chunks, list):
        gold_chunks = []
    return {
        **metadata,
        "sft_task": "agentic_stop",
        "sample_type": sample_type,
        "source_file": str(source_path),
        "source_line": index,
        "final_question": final_question.strip(),
        "final_answer": final_answer.strip(),
        "gold_chunks": [str(chunk_id) for chunk_id in gold_chunks],
    }


def build_full_trace_record(record: dict[str, Any], index: int, source_path: Path) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {index} is missing messages.")
    metadata = base_metadata(record, index, source_path, "full_trace")
    normalized_messages: list[dict[str, str]] = []
    system_added = False
    answer_count = 0
    tool_count = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Record {index} contains a message with non-string content.")
        if role == "system":
            if not system_added:
                normalized_messages.append({"role": "system", "content": AGENTIC_STOP_SYSTEM_PROMPT})
                system_added = True
        elif role == "user":
            normalized_messages.append({"role": "user", "content": content})
        elif role == "tool":
            normalized_messages.append({"role": "user", "content": normalize_tool_response(content)})
        elif role == "assistant":
            if ANSWER_RE.search(content):
                answer = extract_answer(content, metadata)
                normalized_messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
                answer_count += 1
            elif TOOL_CALL_RE.search(content):
                normalized_messages.append({"role": "assistant", "content": normalize_tool_call(content)})
                tool_count += 1
            else:
                raise ValueError(f"Record {index} has unsupported assistant content.")
        else:
            raise ValueError(f"Record {index} has unsupported role: {role!r}.")
    if not system_added:
        normalized_messages.insert(0, {"role": "system", "content": AGENTIC_STOP_SYSTEM_PROMPT})
    if answer_count != 1:
        raise ValueError(f"Record {index} must contain exactly one final answer turn, got {answer_count}.")
    if tool_count < 1:
        raise ValueError(f"Record {index} must contain at least one tool call turn.")
    return {"messages": normalized_messages, "metadata": {**metadata, "tool_turn_count": tool_count}}


def build_finalization_record(record: dict[str, Any], index: int, source_path: Path) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {index} is missing messages.")
    metadata = base_metadata(record, index, source_path, "finalization_only")
    tool_responses: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        role = message.get("role")
        if role == "tool":
            tool_responses.append(normalize_tool_response(content))
        elif role == "user" and content.lstrip().startswith("<tool_response>"):
            tool_responses.append(content)
    if not tool_responses:
        raise ValueError(f"Record {index} is missing tool_response context.")
    final_answer = str(metadata["final_answer"])
    return {
        "messages": [
            {"role": "system", "content": AGENTIC_STOP_SYSTEM_PROMPT},
            {"role": "user", "content": str(metadata["final_question"])},
            *({"role": "user", "content": response} for response in tool_responses),
            {"role": "assistant", "content": f"<answer>{final_answer}</answer>"},
        ],
        "metadata": {**metadata, "tool_response_count": len(tool_responses)},
    }


def build_records(records: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if metadata.get("sample_type") == "finalization_only":
            continue
        output.append(build_full_trace_record(record, index, source_path))
        output.append(build_finalization_record(record, index, source_path))
    return output


def question_set(records: list[dict[str, Any]]) -> set[str]:
    questions: set[str] = set()
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        question = metadata.get("final_question")
        if isinstance(question, str):
            questions.add(question)
    return questions


def output_sample_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        sample_type = metadata.get("sample_type")
        if isinstance(sample_type, str) and sample_type:
            counts[sample_type] = counts.get(sample_type, 0) + 1
    return counts


def build_agentic_stop_sft(
    *,
    train_input: str | Path = DEFAULT_INPUT_DIR / "train_cli.jsonl",
    eval_input: str | Path = DEFAULT_INPUT_DIR / "eval.jsonl",
    train_output: str | Path = DEFAULT_OUTPUT_DIR / "train.jsonl",
    eval_output: str | Path = DEFAULT_OUTPUT_DIR / "eval.jsonl",
    manifest_path: str | Path = DEFAULT_OUTPUT_DIR / "manifest.json",
) -> dict[str, Any]:
    reject_heldout_test(train_input)
    reject_heldout_test(eval_input)
    train_input_path = resolve_project_path(train_input)
    eval_input_path = resolve_project_path(eval_input)
    source_train = load_jsonl(train_input_path)
    source_eval = load_jsonl(eval_input_path)
    train_records = build_records(source_train, train_input_path)
    eval_records = build_records(source_eval, eval_input_path)
    write_jsonl(train_records, train_output)
    write_jsonl(eval_records, eval_output)
    train_questions = question_set(train_records)
    eval_questions = question_set(eval_records)
    train_sample_type_counts = output_sample_type_counts(train_records)
    eval_sample_type_counts = output_sample_type_counts(eval_records)
    manifest = {
        "format": "sharegpt_messages_jsonl",
        "sft_task": "agentic_stop",
        "system_prompt": AGENTIC_STOP_SYSTEM_PROMPT,
        "train_input": str(train_input_path),
        "eval_input": str(eval_input_path),
        "train_output": str(resolve_project_path(train_output)),
        "eval_output": str(resolve_project_path(eval_output)),
        "train_source_count": len(source_train),
        "eval_source_count": len(source_eval),
        "train_source_used_count": train_sample_type_counts.get("full_trace", 0),
        "eval_source_used_count": eval_sample_type_counts.get("full_trace", 0),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "sample_types": {
            "full_trace": {
                "train_count": train_sample_type_counts.get("full_trace", 0),
                "eval_count": eval_sample_type_counts.get("full_trace", 0),
            },
            "finalization_only": {
                "train_count": train_sample_type_counts.get("finalization_only", 0),
                "eval_count": eval_sample_type_counts.get("finalization_only", 0),
            },
        },
        "heldout_test_excluded": True,
        "train_eval_disjoint_by_question": train_questions.isdisjoint(eval_questions),
        "normalization": {
            "tool_assistant": "first_valid_tool_call_without_think",
            "finalization_only": "question_plus_tool_responses_then_answer",
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build agentic-stop SFT calibration data.")
    parser.add_argument("--train-input", default=str(DEFAULT_INPUT_DIR / "train_cli.jsonl"))
    parser.add_argument("--eval-input", default=str(DEFAULT_INPUT_DIR / "eval.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    manifest = build_agentic_stop_sft(
        train_input=args.train_input,
        eval_input=args.eval_input,
        train_output=output_dir / "train.jsonl",
        eval_output=output_dir / "eval.jsonl",
        manifest_path=output_dir / "manifest.json",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
