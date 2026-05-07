from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DIRECT_ANSWER_SYSTEM_PROMPT = (
    "你是中文小说问答评测模型。请直接回答用户问题，只输出 <answer>最终答案</answer>。"
    "不要输出 <tool_call>、分析过程或任何额外文字。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direct-answer SFT train/eval JSONL files from SFT metadata.")
    parser.add_argument("--train-input", default="./data/novel_eval/sft_zh_unsloth/train_cli.jsonl")
    parser.add_argument("--eval-input", default="./data/novel_eval/sft_zh_unsloth/eval.jsonl")
    parser.add_argument("--train-output", default="./data/novel_eval/sft_direct_answer/train.jsonl")
    parser.add_argument("--eval-output", default="./data/novel_eval/sft_direct_answer/eval.jsonl")
    parser.add_argument("--manifest", default="./data/novel_eval/sft_direct_answer/manifest.json")
    parser.add_argument("--system-prompt", default=DIRECT_ANSWER_SYSTEM_PROMPT)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object: {path}")
            payload["_source_line"] = line_number
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def assert_not_heldout_test(path: Path, heldout_test_path: Path) -> None:
    if path.resolve() == heldout_test_path.resolve():
        raise ValueError(f"Direct-answer SFT data must not use held-out test file: {path}")


def metadata_from_record(record: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Record from {source_path}:{record.get('_source_line')} is missing metadata.")

    final_question = metadata.get("final_question")
    final_answer = metadata.get("final_answer")
    if not isinstance(final_question, str) or not final_question.strip():
        raise ValueError(f"Record from {source_path}:{record.get('_source_line')} is missing metadata.final_question.")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError(f"Record from {source_path}:{record.get('_source_line')} is missing metadata.final_answer.")

    cleaned = dict(metadata)
    cleaned["source_file"] = str(source_path)
    cleaned["source_line"] = record.get("_source_line")
    cleaned["sft_task"] = "direct_answer"
    return cleaned


def to_direct_answer_record(record: dict[str, Any], *, source_path: Path, system_prompt: str) -> dict[str, Any]:
    metadata = metadata_from_record(record, source_path=source_path)
    final_question = str(metadata["final_question"])
    final_answer = str(metadata["final_answer"])
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_question},
            {"role": "assistant", "content": f"<answer>{final_answer}</answer>"},
        ],
        "metadata": metadata,
    }


def should_use_source_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return metadata.get("sample_type") != "finalization_only"


def convert_records(records: list[dict[str, Any]], *, source_path: Path, system_prompt: str) -> list[dict[str, Any]]:
    return [
        to_direct_answer_record(record, source_path=source_path, system_prompt=system_prompt)
        for record in records
        if should_use_source_record(record)
    ]


def question_set(records: list[dict[str, Any]]) -> set[str]:
    questions: set[str] = set()
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        question = metadata.get("final_question")
        if isinstance(question, str) and question:
            questions.add(question)
    return questions


def build_direct_answer_sft(
    *,
    train_input: Path,
    eval_input: Path,
    train_output: Path,
    eval_output: Path,
    manifest_path: Path,
    system_prompt: str = DIRECT_ANSWER_SYSTEM_PROMPT,
) -> dict[str, Any]:
    heldout_test_path = ROOT / "data" / "novel_eval" / "test.jsonl"
    assert_not_heldout_test(train_input, heldout_test_path)
    assert_not_heldout_test(eval_input, heldout_test_path)

    train_source = load_jsonl(train_input)
    eval_source = load_jsonl(eval_input)
    train_records = convert_records(train_source, source_path=train_input, system_prompt=system_prompt)
    eval_records = convert_records(eval_source, source_path=eval_input, system_prompt=system_prompt)

    train_questions = question_set(train_records)
    eval_questions = question_set(eval_records)
    overlap_count = len(train_questions & eval_questions)

    write_jsonl(train_output, train_records)
    write_jsonl(eval_output, eval_records)

    manifest = {
        "format": "sharegpt_messages_jsonl",
        "task": "direct_answer_sft",
        "system_prompt": system_prompt,
        "source_train": str(train_input),
        "source_eval": str(eval_input),
        "train_output": str(train_output),
        "eval_output": str(eval_output),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "train_eval_question_overlap_count": overlap_count,
        "train_eval_disjoint_by_question": overlap_count == 0,
        "heldout_test_excluded": True,
        "heldout_test_path": str(heldout_test_path),
    }
    write_json(manifest_path, manifest)

    return manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    manifest = build_direct_answer_sft(
        train_input=project_path(args.train_input),
        eval_input=project_path(args.eval_input),
        train_output=project_path(args.train_output),
        eval_output=project_path(args.eval_output),
        manifest_path=project_path(args.manifest),
        system_prompt=args.system_prompt,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
