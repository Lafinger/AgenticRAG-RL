from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOOL_FRAGMENT_RE = re.compile(r"</?tool_call>", re.IGNORECASE)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())


def normalize_answer(text: str) -> str:
    return "".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and SFT prediction JSONL files.")
    parser.add_argument("--base", required=True, help="Base model prediction JSONL.")
    parser.add_argument("--sft", required=True, help="SFT model prediction JSONL.")
    parser.add_argument("--output", required=True, help="Summary JSON output path.")
    parser.add_argument(
        "--allow-question-mismatch",
        action="store_true",
        help="Allow comparing only the common question subset. By default, question sets must match exactly.",
    )
    return parser.parse_args()


def best_scores(record: dict[str, Any]) -> tuple[float, float]:
    if "em" in record and "f1" in record:
        return float(record["em"]), float(record["f1"])

    prediction = str(record.get("prediction", ""))
    gold = str(record.get("gold", ""))
    aliases = [str(alias) for alias in record.get("aliases", [])]
    candidates = [gold, *aliases]
    return (
        max(exact_match(prediction, candidate) for candidate in candidates),
        max(token_f1(prediction, candidate) for candidate in candidates),
    )


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    em, f1 = best_scores(record)
    raw_prediction = str(record.get("raw_prediction", record.get("prediction", "")))
    return {
        **record,
        "em": em,
        "f1": f1,
        "raw_prediction": raw_prediction,
        "answer_tag_present": bool(record.get("answer_tag_present", False)),
        "tool_call_count": int(record.get("tool_call_count", 0)),
        "valid_tool_call_count": int(record.get("valid_tool_call_count", 0)),
        "tool_fragment_present": bool(record.get("tool_fragment_present", TOOL_FRAGMENT_RE.search(raw_prediction))),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    divisor = max(count, 1)
    return {
        "count": count,
        "avg_em": sum(record["em"] for record in records) / divisor,
        "avg_f1": sum(record["f1"] for record in records) / divisor,
        "answer_tag_rate": sum(1 for record in records if record["answer_tag_present"]) / divisor,
        "tool_call_rate": sum(1 for record in records if record["tool_call_count"] > 0) / divisor,
        "valid_tool_call_rate": sum(1 for record in records if record["valid_tool_call_count"] > 0) / divisor,
        "tool_fragment_rate": sum(1 for record in records if record["tool_fragment_present"]) / divisor,
        "avg_generation_chars": sum(len(record["raw_prediction"]) for record in records) / divisor,
    }


def index_by_question(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question = str(record.get("question", ""))
        if question:
            indexed[question] = enrich(record)
    return indexed


def validate_question_alignment(
    base_records: dict[str, dict[str, Any]],
    sft_records: dict[str, dict[str, Any]],
    *,
    allow_mismatch: bool,
) -> tuple[set[str], set[str]]:
    base_only = set(base_records) - set(sft_records)
    sft_only = set(sft_records) - set(base_records)
    if (base_only or sft_only) and not allow_mismatch:
        raise SystemExit(
            "base 与 sft 的 question 集合不一致；"
            f"base_only_count={len(base_only)}, sft_only_count={len(sft_only)}。"
            "如需仅比较交集，请显式传 --allow-question-mismatch。"
        )
    return base_only, sft_only


def main() -> None:
    args = parse_args()
    base_records = index_by_question(load_jsonl(args.base))
    sft_records = index_by_question(load_jsonl(args.sft))
    base_only, sft_only = validate_question_alignment(
        base_records,
        sft_records,
        allow_mismatch=args.allow_question_mismatch,
    )
    common_questions = [question for question in base_records if question in sft_records]

    per_sample = []
    for question in common_questions:
        base = base_records[question]
        sft = sft_records[question]
        per_sample.append(
            {
                "question": question,
                "gold": sft.get("gold", base.get("gold", "")),
                "base_prediction": base.get("prediction", ""),
                "sft_prediction": sft.get("prediction", ""),
                "base_f1": base["f1"],
                "sft_f1": sft["f1"],
                "f1_delta": sft["f1"] - base["f1"],
                "base_em": base["em"],
                "sft_em": sft["em"],
                "em_delta": sft["em"] - base["em"],
                "base_answer_tag_present": base["answer_tag_present"],
                "sft_answer_tag_present": sft["answer_tag_present"],
                "base_tool_call_count": base["tool_call_count"],
                "sft_tool_call_count": sft["tool_call_count"],
                "base_valid_tool_call_count": base["valid_tool_call_count"],
                "sft_valid_tool_call_count": sft["valid_tool_call_count"],
                "base_tool_fragment_present": base["tool_fragment_present"],
                "sft_tool_fragment_present": sft["tool_fragment_present"],
            }
        )

    aligned_base = [base_records[question] for question in common_questions]
    aligned_sft = [sft_records[question] for question in common_questions]
    base_summary = summarize(aligned_base)
    sft_summary = summarize(aligned_sft)
    payload = {
        "summary": {
            "count": len(common_questions),
            "base": base_summary,
            "sft": sft_summary,
            "delta": {
                "avg_em": sft_summary["avg_em"] - base_summary["avg_em"],
                "avg_f1": sft_summary["avg_f1"] - base_summary["avg_f1"],
                "answer_tag_rate": sft_summary["answer_tag_rate"] - base_summary["answer_tag_rate"],
                "tool_call_rate": sft_summary["tool_call_rate"] - base_summary["tool_call_rate"],
                "valid_tool_call_rate": sft_summary["valid_tool_call_rate"] - base_summary["valid_tool_call_rate"],
                "tool_fragment_rate": sft_summary["tool_fragment_rate"] - base_summary["tool_fragment_rate"],
                "avg_generation_chars": sft_summary["avg_generation_chars"] - base_summary["avg_generation_chars"],
            },
            "base_only_count": len(base_only),
            "sft_only_count": len(sft_only),
            "question_mismatch_allowed": args.allow_question_mismatch,
        },
        "per_sample": per_sample,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
