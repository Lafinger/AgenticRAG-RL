from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SPLIT_PREFIX = "stu" + "dio_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split SFT JSONL into disjoint command-line train/eval files.")
    parser.add_argument("--input", default="./data/novel_eval/sft_zh_unsloth_react_v3/train.jsonl")
    parser.add_argument("--train-output", default="./data/novel_eval/sft_zh_unsloth_react_v3/train_cli.jsonl")
    parser.add_argument("--eval-output", default="./data/novel_eval/sft_zh_unsloth_react_v3/eval.jsonl")
    parser.add_argument("--manifest", default="./data/novel_eval/sft_zh_unsloth_react_v3/manifest.json")
    parser.add_argument("--eval-count", type=int, default=200)
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


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def choose_eval_indices(record_count: int, eval_count: int) -> list[int]:
    if eval_count <= 0:
        raise ValueError("--eval-count must be positive.")
    if record_count <= eval_count:
        raise ValueError(f"Need more than {eval_count} records, got {record_count}.")
    step = max(record_count // eval_count, 1)
    eval_indices: list[int] = []
    for index in range(0, record_count, step):
        eval_indices.append(index)
        if len(eval_indices) >= eval_count:
            break
    return eval_indices


def sample_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    value = metadata.get("sample_type")
    return str(value).strip() if isinstance(value, str) and value.strip() else "default"


def count_by_sample_type(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = sample_type(record)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def allocate_eval_counts(groups: dict[str, list[int]], eval_count: int) -> dict[str, int]:
    total = sum(len(indices) for indices in groups.values())
    raw_allocations = {
        key: eval_count * len(indices) / total
        for key, indices in groups.items()
    }
    eligible_groups = {key: indices for key, indices in groups.items() if len(indices) > 1}
    if eval_count < len(eligible_groups):
        selected = sorted(eligible_groups, key=lambda key: len(eligible_groups[key]), reverse=True)[:eval_count]
        return {key: 1 for key in selected}
    allocations = {
        key: min(max(1, int(value)), len(groups[key]) - 1)
        for key, value in raw_allocations.items()
        if key in eligible_groups
    }
    while sum(allocations.values()) < eval_count:
        candidates = [
            key
            for key in allocations
            if allocations[key] < len(groups[key]) - 1
        ]
        if not candidates:
            break
        best_key = max(candidates, key=lambda key: raw_allocations[key] - allocations[key])
        allocations[best_key] += 1
    while sum(allocations.values()) > eval_count:
        candidates = [key for key, count in allocations.items() if count > 1]
        if not candidates:
            break
        best_key = min(candidates, key=lambda key: raw_allocations[key] - allocations[key])
        allocations[best_key] -= 1
    return allocations


def choose_stratified_eval_indices(records: list[dict[str, Any]], eval_count: int) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(sample_type(record), []).append(index)
    if len(groups) <= 1:
        return choose_eval_indices(len(records), eval_count)

    allocations = allocate_eval_counts(groups, eval_count)
    eval_indices: list[int] = []
    for key in sorted(allocations):
        group_indices = groups[key]
        relative_indices = choose_eval_indices(len(group_indices), allocations[key])
        eval_indices.extend(group_indices[index] for index in relative_indices)
    return sorted(eval_indices)


def split_records(records: list[dict[str, Any]], eval_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    eval_indices = choose_stratified_eval_indices(records, eval_count)
    eval_index_set = set(eval_indices)
    train_records = [record for index, record in enumerate(records) if index not in eval_index_set]
    eval_records = [records[index] for index in eval_indices]
    stride = max(len(records) // eval_count, 1)
    return train_records, eval_records, stride


def split_sft_train_eval(
    input_path: Path,
    train_output: Path,
    eval_output: Path,
    manifest_path: Path,
    eval_count: int,
) -> dict[str, Any]:
    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found: {input_path}")

    train_records, eval_records, stride = split_records(records, eval_count)
    write_jsonl(train_output, train_records)
    write_jsonl(eval_output, eval_records)

    manifest = read_manifest(manifest_path)
    manifest.update(
        {
            "split_train_output": str(train_output),
            "split_eval_output": str(eval_output),
            "split_source": str(input_path),
            "split_train_count": len(train_records),
            "split_eval_count": len(eval_records),
            "split_eval_stride": stride,
            "split_eval_disjoint_from_train": True,
            "split_train_sample_type_counts": count_by_sample_type(train_records),
            "split_eval_sample_type_counts": count_by_sample_type(eval_records),
        }
    )
    for key in list(manifest):
        if key.startswith(LEGACY_SPLIT_PREFIX):
            del manifest[key]
    write_json(manifest_path, manifest)

    return {
        "source": str(input_path),
        "source_count": len(records),
        "train_output": str(train_output),
        "train_count": len(train_records),
        "eval_output": str(eval_output),
        "eval_count": len(eval_records),
        "eval_stride": stride,
        "train_sample_type_counts": count_by_sample_type(train_records),
        "eval_sample_type_counts": count_by_sample_type(eval_records),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    summary = split_sft_train_eval(
        input_path=project_path(args.input),
        train_output=project_path(args.train_output),
        eval_output=project_path(args.eval_output),
        manifest_path=project_path(args.manifest),
        eval_count=args.eval_count,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
