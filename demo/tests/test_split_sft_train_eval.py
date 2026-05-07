from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_sft_train_eval import split_sft_train_eval


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_split_sft_train_eval_writes_disjoint_files_and_split_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    train_output = tmp_path / "train_cli.jsonl"
    eval_output = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records = [{"id": index, "messages": [{"role": "user", "content": f"q{index}"}]} for index in range(10)]
    write_jsonl(input_path, records)
    old_eval_count_key = "stu" + "dio_eval_count"
    manifest_path.write_text(
        json.dumps({old_eval_count_key: 99, "existing": True}, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = split_sft_train_eval(
        input_path=input_path,
        train_output=train_output,
        eval_output=eval_output,
        manifest_path=manifest_path,
        eval_count=3,
    )

    train_records = read_jsonl(train_output)
    eval_records = read_jsonl(eval_output)
    assert summary["source_count"] == 10
    assert summary["train_count"] == 7
    assert summary["eval_count"] == 3
    assert {record["id"] for record in train_records}.isdisjoint({record["id"] for record in eval_records})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["existing"] is True
    assert manifest["split_eval_disjoint_from_train"] is True
    assert manifest["split_eval_count"] == 3
    assert old_eval_count_key not in manifest
