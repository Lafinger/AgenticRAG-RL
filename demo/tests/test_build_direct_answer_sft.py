from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_direct_answer_sft import build_direct_answer_sft


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_source_record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": '<tool_call>{"name":"keyword_search","arguments":{"query":"q"}}</tool_call>'},
            {"role": "user", "content": "<tool_response>证据</tool_response>"},
            {"role": "assistant", "content": f"<answer>{answer}</answer>"},
        ],
        "metadata": {
            "final_question": question,
            "final_answer": answer,
            "answer_aliases": [answer],
            "gold_chunks": ["chunk-a"],
        },
    }


def test_build_direct_answer_sft_writes_three_turn_messages_and_manifest(tmp_path: Path) -> None:
    train_input = tmp_path / "train_studio.jsonl"
    eval_input = tmp_path / "eval.jsonl"
    train_output = tmp_path / "direct" / "train.jsonl"
    eval_output = tmp_path / "direct" / "eval.jsonl"
    manifest_path = tmp_path / "direct" / "manifest.json"
    write_jsonl(train_input, [make_source_record("谁救了段誉？", "古松")])
    write_jsonl(eval_input, [make_source_record("谁去了冰火岛？", "张翠山")])

    manifest = build_direct_answer_sft(
        train_input=train_input,
        eval_input=eval_input,
        train_output=train_output,
        eval_output=eval_output,
        manifest_path=manifest_path,
    )

    train_records = read_jsonl(train_output)
    assert train_records[0]["messages"] == [
        {
            "role": "system",
            "content": "你是中文小说问答评测模型。请直接回答用户问题，只输出 <answer>最终答案</answer>。不要输出 <tool_call>、分析过程或任何额外文字。",
        },
        {"role": "user", "content": "谁救了段誉？"},
        {"role": "assistant", "content": "<answer>古松</answer>"},
    ]
    user_and_assistant = train_records[0]["messages"][1:]
    assert "<tool_call>" not in json.dumps(user_and_assistant, ensure_ascii=False)
    assert "<tool_response>" not in json.dumps(user_and_assistant, ensure_ascii=False)
    assert train_records[0]["metadata"]["final_answer"] == "古松"
    assert train_records[0]["metadata"]["sft_task"] == "direct_answer"

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["heldout_test_excluded"] is True
    assert saved_manifest["train_count"] == 1
    assert saved_manifest["eval_count"] == 1
    assert saved_manifest["train_eval_disjoint_by_question"] is True


def test_build_direct_answer_sft_rejects_heldout_test_file(tmp_path: Path) -> None:
    heldout_test = ROOT / "data" / "novel_eval" / "test.jsonl"
    eval_input = tmp_path / "eval.jsonl"
    write_jsonl(eval_input, [make_source_record("问题", "答案")])

    with pytest.raises(ValueError, match="held-out test"):
        build_direct_answer_sft(
            train_input=heldout_test,
            eval_input=eval_input,
            train_output=tmp_path / "train.jsonl",
            eval_output=tmp_path / "eval_out.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
