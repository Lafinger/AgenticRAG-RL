from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_agentic_stop_sft import build_agentic_stop_sft


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
            {"role": "system", "content": "旧 system prompt"},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": '<think>先搜人物</think><tool_call>{"name":"keyword_search","arguments":{"query":"第一跳"}}</tool_call>',
            },
            {"role": "user", "content": "<tool_response>[chunk-a] 第一段证据</tool_response>"},
            {
                "role": "assistant",
                "content": '<think>再搜答案</think><tool_call>{"name":"keyword_search","arguments":{"query":"第二跳"}}</tool_call>',
            },
            {"role": "user", "content": "<tool_response>[chunk-b] 第二段证据</tool_response>"},
            {"role": "assistant", "content": f"<answer>{answer}</answer>"},
        ],
        "metadata": {
            "final_question": question,
            "final_answer": answer,
            "answer_aliases": [answer],
            "gold_chunks": ["chunk-a", "chunk-b"],
        },
    }


def test_build_agentic_stop_sft_writes_normalized_trace_and_finalization_samples(tmp_path: Path) -> None:
    train_input = tmp_path / "train_studio.jsonl"
    eval_input = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "agentic_stop"
    write_jsonl(train_input, [make_source_record("谁救了段誉？", "古松")])
    write_jsonl(eval_input, [make_source_record("谁去了冰火岛？", "张翠山")])

    manifest = build_agentic_stop_sft(
        train_input=train_input,
        eval_input=eval_input,
        train_output=output_dir / "train.jsonl",
        eval_output=output_dir / "eval.jsonl",
        manifest_path=output_dir / "manifest.json",
    )

    train_records = read_jsonl(output_dir / "train.jsonl")
    assert len(train_records) == 2
    full_trace, finalization = train_records
    assert full_trace["metadata"]["sample_type"] == "full_trace"
    assert finalization["metadata"]["sample_type"] == "finalization_only"
    assert full_trace["metadata"]["final_answer"] == "古松"
    assert full_trace["metadata"]["gold_chunks"] == ["chunk-a", "chunk-b"]

    assert {message["role"] for message in full_trace["messages"]} <= {"system", "user", "assistant"}
    tool_assistant_messages = [
        message["content"]
        for message in full_trace["messages"]
        if message["role"] == "assistant" and "<tool_call>" in message["content"]
    ]
    assert len(tool_assistant_messages) == 2
    assert all("<think>" not in content for content in tool_assistant_messages)
    assert all(content.count("<tool_call>") == 1 for content in tool_assistant_messages)
    assert full_trace["messages"][-1] == {"role": "assistant", "content": "<answer>古松</answer>"}

    finalization_assistant = [message for message in finalization["messages"] if message["role"] == "assistant"]
    assert finalization_assistant == [{"role": "assistant", "content": "<answer>古松</answer>"}]
    assert "<tool_call>" not in json.dumps(finalization_assistant, ensure_ascii=False)
    assert "<tool_response>" in json.dumps(finalization["messages"], ensure_ascii=False)

    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["heldout_test_excluded"] is True
    assert saved_manifest["train_count"] == 2
    assert saved_manifest["eval_count"] == 2
    assert saved_manifest["sample_types"]["full_trace"]["train_count"] == 1
    assert saved_manifest["sample_types"]["finalization_only"]["train_count"] == 1


def test_build_agentic_stop_sft_rejects_heldout_test_file(tmp_path: Path) -> None:
    heldout_test = ROOT / "data" / "novel_eval" / "test.jsonl"
    eval_input = tmp_path / "eval.jsonl"
    write_jsonl(eval_input, [make_source_record("问题", "答案")])

    with pytest.raises(ValueError, match="held-out test"):
        build_agentic_stop_sft(
            train_input=heldout_test,
            eval_input=eval_input,
            train_output=tmp_path / "train.jsonl",
            eval_output=tmp_path / "eval_out.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
