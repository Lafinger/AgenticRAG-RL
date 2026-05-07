from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.grpo_data import build_grpo_rows
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.protocols import TOOL_SCHEMAS
from agentic_rag_rl.traces import build_oracle_traces, convert_traces_to_sharegpt, convert_traces_to_sft_records


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_build_oracle_trace_to_sft_and_grpo_rows() -> None:
    corpus = load_chunks(DATA_DIR / "corpus.jsonl")
    examples = load_multihop_examples(DATA_DIR / "qa_pairs.jsonl")

    traces = build_oracle_traces(examples, corpus, use_zh=True)
    sft_records = convert_traces_to_sft_records(traces)
    sharegpt_records = convert_traces_to_sharegpt(sft_records)
    grpo_rows = build_grpo_rows(examples)

    assert len(traces) == 2
    assert len(sft_records) == 4
    assert len(sharegpt_records) == 4
    assert traces[0]["messages"][0]["role"] == "system"
    assert traces[0]["tools"] == TOOL_SCHEMAS
    assert traces[0]["messages"][2]["content"].startswith("<think>")
    assert "\n<tool_call>\n" in traces[0]["messages"][2]["content"]
    assert traces[0]["messages"][3]["role"] == "tool"
    assert "<tool_response>" not in traces[0]["messages"][3]["content"]
    full_trace = sft_records[0]
    finalization = sft_records[1]
    assert full_trace["metadata"]["sample_type"] == "full_trace"
    assert full_trace["messages"][3]["role"] == "tool"
    assert full_trace["tools"] == TOOL_SCHEMAS
    assert full_trace["messages"][-1]["content"].startswith("<answer>")
    assert finalization["metadata"]["sample_type"] == "finalization_only"
    assert finalization["messages"][0]["role"] == "system"
    assert finalization["messages"][1]["role"] == "user"
    assert finalization["messages"][-1]["role"] == "assistant"
    assert finalization["messages"][-1]["content"].startswith("<answer>")
    assert any(message["role"] == "tool" for message in finalization["messages"])
    assert all(message["role"] != "assistant" or "<tool_call>" not in message["content"] for message in finalization["messages"])
    assert finalization["tools"] == TOOL_SCHEMAS
    assert sharegpt_records[0]["messages"][0]["role"] == "system"
    assert sharegpt_records[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["agent_name"] == "tool_agent"
    assert grpo_rows[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["reward_model"]["ground_truth"]["gold_chunks"] == ["corpus_chunkids_000001", "corpus_chunkids_000001"]
