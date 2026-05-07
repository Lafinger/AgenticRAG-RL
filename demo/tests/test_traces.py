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
    assert traces[0]["messages"][0]["role"] == "system"
    assert traces[0]["tools"] == TOOL_SCHEMAS
    assert traces[0]["messages"][2]["content"].startswith("<think>")
    assert "\n<tool_call>\n" in traces[0]["messages"][2]["content"]
    assert traces[0]["messages"][3]["role"] == "tool"
    assert "<tool_response>" not in traces[0]["messages"][3]["content"]
    assert sft_records[0]["messages"][3]["role"] == "tool"
    assert sft_records[0]["tools"] == TOOL_SCHEMAS
    assert sft_records[0]["messages"][-1]["content"].startswith("<answer>")
    assert sharegpt_records[0]["messages"][0]["role"] == "system"
    assert sharegpt_records[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["agent_name"] == "tool_agent"
    assert grpo_rows[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["reward_model"]["ground_truth"]["gold_chunks"] == ["corpus_chunkids_000001", "corpus_chunkids_000001"]
