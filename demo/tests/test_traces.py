from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.grpo_data import build_grpo_rows
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.traces import build_oracle_traces, convert_traces_to_sharegpt, convert_traces_to_sft_records


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_financial"


def test_build_oracle_trace_to_sft_and_grpo_rows() -> None:
    corpus = load_chunks(DATA_DIR / "corpus.jsonl")
    examples = load_multihop_examples(DATA_DIR / "qa_pairs.jsonl")

    traces = build_oracle_traces(examples, corpus, use_zh=True)
    sft_records = convert_traces_to_sft_records(traces)
    sharegpt_records = convert_traces_to_sharegpt(sft_records)
    grpo_rows = build_grpo_rows(examples)

    assert len(traces) == 2
    assert traces[0]["messages"][0]["role"] == "system"
    assert "<tool_call>" in traces[0]["messages"][2]["content"]
    assert "<tool_response>" in sft_records[0]["messages"][3]["content"]
    assert sharegpt_records[0]["messages"][0]["role"] == "system"
    assert grpo_rows[0]["agent_name"] == "tool_agent"
    assert grpo_rows[0]["reward_model"]["ground_truth"]["gold_chunks"] == ["yh_0002", "hq_0002"]
