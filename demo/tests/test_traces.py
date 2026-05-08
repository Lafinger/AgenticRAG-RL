from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_rl.grpo_data import build_grpo_rows
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.protocols import TOOL_SCHEMAS
from agentic_rag_rl.traces import build_oracle_traces, convert_traces_to_sharegpt, convert_traces_to_sft_records
from agentic_rag_rl.types import Chunk, Hop, MultiHopExample


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_build_oracle_trace_to_sft_and_grpo_rows() -> None:
    corpus = load_chunks(DATA_DIR / "corpus.jsonl")
    examples = load_multihop_examples(DATA_DIR / "qa_pairs.jsonl")

    traces = build_oracle_traces(examples, corpus, use_zh=True)
    sft_records = convert_traces_to_sft_records(traces)
    sharegpt_records = convert_traces_to_sharegpt(sft_records)
    grpo_rows = build_grpo_rows(examples)

    assert len(traces) == 2
    assert len(sft_records) == 10
    assert len(sharegpt_records) == 10
    assert examples[0].hops[0].search_tools == ["keyword_search"]
    assert traces[0]["messages"][0]["role"] == "system"
    assert traces[0]["tools"] == TOOL_SCHEMAS
    assert traces[0]["plan"][0] == {
        "id": 1,
        "sub_query": examples[0].hops[0].question,
        "tool": "keyword_search",
        "depends_on": [],
        "gold_answer": examples[0].hops[0].answer,
        "gold_chunk_id": examples[0].hops[0].doc_chunk_id,
    }
    assert traces[0]["tool_calls"][0]["tool"] == "keyword_search"
    assert traces[0]["evidence"][0]["results"][0]["chunk_id"] == examples[0].hops[0].doc_chunk_id
    assert traces[0]["messages"][2]["content"].startswith("<think>要回答最终问题，先查：")
    assert traces[0]["messages"][2]["content"].count("<think>") == 1
    assert traces[0]["messages"][2]["content"].count("</think>") == 1
    assert "\n<tool_call>\n" in traces[0]["messages"][2]["content"]
    assert "\n</tool_call>" in traces[0]["messages"][2]["content"]
    trace_payload = traces[0]["messages"][2]["content"].split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    assert json.loads(trace_payload) == {
        "name": "keyword_search",
        "arguments": {"query": examples[0].hops[0].question},
    }
    assert traces[0]["messages"][3]["role"] == "tool"
    assert "<tool_response>" not in traces[0]["messages"][3]["content"]
    full_trace = sft_records[0]
    first_action = sft_records[1]
    first_action_repeat = sft_records[2]
    next_action = sft_records[3]
    final_answer = sft_records[4]
    assert full_trace["metadata"]["sample_type"] == "full_trace"
    assert full_trace["messages"][3]["role"] == "tool"
    assert full_trace["tools"] == TOOL_SCHEMAS
    tool_turns = [message for message in full_trace["messages"] if message["role"] == "assistant" and "<tool_call>" in message["content"]]
    assert tool_turns
    for index, message in enumerate(tool_turns):
        expected_prefix = "<think>要回答最终问题，先查：" if index == 0 else "<think>已获得上一跳线索“"
        assert message["content"].startswith(expected_prefix)
        assert message["content"].count("<think>") == 1
        assert message["content"].count("</think>") == 1
        payload = message["content"].split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
        parsed = json.loads(payload)
        assert parsed["name"] == "keyword_search"
        assert isinstance(parsed["arguments"]["query"], str)
    assert full_trace["messages"][-1]["content"].startswith("<answer>")
    assert "<think>" not in full_trace["messages"][-1]["content"]
    assert first_action["metadata"]["sample_type"] == "first_action_only"
    assert first_action["metadata"]["repeat_index"] == 1
    assert first_action_repeat["metadata"]["sample_type"] == "first_action_only"
    assert first_action_repeat["metadata"]["repeat_index"] == 2
    assert [message["role"] for message in first_action["messages"]] == ["system", "user", "assistant"]
    assert first_action["messages"][-1]["content"].startswith("<think>要回答最终问题，先查：")
    assert next_action["metadata"]["sample_type"] == "next_action_only"
    assert next_action["metadata"]["target_tool_turn"] == 2
    assert next_action["messages"][2]["loss"] is False
    assert next_action["messages"][-1]["loss"] is True
    assert next_action["messages"][-1]["content"].startswith("<think>已获得上一跳线索“")
    assert final_answer["metadata"]["sample_type"] == "final_answer_only"
    assert final_answer["messages"][2]["loss"] is False
    assert final_answer["messages"][4]["loss"] is False
    assert final_answer["messages"][-1]["loss"] is True
    assert final_answer["messages"][-1]["content"].startswith("<answer>")
    assert final_answer["tools"] == TOOL_SCHEMAS
    assert sharegpt_records[0]["messages"][0]["role"] == "system"
    assert sharegpt_records[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["agent_name"] == "tool_agent"
    assert grpo_rows[0]["tools"] == TOOL_SCHEMAS
    assert grpo_rows[0]["reward_model"]["ground_truth"]["gold_chunks"] == ["corpus_chunkids_000001", "corpus_chunkids_000001"]


def test_convert_legacy_trace_rewrites_existing_think() -> None:
    trace = {
        "messages": [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "最终问题"},
            {
                "role": "assistant",
                "content": '<think>旧的随意 think</think><tool_call>{"name":"keyword_search","arguments":{"query":"第一跳"}}</tool_call>',
            },
            {"role": "tool", "content": "<tool_response>[chunk-a] 证据</tool_response>"},
            {"role": "assistant", "content": "<answer>答案</answer>"},
        ],
        "metadata": {"final_question": "最终问题", "final_answer": "答案"},
    }

    records = convert_traces_to_sft_records([trace])
    first_tool_turn = records[0]["messages"][2]["content"]

    assert "旧的随意 think" not in first_tool_turn
    assert first_tool_turn.startswith("<think>要回答最终问题，先查：第一跳</think>")
    assert records[0]["messages"][3]["content"] == "[chunk-a] 证据"
    assert records[-1]["metadata"]["sample_type"] == "final_answer_only"
    assert records[-1]["messages"][2]["loss"] is False


def test_oracle_trace_uses_hybrid_search_for_multi_tool_hops() -> None:
    chunks = [Chunk(chunk_id="chunk-1", title="标题", text="证据")]
    examples = [
        MultiHopExample(
            final_question="最终问题",
            final_answer="答案",
            hop_count=1,
            qa_type="inference",
            subset="smoke",
            hops=[
                Hop(
                    hop_idx=1,
                    question="需要多路检索的问题？",
                    answer="答案",
                    doc_chunk_id="chunk-1",
                    qa_type="inference",
                    search_tools=["keyword_search", "dense_search"],
                )
            ],
        )
    ]

    trace = build_oracle_traces(examples, chunks, use_zh=True)[0]

    assert trace["plan"][0]["tool"] == "hybrid_search"
    payload = trace["messages"][2]["content"].split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    assert json.loads(payload)["name"] == "hybrid_search"
