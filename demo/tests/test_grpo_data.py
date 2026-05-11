from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.grpo_data import build_grpo_rows
from agentic_rag_rl.io import load_multihop_examples


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_build_grpo_rows_matches_verl_tool_agent_schema() -> None:
    examples = load_multihop_examples(DATA_DIR / "qa_pairs.jsonl")

    row = build_grpo_rows(examples, split="train")[0]

    assert set(row) == {"data_source", "prompt", "ability", "reward_model", "extra_info", "metadata"}
    assert row["data_source"] == "novel_agentic_rag"
    assert row["ability"] == "multi_hop_qa"
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1] == {"role": "user", "content": examples[0].final_question}
    assert "gold_chunks" not in row["prompt"][0]["content"]
    assert "gold_chunks" not in row["prompt"][1]["content"]

    ground_truth = row["reward_model"]["ground_truth"]
    assert ground_truth["target"] == examples[0].final_answer
    assert ground_truth["answer"] == examples[0].final_answer
    assert ground_truth["question"] == examples[0].final_question
    assert ground_truth["answer_aliases"][0] == examples[0].final_answer
    assert ground_truth["gold_chunks"] == [hop.doc_chunk_id for hop in examples[0].hops]
    assert ground_truth["hop_count"] == examples[0].hop_count

    extra_info = row["extra_info"]
    assert extra_info["need_tools_kwargs"] is True
    assert extra_info["split"] == "train"
    assert set(extra_info["tools_kwargs"]) == {"keyword_search", "dense_search", "hybrid_search"}
    assert extra_info["tools_kwargs"]["keyword_search"]["create_kwargs"] == {
        "ground_truth": examples[0].final_answer,
        "question": examples[0].final_question,
        "data_source": "novel_agentic_rag",
    }
    assert row["metadata"]["tool_names"] == ["keyword_search", "dense_search", "hybrid_search"]

