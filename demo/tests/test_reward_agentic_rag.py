from __future__ import annotations

from training import reward_agentic_rag


def test_reward_extracts_tool_response_chunks_and_scores_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        reward_agentic_rag,
        "judge_agentic_answer",
        lambda **kwargs: {"correctness": 1.0, "faithfulness": 1.0},
    )
    solution = (
        '<think>要回答最终问题，先查：侯监集</think>\n'
        '<tool_call>{"name":"keyword_search","arguments":{"query":"侯监集"}}</tool_call>\n'
        "<tool_response>\n[chunk-a] 侯监集因侯赢得名。\n</tool_response>\n"
        "<answer>侯赢</answer>"
    )

    score = reward_agentic_rag.compute_score(
        solution_str=solution,
        ground_truth={
            "target": "侯赢",
            "question": "侯监集因谁得名？",
            "answer_aliases": ["侯赢"],
            "gold_chunks": ["chunk-a"],
            "hop_count": 1,
        },
    )

    assert score > 0.9


def test_reward_does_not_treat_missing_answer_tag_as_prediction(monkeypatch) -> None:
    called = False

    def fake_judge(**kwargs):
        nonlocal called
        called = True
        return {"correctness": 1.0, "faithfulness": 1.0}

    monkeypatch.setattr(reward_agentic_rag, "judge_agentic_answer", fake_judge)

    score = reward_agentic_rag.compute_score(
        solution_str="侯赢",
        ground_truth={"target": "侯赢", "gold_chunks": ["chunk-a"], "hop_count": 1},
    )

    assert called is False
    assert score == 0.0


def test_query_only_tool_call_is_not_counted_as_valid_tool_call() -> None:
    assert reward_agentic_rag._valid_tool_call_count("<tool_call>侯监集</tool_call>") == 0
    assert (
        reward_agentic_rag._valid_tool_call_count(
            '<tool_call>{"name":"keyword_search","arguments":{"query":"侯监集"}}</tool_call>'
        )
        == 1
    )
