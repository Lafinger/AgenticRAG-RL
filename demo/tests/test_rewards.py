from __future__ import annotations

from agentic_rag_rl.rewards import RewardInputs, score_response


def test_reward_v5a_prefers_correctness_and_hop_recall() -> None:
    inputs = RewardInputs(
        prediction="<answer>永辉超市</answer>",
        target="永辉超市",
        answer_aliases=["永辉超市", "永辉"],
        gold_chunks=["yh_0002", "hq_0002"],
        retrieved_chunk_ids=["yh_0002", "hq_0002"],
        hop_count=2,
        tool_calls=2,
        judge_correctness=1.0,
        judge_faithfulness=0.4,
        grounded_answer=1.0,
    )

    result = score_response(inputs, reward_version="v5a")

    assert result.score == 0.91
    assert result.breakdown["hop_recall"] == 1.0


def test_reward_v6a_penalizes_insufficient_search() -> None:
    inputs = RewardInputs(
        prediction="<answer>彩食鲜</answer>",
        target="彩食鲜",
        answer_aliases=["彩食鲜"],
        gold_chunks=["yh_0001", "csx_0001"],
        retrieved_chunk_ids=["yh_0001"],
        hop_count=2,
        tool_calls=1,
        judge_correctness=0.8,
        judge_faithfulness=0.9,
        grounded_answer=1.0,
    )

    result = score_response(inputs, reward_version="v6a")

    assert result.breakdown["insufficient_search_penalty"] == -0.05
    assert round(result.score, 2) == 0.80


def test_reward_v9a_uses_hop_f1() -> None:
    inputs = RewardInputs(
        prediction="<answer>永辉超市</answer>",
        target="永辉超市",
        answer_aliases=["永辉超市", "永辉"],
        gold_chunks=["yh_0002", "hq_0002"],
        retrieved_chunk_ids=["yh_0002", "noise_0001"],
        hop_count=2,
        tool_calls=2,
        judge_correctness=0.7,
        judge_faithfulness=0.6,
        grounded_answer=1.0,
    )

    result = score_response(inputs, reward_version="v9a")

    assert round(result.breakdown["hop_precision_recall"], 2) == 0.5
    assert round(result.score, 2) == 0.68
