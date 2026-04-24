from __future__ import annotations

from agentic_rag_rl.rewards import RewardInputs, score_response


def test_reward_v5a_prefers_correctness_and_hop_recall() -> None:
    inputs = RewardInputs(
        prediction="<answer>东拉河和哭咽河</answer>",
        target="东拉河和哭咽河",
        answer_aliases=["因东拉河和哭咽河得名"],
        gold_chunks=["corpus_chunkids_000002", "corpus_chunkids_000003"],
        retrieved_chunk_ids=["corpus_chunkids_000002", "corpus_chunkids_000003"],
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
        prediction="<answer>双水村小学</answer>",
        target="双水村小学",
        answer_aliases=["在双水村小学同班读书"],
        gold_chunks=["corpus_chunkids_000003", "corpus_chunkids_000002"],
        retrieved_chunk_ids=["corpus_chunkids_000003"],
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
        prediction="<answer>东拉河和哭咽河</answer>",
        target="东拉河和哭咽河",
        answer_aliases=["因东拉河和哭咽河得名"],
        gold_chunks=["corpus_chunkids_000002", "corpus_chunkids_000003"],
        retrieved_chunk_ids=["corpus_chunkids_000002", "noise_0001"],
        hop_count=2,
        tool_calls=2,
        judge_correctness=0.7,
        judge_faithfulness=0.6,
        grounded_answer=1.0,
    )

    result = score_response(inputs, reward_version="v9a")

    assert round(result.breakdown["hop_precision_recall"], 2) == 0.5
    assert round(result.score, 2) == 0.68
