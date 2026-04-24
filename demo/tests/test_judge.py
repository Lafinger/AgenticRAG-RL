from __future__ import annotations

from agentic_rag_rl.judge import judge_agentic_answer


def test_judge_agentic_answer_scores_grounded_predictions() -> None:
    result = judge_agentic_answer(
        question="永辉超市和红旗连锁哪家的营业收入更高？",
        prediction="永辉超市",
        gold="永辉超市",
        aliases=["永辉超市", "永辉"],
        evidence_text="永辉超市营业收入 377.79 亿元；红旗连锁营业收入 52.33 亿元。",
        gold_chunks=["yh_0002", "hq_0002"],
        retrieved_chunk_ids=["yh_0002", "hq_0002"],
    )

    assert result["correctness"] == 1.0
    assert result["faithfulness"] > 0.5
    assert result["context_precision"] == 1.0
