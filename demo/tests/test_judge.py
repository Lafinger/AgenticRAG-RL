from __future__ import annotations

from agentic_rag_rl.judge import judge_agentic_answer


def test_judge_agentic_answer_scores_grounded_predictions() -> None:
    result = judge_agentic_answer(
        question="双水村为什么叫双水村？",
        prediction="因为有东拉河和哭咽河。",
        gold="因为有东拉河和哭咽河。",
        aliases=["东拉河和哭咽河", "因东拉河和哭咽河得名"],
        evidence_text="村边有东拉河和哭咽河，因此这个村子取名叫双水村。",
        gold_chunks=["corpus_chunkids_000002"],
        retrieved_chunk_ids=["corpus_chunkids_000002"],
    )

    assert result["correctness"] == 1.0
    assert result["faithfulness"] > 0.5
    assert result["context_precision"] == 1.0
