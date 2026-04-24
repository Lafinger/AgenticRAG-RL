from __future__ import annotations

from .evaluation import exact_match, hop_precision, token_f1


def judge_agentic_answer(
    question: str,
    prediction: str,
    gold: str,
    aliases: list[str],
    evidence_text: str,
    gold_chunks: list[str],
    retrieved_chunk_ids: list[str],
) -> dict[str, float]:
    del question
    correctness = max([exact_match(prediction, gold), token_f1(prediction, gold), *[token_f1(prediction, alias) for alias in aliases]])
    evidence_f1 = token_f1(prediction, evidence_text)
    faithfulness = 1.0 if prediction and prediction in evidence_text else max(evidence_f1, 0.0)
    context_precision = hop_precision(retrieved_chunk_ids, gold_chunks)
    return {
        "correctness": round(correctness, 4),
        "faithfulness": round(faithfulness, 4),
        "context_precision": round(context_precision, 4),
    }
