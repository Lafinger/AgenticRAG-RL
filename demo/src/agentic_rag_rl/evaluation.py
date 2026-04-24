from __future__ import annotations

from typing import Sequence

from .retrieval import tokenize


def normalize_answer(text: str) -> str:
    return "".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hop_recall(predicted_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    predicted = set(predicted_chunk_ids)
    gold = set(gold_chunk_ids)
    return len(predicted & gold) / len(gold)


def hop_precision(predicted_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str]) -> float:
    if not predicted_chunk_ids:
        return 0.0
    predicted = set(predicted_chunk_ids)
    gold = set(gold_chunk_ids)
    return len(predicted & gold) / len(predicted)


def hop_precision_recall_f1(predicted_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str]) -> float:
    precision = hop_precision(predicted_chunk_ids, gold_chunk_ids)
    recall = hop_recall(predicted_chunk_ids, gold_chunk_ids)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def premature_collapse(predicted_steps: int, gold_hops: int) -> float:
    return 1.0 if predicted_steps < gold_hops else 0.0


def over_extension(predicted_steps: int, gold_hops: int) -> float:
    if gold_hops <= 0:
        return 0.0
    return max(predicted_steps - gold_hops, 0) / gold_hops


def step_alignment(predicted_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    matches = 0
    for predicted, gold in zip(predicted_chunk_ids, gold_chunk_ids):
        if predicted == gold:
            matches += 1
    return matches / len(gold_chunk_ids)
