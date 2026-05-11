from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .evaluation import hop_precision_recall_f1, hop_recall
from .protocols import extract_answer_tag


@dataclass(slots=True)
class RewardInputs:
    prediction: str
    target: str
    answer_aliases: list[str]
    gold_chunks: list[str]
    retrieved_chunk_ids: list[str]
    hop_count: int
    tool_calls: int
    judge_correctness: float
    judge_faithfulness: float
    grounded_answer: float


@dataclass(slots=True)
class RewardResult:
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _format_score(prediction: str) -> float:
    score = 0.0
    if "<answer>" in prediction and "</answer>" in prediction:
        score += 0.10
    if _has_valid_json_tool_call(prediction):
        score += 0.04
    return min(score, 0.10)


def _has_valid_json_tool_call(prediction: str) -> bool:
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", prediction, flags=re.DOTALL):
        try:
            payload = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("name") and isinstance(payload.get("arguments"), dict):
            return True
    return False


def _search_effort(tool_calls: int, hop_count: int) -> float:
    if hop_count <= 0:
        return 1.0
    return 1.0 if tool_calls >= hop_count else max(tool_calls / hop_count, 0.0)


def _insufficient_search_penalty(tool_calls: int, hop_count: int) -> float:
    return -0.05 if tool_calls < hop_count else 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_response(inputs: RewardInputs, reward_version: str = "v5a") -> RewardResult:
    _ = extract_answer_tag(inputs.prediction)
    format_score = _format_score(inputs.prediction)
    hop_recall_score = hop_recall(inputs.retrieved_chunk_ids, inputs.gold_chunks)
    hop_f1 = hop_precision_recall_f1(inputs.retrieved_chunk_ids, inputs.gold_chunks)
    search_effort = _search_effort(inputs.tool_calls, inputs.hop_count)
    insufficient_penalty = _insufficient_search_penalty(inputs.tool_calls, inputs.hop_count)

    breakdown = {
        "judge_correctness": _bounded(inputs.judge_correctness),
        "judge_faithfulness": _bounded(inputs.judge_faithfulness),
        "grounded_answer": _bounded(inputs.grounded_answer),
        "format": format_score,
        "hop_recall": hop_recall_score,
        "hop_precision_recall": hop_f1,
        "search_effort": search_effort,
        "insufficient_search_penalty": insufficient_penalty,
    }

    if reward_version == "v5a":
        score = (
            breakdown["judge_correctness"] * 0.40
            + breakdown["hop_recall"] * 0.25
            + breakdown["judge_faithfulness"] * 0.15
            + breakdown["format"] * 1.0
            + breakdown["search_effort"] * 0.10
        )
    elif reward_version == "v6a":
        score = (
            breakdown["judge_faithfulness"] * 0.30
            + breakdown["judge_correctness"] * 0.25
            + breakdown["hop_precision_recall"] * 0.20
            + breakdown["grounded_answer"] * 0.15
            + breakdown["format"] * 1.0
            + breakdown["insufficient_search_penalty"]
        )
    elif reward_version == "v9a":
        score = (
            breakdown["hop_precision_recall"] * 0.30
            + breakdown["judge_faithfulness"] * 0.25
            + breakdown["judge_correctness"] * 0.25
            + breakdown["grounded_answer"] * 0.10
            + breakdown["format"] * 1.0
        )
    else:
        raise ValueError(f"Unsupported reward_version: {reward_version}")

    return RewardResult(score=round(score, 4), breakdown=breakdown)
