from __future__ import annotations

import os
import json
import re
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.judge import judge_agentic_answer
from agentic_rag_rl.protocols import extract_answer_tag
from agentic_rag_rl.rewards import RewardInputs, score_response


CHUNK_ID_PATTERN = re.compile(r"\[([A-Za-z0-9_:-]+)\]")
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _normalize_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        return {"target": ground_truth, "answer_aliases": [ground_truth], "gold_chunks": [], "hop_count": 1}
    return {"target": "", "answer_aliases": [], "gold_chunks": [], "hop_count": 1}


def _extract_retrieved_chunk_ids(solution_str: str, extra_info: dict[str, Any]) -> list[str]:
    if extra_info.get("retrieved_chunk_ids"):
        return list(extra_info["retrieved_chunk_ids"])
    evidence_text = str(extra_info.get("evidence_text", ""))
    merged = f"{solution_str}\n{evidence_text}"
    return list(dict.fromkeys(CHUNK_ID_PATTERN.findall(merged)))


def _extract_evidence_text(solution_str: str, extra_info: dict[str, Any]) -> str:
    pieces: list[str] = []
    if extra_info.get("evidence_text"):
        pieces.append(str(extra_info["evidence_text"]))
    pieces.extend(match.strip() for match in TOOL_RESPONSE_PATTERN.findall(solution_str) if match.strip())
    chunk_lines = [line.strip() for line in solution_str.splitlines() if CHUNK_ID_PATTERN.search(line)]
    pieces.extend(chunk_lines)
    return "\n".join(dict.fromkeys(piece for piece in pieces if piece))


def _valid_tool_call_count(solution_str: str) -> int:
    count = 0
    for match in TOOL_CALL_PATTERN.findall(solution_str):
        try:
            payload = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("name") and isinstance(payload.get("arguments"), dict):
            count += 1
    return count


def _count_tool_calls(solution_str: str, extra_info: dict[str, Any]) -> int:
    if "tool_calls" in extra_info:
        return int(extra_info["tool_calls"])
    return _valid_tool_call_count(solution_str)


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    del data_source, kwargs
    extra_info = extra_info or {}
    ground_truth_dict = _normalize_ground_truth(ground_truth)
    target = str(ground_truth_dict.get("target", ""))
    aliases = list(ground_truth_dict.get("answer_aliases", [target]))
    gold_chunks = list(ground_truth_dict.get("gold_chunks", []))
    hop_count = int(ground_truth_dict.get("hop_count", 1) or 1)
    prediction = extract_answer_tag(solution_str)
    retrieved_chunk_ids = _extract_retrieved_chunk_ids(solution_str, extra_info)
    tool_calls = _count_tool_calls(solution_str, extra_info)
    evidence_text = _extract_evidence_text(solution_str, extra_info)

    if prediction:
        judge_scores = judge_agentic_answer(
            question=str(ground_truth_dict.get("question", "")),
            prediction=prediction,
            gold=target,
            aliases=aliases,
            evidence_text=evidence_text,
            gold_chunks=gold_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
    else:
        judge_scores = {"correctness": 0.0, "faithfulness": 0.0}
    grounded_answer = 1.0 if prediction and prediction in evidence_text else judge_scores["faithfulness"]
    reward_version = os.environ.get("AGENTIC_RAG_REWARD_VERSION", "v9a")

    result = score_response(
        RewardInputs(
            prediction=solution_str,
            target=target,
            answer_aliases=aliases,
            gold_chunks=gold_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            hop_count=hop_count,
            tool_calls=tool_calls,
            judge_correctness=judge_scores["correctness"],
            judge_faithfulness=judge_scores["faithfulness"],
            grounded_answer=grounded_answer,
        ),
        reward_version=reward_version,
    )
    return float(result.score)


def compute_score_breakdown(solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_info = extra_info or {}
    score = compute_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    return {"score": score, "reward_version": os.environ.get("AGENTIC_RAG_REWARD_VERSION", "v5a")}
