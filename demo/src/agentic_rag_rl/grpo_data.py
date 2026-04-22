from __future__ import annotations

from typing import Any, Iterable

from .protocols import DEFAULT_AGENT_NAME, SYSTEM_PROMPT_ZH
from .types import MultiHopExample


def build_grpo_rows(examples: Iterable[MultiHopExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_ZH},
                    {"role": "user", "content": example.final_question},
                ],
                "agent_name": DEFAULT_AGENT_NAME,
                "reward_model": {
                    "ground_truth": {
                        "target": example.final_answer,
                        "question": example.final_question,
                        "answer_aliases": example.answer_aliases or [example.final_answer],
                        "gold_chunks": [hop.doc_chunk_id for hop in example.hops],
                        "hop_count": example.hop_count,
                    }
                },
            }
        )
    return rows
