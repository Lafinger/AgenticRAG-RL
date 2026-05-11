from __future__ import annotations

from typing import Any, Iterable

from .protocols import SYSTEM_PROMPT_ZH
from .types import MultiHopExample


GRPO_DATA_SOURCE = "novel_agentic_rag"
GRPO_ABILITY = "multi_hop_qa"
GRPO_TOOL_NAMES = ("keyword_search", "dense_search", "hybrid_search")


def _answer_aliases(example: MultiHopExample) -> list[str]:
    aliases = list(example.answer_aliases or [])
    if example.final_answer and example.final_answer not in aliases:
        aliases.insert(0, example.final_answer)
    return aliases or [example.final_answer]


def _ground_truth(example: MultiHopExample) -> dict[str, Any]:
    return {
        "target": example.final_answer,
        "answer": example.final_answer,
        "question": example.final_question,
        "answer_aliases": _answer_aliases(example),
        "gold_chunks": [hop.doc_chunk_id for hop in example.hops if hop.doc_chunk_id],
        "hop_count": example.hop_count,
    }


def _tools_kwargs(tool_names: Iterable[str], *, example: MultiHopExample, data_source: str) -> dict[str, Any]:
    return {
        tool_name: {
            "create_kwargs": {
                "ground_truth": example.final_answer,
                "question": example.final_question,
                "data_source": data_source,
            }
        }
        for tool_name in tool_names
    }


def build_grpo_rows(
    examples: Iterable[MultiHopExample],
    *,
    split: str = "train",
    data_source: str = GRPO_DATA_SOURCE,
    tool_names: Iterable[str] = GRPO_TOOL_NAMES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tool_name_list = list(tool_names)
    for index, example in enumerate(examples):
        ground_truth = _ground_truth(example)
        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_ZH},
                    {"role": "user", "content": example.final_question},
                ],
                "ability": GRPO_ABILITY,
                "reward_model": {
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index": index,
                    "need_tools_kwargs": True,
                    "question": example.final_question,
                    "split": split,
                    "subset": example.subset,
                    "hop_count": example.hop_count,
                    "tools_kwargs": _tools_kwargs(tool_name_list, example=example, data_source=data_source),
                },
                "metadata": {
                    "subset": example.subset,
                    "hop_count": example.hop_count,
                    "qa_type": example.qa_type,
                    "tool_names": tool_name_list,
                },
            }
        )
    return rows
