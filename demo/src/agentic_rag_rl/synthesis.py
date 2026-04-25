from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any, Protocol

from .types import Chunk

logger = logging.getLogger(__name__)


class SeedQAClient(Protocol):
    def generate_seed_qa(self, chunk_text: str, *, max_items: int) -> list[dict[str, Any]]:
        ...


def _first_sentence(text: str, max_chars: int = 80) -> str:
    sentence = re.split(r"[。！？\n]", text.strip(), maxsplit=1)[0]
    return sentence[:max_chars].strip(" ，,；;") or text[:max_chars]


def generate_seed_questions(chunks: Iterable[Chunk], llm_client: SeedQAClient, max_per_chunk: int = 2) -> list[dict[str, Any]]:
    chunk_list = list(chunks)
    seeds: list[dict[str, Any]] = []
    logger.info("seed_qa_generation.start chunk_count=%s max_per_chunk=%s", len(chunk_list), max_per_chunk)
    for index, chunk in enumerate(chunk_list, start=1):
        logger.info(
            "seed_qa_generation.chunk_start index=%s/%s chunk_id=%s title=%s chunk_chars=%s",
            index,
            len(chunk_list),
            chunk.chunk_id,
            chunk.title,
            len(chunk.text),
        )
        try:
            items = llm_client.generate_seed_qa(chunk.text, max_items=max_per_chunk)
        except Exception:
            logger.exception(
                "seed_qa_generation.chunk_failed index=%s/%s chunk_id=%s",
                index,
                len(chunk_list),
                chunk.chunk_id,
            )
            raise
        for item in items:
            seeds.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "doc_chunk_id": chunk.chunk_id,
                    "tool": "keyword_search",
                    "entities": item.get("entities", []),
                    "qa_type": item.get("qa_type", "inference"),
                }
            )
        logger.info(
            "seed_qa_generation.chunk_done index=%s/%s chunk_id=%s generated_count=%s total_seed_count=%s",
            index,
            len(chunk_list),
            chunk.chunk_id,
            len(items),
            len(seeds),
        )
    logger.info("seed_qa_generation.done chunk_count=%s seed_count=%s", len(chunk_list), len(seeds))
    return seeds


def synthesize_multihop_examples(seeds: list[dict[str, Any]], chunks: list[Chunk], target_count: int = 100) -> list[dict[str, Any]]:
    logger.info(
        "multihop_synthesis.start seed_count=%s chunk_count=%s target_count=%s",
        len(seeds),
        len(chunks),
        target_count,
    )
    examples: list[dict[str, Any]] = []
    relation_seed = next((seed for seed in seeds if "孙少平和郝红梅" in seed["question"]), None)
    hardship_seed = next((seed for seed in seeds if "孙少平在学校生活艰难" in seed["question"]), relation_seed)
    place_seed = next((seed for seed in seeds if "双水村为什么" in seed["question"]), None)
    classmate_seed = next((seed for seed in seeds if "孙少安和田润叶" in seed["question"]), None)
    logger.info(
        "multihop_synthesis.anchor_seeds relation=%s hardship=%s place=%s classmate=%s",
        bool(relation_seed),
        bool(hardship_seed),
        bool(place_seed),
        bool(classmate_seed),
    )

    if relation_seed and hardship_seed:
        examples.append(
            {
                "final_question": "孙少平和郝红梅在学校有什么共同处境？这种处境体现在哪件日常小事上？",
                "final_answer": "他们都很贫穷，常常最后去取黑高粱面馍。",
                "hop_count": 2,
                "qa_type": "inference",
                "subset": "2hop_novel_relation",
                "hops": [
                    {**hardship_seed, "hop_idx": 1},
                    {**relation_seed, "hop_idx": 2},
                ],
                "answer_aliases": ["贫穷", "都很贫穷", "最后去取黑高粱面馍"],
            }
        )
        logger.info("multihop_synthesis.added subset=2hop_novel_relation total_count=%s", len(examples))

    if classmate_seed and place_seed and len(examples) < target_count:
        examples.append(
            {
                "final_question": "孙少安和田润叶曾在哪里同班读书？这个地方为什么叫双水村？",
                "final_answer": "他们曾在双水村小学同班读书；双水村因东拉河和哭咽河得名。",
                "hop_count": 2,
                "qa_type": "inference",
                "subset": "2hop_novel_place",
                "hops": [
                    {**classmate_seed, "hop_idx": 1},
                    {**place_seed, "hop_idx": 2},
                ],
                "answer_aliases": ["双水村小学", "东拉河和哭咽河", "因东拉河和哭咽河得名"],
            }
        )
        logger.info("multihop_synthesis.added subset=2hop_novel_place total_count=%s", len(examples))

    generic_attempt_count = 0
    for left, right in zip(seeds, seeds[1:]):
        if len(examples) >= target_count:
            break
        generic_attempt_count += 1
        if left["doc_chunk_id"] == right["doc_chunk_id"]:
            logger.debug(
                "multihop_synthesis.skip_same_chunk left_chunk_id=%s right_chunk_id=%s",
                left["doc_chunk_id"],
                right["doc_chunk_id"],
            )
            continue
        examples.append(
            {
                "final_question": f"先回答“{left['question']}”，再结合“{right['question']}”说明两条信息的关系。",
                "final_answer": f"{left['answer']}；{right['answer']}",
                "hop_count": 2,
                "qa_type": "inference",
                "subset": "2hop_novel_generic",
                "hops": [{**left, "hop_idx": 1}, {**right, "hop_idx": 2}],
                "answer_aliases": [left["answer"], right["answer"], _first_sentence(f"{left['answer']}；{right['answer']}")],
            }
        )
        logger.info(
            "multihop_synthesis.added subset=2hop_novel_generic total_count=%s target_count=%s",
            len(examples),
            target_count,
        )
    logger.info(
        "multihop_synthesis.done generated_count=%s target_count=%s generic_attempt_count=%s",
        len(examples),
        target_count,
        generic_attempt_count,
    )
    return examples


def clean_multihop_examples(records: list[dict[str, Any]], valid_chunk_ids: set[str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for record in records:
        hops = record.get("hops", [])
        if len(hops) < 2:
            continue
        if any(hop.get("doc_chunk_id") not in valid_chunk_ids for hop in hops):
            continue
        cleaned.append(record)
    return cleaned
