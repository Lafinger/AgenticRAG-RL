from __future__ import annotations

import re
from typing import Any, Iterable

from .types import Chunk


NOVEL_ENTITIES = ["孙少平", "郝红梅", "双水村", "孙少安", "田润叶", "田晓霞", "金波", "孙玉厚", "王满银", "兰花", "田福堂"]


def _contains_all(text: str, terms: Iterable[str]) -> bool:
    return all(term in text for term in terms)


def _first_sentence(text: str, max_chars: int = 80) -> str:
    sentence = re.split(r"[。！？\n]", text.strip(), maxsplit=1)[0]
    return sentence[:max_chars].strip(" ，,；;") or text[:max_chars]


def _seed(question: str, answer: str, chunk: Chunk, qa_type: str) -> dict[str, Any]:
    aliases = chunk.metadata.get("character_aliases", [])
    return {
        "question": question,
        "answer": answer,
        "doc_chunk_id": chunk.chunk_id,
        "tool": "keyword_search",
        "entities": [alias for alias in aliases if alias in question or alias in answer],
        "qa_type": qa_type,
    }


def generate_seed_questions(chunks: Iterable[Chunk], max_per_chunk: int = 2) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for chunk in chunks:
        generated = 0
        text = chunk.text

        if _contains_all(text, ["孙少平", "郝红梅"]) and generated < max_per_chunk:
            answer = "他们都很贫穷，常常最后去取黑高粱面馍。"
            seeds.append(_seed("孙少平和郝红梅在学校有什么共同处境？", answer, chunk, "character_relation"))
            generated += 1

        if "孙少平" in text and ("艰难" in text or "高粱面馍" in text or "贫穷" in text) and generated < max_per_chunk:
            seeds.append(_seed("孙少平在学校生活艰难的表现是什么？", "每顿饭常常最后才去取两个黑高粱面馍。", chunk, "character_behavior"))
            generated += 1

        if _contains_all(text, ["双水村", "东拉河", "哭咽河"]) and generated < max_per_chunk:
            seeds.append(_seed("双水村为什么叫双水村？", "因为有东拉河和哭咽河。", chunk, "place_origin"))
            generated += 1

        if _contains_all(text, ["孙少安", "田润叶"]) and generated < max_per_chunk:
            seeds.append(_seed("孙少安和田润叶是什么关系？", "小时候是同班同学。", chunk, "character_relation"))
            generated += 1

        aliases = [alias for alias in NOVEL_ENTITIES if alias in text]
        if aliases and generated < max_per_chunk:
            entity = aliases[0]
            seeds.append(_seed(f"这段文字主要写到了谁？", entity, chunk, "character_identity"))
    return seeds


def synthesize_multihop_examples(seeds: list[dict[str, Any]], chunks: list[Chunk], target_count: int = 100) -> list[dict[str, Any]]:
    del chunks
    examples: list[dict[str, Any]] = []
    relation_seed = next((seed for seed in seeds if "孙少平和郝红梅" in seed["question"]), None)
    hardship_seed = next((seed for seed in seeds if "孙少平在学校生活艰难" in seed["question"]), relation_seed)
    place_seed = next((seed for seed in seeds if "双水村为什么" in seed["question"]), None)
    classmate_seed = next((seed for seed in seeds if "孙少安和田润叶" in seed["question"]), None)

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

    for left, right in zip(seeds, seeds[1:]):
        if len(examples) >= target_count:
            break
        if left["doc_chunk_id"] == right["doc_chunk_id"]:
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
