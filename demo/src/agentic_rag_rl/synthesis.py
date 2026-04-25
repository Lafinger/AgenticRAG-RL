from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from .llm_client import ChatMessage, LLMClient
from .types import Chunk

logger = logging.getLogger(__name__)


def _first_sentence(text: str, max_chars: int = 80) -> str:
    sentence = re.split(r"[。！？\n]", text.strip(), maxsplit=1)[0]
    return sentence[:max_chars].strip(" ，,；;") or text[:max_chars]


def generate_seed_questions(chunks: Iterable[Chunk], llm_client: LLMClient, max_per_chunk: int = 2) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for seed_batch in iter_seed_question_batches(chunks, llm_client, max_per_chunk=max_per_chunk):
        seeds.extend(seed_batch)
    return seeds


def iter_seed_question_batches(
    chunks: Iterable[Chunk],
    llm_client: LLMClient,
    max_per_chunk: int = 2,
) -> Iterable[list[dict[str, Any]]]:
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
            items = generate_seed_qa(llm_client, chunk.text, max_items=max_per_chunk)
        except Exception:
            logger.exception(
                "seed_qa_generation.chunk_failed index=%s/%s chunk_id=%s",
                index,
                len(chunk_list),
                chunk.chunk_id,
            )
            raise
        batch: list[dict[str, Any]] = []
        for item in items:
            seed = {
                "question": item["question"],
                "answer": item["answer"],
                "doc_chunk_id": chunk.chunk_id,
                "tool": "keyword_search",
                "entities": item.get("entities", []),
                "qa_type": item.get("qa_type", "inference"),
            }
            batch.append(seed)
            seeds.append(seed)
        logger.info(
            "seed_qa_generation.chunk_done index=%s/%s chunk_id=%s generated_count=%s total_seed_count=%s",
            index,
            len(chunk_list),
            chunk.chunk_id,
            len(items),
            len(seeds),
        )
        yield batch
    logger.info("seed_qa_generation.done chunk_count=%s seed_count=%s", len(chunk_list), len(seeds))


def generate_seed_qa(llm_client: LLMClient, chunk_text: str, *, max_items: int) -> list[dict[str, Any]]:
    logger.info("seed_qa.start max_items=%s chunk_chars=%s", max_items, len(chunk_text))
    content = llm_client.chat(_build_seed_qa_messages(chunk_text, max_items=max_items), temperature=0.2)
    records = _extract_json_array(content)
    normalized_records = [
        _normalize_seed_qa_record(record) for record in records[:max_items] if _is_usable_seed_qa_record(record)
    ]
    logger.info(
        "seed_qa.done raw_count=%s returned_count=%s response_chars=%s",
        len(records),
        len(normalized_records),
        len(content),
    )
    return normalized_records


def _build_seed_qa_messages(chunk_text: str, *, max_items: int) -> list[ChatMessage]:
    system_prompt = (
        "你是中文小说阅读问答数据构造专家。"
        "请只基于给定片段生成可由该片段直接回答的 seed QA。"
        "问题类型应覆盖人物身份、人物关系、地点归属、事件原因、事件结果、人物行为。"
        "只能输出 JSON 数组，不要输出解释。"
    )
    user_prompt = f"""请为下面《平凡的世界》片段生成最多 {max_items} 条 seed QA。

要求：
1. 每条必须能从片段中直接找到证据。
2. answer 要短，避免长段摘抄。
3. qa_type 只能取 character_identity、character_relation、place_origin、event_cause、event_result、character_behavior、inference。
4. entities 写出问题涉及的人物、地点或事件关键词。
5. 输出 JSON 数组，字段为 question、answer、qa_type、entities。

片段：
{chunk_text}
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    payload = json.loads(cleaned)
    if not isinstance(payload, list):
        raise ValueError("LLM seed QA response must be a JSON array.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Every seed QA item must be a JSON object.")
    return payload


def _is_usable_seed_qa_record(record: dict[str, Any]) -> bool:
    return bool(str(record.get("question", "")).strip()) and bool(str(record.get("answer", "")).strip())


def _normalize_seed_qa_record(record: dict[str, Any]) -> dict[str, Any]:
    entities = record.get("entities", [])
    if isinstance(entities, str):
        entities = [entities]
    if not isinstance(entities, list):
        entities = []
    return {
        "question": str(record.get("question", "")).strip(),
        "answer": str(record.get("answer", "")).strip(),
        "qa_type": str(record.get("qa_type", "inference")).strip() or "inference",
        "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
    }


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
