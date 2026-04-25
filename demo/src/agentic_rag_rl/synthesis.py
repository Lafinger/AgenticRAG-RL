from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from .llm_client import ChatMessage, LLMClient
from .retrieval import HybridRetriever, RetrievalResult
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


def synthesize_multihop_examples(
    seeds: list[dict[str, Any]],
    chunks: list[Chunk],
    target_count: int = 100,
    merge_llm_client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    logger.info(
        "multihop_synthesis.start seed_count=%s chunk_count=%s target_count=%s",
        len(seeds),
        len(chunks),
        target_count,
    )
    examples: list[dict[str, Any]] = []
    retriever = HybridRetriever(chunks)
    seeds_by_chunk = _group_seeds_by_chunk(seeds)
    seen_questions: set[str] = set()
    extension_attempt_count = 0

    for seed in seeds:
        if len(examples) >= target_count:
            break
        chain = [_seed_to_hop(seed, hop_idx=1, qa_type="initial_qa")]
        for hop_idx in range(2, 4):
            if len(examples) >= target_count:
                break
            extension_attempt_count += 1
            next_seed = _select_next_seed(chain, seeds_by_chunk, retriever)
            if next_seed is None:
                logger.debug(
                    "multihop_synthesis.extension_miss seed_chunk_id=%s hop_idx=%s",
                    seed["doc_chunk_id"],
                    hop_idx,
                )
                break
            chain.append(_seed_to_hop(next_seed, hop_idx=hop_idx, qa_type="inference"))
            example = _build_stepwise_example(chain, merge_llm_client=merge_llm_client)
            if example["final_question"] in seen_questions:
                continue
            seen_questions.add(example["final_question"])
            examples.append(example)
            logger.info(
                "multihop_synthesis.added subset=%s total_count=%s target_count=%s",
                example["subset"],
                len(examples),
                target_count,
            )

    logger.info(
        "multihop_synthesis.done generated_count=%s target_count=%s extension_attempt_count=%s",
        len(examples),
        target_count,
        extension_attempt_count,
    )
    return examples


def _group_seeds_by_chunk(seeds: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        grouped.setdefault(seed["doc_chunk_id"], []).append(seed)
    return grouped


def _seed_to_hop(seed: dict[str, Any], *, hop_idx: int, qa_type: str) -> dict[str, Any]:
    return {
        "hop_idx": hop_idx,
        "question": seed["question"],
        "answer": seed["answer"],
        "doc_chunk_id": seed["doc_chunk_id"],
        "qa_type": qa_type,
        "search_tools": seed.get("search_tools", [seed.get("tool", "hybrid_search")]),
    }


def _select_next_seed(
    chain: list[dict[str, Any]],
    seeds_by_chunk: dict[str, list[dict[str, Any]]],
    retriever: HybridRetriever,
) -> dict[str, Any] | None:
    used_chunk_ids = {hop["doc_chunk_id"] for hop in chain}
    last_hop = chain[-1]
    for result in _retrieve_extension_candidates(last_hop, retriever):
        if result.chunk_id in used_chunk_ids:
            continue
        for seed in seeds_by_chunk.get(result.chunk_id, []):
            return {**seed, "search_tools": _result_search_tools(result)}
    return None


def _retrieve_extension_candidates(hop: dict[str, Any], retriever: HybridRetriever) -> list[RetrievalResult]:
    primary_results = retriever.hybrid_search(hop["question"], top_k=8, candidate_k=12)
    secondary_results: list[RetrievalResult] = []
    if not _is_low_signal_query(hop["answer"]):
        secondary_results = retriever.hybrid_search(hop["answer"], top_k=4, candidate_k=8)
    return _merge_retrieval_results(primary_results, secondary_results)


def _is_low_signal_query(text: str) -> bool:
    stripped = str(text).strip()
    if len(stripped) < 2:
        return True
    return bool(re.fullmatch(r"[\d\s.,，。:：;；%％元年月日+-]+", stripped))


def _merge_retrieval_results(*results_lists: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for results in results_lists:
        for result in results:
            if result.chunk_id not in merged:
                merged[result.chunk_id] = result
    return list(merged.values())


def _result_search_tools(result: RetrievalResult) -> list[str]:
    tools: list[str] = []
    if "keyword" in result.source:
        tools.append("keyword_search")
    if "dense" in result.source:
        tools.append("dense_search")
    return tools or ["hybrid_search"]


def _build_stepwise_example(chain: list[dict[str, Any]], merge_llm_client: LLMClient | None = None) -> dict[str, Any]:
    hop_count = len(chain)
    merged = _merge_stepwise_question_with_llm(chain, merge_llm_client) if merge_llm_client else {}
    final_answer = merged.get("final_answer") or chain[-1]["answer"]
    final_question = merged.get("final_question") or _build_stepwise_question(chain)
    qa_type = merged.get("qa_type") or "inference"
    answer_aliases = merged.get("answer_aliases") or [final_answer, _first_sentence(final_answer)]
    return {
        "final_question": final_question,
        "final_answer": final_answer,
        "hop_count": hop_count,
        "qa_type": qa_type,
        "subset": f"{hop_count}hop_novel_stepwise",
        "hops": [dict(hop) for hop in chain],
        "answer_aliases": answer_aliases,
    }


def _merge_stepwise_question_with_llm(chain: list[dict[str, Any]], llm_client: LLMClient) -> dict[str, Any]:
    logger.info("multihop_merge.start hop_count=%s", len(chain))
    content = llm_client.chat(_build_multihop_merge_messages(chain), temperature=0.2)
    try:
        payload = _extract_json_object(content)
    except Exception:
        logger.exception("multihop_merge.invalid_json response_chars=%s", len(content))
        return {}
    final_question = str(payload.get("final_question", "")).strip()
    final_answer = str(payload.get("final_answer", "")).strip()
    if not final_question or not final_answer:
        logger.warning("multihop_merge.missing_required_fields keys=%s", sorted(payload))
        return {}
    answer_aliases = payload.get("answer_aliases", [final_answer])
    if isinstance(answer_aliases, str):
        answer_aliases = [answer_aliases]
    if not isinstance(answer_aliases, list):
        answer_aliases = [final_answer]
    logger.info("multihop_merge.done hop_count=%s final_question_chars=%s", len(chain), len(final_question))
    return {
        "final_question": final_question,
        "final_answer": final_answer,
        "qa_type": str(payload.get("qa_type", "inference")).strip() or "inference",
        "answer_aliases": [str(alias).strip() for alias in answer_aliases if str(alias).strip()],
    }


def _build_multihop_merge_messages(chain: list[dict[str, Any]]) -> list[ChatMessage]:
    hop_lines = "\n".join(
        f"hop{hop['hop_idx']}:\n"
        f"- question: {hop['question']}\n"
        f"- answer: {hop['answer']}\n"
        f"- doc_chunk_id: {hop['doc_chunk_id']}"
        for hop in chain
    )
    system_prompt = (
        "你是中文小说多跳阅读问答合成专家。"
        "请把给定的逐跳 QA 链合并成一个自然、明确、必须按顺序检索才能回答的多跳问题。"
        "只能输出 JSON 对象，不要输出解释。"
    )
    user_prompt = f"""请基于以下逐跳 QA 链生成多跳 QA。

要求：
1. final_question 必须自然，不能直接暴露“第1步/第2步”。
2. final_question 必须需要全部 hop 才能回答。
3. final_answer 默认使用最后一跳 answer，除非链路逻辑要求更准确的短答案。
4. qa_type 取 inference。
5. answer_aliases 给出 1-3 个可接受短答案。
6. 输出 JSON 对象，字段为 final_question、final_answer、qa_type、answer_aliases。

逐跳 QA：
{hop_lines}
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM merge response must be a JSON object.")
    return payload


def _build_stepwise_question(chain: list[dict[str, Any]]) -> str:
    steps = "；".join(f"第{hop['hop_idx']}步回答“{hop['question']}”" for hop in chain)
    return f"{steps}。请沿着这些线索逐步检索，最终答案是什么？"


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
