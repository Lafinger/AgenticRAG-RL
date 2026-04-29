from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from .llm_client import ChatMessage, LLMClient
from .retrieval import HybridRetriever, RetrievalResult
from .types import Chunk

logger = logging.getLogger(__name__)

SEED_QA_TYPES = {"character", "place", "object", "relation", "action_result"}
SEED_ANSWER_MAX_CHARS = 20
SEED_QUESTION_MIN_CHARS = 8
LEGACY_SEED_QA_TYPE_MAP = {
    "character_identity": "character",
    "place_origin": "place",
    "object_reference": "object",
    "character_relation": "relation",
    "event_result": "action_result",
    "event_cause": "action_result",
    "character_behavior": "action_result",
    "inference": "action_result",
}
SEED_DOC_REFERENCE_RE = re.compile(r"(根据|据|文档|片段|文本|材料|上文|原文)")
SEED_AMBIGUOUS_PRONOUN_RE = re.compile(
    r"(?:^|[，。？?；;、\s])(?:他|她|这个人|那个人|此人|该人|这位|那位|这个青年|那个青年)"
    r"(?:[，。？?；;、\s]|的|在|被|是|做|去|给|把|和)"
)
SEED_ABSTRACT_RE = re.compile(r"(感悟|意义|重要性|体现|说明|反映|象征|态度|心情|感受|评价)")
SEED_HARD_COMPOUND_RE = re.compile(r"(；|;|、|\n|以及|并且|同时|[，,]|\d+[.．、])")
SEED_SOFT_COMPOUND_RE = re.compile(r"(和|或)")


def _first_sentence(text: str, max_chars: int = 80) -> str:
    sentence = re.split(r"[。！？\n]", text.strip(), maxsplit=1)[0]
    return sentence[:max_chars].strip(" ，,；;") or text[:max_chars]


def generate_seed_questions(
    chunks: Iterable[Chunk],
    llm_client: LLMClient,
    max_per_chunk: int = 2,
    max_concurrency: int = 1,
) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for seed_batch in iter_seed_question_batches(
        chunks,
        llm_client,
        max_per_chunk=max_per_chunk,
        max_concurrency=max_concurrency,
    ):
        seeds.extend(seed_batch)
    return seeds


def iter_seed_question_batches(
    chunks: Iterable[Chunk],
    llm_client: LLMClient,
    max_per_chunk: int = 2,
    max_attempts: int = 2,
    continue_on_error: bool = False,
    on_chunk_failed: Callable[[Chunk, Exception], None] | None = None,
    max_concurrency: int = 1,
) -> Iterable[list[dict[str, Any]]]:
    chunk_list = list(chunks)
    seeds: list[dict[str, Any]] = []
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1.")
    logger.info(
        "seed_qa_generation.start chunk_count=%s max_per_chunk=%s max_concurrency=%s",
        len(chunk_list),
        max_per_chunk,
        max_concurrency,
    )
    if max_concurrency == 1:
        yield from _iter_seed_question_batches_sequential(
            chunk_list,
            llm_client,
            max_per_chunk=max_per_chunk,
            max_attempts=max_attempts,
            continue_on_error=continue_on_error,
            on_chunk_failed=on_chunk_failed,
            seeds=seeds,
        )
        logger.info("seed_qa_generation.done chunk_count=%s seed_count=%s", len(chunk_list), len(seeds))
        return

    worker_count = min(max_concurrency, len(chunk_list))
    if worker_count == 0:
        logger.info("seed_qa_generation.done chunk_count=0 seed_count=0")
        return

    next_chunk_index = 0
    futures: dict[Future[list[dict[str, Any]]], tuple[int, Chunk]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal next_chunk_index
        if next_chunk_index >= len(chunk_list):
            return
        chunk = chunk_list[next_chunk_index]
        index = next_chunk_index + 1
        next_chunk_index += 1
        logger.info(
            "seed_qa_generation.chunk_start index=%s/%s chunk_id=%s title=%s chunk_chars=%s",
            index,
            len(chunk_list),
            chunk.chunk_id,
            chunk.title,
            len(chunk.text),
        )
        future = executor.submit(
            generate_seed_qa,
            llm_client,
            chunk.text,
            max_items=max_per_chunk,
            max_attempts=max_attempts,
        )
        futures[future] = (index, chunk)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _ in range(worker_count):
            submit_next(executor)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index, chunk = futures.pop(future)
                try:
                    items = future.result()
                except Exception as exc:
                    logger.exception(
                        "seed_qa_generation.chunk_failed index=%s/%s chunk_id=%s",
                        index,
                        len(chunk_list),
                        chunk.chunk_id,
                    )
                    if on_chunk_failed:
                        on_chunk_failed(chunk, exc)
                    if continue_on_error:
                        submit_next(executor)
                        continue
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                batch = _build_seed_batch(chunk, items)
                seeds.extend(batch)
                logger.info(
                    "seed_qa_generation.chunk_done index=%s/%s chunk_id=%s generated_count=%s total_seed_count=%s",
                    index,
                    len(chunk_list),
                    chunk.chunk_id,
                    len(items),
                    len(seeds),
                )
                yield batch
                submit_next(executor)
    logger.info("seed_qa_generation.done chunk_count=%s seed_count=%s", len(chunk_list), len(seeds))


def _iter_seed_question_batches_sequential(
    chunk_list: list[Chunk],
    llm_client: LLMClient,
    *,
    max_per_chunk: int,
    max_attempts: int,
    continue_on_error: bool,
    on_chunk_failed: Callable[[Chunk, Exception], None] | None,
    seeds: list[dict[str, Any]],
) -> Iterable[list[dict[str, Any]]]:
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
            items = generate_seed_qa(llm_client, chunk.text, max_items=max_per_chunk, max_attempts=max_attempts)
        except Exception as exc:
            logger.exception(
                "seed_qa_generation.chunk_failed index=%s/%s chunk_id=%s",
                index,
                len(chunk_list),
                chunk.chunk_id,
            )
            if on_chunk_failed:
                on_chunk_failed(chunk, exc)
            if continue_on_error:
                continue
            raise
        batch = _build_seed_batch(chunk, items)
        seeds.extend(batch)
        logger.info(
            "seed_qa_generation.chunk_done index=%s/%s chunk_id=%s generated_count=%s total_seed_count=%s",
            index,
            len(chunk_list),
            chunk.chunk_id,
            len(items),
            len(seeds),
        )
        yield batch


def _build_seed_batch(chunk: Chunk, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        seed = {
            "question": item["question"],
            "answer": item["answer"],
            "doc_chunk_id": chunk.chunk_id,
            "tool": "keyword_search",
            "entities": item.get("entities", []),
            "qa_type": item.get("qa_type", "action_result"),
        }
        batch.append(seed)
    return batch


def generate_seed_qa(llm_client: LLMClient, chunk_text: str, *, max_items: int, max_attempts: int = 2) -> list[dict[str, Any]]:
    logger.info("seed_qa.start max_items=%s chunk_chars=%s", max_items, len(chunk_text))
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        content = ""
        try:
            content = llm_client.chat(_build_seed_qa_messages(chunk_text, max_items=max_items), temperature=0.2)
            records = _extract_json_array(content)
            normalized_records, dropped_records = clean_seed_qa_records(records, max_records=max_items)
            logger.info(
                "seed_qa.done attempt=%s raw_count=%s returned_count=%s dropped_count=%s response_chars=%s",
                attempt,
                len(records),
                len(normalized_records),
                len(dropped_records),
                len(content),
            )
            return normalized_records
        except Exception as exc:
            last_error = exc
            logger.warning(
                "seed_qa.attempt_failed attempt=%s/%s error_type=%s error=%s response_preview=%r",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                content[:300],
            )
    if last_error is None:
        raise RuntimeError("Seed QA generation failed without an exception.")
    raise last_error


def _build_seed_qa_messages(chunk_text: str, *, max_items: int) -> list[ChatMessage]:
    system_prompt = (
        "你是中文小说信息抽取和阅读问答数据构造专家。"
        "请只基于给定小说片段提取原子化、可验证、唯一答案的 seed QA。"
        "只能输出 JSON 数组，不要输出解释。"
    )
    user_prompt = f"""请为下面《平凡的世界》片段生成最多 {max_items} 条 seed QA。

# 任务
给定一段小说文本，提取一组原子化、可验证的事实，并将每个事实转化为问答对。

# 问答生成规则
1. 原子性
- 每个 QA 只包含一个不可拆分的事实，不能把多个动作、多个原因、多个关系并列在一个答案里。
- 错误示例："孙少平为什么最后取饭？" -> "因为贫穷、敏感、自尊心强"。
- 正确做法：拆成更单一的问题，例如"孙少平最后取饭体现了怎样的生活处境？"。

2. 可验证性
- answer 必须直接来自片段，且只能属于以下类型之一：
  - 人物名：如 孙少平、郝红梅。
  - 地点名：如 双水村、罐子村。
  - 物品名：如 粮票、黑高粱面馍。
  - 明确关系：如 同班同学、兄妹、父子。
  - 明确行为结果：如 借书、取走两个高粱面馍、被老师没收。
- 拒绝主观判断、抽象感悟、长段解释或多原因概括。

3. 时间 / 阶段明确性
- 如果片段中出现明确时间、年代、季节、上学阶段、事件阶段，question 必须写入该时间或阶段。
- 如果片段没有明确时间或阶段，不强制补时间，不能编造时间。

4. 唯一答案
- question 必须足够具体，使片段中只有一个明确答案。
- 避免"他/她/这个人"这类指代不清的问题；必须写出人物名或明确称谓。

5. 问题类型
- qa_type 只能取以下 5 类之一：
  - character：答案是人物名或人物身份。
  - place：答案是地点名。
  - object：答案是物品名。
  - relation：答案是明确人物关系。
  - action_result：答案是明确行为或明确结果。

6. 输出格式
- 输出 JSON 数组，每个元素包含 question、answer、qa_type、entities。
- answer 要短，不要摘抄长句。
- entities 写出问题涉及的人物、地点、物品或事件关键词。

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
    return {
        "question": str(record.get("question", "")).strip(),
        "answer": _refine_seed_answer(record.get("answer", "")),
        "qa_type": _normalize_seed_qa_type(record.get("qa_type")),
        "entities": _normalize_seed_entities(record.get("entities", [])),
    }


def _refine_seed_answer(value: Any) -> str:
    answer = re.sub(r"\s+", " ", str(value or "")).strip()
    answer = re.sub(r"^(?:答案[:：]?|答[:：]?|回答[:：]?)\s*", "", answer)
    answer = answer.strip(" \t\r\n\"'“”‘’")
    answer = answer.strip("。！？!?；;，,、")
    return answer.strip()


def _normalize_seed_entities(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_entities = [value]
    elif isinstance(value, list):
        raw_entities = value
    else:
        raw_entities = []

    entities: list[str] = []
    seen: set[str] = set()
    for raw_entity in raw_entities:
        for entity in re.split(r"[,;；]", str(raw_entity)):
            cleaned = entity.strip()
            if cleaned and cleaned not in seen:
                entities.append(cleaned)
                seen.add(cleaned)
    return entities


def _normalize_seed_qa_type(value: Any) -> str:
    qa_type = str(value or "").strip()
    if qa_type in SEED_QA_TYPES:
        return qa_type
    return LEGACY_SEED_QA_TYPE_MAP.get(qa_type, "action_result")


def validate_seed_qa_record(record: dict[str, Any], valid_chunk_ids: set[str] | None = None) -> list[str]:
    reasons: list[str] = []
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    qa_type = _normalize_seed_qa_type(record.get("qa_type"))
    doc_chunk_id = str(record.get("doc_chunk_id", "")).strip()
    normalized_question = re.sub(r"\s+", "", question)
    normalized_answer = re.sub(r"\s+", "", answer)

    if not question:
        reasons.append("empty_question")
    if not answer:
        reasons.append("empty_answer")
    if len(question) < SEED_QUESTION_MIN_CHARS:
        reasons.append("question_too_short")
    if qa_type not in SEED_QA_TYPES:
        reasons.append("invalid_qa_type")
    if valid_chunk_ids is not None and doc_chunk_id not in valid_chunk_ids:
        reasons.append("doc_chunk_id_not_found")
    if len(answer) > SEED_ANSWER_MAX_CHARS:
        reasons.append("answer_too_long")
    if len(normalized_answer) >= 2 and normalized_answer in normalized_question:
        reasons.append("answer_leaked_in_question")
    if SEED_DOC_REFERENCE_RE.search(question):
        reasons.append("question_mentions_context")
    if SEED_AMBIGUOUS_PRONOUN_RE.search(question):
        reasons.append("ambiguous_pronoun")
    if SEED_ABSTRACT_RE.search(question) or SEED_ABSTRACT_RE.search(answer):
        reasons.append("subjective_or_abstract")
    if SEED_HARD_COMPOUND_RE.search(answer):
        reasons.append("compound_answer")
    if qa_type != "relation" and SEED_SOFT_COMPOUND_RE.search(answer):
        reasons.append("compound_answer")
    return reasons


def clean_seed_qa_records(
    records: Iterable[dict[str, Any]],
    valid_chunk_ids: set[str] | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_doc_questions: set[tuple[str, str]] = set()

    for record in records:
        if not _is_usable_seed_qa_record(record):
            dropped.append({"record": record, "drop_reasons": ["empty_question_or_answer"]})
            continue
        normalized = _normalize_seed_qa_record(record)
        if record.get("doc_chunk_id") is not None:
            normalized["doc_chunk_id"] = str(record.get("doc_chunk_id", "")).strip()
        if record.get("tool") is not None:
            normalized["tool"] = str(record.get("tool", "")).strip() or "keyword_search"

        reasons = validate_seed_qa_record(normalized, valid_chunk_ids=valid_chunk_ids)
        question_key = normalized["question"]
        pair_key = (normalized["question"], normalized["answer"])
        doc_question_key = (str(normalized.get("doc_chunk_id", "")), normalized["question"])
        if question_key in seen_questions:
            reasons.append("duplicate_question")
        if pair_key in seen_pairs:
            reasons.append("duplicate_question_answer")
        if normalized.get("doc_chunk_id") and doc_question_key in seen_doc_questions:
            reasons.append("duplicate_doc_question")

        if reasons:
            dropped.append({**normalized, "drop_reasons": reasons})
            continue

        seen_questions.add(question_key)
        seen_pairs.add(pair_key)
        if normalized.get("doc_chunk_id"):
            seen_doc_questions.add(doc_question_key)
        cleaned.append(normalized)
        if max_records is not None and len(cleaned) >= max_records:
            break

    return cleaned, dropped


def synthesize_multihop_examples(
    seeds: list[dict[str, Any]],
    chunks: list[Chunk],
    target_count: int = 100,
    merge_llm_client: LLMClient | None = None,
    skip_chain_keys: set[str] | None = None,
    existing_questions: set[str] | None = None,
    max_concurrency: int = 1,
) -> list[dict[str, Any]]:
    return list(
        iter_synthesize_multihop_examples(
            seeds,
            chunks,
            target_count=target_count,
            merge_llm_client=merge_llm_client,
            skip_chain_keys=skip_chain_keys,
            existing_questions=existing_questions,
            max_concurrency=max_concurrency,
        )
    )


def iter_synthesize_multihop_examples(
    seeds: list[dict[str, Any]],
    chunks: list[Chunk],
    target_count: int = 100,
    merge_llm_client: LLMClient | None = None,
    skip_chain_keys: set[str] | None = None,
    existing_questions: set[str] | None = None,
    max_concurrency: int = 1,
) -> Iterable[dict[str, Any]]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1.")
    logger.info(
        "multihop_synthesis.start seed_count=%s chunk_count=%s target_count=%s max_concurrency=%s",
        len(seeds),
        len(chunks),
        target_count,
        max_concurrency,
    )
    retriever = HybridRetriever(chunks)
    seeds_by_chunk = _group_seeds_by_chunk(seeds)
    seen_questions = set(existing_questions or set())
    seen_chain_keys = set(skip_chain_keys or set())
    generated_count = 0
    stats = {"extension_attempt_count": 0}
    candidate_chains = _iter_multihop_candidate_chains(seeds, seeds_by_chunk, retriever, seen_chain_keys, stats)

    if merge_llm_client is None or max_concurrency == 1:
        for chain in candidate_chains:
            if generated_count >= target_count:
                break
            example = _build_stepwise_example(chain, merge_llm_client=merge_llm_client)
            if example["final_question"] in seen_questions:
                continue
            seen_questions.add(example["final_question"])
            generated_count += 1
            logger.info(
                "multihop_synthesis.added subset=%s total_count=%s target_count=%s",
                example["subset"],
                generated_count,
                target_count,
            )
            yield example
    else:
        worker_count = max_concurrency
        futures: dict[Future[dict[str, Any]], list[dict[str, Any]]] = {}
        candidate_iter = iter(candidate_chains)

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            try:
                chain = next(candidate_iter)
            except StopIteration:
                return False
            futures[executor.submit(_build_stepwise_example, chain, merge_llm_client)] = chain
            return True

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(worker_count):
                if not submit_next(executor):
                    break
            while futures and generated_count < target_count:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    example = future.result()
                    if example["final_question"] in seen_questions:
                        if generated_count < target_count:
                            submit_next(executor)
                        continue
                    seen_questions.add(example["final_question"])
                    generated_count += 1
                    logger.info(
                        "multihop_synthesis.added subset=%s total_count=%s target_count=%s",
                        example["subset"],
                        generated_count,
                        target_count,
                    )
                    yield example
                    if generated_count >= target_count:
                        break
                    submit_next(executor)
            for pending_future in futures:
                pending_future.cancel()

    logger.info(
        "multihop_synthesis.done generated_count=%s target_count=%s extension_attempt_count=%s",
        generated_count,
        target_count,
        stats["extension_attempt_count"],
    )


def _iter_multihop_candidate_chains(
    seeds: list[dict[str, Any]],
    seeds_by_chunk: dict[str, list[dict[str, Any]]],
    retriever: HybridRetriever,
    seen_chain_keys: set[str],
    stats: dict[str, int],
) -> Iterable[list[dict[str, Any]]]:
    for seed in seeds:
        chain = [_seed_to_hop(seed, hop_idx=1)]
        for hop_idx in range(2, 4):
            stats["extension_attempt_count"] += 1
            next_seed = _select_next_seed(chain, seeds_by_chunk, retriever)
            if next_seed is None:
                logger.debug(
                    "multihop_synthesis.extension_miss seed_chunk_id=%s hop_idx=%s",
                    seed["doc_chunk_id"],
                    hop_idx,
                )
                break
            chain.append(_seed_to_hop(next_seed, hop_idx=hop_idx))
            chain_key = multihop_chain_key(chain)
            if chain_key in seen_chain_keys:
                logger.info(
                    "multihop_synthesis.skip_existing_chain hop_count=%s chain_key=%s",
                    len(chain),
                    chain_key,
                )
                continue
            seen_chain_keys.add(chain_key)
            yield [dict(hop) for hop in chain]


def _group_seeds_by_chunk(seeds: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        grouped.setdefault(seed["doc_chunk_id"], []).append(seed)
    return grouped


def _seed_to_hop(seed: dict[str, Any], *, hop_idx: int) -> dict[str, Any]:
    return {
        "hop_idx": hop_idx,
        "question": seed["question"],
        "answer": seed["answer"],
        "doc_chunk_id": seed["doc_chunk_id"],
        "qa_type": _normalize_seed_qa_type(seed.get("qa_type")),
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


def multihop_chain_key(hops: Iterable[dict[str, Any]]) -> str:
    return "|".join(f"{hop.get('doc_chunk_id', '')}::{hop.get('question', '')}" for hop in hops)


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
    try:
        content = llm_client.chat(_build_multihop_merge_messages(chain), temperature=0.2)
    except Exception:
        logger.exception("multihop_merge.failed fallback=rule hop_count=%s", len(chain))
        return {}
    try:
        payload = _extract_json_object(content)
    except Exception:
        logger.exception("multihop_merge.invalid_json fallback=rule response_chars=%s", len(content))
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
