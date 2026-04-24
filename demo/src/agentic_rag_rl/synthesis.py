from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .types import Chunk


def _extract_metric_answer(text: str, metric: str) -> str | None:
    matched = re.search(rf"{re.escape(metric)}[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(亿元|元|家|个)?", text)
    if not matched:
        return None
    return f"{matched.group(1)} {matched.group(2) or ''}".strip()


def _extract_legal_representative(text: str) -> str | None:
    matched = re.search(r"法定代表人(?:为)?([^\s，。,；]+)", text)
    return matched.group(1) if matched else None


def _extract_subcompany(text: str) -> str | None:
    matched = re.search(r"旗下.*?公司([^\s，。,；]+)", text)
    return matched.group(1) if matched else ("彩食鲜" if "彩食鲜" in text else None)


def generate_seed_questions(chunks: Iterable[Chunk], max_per_chunk: int = 2) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for chunk in chunks:
        generated = 0
        revenue = _extract_metric_answer(chunk.text, "营业收入")
        if revenue and generated < max_per_chunk:
            seeds.append(
                {
                    "question": f"{chunk.company or chunk.title} 2025 年上半年营业收入是多少？",
                    "answer": revenue,
                    "doc_chunk_id": chunk.chunk_id,
                    "tool": "keyword_search",
                    "company": chunk.company,
                    "qa_type": "initial_qa",
                }
            )
            generated += 1

        representative = _extract_legal_representative(chunk.text)
        if representative and generated < max_per_chunk:
            seeds.append(
                {
                    "question": f"{chunk.company or chunk.title} 的法定代表人是谁？",
                    "answer": representative,
                    "doc_chunk_id": chunk.chunk_id,
                    "tool": "keyword_search",
                    "company": chunk.company,
                    "qa_type": "initial_qa",
                }
            )
            generated += 1

        subcompany = _extract_subcompany(chunk.text)
        if subcompany and generated < max_per_chunk:
            seeds.append(
                {
                    "question": f"{chunk.company or chunk.title} 旗下的生鲜供应链公司是什么？",
                    "answer": subcompany,
                    "doc_chunk_id": chunk.chunk_id,
                    "tool": "keyword_search",
                    "company": chunk.company,
                    "qa_type": "initial_qa",
                }
            )
    return seeds


def synthesize_multihop_examples(seeds: list[dict[str, Any]], chunks: list[Chunk], target_count: int = 100) -> list[dict[str, Any]]:
    del chunks
    examples: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        if "营业收入" in seed["question"]:
            by_metric["营业收入"].append(seed)

    revenue_seeds = by_metric["营业收入"]
    for left in revenue_seeds:
        for right in revenue_seeds:
            if left["company"] == right["company"]:
                continue
            examples.append(
                {
                    "final_question": f"{left['company']}和{right['company']}哪家的营业收入更高？",
                    "final_answer": left["company"] if float(left["answer"].split()[0]) >= float(right["answer"].split()[0]) else right["company"],
                    "hop_count": 2,
                    "qa_type": "comparison",
                    "subset": "2hop_comparison",
                    "hops": [
                        {
                            "hop_idx": 1,
                            "question": left["question"],
                            "answer": left["answer"],
                            "doc_chunk_id": left["doc_chunk_id"],
                            "qa_type": left["qa_type"],
                        },
                        {
                            "hop_idx": 2,
                            "question": right["question"],
                            "answer": right["answer"],
                            "doc_chunk_id": right["doc_chunk_id"],
                            "qa_type": right["qa_type"],
                        },
                    ],
                    "answer_aliases": [left["company"], right["company"]],
                }
            )
            if len(examples) >= target_count:
                return examples
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
