from __future__ import annotations

import re
from typing import Any

from .retrieval import HybridRetriever


def _extract_metric(query: str) -> str:
    for metric in ("营业收入", "净利润", "法定代表人"):
        if metric in query:
            return metric
    return "营业收入"


def _extract_companies(query: str, retriever: HybridRetriever) -> list[str]:
    companies: list[str] = []
    for chunk in retriever.chunks:
        company = chunk.company
        if company and company in query and company not in companies:
            companies.append(company)
    return companies


def _extract_numeric_value(text: str, metric: str) -> float | None:
    matched = re.search(rf"{re.escape(metric)}[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    return float(matched.group(1)) if matched else None


def run_agentic_episode(query: str, retriever: HybridRetriever, max_turns: int = 7) -> dict[str, Any]:
    metric = _extract_metric(query)
    companies = _extract_companies(query, retriever)
    evidence: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    tool_calls = 0

    if len(companies) >= 2 and "哪家" in query:
        comparisons: list[tuple[str, float]] = []
        for company in companies[:max_turns]:
            tool_calls += 1
            results = retriever.hybrid_search(f"{company} {metric}", top_k=1)
            if not results:
                continue
            top = results[0]
            evidence.append({"query": f"{company} {metric}", "results": [top.to_record()]})
            retrieved_chunk_ids.append(top.chunk_id)
            value = _extract_numeric_value(top.text, metric)
            if value is not None:
                comparisons.append((company, value))
        final_answer = sorted(comparisons, key=lambda item: item[1], reverse=True)[0][0] if comparisons else ""
    else:
        tool_calls += 1
        results = retriever.hybrid_search(query, top_k=1)
        top = results[0] if results else None
        if top is not None:
            evidence.append({"query": query, "results": [top.to_record()]})
            retrieved_chunk_ids.append(top.chunk_id)
            final_answer = top.metadata.get("company") or top.text[:40]
        else:
            final_answer = ""

    return {
        "query": query,
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "evidence": evidence,
    }
