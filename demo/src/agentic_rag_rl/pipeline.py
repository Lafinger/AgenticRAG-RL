from __future__ import annotations

import re
from typing import Any

from .retrieval import HybridRetriever


def _extract_companies(query: str, retriever: HybridRetriever) -> list[str]:
    companies: list[str] = []
    for chunk in retriever.chunks:
        company = chunk.company
        if company and company in query and company not in companies:
            companies.append(company)
    return companies


def _extract_metric(query: str) -> str:
    for metric in ("营业收入", "净利润", "法定代表人"):
        if metric in query:
            return metric
    return "营业收入"


def _extract_numeric_value(text: str, metric: str) -> float | None:
    matched = re.search(rf"{re.escape(metric)}[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    return float(matched.group(1)) if matched else None


def run_pipeline_episode(query: str, retriever: HybridRetriever) -> dict[str, Any]:
    companies = _extract_companies(query, retriever)
    metric = _extract_metric(query)

    if len(companies) >= 2 and "哪家" in query:
        retrieved: list[dict[str, Any]] = []
        scored: list[tuple[str, float]] = []
        for company in companies:
            results = retriever.hybrid_search(f"{company} {metric}", top_k=1)
            if not results:
                continue
            top = results[0]
            value = _extract_numeric_value(top.text, metric)
            if value is not None:
                scored.append((company, value))
            retrieved.append(top.to_record())
        answer = sorted(scored, key=lambda item: item[1], reverse=True)[0][0] if scored else ""
        return {"query": query, "final_answer": answer, "evidence": retrieved}

    results = retriever.hybrid_search(query, top_k=3)
    answer = results[0].metadata.get("company") if results else ""
    return {"query": query, "final_answer": answer or (results[0].text[:40] if results else ""), "evidence": [item.to_record() for item in results]}
