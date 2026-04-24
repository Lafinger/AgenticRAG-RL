from __future__ import annotations

from typing import Any

from .agentic import answer_from_evidence
from .retrieval import HybridRetriever


def run_pipeline_episode(query: str, retriever: HybridRetriever) -> dict[str, Any]:
    results = retriever.hybrid_search(query, top_k=3)
    evidence_text = "\n".join(item.text for item in results)
    answer = answer_from_evidence(query, evidence_text)
    return {"query": query, "final_answer": answer, "evidence": [item.to_record() for item in results]}
