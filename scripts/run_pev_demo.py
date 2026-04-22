from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_study.demo_data import load_chunks, load_examples
from agentic_rag_study.pev_graph import build_pev_graph
from agentic_rag_study.retrieval import HybridRetriever


def main() -> None:
    data_dir = ROOT / "data" / "demo_financial"
    chunks = load_chunks(data_dir / "corpus.jsonl")
    examples = load_examples(data_dir / "questions.jsonl")
    retriever = HybridRetriever(chunks)
    graph = build_pev_graph(retriever)

    for example in examples:
        result = graph.invoke({"query": example.final_question, "query_type": "multi_hop", "iteration_count": 0, "total_tool_calls": 0})
        print("=" * 80)
        print(f"问题: {example.final_question}")
        print(f"标准答案: {example.final_answer}")
        print(f"预测答案: {result.get('final_answer', '')}")
        print("计划:")
        print(json.dumps(result.get("plan", []), ensure_ascii=False, indent=2))
        print("Trace:")
        print(json.dumps(result.get("trace", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

