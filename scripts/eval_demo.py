from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_study.demo_data import load_chunks, load_examples
from agentic_rag_study.evaluation import exact_match, hop_recall, over_extension, premature_collapse, step_alignment, token_f1
from agentic_rag_study.pev_graph import build_pev_graph
from agentic_rag_study.retrieval import HybridRetriever


def main() -> None:
    data_dir = ROOT / "data" / "demo_financial"
    chunks = load_chunks(data_dir / "corpus.jsonl")
    examples = load_examples(data_dir / "questions.jsonl")
    retriever = HybridRetriever(chunks)
    graph = build_pev_graph(retriever)

    header = "Question | EM | F1 | HopRecall | PrematureCollapse | OverExtension | StepAlignment"
    print(header)
    print("-" * len(header))

    for example in examples:
        result = graph.invoke({"query": example.final_question, "query_type": "multi_hop", "iteration_count": 0, "total_tool_calls": 0})
        evidence = result.get("evidence", [])
        predicted_chunk_ids = [step["results"][0]["chunk_id"] for step in evidence if step.get("results")]
        gold_chunk_ids = [hop.doc_chunk_id for hop in example.hops]
        predicted_steps = len(predicted_chunk_ids)
        print(
            f"{example.final_question[:16]:<16} | "
            f"{exact_match(result.get('final_answer', ''), example.final_answer):.2f} | "
            f"{token_f1(result.get('final_answer', ''), example.final_answer):.2f} | "
            f"{hop_recall(predicted_chunk_ids, gold_chunk_ids):.2f} | "
            f"{premature_collapse(predicted_steps, example.hop_count):.2f} | "
            f"{over_extension(predicted_steps, example.hop_count):.2f} | "
            f"{step_alignment(predicted_chunk_ids, gold_chunk_ids):.2f}"
        )


if __name__ == "__main__":
    main()

