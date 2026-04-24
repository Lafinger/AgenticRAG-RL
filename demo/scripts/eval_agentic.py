from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.agentic import run_agentic_episode
from agentic_rag_rl.evaluation import exact_match, hop_recall, token_f1
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic search on a QA dataset.")
    parser.add_argument("--data", default=str(ROOT / "data" / "smoke_financial" / "qa_pairs.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "smoke_financial" / "corpus.jsonl"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()

    examples = load_multihop_examples(args.data)[: args.max_samples]
    retriever = HybridRetriever(load_chunks(args.corpus))
    results = []
    for example in examples:
        rollout = run_agentic_episode(example.final_question, retriever, max_turns=args.max_turns)
        results.append(
            {
                "question": example.final_question,
                "gold": example.final_answer,
                "prediction": rollout["final_answer"],
                "retrieved_chunk_ids": rollout["retrieved_chunk_ids"],
                "gold_chunks": [hop.doc_chunk_id for hop in example.hops],
                "tool_calls": rollout["tool_calls"],
                "em": exact_match(rollout["final_answer"], example.final_answer),
                "f1": token_f1(rollout["final_answer"], example.final_answer),
                "hop_recall": hop_recall(rollout["retrieved_chunk_ids"], [hop.doc_chunk_id for hop in example.hops]),
                "evidence": rollout["evidence"],
            }
        )

    summary = {
        "count": len(results),
        "avg_em": sum(item["em"] for item in results) / max(len(results), 1),
        "avg_f1": sum(item["f1"] for item in results) / max(len(results), 1),
        "avg_hop_recall": sum(item["hop_recall"] for item in results) / max(len(results), 1),
    }
    payload = {"summary": summary, "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
