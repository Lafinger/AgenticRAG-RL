from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.evaluation import exact_match, token_f1
from agentic_rag_rl.io import load_chunks, load_multihop_examples
from agentic_rag_rl.pipeline import run_pipeline_episode
from agentic_rag_rl.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dataset-level pipeline evaluation.")
    parser.add_argument("--data", default=str(ROOT / "data" / "novel_eval" / "qa_pairs.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "results" / "pipeline_eval.json"))
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    examples = load_multihop_examples(args.data)[: args.max_samples]
    retriever = HybridRetriever(load_chunks(args.corpus))
    results = []
    for example in examples:
        result = run_pipeline_episode(example.final_question, retriever)
        results.append(
            {
                "question": example.final_question,
                "gold": example.final_answer,
                "prediction": result["final_answer"],
                "em": exact_match(result["final_answer"], example.final_answer),
                "f1": token_f1(result["final_answer"], example.final_answer),
                "evidence": result["evidence"],
            }
        )

    payload = {
        "summary": {
            "count": len(results),
            "avg_em": sum(item["em"] for item in results) / max(len(results), 1),
            "avg_f1": sum(item["f1"] for item in results) / max(len(results), 1),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
