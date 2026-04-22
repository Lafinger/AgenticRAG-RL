from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.judge import judge_agentic_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic judge on evaluation results.")
    parser.add_argument("results_file")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    judged_results = []
    for item in payload["results"]:
        evidence_text = " ".join(result["text"] for evidence in item.get("evidence", []) for result in evidence.get("results", []))
        item["judge"] = judge_agentic_answer(
            question=item["question"],
            prediction=item["prediction"],
            gold=item["gold"],
            aliases=[item["gold"]],
            evidence_text=evidence_text,
            gold_chunks=item.get("gold_chunks", []),
            retrieved_chunk_ids=item.get("retrieved_chunk_ids", []),
        )
        judged_results.append(item)

    summary = {
        "count": len(judged_results),
        "avg_correctness": sum(item["judge"]["correctness"] for item in judged_results) / max(len(judged_results), 1),
        "avg_faithfulness": sum(item["judge"]["faithfulness"] for item in judged_results) / max(len(judged_results), 1),
        "avg_context_precision": sum(item["judge"]["context_precision"] for item in judged_results) / max(len(judged_results), 1),
    }
    output_payload = {"summary": summary, "results": judged_results}
    output_path = Path(args.output) if args.output else Path(args.results_file).with_name(Path(args.results_file).stem + "_judged.json")
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
