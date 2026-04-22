from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.pipeline import run_pipeline_episode
from agentic_rag_rl.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline-style evaluation query.")
    parser.add_argument("query")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "smoke_financial" / "corpus.jsonl"))
    args = parser.parse_args()

    retriever = HybridRetriever(load_chunks(args.corpus))
    result = run_pipeline_episode(args.query, retriever)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
