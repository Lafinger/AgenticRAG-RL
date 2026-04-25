from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks, write_jsonl
from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.llm_client import DoubaoSeedQAClient, get_doubao_base_url, get_doubao_model
from agentic_rag_rl.synthesis import generate_seed_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed QA from novel corpus chunks.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--max-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    load_env_file(args.env_file)
    chunks = load_chunks(args.corpus)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    client = DoubaoSeedQAClient(api_key=args.api_key, model=get_doubao_model(args.model), base_url=get_doubao_base_url(args.base_url))
    seeds = generate_seed_questions(chunks, client, max_per_chunk=args.max_per_chunk)
    write_jsonl(seeds, args.output)
    print(f"seed_count={len(seeds)}")


if __name__ == "__main__":
    main()
