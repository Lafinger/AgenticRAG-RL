from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.io import load_chunks, write_jsonl_record
from agentic_rag_rl.llm_client import DEFAULT_LLM_PROVIDER, create_llm_client, get_doubao_base_url, get_doubao_model
from agentic_rag_rl.synthesis import iter_seed_question_batches


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed QA from novel corpus chunks.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--max-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=["doubao"])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logging.info(
        "gen_seed_qa.start corpus=%s output=%s max_per_chunk=%s max_chunks=%s",
        args.corpus,
        args.output,
        args.max_per_chunk,
        args.max_chunks,
    )
    load_env_file(args.env_file)
    chunks = load_chunks(args.corpus)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    logging.info(
        "gen_seed_qa.loaded_chunks chunk_count=%s provider=%s model=%s base_url=%s",
        len(chunks),
        args.llm_provider,
        get_doubao_model(args.model),
        get_doubao_base_url(args.base_url),
    )
    llm_client = create_llm_client(
        args.llm_provider,
        api_key=args.api_key,
        model=get_doubao_model(args.model),
        base_url=get_doubao_base_url(args.base_url),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for seed_batch in iter_seed_question_batches(chunks, llm_client, max_per_chunk=args.max_per_chunk):
            for seed in seed_batch:
                write_jsonl_record(handle, seed)
                seed_count += 1
            handle.flush()
            logging.info("gen_seed_qa.appended batch_count=%s total_seed_count=%s", len(seed_batch), seed_count)
    logging.info("gen_seed_qa.done output=%s seed_count=%s", args.output, seed_count)
    print(f"seed_count={seed_count}")


if __name__ == "__main__":
    main()
