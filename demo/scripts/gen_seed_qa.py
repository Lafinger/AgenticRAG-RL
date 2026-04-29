from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl_record
from agentic_rag_rl.llm_client import DEFAULT_LLM_PROVIDER, create_llm_client, get_doubao_base_url, get_doubao_model
from agentic_rag_rl.synthesis import iter_seed_question_batches


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def _default_failed_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed.jsonl")


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
    parser.add_argument("--overwrite", action="store_true", help="Clear output and regenerate from the beginning.")
    parser.add_argument("--failed-output", help="JSONL file for chunks that still fail after retries.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one chunk fails.")
    parser.add_argument("--max-attempts", type=int, default=2, help="LLM attempts per chunk before marking it failed.")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent LLM requests.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1.")

    _configure_logging(args.log_level)
    logging.info(
        "gen_seed_qa.start corpus=%s output=%s max_per_chunk=%s max_chunks=%s max_concurrency=%s",
        args.corpus,
        args.output,
        args.max_per_chunk,
        args.max_chunks,
        args.max_concurrency,
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path = Path(args.failed_output) if args.failed_output else _default_failed_output(output_path)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and failed_output_path.exists():
        failed_output_path.unlink()
    existing_records = [] if args.overwrite or not output_path.exists() else load_jsonl(output_path)
    completed_chunk_ids = {str(record.get("doc_chunk_id")) for record in existing_records if record.get("doc_chunk_id")}
    pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in completed_chunk_ids]
    logging.info(
        "gen_seed_qa.resume existing_seed_count=%s completed_chunk_count=%s pending_chunk_count=%s overwrite=%s",
        len(existing_records),
        len(completed_chunk_ids),
        len(pending_chunks),
        args.overwrite,
    )
    seed_count = len(existing_records)
    if not pending_chunks:
        logging.info("gen_seed_qa.done output=%s seed_count=%s", args.output, seed_count)
        print(f"seed_count={seed_count}")
        return
    llm_client = create_llm_client(
        args.llm_provider,
        api_key=args.api_key,
        model=get_doubao_model(args.model),
        base_url=get_doubao_base_url(args.base_url),
    )
    mode = "w" if args.overwrite else "a"
    failed_count = 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:

        def record_failed_chunk(chunk: Any, exc: Exception) -> None:
            nonlocal failed_count
            with failed_output_path.open("a", encoding="utf-8", newline="") as failed_handle:
                write_jsonl_record(
                    failed_handle,
                    {
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                failed_handle.flush()
            failed_count += 1

        for seed_batch in iter_seed_question_batches(
            pending_chunks,
            llm_client,
            max_per_chunk=args.max_per_chunk,
            max_attempts=args.max_attempts,
            continue_on_error=not args.fail_fast,
            on_chunk_failed=record_failed_chunk,
            max_concurrency=args.max_concurrency,
        ):
            for seed in seed_batch:
                write_jsonl_record(handle, seed)
                seed_count += 1
            handle.flush()
            logging.info("gen_seed_qa.appended batch_count=%s total_seed_count=%s", len(seed_batch), seed_count)
    logging.info(
        "gen_seed_qa.done output=%s seed_count=%s failed_output=%s failed_count=%s",
        args.output,
        seed_count,
        failed_output_path,
        failed_count,
    )
    print(f"seed_count={seed_count}")
    if failed_count:
        print(f"failed_count={failed_count}")
        print(f"failed_output={failed_output_path}")


if __name__ == "__main__":
    main()
