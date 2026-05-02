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
from agentic_rag_rl.llm_client import (
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_CHOICES,
    create_batch_job_client,
    create_llm_client,
    get_doubao_batch_job_config,
    get_doubao_use_batch_inference,
    resolve_llm_base_url,
    resolve_llm_model,
    split_doubao_model_version,
)
from agentic_rag_rl.synthesis import (
    build_seed_batch,
    build_seed_qa_messages,
    iter_seed_question_batches,
    parse_seed_qa_content,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def _default_failed_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed.jsonl")


def _default_checkpoint_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.checkpoint.jsonl")


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        write_jsonl_record(handle, record)
        handle.flush()


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        chunk_id = str(record.get("chunk_id", "")).strip()
        if chunk_id:
            latest[chunk_id] = record
    return latest


def _group_seed_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        chunk_id = str(record.get("doc_chunk_id", "")).strip()
        if chunk_id:
            grouped.setdefault(chunk_id, []).append(record)
    return grouped


def _write_ordered_seed_records(output: Path, chunks: list[Any], records_by_chunk: dict[str, list[dict[str, Any]]]) -> int:
    temp_output = output.with_name(f"{output.name}.tmp")
    seed_count = 0
    with temp_output.open("w", encoding="utf-8", newline="") as handle:
        for chunk in chunks:
            for record in records_by_chunk.get(chunk.chunk_id, []):
                write_jsonl_record(handle, record)
                seed_count += 1
    temp_output.replace(output)
    return seed_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed QA from novel corpus chunks.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--max-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=LLM_PROVIDER_CHOICES)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, help="Online LLM request timeout for seed QA generation.")
    parser.add_argument(
        "--use-batch-inference",
        action="store_true",
        help="Use Doubao Batch Inference Job API for seed QA generation.",
    )
    parser.add_argument("--batch-foundation-model", help="Foundation model name for Batch Job, e.g. doubao-seed-2-0-pro.")
    parser.add_argument("--batch-model-version", help="Foundation model version for Batch Job, e.g. 260215.")
    parser.add_argument("--batch-custom-model-id", help="Custom model ID for Batch Job. Overrides foundation model.")
    parser.add_argument("--batch-tos-bucket", help="TOS bucket for Batch Job input and output.")
    parser.add_argument("--batch-input-prefix", help="TOS input object key prefix for Batch Job JSONL files.")
    parser.add_argument("--batch-output-prefix", help="TOS output object key prefix for Batch Job results.")
    parser.add_argument("--batch-project-name", help="Volcengine project name for Batch Job.")
    parser.add_argument("--batch-completion-window", help="Batch Job completion window, e.g. 1d.")
    parser.add_argument("--batch-poll-interval", type=float, default=60.0, help="Seconds between Batch Job status polls.")
    parser.add_argument("--batch-wait-timeout", type=float, help="Maximum seconds to wait for Batch Job completion.")
    parser.add_argument("--batch-work-dir", default=str(ROOT / "data" / "batch_jobs" / "seed_qa"), help="Local Batch Job work dir.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output and regenerate from the beginning.")
    parser.add_argument("--failed-output", help="JSONL file for chunks that still fail after retries.")
    parser.add_argument("--checkpoint-output", help="JSONL checkpoint file for successful and failed chunks.")
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
    batch_inference_enabled = get_doubao_use_batch_inference(args.use_batch_inference)
    if batch_inference_enabled and args.llm_provider != "doubao":
        parser.error("--use-batch-inference only supports --llm-provider doubao.")
    model = resolve_llm_model(args.llm_provider, args.model)
    base_url = resolve_llm_base_url(args.llm_provider, args.base_url)
    chunks = load_chunks(args.corpus)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    logging.info(
        "gen_seed_qa.loaded_chunks chunk_count=%s provider=%s model=%s base_url=%s batch_inference=%s",
            len(chunks),
            args.llm_provider,
            model,
            base_url,
            batch_inference_enabled,
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path = Path(args.failed_output) if args.failed_output else _default_failed_output(output_path)
    checkpoint_output_path = Path(args.checkpoint_output) if args.checkpoint_output else _default_checkpoint_output(output_path)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and failed_output_path.exists():
        failed_output_path.unlink()
    if args.overwrite and checkpoint_output_path.exists():
        checkpoint_output_path.unlink()
    existing_records = [] if args.overwrite or not output_path.exists() else load_jsonl(output_path)
    records_by_chunk = _group_seed_records(existing_records)
    checkpoint = {} if args.overwrite else _load_checkpoint(checkpoint_output_path)
    completed_chunk_ids = {chunk_id for chunk_id, record in checkpoint.items() if record.get("status") == "ok"}
    completed_chunk_ids.update(records_by_chunk)
    pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in completed_chunk_ids]
    seed_count = _write_ordered_seed_records(output_path, chunks, records_by_chunk) if records_by_chunk or completed_chunk_ids else 0
    logging.info(
        "gen_seed_qa.resume existing_seed_count=%s completed_chunk_count=%s pending_chunk_count=%s checkpoint=%s overwrite=%s",
        seed_count,
        len(completed_chunk_ids),
        len(pending_chunks),
        checkpoint_output_path,
        args.overwrite,
    )
    if not pending_chunks:
        logging.info("gen_seed_qa.done output=%s seed_count=%s", args.output, seed_count)
        print(f"seed_count={seed_count}")
        return
    failed_count = 0

    def record_failed_chunk(chunk: Any, exc: Exception) -> None:
        nonlocal failed_count
        failed_record = {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        with failed_output_path.open("a", encoding="utf-8", newline="") as failed_handle:
            write_jsonl_record(failed_handle, failed_record)
            failed_handle.flush()
        _append_checkpoint(checkpoint_output_path, failed_record)
        failed_count += 1

    def record_completed_chunk(chunk: Any, seed_batch: list[dict[str, Any]]) -> None:
        nonlocal seed_count
        records_by_chunk[chunk.chunk_id] = seed_batch
        seed_count = _write_ordered_seed_records(output_path, chunks, records_by_chunk)
        _append_checkpoint(
            checkpoint_output_path,
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "status": "ok",
                "seed_count": len(seed_batch),
            },
        )

    if batch_inference_enabled:
        batch_model = model
        default_batch_name, default_batch_version = split_doubao_model_version(batch_model)
        batch_config = get_doubao_batch_job_config(
            foundation_model_name=args.batch_foundation_model or default_batch_name,
            foundation_model_version=args.batch_model_version or default_batch_version,
            custom_model_id=args.batch_custom_model_id,
            tos_bucket=args.batch_tos_bucket,
            input_key_prefix=args.batch_input_prefix,
            output_key_prefix=args.batch_output_prefix,
            project_name=args.batch_project_name,
            completion_window=args.batch_completion_window,
        )
        batch_client = create_batch_job_client(batch_config)
        requests = [
            {
                "custom_id": chunk.chunk_id,
                "messages": build_seed_qa_messages(chunk.text, max_items=args.max_per_chunk),
                "temperature": 0.2,
            }
            for chunk in pending_chunks
        ]
        results = batch_client.run_chat_batch(
            requests,
            local_dir=args.batch_work_dir,
            job_name="agentic-rag-seed-qa",
            model=batch_model,
            poll_interval_seconds=args.batch_poll_interval,
            wait_timeout_seconds=args.batch_wait_timeout,
        )
        for index, chunk in enumerate(pending_chunks, start=1):
            try:
                result = results.get(chunk.chunk_id)
                if result is None:
                    raise RuntimeError("Batch job result missing for chunk.")
                seed_items = parse_seed_qa_content(result.content, max_items=args.max_per_chunk)
                seed_batch = build_seed_batch(chunk, seed_items)
            except Exception as exc:
                record_failed_chunk(chunk, exc)
                logging.exception(
                    "gen_seed_qa.batch_parse_failed progress=%s/%s chunk_id=%s",
                    index,
                    len(pending_chunks),
                    chunk.chunk_id,
                )
                if args.fail_fast:
                    raise
                continue
            record_completed_chunk(chunk, seed_batch)
            logging.info(
                "gen_seed_qa.batch_recorded progress=%s/%s chunk_id=%s batch_count=%s total_seed_count=%s",
                index,
                len(pending_chunks),
                chunk.chunk_id,
                len(seed_batch),
                seed_count,
            )
        logging.info(
            "gen_seed_qa.done output=%s seed_count=%s checkpoint=%s failed_output=%s failed_count=%s",
            args.output,
            seed_count,
            checkpoint_output_path,
            failed_output_path,
            failed_count,
        )
        print(f"seed_count={seed_count}")
        print(f"checkpoint_output={checkpoint_output_path}")
        if failed_count:
            print(f"failed_count={failed_count}")
            print(f"failed_output={failed_output_path}")
        return

    llm_client = create_llm_client(
        args.llm_provider,
        api_key=args.api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=args.timeout_seconds,
    )

    for seed_batch in iter_seed_question_batches(
        pending_chunks,
        llm_client,
        max_per_chunk=args.max_per_chunk,
        max_attempts=args.max_attempts,
        continue_on_error=not args.fail_fast,
        on_chunk_failed=record_failed_chunk,
        on_chunk_done=record_completed_chunk,
        max_concurrency=args.max_concurrency,
    ):
        logging.info("gen_seed_qa.recorded batch_count=%s total_seed_count=%s", len(seed_batch), seed_count)
    logging.info(
        "gen_seed_qa.done output=%s seed_count=%s checkpoint=%s failed_output=%s failed_count=%s",
        args.output,
        seed_count,
        checkpoint_output_path,
        failed_output_path,
        failed_count,
    )
    print(f"seed_count={seed_count}")
    print(f"checkpoint_output={checkpoint_output_path}")
    if failed_count:
        print(f"failed_count={failed_count}")
        print(f"failed_output={failed_output_path}")


if __name__ == "__main__":
    main()
