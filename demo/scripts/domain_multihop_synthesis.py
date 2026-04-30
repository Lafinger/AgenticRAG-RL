from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl_record
from agentic_rag_rl.llm_client import (
    DEFAULT_LLM_PROVIDER,
    create_llm_client,
    get_doubao_base_url,
    get_doubao_thinking_model,
)
from agentic_rag_rl.synthesis import iter_synthesize_multihop_examples, multihop_chain_key


def _default_failed_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed.jsonl")


def _default_checkpoint_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.checkpoint.jsonl")


def _append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        write_jsonl_record(handle, record)
        handle.flush()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize multi-hop QA examples from seeds.")
    parser.add_argument("--seeds", default=str(ROOT / "data" / "novel_eval" / "seeds_clean.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "qa_pairs.jsonl"))
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=["doubao"])
    parser.add_argument("--merge-model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, help="Doubao request timeout for LLM merge.")
    parser.add_argument("--disable-llm-merge", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent LLM merge requests.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output and regenerate from the beginning.")
    parser.add_argument("--failed-output", help="JSONL file for failed LLM merge chains.")
    parser.add_argument("--checkpoint-output", help="JSONL checkpoint file for successful and failed chains.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1.")

    _configure_logging(args.log_level)
    load_env_file(args.env_file)
    logging.info(
        "domain_multihop_synthesis.start seeds=%s corpus=%s output=%s target_count=%s llm_merge=%s merge_model=%s max_concurrency=%s",
        args.seeds,
        args.corpus,
        args.output,
        args.target_count,
        not args.disable_llm_merge,
        get_doubao_thinking_model(args.merge_model),
        args.max_concurrency,
    )
    seeds = load_jsonl(args.seeds)
    chunks = load_chunks(args.corpus)
    logging.info("domain_multihop_synthesis.loaded_inputs seed_count=%s chunk_count=%s", len(seeds), len(chunks))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path = Path(args.failed_output) if args.failed_output else _default_failed_output(output_path)
    checkpoint_output_path = Path(args.checkpoint_output) if args.checkpoint_output else _default_checkpoint_output(output_path)
    if args.overwrite and failed_output_path.exists():
        failed_output_path.unlink()
    if args.overwrite and checkpoint_output_path.exists():
        checkpoint_output_path.unlink()
    existing_examples = [] if args.overwrite or not output_path.exists() else load_jsonl(output_path)
    existing_chain_keys = {multihop_chain_key(record.get("hops", [])) for record in existing_examples}
    existing_questions = {
        str(record.get("final_question", "")).strip()
        for record in existing_examples
        if str(record.get("final_question", "")).strip()
    }
    remaining_count = max(args.target_count - len(existing_examples), 0)
    logging.info(
        "domain_multihop_synthesis.resume existing_count=%s remaining_count=%s overwrite=%s",
        len(existing_examples),
        remaining_count,
        args.overwrite,
    )
    if remaining_count == 0:
        logging.info("domain_multihop_synthesis.done output=%s multihop_count=%s", args.output, len(existing_examples))
        print(f"multihop_count={len(existing_examples)}")
        return
    merge_llm_client = None
    if not args.disable_llm_merge:
        merge_llm_client = create_llm_client(
            args.llm_provider,
            api_key=args.api_key,
            model=get_doubao_thinking_model(args.merge_model),
            base_url=get_doubao_base_url(args.base_url),
            timeout_seconds=args.timeout_seconds,
        )
    generated_count = 0
    failed_count = 0
    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        def record_failed_chain(chain: list[dict], exc: Exception) -> None:
            nonlocal failed_count
            failed_record = {
                "chain_key": multihop_chain_key(chain),
                "status": "failed",
                "hops": chain,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _append_record(failed_output_path, failed_record)
            _append_record(checkpoint_output_path, failed_record)
            failed_count += 1

        for example in iter_synthesize_multihop_examples(
            seeds,
            chunks,
            target_count=remaining_count,
            merge_llm_client=merge_llm_client,
            skip_chain_keys=existing_chain_keys,
            existing_questions=existing_questions,
            max_concurrency=args.max_concurrency,
            raise_on_merge_failure=merge_llm_client is not None,
            on_chain_failed=record_failed_chain,
        ):
            write_jsonl_record(handle, example)
            handle.flush()
            generated_count += 1
            _append_record(
                checkpoint_output_path,
                {
                    "chain_key": multihop_chain_key(example.get("hops", [])),
                    "status": "ok",
                    "final_question": example.get("final_question"),
                },
            )
            logging.info(
                "domain_multihop_synthesis.appended generated_count=%s total_count=%s failed_count=%s",
                generated_count,
                len(existing_examples) + generated_count,
                failed_count,
            )
    total_count = len(existing_examples) + generated_count
    logging.info("domain_multihop_synthesis.done output=%s multihop_count=%s", args.output, total_count)
    print(f"multihop_count={total_count}")
    if failed_count:
        print(f"failed_count={failed_count}")
        print(f"failed_output={failed_output_path}")


if __name__ == "__main__":
    main()
