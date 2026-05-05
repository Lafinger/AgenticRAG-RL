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
    LLM_PROVIDER_CHOICES,
    create_batch_job_client,
    create_llm_client,
    get_doubao_batch_job_config,
    get_doubao_use_batch_inference,
    resolve_llm_base_url,
    resolve_judge_model,
    resolve_thinking_model,
    split_doubao_model_version,
)
from agentic_rag_rl.retrieval import HybridRetriever
from agentic_rag_rl.synthesis import (
    _group_seeds_by_chunk,
    _iter_multihop_candidate_chains,
    build_multihop_merge_messages,
    build_stepwise_example_from_merge,
    judge_multihop_example,
    iter_synthesize_multihop_examples,
    multihop_chain_key,
    parse_multihop_merge_content,
    seed_record_key,
    validate_multihop_example,
)


def _default_failed_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed.jsonl")


def _default_checkpoint_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.checkpoint.jsonl")


def _default_rejected_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.rejected.jsonl")


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
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=LLM_PROVIDER_CHOICES)
    parser.add_argument("--merge-model")
    parser.add_argument("--judge-model", help="Judge model for LLM quality gate. Defaults to merge model.")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, help="Online LLM request timeout for LLM merge.")
    parser.add_argument(
        "--use-batch-inference",
        action="store_true",
        help="Use Doubao Batch Inference Job API for LLM merge.",
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
    parser.add_argument("--batch-work-dir", default=str(ROOT / "data" / "batch_jobs" / "multihop_merge"), help="Local Batch Job work dir.")
    parser.add_argument("--batch-max-candidates", type=int, help="Maximum candidate chains submitted to one Batch Job.")
    parser.add_argument("--quality-gate", default="llm", choices=["rules", "llm"], help="Quality gate before appending examples.")
    parser.add_argument("--candidate-multiplier", type=int, help="Candidate chain multiplier for stricter quality gates.")
    parser.add_argument("--disable-llm-merge", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent LLM merge requests.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output and regenerate from the beginning.")
    parser.add_argument("--failed-output", help="JSONL file for failed LLM merge chains.")
    parser.add_argument("--rejected-output", help="JSONL file for rejected merged examples.")
    parser.add_argument("--checkpoint-output", help="JSONL checkpoint file for successful and failed chains.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1.")
    if args.candidate_multiplier is not None and args.candidate_multiplier < 1:
        parser.error("--candidate-multiplier must be >= 1.")
    if args.quality_gate == "llm" and args.disable_llm_merge:
        parser.error("--quality-gate llm requires LLM merge; remove --disable-llm-merge or use --quality-gate rules.")

    _configure_logging(args.log_level)
    load_env_file(args.env_file)
    batch_inference_enabled = get_doubao_use_batch_inference(args.use_batch_inference)
    if batch_inference_enabled and args.llm_provider != "doubao":
        parser.error("--use-batch-inference only supports --llm-provider doubao.")
    merge_model = resolve_thinking_model(args.llm_provider, args.merge_model)
    judge_model = resolve_judge_model(args.llm_provider, args.judge_model) if args.judge_model else merge_model
    base_url = resolve_llm_base_url(args.llm_provider, args.base_url)
    candidate_multiplier = args.candidate_multiplier or (5 if args.quality_gate == "llm" else 10)
    logging.info(
        "domain_multihop_synthesis.start seeds=%s corpus=%s output=%s target_count=%s llm_merge=%s merge_model=%s quality_gate=%s judge_model=%s max_concurrency=%s batch_inference=%s",
        args.seeds,
        args.corpus,
        args.output,
        args.target_count,
        not args.disable_llm_merge,
        merge_model,
        args.quality_gate,
        judge_model,
        args.max_concurrency,
        batch_inference_enabled,
    )
    seeds = load_jsonl(args.seeds)
    chunks = load_chunks(args.corpus)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    valid_chunk_ids = set(chunks_by_id)
    seed_keys = {seed_record_key(seed) for seed in seeds}
    logging.info("domain_multihop_synthesis.loaded_inputs seed_count=%s chunk_count=%s", len(seeds), len(chunks))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path = Path(args.failed_output) if args.failed_output else _default_failed_output(output_path)
    checkpoint_output_path = Path(args.checkpoint_output) if args.checkpoint_output else _default_checkpoint_output(output_path)
    rejected_output_path = Path(args.rejected_output) if args.rejected_output else _default_rejected_output(output_path)
    if args.overwrite and failed_output_path.exists():
        failed_output_path.unlink()
    if args.overwrite and checkpoint_output_path.exists():
        checkpoint_output_path.unlink()
    if args.overwrite and rejected_output_path.exists():
        rejected_output_path.unlink()
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

    def record_failed_chain(chain: list[dict], exc: Exception) -> None:
        failed_record = {
            "chain_key": multihop_chain_key(chain),
            "status": "failed",
            "hops": chain,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _append_record(failed_output_path, failed_record)
        _append_record(checkpoint_output_path, failed_record)

    rejected_count = 0

    def record_rejected_example(record: dict) -> None:
        nonlocal rejected_count
        rejected_count += 1
        _append_record(rejected_output_path, {**record, "status": "rejected"})

    judge_llm_client = None
    if args.quality_gate == "llm":
        judge_llm_client = create_llm_client(
            args.llm_provider,
            api_key=args.api_key,
            model=judge_model,
            base_url=base_url,
            timeout_seconds=args.timeout_seconds,
        )

    if batch_inference_enabled and not args.disable_llm_merge:
        stats = {"extension_attempt_count": 0}
        seen_chain_keys = set(existing_chain_keys)
        candidate_limit = args.batch_max_candidates or max(remaining_count * candidate_multiplier, remaining_count)
        candidate_chains = []
        candidate_iter = _iter_multihop_candidate_chains(
            seeds,
            _group_seeds_by_chunk(seeds),
            HybridRetriever(chunks),
            seen_chain_keys,
            stats,
        )
        for chain in candidate_iter:
            candidate_chains.append(chain)
            if len(candidate_chains) >= candidate_limit:
                break
        batch_model = merge_model
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
        chain_by_id = {multihop_chain_key(chain): chain for chain in candidate_chains}
        requests = [
            {
                "custom_id": custom_id,
                "messages": build_multihop_merge_messages(chain),
                "temperature": 0.2,
            }
            for custom_id, chain in chain_by_id.items()
        ]
        results = batch_client.run_chat_batch(
            requests,
            local_dir=args.batch_work_dir,
            job_name="agentic-rag-multihop-merge",
            model=batch_model,
            poll_interval_seconds=args.batch_poll_interval,
            wait_timeout_seconds=args.batch_wait_timeout,
        )
        generated_count = 0
        failed_count = 0
        seen_questions = set(existing_questions)
        mode = "w" if args.overwrite else "a"
        with output_path.open(mode, encoding="utf-8", newline="") as handle:
            for chain_key, chain in chain_by_id.items():
                if generated_count >= remaining_count:
                    break
                try:
                    result = results.get(chain_key)
                    if result is None:
                        raise RuntimeError("Batch job result missing for chain.")
                    merged = parse_multihop_merge_content(result.content)
                    example = build_stepwise_example_from_merge(chain, merged)
                    if example["final_question"] in seen_questions:
                        record_rejected_example(
                            {
                                "chain_key": chain_key,
                                "stage": "duplicate",
                                "problem_codes": ["DUPLICATE_FINAL_QUESTION"],
                                "final_question": example.get("final_question", ""),
                                "final_answer": example.get("final_answer", ""),
                                "hops": chain,
                                "detail": "",
                            }
                        )
                        continue
                    rule_codes = validate_multihop_example(
                        example,
                        valid_chunk_ids=valid_chunk_ids,
                        seed_keys=seed_keys,
                    )
                    if rule_codes:
                        record_rejected_example(
                            {
                                "chain_key": chain_key,
                                "stage": "rules",
                                "problem_codes": rule_codes,
                                "final_question": example.get("final_question", ""),
                                "final_answer": example.get("final_answer", ""),
                                "hops": chain,
                                "detail": "",
                            }
                        )
                        continue
                    if args.quality_gate == "llm":
                        judge_codes = judge_multihop_example(example, judge_llm_client, chunks_by_id)  # type: ignore[arg-type]
                        if judge_codes:
                            record_rejected_example(
                                {
                                    "chain_key": chain_key,
                                    "stage": "judge",
                                    "problem_codes": judge_codes,
                                    "final_question": example.get("final_question", ""),
                                    "final_answer": example.get("final_answer", ""),
                                    "hops": chain,
                                    "detail": "",
                                }
                            )
                            continue
                except Exception as exc:
                    failed_count += 1
                    record_failed_chain(chain, exc)
                    logging.exception("domain_multihop_synthesis.batch_merge_failed chain_key=%s", chain_key)
                    continue
                seen_questions.add(example["final_question"])
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
                    "domain_multihop_synthesis.batch_appended generated_count=%s total_count=%s failed_count=%s rejected_count=%s",
                    generated_count,
                    len(existing_examples) + generated_count,
                    failed_count,
                    rejected_count,
                )
        total_count = len(existing_examples) + generated_count
        logging.info(
            "domain_multihop_synthesis.done output=%s multihop_count=%s rejected_count=%s remaining_count=%s",
            args.output,
            total_count,
            rejected_count,
            max(args.target_count - total_count, 0),
        )
        print(f"multihop_count={total_count}")
        print(f"rejected_count={rejected_count}")
        print(f"remaining_count={max(args.target_count - total_count, 0)}")
        if failed_count:
            print(f"failed_count={failed_count}")
            print(f"failed_output={failed_output_path}")
        if rejected_count:
            print(f"rejected_output={rejected_output_path}")
        return

    merge_llm_client = None
    if not args.disable_llm_merge:
        merge_llm_client = create_llm_client(
            args.llm_provider,
            api_key=args.api_key,
            model=merge_model,
            base_url=base_url,
            timeout_seconds=args.timeout_seconds,
        )
    generated_count = 0
    failed_count = 0
    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        def record_failed_chain_with_count(chain: list[dict], exc: Exception) -> None:
            nonlocal failed_count
            record_failed_chain(chain, exc)
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
            on_chain_failed=record_failed_chain_with_count,
            quality_gate=args.quality_gate,
            judge_llm_client=judge_llm_client,
            seed_keys=seed_keys,
            on_example_rejected=record_rejected_example,
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
                "domain_multihop_synthesis.appended generated_count=%s total_count=%s failed_count=%s rejected_count=%s",
                generated_count,
                len(existing_examples) + generated_count,
                failed_count,
                rejected_count,
            )
    total_count = len(existing_examples) + generated_count
    logging.info(
        "domain_multihop_synthesis.done output=%s multihop_count=%s rejected_count=%s remaining_count=%s",
        args.output,
        total_count,
        rejected_count,
        max(args.target_count - total_count, 0),
    )
    print(f"multihop_count={total_count}")
    print(f"rejected_count={rejected_count}")
    print(f"remaining_count={max(args.target_count - total_count, 0)}")
    if failed_count:
        print(f"failed_count={failed_count}")
        print(f"failed_output={failed_output_path}")
    if rejected_count:
        print(f"rejected_output={rejected_output_path}")


if __name__ == "__main__":
    main()
