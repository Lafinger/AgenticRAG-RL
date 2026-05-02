from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.indexing import build_index_bundle, save_index_bundle
from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.llm_client import (
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_CHOICES,
    create_batch_job_client,
    create_llm_client,
    get_doubao_batch_job_config,
    get_doubao_use_batch_inference,
    resolve_kg_model,
    resolve_llm_base_url,
    split_doubao_model_version,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def _looks_like_local_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith((".", "~")) or "\\" in value or value.startswith("models/")


def _validate_embedding_model(value: str) -> None:
    if _looks_like_local_path(value) and not Path(value).exists():
        raise SystemExit(
            "Embedding model path not found: "
            f"{value}\n"
            "Download it first:\n"
            "  uv run hf download BAAI/bge-m3 --local-dir .\\models\\bge-m3\n"
            "Or use the Hugging Face model id directly:\n"
            "  --embedding-model BAAI/bge-m3"
        )


def _warn_missing_reranker_model(value: str | None) -> None:
    if value and _looks_like_local_path(value) and not Path(value).exists():
        print(f"warning: reranker model path not found: {value}", file=sys.stderr)
        print(
            "warning: download it with: "
            "uv run hf download BAAI/bge-reranker-v2-m3 --local-dir .\\models\\bge-reranker-v2-m3",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS+BGE-M3, BM25 and optional KG indexes from corpus chunks.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--reranker-model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-kg", action="store_true")
    parser.add_argument("--kg-cache")
    parser.add_argument("--kg-model")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent online LLM KG extraction requests.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=LLM_PROVIDER_CHOICES)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, help="Online LLM request timeout for KG extraction.")
    parser.add_argument(
        "--use-batch-inference",
        action="store_true",
        help="Use Doubao Batch Inference Job API for KG extraction.",
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
    parser.add_argument("--batch-work-dir", default=str(ROOT / "data" / "batch_jobs" / "kg"), help="Local Batch Job work dir.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1.")
    _configure_logging(args.log_level)
    _validate_embedding_model(args.embedding_model)
    _warn_missing_reranker_model(args.reranker_model)

    logging.info("build_index.progress stage=load_env env_file=%s", args.env_file)
    load_env_file(args.env_file)
    batch_inference_enabled = get_doubao_use_batch_inference(args.use_batch_inference)
    if batch_inference_enabled and args.llm_provider != "doubao":
        parser.error("--use-batch-inference only supports --llm-provider doubao.")
    logging.info("build_index.progress stage=load_corpus_start corpus=%s", args.corpus)
    chunks = load_chunks(args.corpus)
    logging.info(
        "build_index.start chunk_count=%s skip_kg=%s embedding_model=%s max_concurrency=%s",
        len(chunks),
        args.skip_kg,
        args.embedding_model,
        args.max_concurrency,
    )
    kg_llm_client = None
    kg_batch_job_client = None
    batch_model = None
    kg_cache = args.kg_cache or str(Path(args.index_dir) / "triples_cache.jsonl")
    if not args.skip_kg:
        kg_model = resolve_kg_model(args.llm_provider, args.kg_model)
        logging.info(
            "build_index.progress stage=create_kg_client provider=%s model=%s cache=%s batch_inference=%s",
            args.llm_provider,
            kg_model,
            kg_cache,
            batch_inference_enabled,
        )
        batch_model = kg_model
        if batch_inference_enabled:
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
            kg_batch_job_client = create_batch_job_client(batch_config)
        else:
            kg_llm_client = create_llm_client(
                args.llm_provider,
                api_key=args.api_key,
                model=batch_model,
                base_url=resolve_llm_base_url(args.llm_provider, args.base_url),
                timeout_seconds=args.timeout_seconds,
            )

    logging.info("build_index.progress stage=build_bundle_start")
    bundle = build_index_bundle(
        chunks,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        skip_kg=args.skip_kg,
        kg_llm_client=kg_llm_client,
        kg_batch_job_client=kg_batch_job_client,
        kg_batch_job_dir=args.batch_work_dir,
        kg_batch_job_name="agentic-rag-kg",
        kg_batch_model=batch_model,
        kg_batch_poll_interval_seconds=args.batch_poll_interval,
        kg_batch_wait_timeout_seconds=args.batch_wait_timeout,
        kg_cache_path=kg_cache,
        kg_max_concurrency=args.max_concurrency,
    )
    bundle["manifest"]["reranker_model"] = args.reranker_model
    logging.info("build_index.progress stage=save_bundle_start index_dir=%s", args.index_dir)
    save_index_bundle(bundle, args.index_dir)
    logging.info("build_index.progress stage=done index_dir=%s", args.index_dir)
    print(f"chunk_count={bundle['manifest']['chunk_count']}")
    print(f"faiss={bundle['manifest']['faiss']}")
    print(f"bm25={bundle['manifest']['bm25']}")
    print(f"knowledge_graph={bundle['manifest']['knowledge_graph']}")
    print(f"max_concurrency={args.max_concurrency}")
    print(f"index_dir={args.index_dir}")


if __name__ == "__main__":
    main()
