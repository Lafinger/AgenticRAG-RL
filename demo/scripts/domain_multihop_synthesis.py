from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.env import load_env_file
from agentic_rag_rl.io import load_chunks, load_jsonl, write_jsonl
from agentic_rag_rl.llm_client import (
    DEFAULT_LLM_PROVIDER,
    create_llm_client,
    get_doubao_base_url,
    get_doubao_thinking_model,
)
from agentic_rag_rl.synthesis import synthesize_multihop_examples


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize multi-hop QA examples from seeds.")
    parser.add_argument("--seeds", default=str(ROOT / "data" / "novel_eval" / "seeds.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel_eval" / "qa_pairs.jsonl"))
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=["doubao"])
    parser.add_argument("--merge-model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--disable-llm-merge", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    _configure_logging(args.log_level)
    load_env_file(args.env_file)
    logging.info(
        "domain_multihop_synthesis.start seeds=%s corpus=%s output=%s target_count=%s llm_merge=%s merge_model=%s",
        args.seeds,
        args.corpus,
        args.output,
        args.target_count,
        not args.disable_llm_merge,
        get_doubao_thinking_model(args.merge_model),
    )
    seeds = load_jsonl(args.seeds)
    chunks = load_chunks(args.corpus)
    logging.info("domain_multihop_synthesis.loaded_inputs seed_count=%s chunk_count=%s", len(seeds), len(chunks))
    merge_llm_client = None
    if not args.disable_llm_merge:
        merge_llm_client = create_llm_client(
            args.llm_provider,
            api_key=args.api_key,
            model=get_doubao_thinking_model(args.merge_model),
            base_url=get_doubao_base_url(args.base_url),
        )
    examples = synthesize_multihop_examples(
        seeds,
        chunks,
        target_count=args.target_count,
        merge_llm_client=merge_llm_client,
    )
    write_jsonl(examples, args.output)
    logging.info("domain_multihop_synthesis.done output=%s multihop_count=%s", args.output, len(examples))
    print(f"multihop_count={len(examples)}")


if __name__ == "__main__":
    main()
