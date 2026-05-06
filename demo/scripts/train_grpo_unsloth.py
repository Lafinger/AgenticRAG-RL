from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.monitoring import create_jsonl_metrics_callback, normalize_report_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO/RL with Unsloth + TRL.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_grpo.yaml"))
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--train-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--report-to", help="Comma-separated Trainer integrations, for example: tensorboard.")
    parser.add_argument("--logging-dir", help="TensorBoard event output directory.")
    parser.add_argument("--metrics-output", help="JSONL metrics output path for the local dashboard.")
    parser.add_argument("--disable-jsonl-metrics", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def project_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def require_grpo_stack() -> tuple[Any, Any, Any]:
    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "当前环境未安装 Unsloth GRPO 训练依赖。请先安装 unsloth、trl、datasets 后再运行。"
        ) from exc
    return load_dataset, GRPOConfig, GRPOTrainer, FastLanguageModel


def reward_func(completions: list[Any], reward_model: list[dict[str, Any]] | None = None, **kwargs: Any) -> list[float]:
    from training.reward_agentic_rag import compute_score

    del kwargs
    reward_model = reward_model or [{} for _ in completions]
    scores: list[float] = []
    for completion, reward_payload in zip(completions, reward_model):
        if isinstance(completion, list):
            solution = "".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in completion)
        else:
            solution = str(completion)
        ground_truth = reward_payload.get("ground_truth", reward_payload) if isinstance(reward_payload, dict) else reward_payload
        scores.append(float(compute_score(solution_str=solution, ground_truth=ground_truth, extra_info={})))
    return scores


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.train_file:
        config["train_file"] = args.train_file
    if args.output_dir:
        config["output_dir"] = args.output_dir

    train_file = project_path(config["train_file"])
    if train_file.lower().endswith(".parquet"):
        dataset_kind = "parquet"
    elif train_file.lower().endswith(".json") or train_file.lower().endswith(".jsonl"):
        dataset_kind = "json"
    else:
        raise SystemExit("Unsloth GRPO 输入目前只支持 parquet/json/jsonl。请先转换训练数据。")

    os.environ["AGENTIC_RAG_REWARD_VERSION"] = str(config.get("reward_version", "v6a"))
    load_dataset, GRPOConfig, GRPOTrainer, FastLanguageModel = require_grpo_stack()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(config["model_name_or_path"]),
        max_seq_length=int(config.get("max_prompt_length", 1024)) + int(config.get("max_completion_length", 1024)),
        load_in_4bit=True,
    )
    dataset = load_dataset(dataset_kind, data_files=train_file, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_dir = project_path(config["output_dir"])
    report_to = normalize_report_to(args.report_to if args.report_to is not None else config.get("report_to", ["tensorboard"]))
    logging_dir = (
        project_path(args.logging_dir)
        if args.logging_dir
        else project_path(config["logging_dir"])
        if config.get("logging_dir")
        else str(Path(output_dir) / "tensorboard")
    )
    metrics_output = (
        project_path(args.metrics_output)
        if args.metrics_output
        else project_path(config["metrics_output"])
        if config.get("metrics_output")
        else str(Path(output_dir) / "metrics.jsonl")
    )
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 5e-6)),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        max_prompt_length=int(config.get("max_prompt_length", 1024)),
        max_completion_length=int(config.get("max_completion_length", 1024)),
        num_generations=int(config.get("num_generations", 4)),
        temperature=float(config.get("temperature", 0.7)),
        seed=int(config.get("seed", 3407)),
        report_to=report_to,
        logging_dir=logging_dir,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=dataset,
    )
    if not args.disable_jsonl_metrics:
        trainer.add_callback(create_jsonl_metrics_callback(metrics_output))
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
