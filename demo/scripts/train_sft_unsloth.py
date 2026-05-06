from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.monitoring import create_jsonl_metrics_callback, normalize_report_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT LoRA with Unsloth.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
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


def require_unsloth_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "当前环境未安装 Unsloth SFT 训练依赖。请先在本环境安装 unsloth、trl、datasets 等训练栈后再运行。"
        ) from exc
    return Dataset, SFTConfig, SFTTrainer, FastLanguageModel


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def render_messages(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Record {index} is missing messages.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        rendered.append({"text": text})
    return rendered


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    config = load_config(args.config)
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir

    Dataset, SFTConfig, SFTTrainer, FastLanguageModel = require_unsloth_stack()

    model_name = str(config["model_name_or_path"])
    max_seq_length = int(config.get("max_seq_length", 2048))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.get("lora_rank", 64)),
        target_modules=list(config.get("lora_target_modules", [])),
        lora_alpha=int(config.get("lora_alpha", config.get("lora_rank", 64))),
        lora_dropout=float(config.get("lora_dropout", 0)),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(config.get("seed", 3407)),
    )

    records = load_jsonl(project_path(config["data_path"]), max_samples=args.max_samples)
    dataset = Dataset.from_list(render_messages(records, tokenizer))

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
    training_config: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 45)),
        "seed": int(config.get("seed", 3407)),
        "report_to": report_to,
        "logging_dir": logging_dir,
        "max_seq_length": max_seq_length,
        "dataset_text_field": "text",
        "packing": bool(config.get("packing", False)),
        "dataset_num_proc": None,
    }
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
    elif config.get("max_steps") is not None:
        training_config["max_steps"] = int(config["max_steps"])
    if config.get("warmup_steps") is not None:
        training_config["warmup_steps"] = int(config["warmup_steps"])
    elif config.get("warmup_ratio") is not None:
        training_config["warmup_ratio"] = float(config["warmup_ratio"])

    training_args = SFTConfig(**training_config)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    if not args.disable_jsonl_metrics:
        trainer.add_callback(create_jsonl_metrics_callback(metrics_output))
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    freeze_support()
    main()
