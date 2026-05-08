from __future__ import annotations

import argparse
import inspect
import json
import sys
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.checkpointing import reset_output_dir, resolve_trainer_resume_checkpoint
from training.monitoring import (
    checkpoint_step_from_path,
    configure_swanlab_environment,
    create_swanlab_history_replay_callback,
    create_jsonl_metrics_callback,
    is_swanlab_enabled,
    normalize_report_to,
    normalize_swanlab_mode,
    normalize_swanlab_resume,
    require_swanlab,
    resolve_swanlab_run_id,
    should_replay_swanlab_history,
)
from training.sft_label_mask import tokenize_chat_with_assistant_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT LoRA with Unsloth.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--data-path")
    parser.add_argument("--eval-data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--report-to", help="Comma-separated Trainer integrations, for example: swanlab.")
    parser.add_argument("--swanlab-project", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", help="SwanLab workspace name.")
    parser.add_argument("--swanlab-mode", help="SwanLab mode, for example: cloud.")
    parser.add_argument("--swanlab-logdir", help="SwanLab local log directory.")
    parser.add_argument("--swanlab-experiment-name", help="SwanLab experiment name.")
    parser.add_argument("--swanlab-run-id", help="Explicit SwanLab run id to resume or create.")
    parser.add_argument("--swanlab-resume", choices=["allow", "must", "never"], help="SwanLab resume mode.")
    parser.add_argument(
        "--swanlab-replay-history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay local metrics history to SwanLab before resuming training.",
    )
    parser.add_argument("--metrics-output", help="JSONL metrics output path for the local dashboard.")
    parser.add_argument("--disable-jsonl-metrics", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", dest="train_mode", action="store_const", const="resume", default="resume")
    mode_group.add_argument("--overwrite", dest="train_mode", action="store_const", const="overwrite")
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


def build_masked_dataset_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Record {index} is missing messages.")
        tools = record.get("tools") if isinstance(record.get("tools"), list) else None
        sample = tokenize_chat_with_assistant_labels(tokenizer, messages, tools=tools, max_length=max_seq_length)
        rendered.append(
            {
                "input_ids": sample.input_ids,
                "attention_mask": sample.attention_mask,
                "labels": sample.labels,
            }
        )
    return rendered


def sft_config_supports_argument(sft_config_class: Any, name: str) -> bool:
    try:
        return name in inspect.signature(sft_config_class.__init__).parameters
    except (TypeError, ValueError):
        return False


def add_eval_training_args(training_config: dict[str, Any], config: dict[str, Any], sft_config_class: Any) -> None:
    eval_steps = int(config.get("eval_steps") or config.get("logging_steps", 5))
    strategy_key = (
        "eval_strategy"
        if sft_config_supports_argument(sft_config_class, "eval_strategy")
        else "evaluation_strategy"
    )
    training_config[strategy_key] = "steps"
    training_config["eval_steps"] = eval_steps
    if sft_config_supports_argument(sft_config_class, "do_eval"):
        training_config["do_eval"] = True


def train_with_resume_mode(trainer: Any, resume_checkpoint: Path | None) -> Any:
    if resume_checkpoint is None:
        return trainer.train()
    return trainer.train(resume_from_checkpoint=str(resume_checkpoint))


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
    if args.eval_data_path:
        config["eval_data_path"] = args.eval_data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir

    model_name = str(config["model_name_or_path"])
    max_seq_length = int(config.get("max_seq_length", 2048))
    output_dir = project_path(config["output_dir"])
    output_path = Path(output_dir)
    report_to = normalize_report_to(args.report_to if args.report_to is not None else config.get("report_to", ["swanlab"]))
    metrics_output = (
        project_path(args.metrics_output)
        if args.metrics_output
        else project_path(config["metrics_output"])
        if config.get("metrics_output")
        else str(Path(output_dir) / "metrics.jsonl")
    )
    swanlab_project = args.swanlab_project or config.get("swanlab_project") or "agentic-rag-rl"
    swanlab_workspace = args.swanlab_workspace or config.get("swanlab_workspace")
    swanlab_mode = normalize_swanlab_mode(args.swanlab_mode or config.get("swanlab_mode") or "cloud") or "cloud"
    swanlab_logdir = project_path(args.swanlab_logdir or config.get("swanlab_logdir") or "./training/swanlab")
    swanlab_experiment_name = (
        args.swanlab_experiment_name
        or config.get("swanlab_experiment_name")
        or Path(output_dir).name
    )
    resume_checkpoint = (
        resolve_trainer_resume_checkpoint(output_path)
        if args.train_mode == "resume"
        else None
    )
    if args.train_mode == "overwrite":
        reset_output_dir(output_path)

    swanlab_run_id: str | None = None
    swanlab_resume = normalize_swanlab_resume(args.swanlab_resume or config.get("swanlab_resume") or "allow") or "allow"
    swanlab_replay_setting = (
        args.swanlab_replay_history
        if args.swanlab_replay_history is not None
        else config.get("swanlab_replay_history", "auto")
    )
    replay_swanlab_history = should_replay_swanlab_history(
        swanlab_replay_setting,
        resume_checkpoint=resume_checkpoint,
    )
    if is_swanlab_enabled(report_to):
        swanlab_run_id = resolve_swanlab_run_id(
            output_path,
            explicit_run_id=args.swanlab_run_id,
            reset=args.train_mode == "overwrite",
        )
        configure_swanlab_environment(
            project=swanlab_project,
            workspace=swanlab_workspace,
            mode=swanlab_mode,
            logdir=swanlab_logdir,
            experiment_name=swanlab_experiment_name,
            run_id=swanlab_run_id,
            resume=swanlab_resume,
        )
        require_swanlab(report_to)

    Dataset, SFTConfig, SFTTrainer, FastLanguageModel = require_unsloth_stack()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    if bool(config.get("packing", False)):
        raise ValueError("assistant-only label mask does not support packing; set packing: false.")
    dataset = Dataset.from_list(build_masked_dataset_records(records, tokenizer, max_seq_length=max_seq_length))
    eval_dataset = None
    if config.get("eval_data_path"):
        eval_records = load_jsonl(project_path(config["eval_data_path"]))
        eval_dataset = Dataset.from_list(
            build_masked_dataset_records(eval_records, tokenizer, max_seq_length=max_seq_length)
        )

    training_config: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_strategy": "steps",
        "save_steps": int(config.get("save_steps", 45)),
        "seed": int(config.get("seed", 3407)),
        "report_to": report_to,
        "max_length": max_seq_length,
        "max_grad_norm": float(config.get("max_grad_norm") if config.get("max_grad_norm") is not None else 1.0),
        "packing": False,
        "dataset_num_proc": None,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
    }
    if is_swanlab_enabled(report_to):
        training_config["run_name"] = swanlab_experiment_name
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
    elif config.get("max_steps") is not None:
        training_config["max_steps"] = int(config["max_steps"])
    if config.get("warmup_steps") is not None:
        training_config["warmup_steps"] = int(config["warmup_steps"])
    elif config.get("warmup_ratio") is not None:
        training_config["warmup_ratio"] = float(config["warmup_ratio"])
    if eval_dataset is not None:
        add_eval_training_args(training_config, config, SFTConfig)

    training_args = SFTConfig(**training_config)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": dataset,
        "args": training_args,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = SFTTrainer(**trainer_kwargs)
    if is_swanlab_enabled(report_to) and replay_swanlab_history and swanlab_run_id:
        trainer.add_callback(
            create_swanlab_history_replay_callback(
                output_dir=output_path,
                run_id=swanlab_run_id,
                max_step=checkpoint_step_from_path(resume_checkpoint),
                metrics_path=metrics_output,
                checkpoint_path=resume_checkpoint,
            )
        )
    if not args.disable_jsonl_metrics:
        trainer.add_callback(create_jsonl_metrics_callback(metrics_output, reset=args.train_mode == "overwrite"))
    train_with_resume_mode(trainer, resume_checkpoint)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    freeze_support()
    main()
