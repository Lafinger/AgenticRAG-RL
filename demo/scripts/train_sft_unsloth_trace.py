from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.checkpointing import (
    TRACE_CHECKPOINT_STATE_FILE,
    reset_output_dir,
    resolve_trace_resume_checkpoint,
    validate_trace_checkpoint_state,
)
from training.monitoring import (
    SwanLabScalarLogger,
    build_training_progress_metrics,
    checkpoint_step_from_path,
    configure_swanlab_environment,
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
    parser = argparse.ArgumentParser(description="Train SFT LoRA with exact optimizer-step sample tracing.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--data-path")
    parser.add_argument("--eval-data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--trace-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--report-to", help="Comma-separated integrations for the custom loop, for example: swanlab.")
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
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--optim", choices=["adamw_8bit", "adamw_torch"], default="adamw_8bit")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable random sampling for deterministic line-range debugging.")
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


def apply_cli_config_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.data_path:
        config["data_path"] = args.data_path
    if args.eval_data_path:
        config["eval_data_path"] = args.eval_data_path
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        config["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.save_steps is not None:
        config["save_steps"] = args.save_steps
    if args.logging_steps is not None:
        config["logging_steps"] = args.logging_steps
    if args.eval_steps is not None:
        config["eval_steps"] = args.eval_steps
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def require_training_stack() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
        from transformers import get_linear_schedule_with_warmup
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "当前环境未安装 Unsloth SFT 训练依赖。请先安装 torch、unsloth、transformers 等训练栈后再运行。"
        ) from exc
    return torch, DataLoader, Dataset, FastLanguageModel, RandomSampler, SequentialSampler, get_linear_schedule_with_warmup


def load_records(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    records: list[dict[str, Any]] = []
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"JSON eval file must contain a list: {path}")
        records = [record for record in payload if isinstance(record, dict)]
    else:
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    if max_samples is not None:
        records = records[:max_samples]
    return records


@dataclass
class TraceSample:
    sample_id: str
    source_line: int
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_length: int
    supervised_token_count: int
    metadata: dict[str, Any]


def build_samples(records: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> list[TraceSample]:
    samples: list[TraceSample] = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Record {index} is missing messages.")
        tools = record.get("tools") if isinstance(record.get("tools"), list) else None
        masked = tokenize_chat_with_assistant_labels(tokenizer, messages, tools=tools, max_length=max_seq_length)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        sample_id = str(metadata.get("sample_id") or f"sft_{index:06d}")
        samples.append(
            TraceSample(
                sample_id=sample_id,
                source_line=index,
                text=masked.text,
                input_ids=masked.input_ids,
                attention_mask=masked.attention_mask,
                labels=masked.labels,
                token_length=masked.token_length,
                supervised_token_count=masked.supervised_token_count,
                metadata=metadata,
            )
        )
    return samples


def first_model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if getattr(parameter, "device", None) is not None and parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Cannot find model device.")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\r\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def create_optimizer(optim_name: str, params: list[Any], learning_rate: float) -> Any:
    if optim_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=learning_rate)
        except ModuleNotFoundError:
            print("bitsandbytes is not installed; falling back to torch.optim.AdamW.", file=sys.stderr)

    import torch

    return torch.optim.AdamW(params, lr=learning_rate)


TRACE_RESUME_CONFIG_KEYS = (
    "model_name_or_path",
    "data_path",
    "sample_count",
    "max_seq_length",
    "batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "total_steps",
    "learning_rate",
    "warmup_steps",
    "seed",
    "shuffle",
    "eval_steps",
    "max_grad_norm",
    "optim",
)


def assert_trace_resume_config_compatible(saved_config: dict[str, Any], current_config: dict[str, Any]) -> None:
    mismatches = []
    for key in TRACE_RESUME_CONFIG_KEYS:
        if saved_config.get(key) != current_config.get(key):
            mismatches.append(f"{key}: checkpoint={saved_config.get(key)!r}, current={current_config.get(key)!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"严格断点续训失败：当前训练配置与 checkpoint 不一致：{details}。")


def build_trace_checkpoint_state(
    torch_module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    global_step: int,
    completed_micro_batches: int,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    cuda_module = getattr(torch_module, "cuda", None)
    cuda_rng_state = (
        cuda_module.get_rng_state_all()
        if cuda_module is not None and cuda_module.is_available()
        else None
    )
    return {
        "version": 1,
        "global_step": int(global_step),
        "completed_micro_batches": int(completed_micro_batches),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": {
            "torch": torch_module.get_rng_state(),
            "cuda": cuda_rng_state,
        },
        "run_config": dict(run_config),
    }


def load_trace_checkpoint_state(torch_module: Any, checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / TRACE_CHECKPOINT_STATE_FILE
    try:
        payload = torch_module.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch_module.load(state_path, map_location="cpu")
    return validate_trace_checkpoint_state(payload, checkpoint_path=state_path)


def restore_trace_checkpoint_state(torch_module: Any, optimizer: Any, scheduler: Any, state: dict[str, Any]) -> None:
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    rng_state = state["rng_state"]
    if "torch" not in rng_state:
        raise ValueError("严格断点续训失败：trace checkpoint 缺少 torch RNG 状态。")
    torch_module.set_rng_state(rng_state["torch"])
    cuda_rng_state = rng_state.get("cuda")
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_rng_state is not None and cuda_module is not None and cuda_module.is_available():
        cuda_module.set_rng_state_all(cuda_rng_state)


def save_trace_checkpoint(
    torch_module: Any,
    checkpoint_dir: Path,
    model: Any,
    tokenizer: Any,
    state: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    torch_module.save(state, checkpoint_dir / TRACE_CHECKPOINT_STATE_FILE)


def collate_trace_features(features: list[dict[str, Any]], tokenizer: Any, torch_module: Any) -> dict[str, Any]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    labels = []
    for feature in features:
        ids = list(feature["input_ids"])
        mask = list(feature["attention_mask"])
        feature_labels = list(feature["labels"])
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)
        labels.append(feature_labels + [-100] * pad_len)

    return {
        "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
        "attention_mask": torch_module.tensor(attention_mask, dtype=torch_module.long),
        "labels": torch_module.tensor(labels, dtype=torch_module.long),
        "trace_items": [
            {
                "sample_id": feature["sample_id"],
                "source_line": feature["source_line"],
                "token_length": feature["token_length"],
                "truncated": feature["truncated"],
                "supervised_token_count": feature["supervised_token_count"],
                "question": feature["question"],
                "answer": feature["answer"],
                "hop_count": feature["hop_count"],
                "gold_chunks": feature["gold_chunks"],
            }
            for feature in features
        ],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    config = load_config(args.config)
    apply_cli_config_overrides(config, args)

    model_name = str(config["model_name_or_path"])
    max_seq_length = int(config.get("max_seq_length", 2048))
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        configured_output_dir = project_path(config["output_dir"])
        output_dir = configured_output_dir.with_name(f"{configured_output_dir.name}_trace")
    resume_checkpoint = (
        resolve_trace_resume_checkpoint(output_dir)
        if args.train_mode == "resume"
        else None
    )
    trace_output = project_path(args.trace_output) if args.trace_output else output_dir / "step_sample_trace.jsonl"
    metrics_output = project_path(args.metrics_output) if args.metrics_output else output_dir / "trace_metrics.jsonl"
    state_output = output_dir / "trace_training_state.json"
    report_to = normalize_report_to(args.report_to if args.report_to is not None else config.get("report_to", ["swanlab"]))
    swanlab_project = args.swanlab_project or config.get("swanlab_project") or "agentic-rag-rl"
    swanlab_workspace = args.swanlab_workspace or config.get("swanlab_workspace")
    swanlab_mode = normalize_swanlab_mode(args.swanlab_mode or config.get("swanlab_mode") or "cloud") or "cloud"
    swanlab_logdir = project_path(args.swanlab_logdir or config.get("swanlab_logdir") or "./training/swanlab")
    swanlab_experiment_name = (
        args.swanlab_experiment_name
        or config.get("swanlab_experiment_name")
        or output_dir.name
    )
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

    (
        torch,
        DataLoader,
        TorchDataset,
        FastLanguageModel,
        RandomSampler,
        SequentialSampler,
        get_linear_schedule_with_warmup,
    ) = require_training_stack()
    if args.train_mode == "overwrite":
        reset_output_dir(output_dir)
        for stale_path in (trace_output, metrics_output):
            if stale_path.exists():
                stale_path.unlink()
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    swanlab_run_id: str | None = None
    if is_swanlab_enabled(report_to):
        swanlab_run_id = resolve_swanlab_run_id(
            output_dir,
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

    resume_state = (
        load_trace_checkpoint_state(torch, resume_checkpoint)
        if resume_checkpoint is not None
        else None
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(resume_checkpoint) if resume_checkpoint is not None else model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if resume_checkpoint is None:
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
    model.train()

    records = load_records(project_path(config["data_path"]), max_samples=args.max_samples)
    samples = build_samples(records, tokenizer, max_seq_length=max_seq_length)

    class TraceDataset(TorchDataset):  # type: ignore[misc, valid-type]
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = samples[index]
            metadata = sample.metadata
            return {
                "input_ids": sample.input_ids,
                "attention_mask": sample.attention_mask,
                "labels": sample.labels,
                "sample_id": sample.sample_id,
                "source_line": sample.source_line,
                "token_length": sample.token_length,
                "truncated": sample.token_length > max_seq_length,
                "supervised_token_count": sample.supervised_token_count,
                "question": str(metadata.get("final_question", "")),
                "answer": str(metadata.get("final_answer", "")),
                "hop_count": metadata.get("hop_count"),
                "gold_chunks": metadata.get("gold_chunks", []),
            }

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_trace_features(features, tokenizer, torch)

    seed = int(config.get("seed", 3407))
    batch_size = int(config.get("per_device_train_batch_size", 2))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 8))
    num_epochs = float(config.get("num_train_epochs", 3))
    learning_rate = float(config.get("learning_rate", 5e-5))
    logging_steps = int(config.get("logging_steps", 5))
    save_steps = int(config.get("save_steps", 45))
    warmup_steps = int(config.get("warmup_steps", 0) or 0)
    max_grad_norm = float(config.get("max_grad_norm") if config.get("max_grad_norm") is not None else 1.0)

    dataset = TraceDataset()
    micro_batches_per_epoch = math.ceil(len(dataset) / batch_size)
    update_steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    max_steps = args.max_steps if args.max_steps is not None else config.get("max_steps")
    total_steps = int(max_steps) if max_steps else int(math.ceil(update_steps_per_epoch * num_epochs))
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有可训练参数，无法进行 LoRA SFT 训练。")
    optimizer = create_optimizer(args.optim, trainable_params, learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if resume_state is not None:
        restore_trace_checkpoint_state(torch, optimizer, scheduler, resume_state)

    eval_samples: list[TraceSample] = []
    configured_eval_steps = config.get("eval_steps")
    eval_steps = int(configured_eval_steps) if configured_eval_steps is not None else None
    eval_data_path = config.get("eval_data_path")
    if eval_data_path:
        eval_records = load_records(project_path(eval_data_path))
        eval_samples = build_samples(eval_records, tokenizer, max_seq_length=max_seq_length)
        if eval_steps is None:
            eval_steps = logging_steps

    def run_eval() -> float | None:
        if not eval_samples:
            return None
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for sample in eval_samples:
                device = first_model_device(model)
                encoded = {
                    "input_ids": torch.tensor([sample.input_ids], dtype=torch.long, device=device),
                    "attention_mask": torch.tensor([sample.attention_mask], dtype=torch.long, device=device),
                }
                labels = torch.tensor([sample.labels], dtype=torch.long, device=device)
                outputs = model(**encoded, labels=labels)
                losses.append(float(outputs.loss.detach().cpu()))
        model.train()
        return sum(losses) / max(len(losses), 1)

    run_config = {
        "model_name_or_path": model_name,
        "data_path": str(project_path(config["data_path"])),
        "eval_data_path": str(project_path(eval_data_path)) if eval_data_path else None,
        "output_dir": str(output_dir),
        "trace_output": str(trace_output),
        "metrics_output": str(metrics_output),
        "sample_count": len(samples),
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "samples_per_optimizer_step": batch_size * gradient_accumulation_steps,
        "num_train_epochs": num_epochs,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "shuffle": not args.no_shuffle,
        "eval_steps": eval_steps,
        "max_grad_norm": max_grad_norm,
        "optim": args.optim,
        "report_to": report_to,
        "swanlab_project": swanlab_project if is_swanlab_enabled(report_to) else None,
        "swanlab_workspace": swanlab_workspace if is_swanlab_enabled(report_to) else None,
        "swanlab_mode": swanlab_mode if is_swanlab_enabled(report_to) else None,
        "swanlab_logdir": str(swanlab_logdir) if is_swanlab_enabled(report_to) else None,
        "swanlab_experiment_name": swanlab_experiment_name if is_swanlab_enabled(report_to) else None,
        "swanlab_run_id": swanlab_run_id if is_swanlab_enabled(report_to) else None,
        "swanlab_resume": swanlab_resume if is_swanlab_enabled(report_to) else None,
        "swanlab_replay_history": replay_swanlab_history if is_swanlab_enabled(report_to) else None,
    }
    if resume_state is not None:
        assert_trace_resume_config_compatible(resume_state["run_config"], run_config)
    write_json(output_dir / "trace_run_config.json", run_config)
    swanlab_logger = (
        SwanLabScalarLogger(
            project=swanlab_project,
            workspace=swanlab_workspace,
            experiment_name=swanlab_experiment_name,
            mode=swanlab_mode,
            logdir=swanlab_logdir,
            config=run_config,
            run_id=swanlab_run_id,
            resume=swanlab_resume,
        )
        if is_swanlab_enabled(report_to)
        else None
    )
    if swanlab_logger is not None and replay_swanlab_history:
        replay_until_step = None
        if resume_state is not None:
            replay_until_step = checkpoint_step_from_path(resume_checkpoint) or int(resume_state["global_step"])
        swanlab_logger.replay_history(
            output_dir,
            max_step=replay_until_step,
            metrics_path=metrics_output,
            checkpoint_path=resume_checkpoint,
        )

    resume_completed_micro_batches = (
        int(resume_state["completed_micro_batches"])
        if resume_state is not None
        else 0
    )
    global_step = int(resume_state["global_step"]) if resume_state is not None else 0
    completed_micro_batches = resume_completed_micro_batches
    running_loss = 0.0
    trace_buffer: list[dict[str, Any]] = []
    device = first_model_device(model)
    skipped_resume_micro_batches = 0

    for epoch_index in range(int(math.ceil(num_epochs))):
        if args.no_shuffle:
            sampler = SequentialSampler(dataset)
        else:
            generator = torch.Generator()
            generator.manual_seed(seed + epoch_index)
            sampler = RandomSampler(dataset, generator=generator)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate, num_workers=0)

        for micro_batch_index, batch in enumerate(dataloader, start=1):
            if global_step >= total_steps:
                break
            if skipped_resume_micro_batches < resume_completed_micro_batches:
                skipped_resume_micro_batches += 1
                continue

            trace_items = batch.pop("trace_items")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / gradient_accumulation_steps
            loss.backward()
            running_loss += float(raw_loss.detach().cpu())
            completed_micro_batches += 1
            trace_buffer.append(
                {
                    "micro_batch_index": completed_micro_batches,
                    "epoch_index": epoch_index + 1,
                    "epoch_progress": epoch_index + micro_batch_index / max(micro_batches_per_epoch, 1),
                    "items": trace_items,
                }
            )

            should_step = completed_micro_batches % gradient_accumulation_steps == 0
            is_epoch_end = micro_batch_index == micro_batches_per_epoch
            if not should_step and not is_epoch_end:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            avg_loss = running_loss / max(len(trace_buffer), 1)
            eval_loss = run_eval() if eval_steps and global_step % eval_steps == 0 else None
            flat_items = [item for micro_batch in trace_buffer for item in micro_batch["items"]]
            progress_metrics = build_training_progress_metrics(global_step, total_steps)
            trace_payload = {
                "step": global_step,
                "epoch": epoch_index + micro_batch_index / max(micro_batches_per_epoch, 1),
                "loss": avg_loss,
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                "eval_loss": eval_loss,
                **progress_metrics,
                "micro_batches": trace_buffer,
                "sample_ids": [item["sample_id"] for item in flat_items],
                "source_lines": [item["source_line"] for item in flat_items],
            }
            append_jsonl(trace_output, trace_payload)
            append_jsonl(
                metrics_output,
                {
                    "step": global_step,
                    "epoch": trace_payload["epoch"],
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "grad_norm": trace_payload["grad_norm"],
                    "eval_loss": eval_loss,
                    **progress_metrics,
                    "sample_count": len(flat_items),
                    "source_lines": trace_payload["source_lines"],
                    "sample_ids": trace_payload["sample_ids"],
                },
            )
            if swanlab_logger is not None:
                swanlab_logger.log(trace_payload, global_step)

            if global_step % logging_steps == 0 or global_step == 1:
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": round(avg_loss, 6),
                            "grad_norm": round(trace_payload["grad_norm"], 6),
                            "learning_rate": current_lr,
                            "eval_loss": eval_loss,
                        },
                        ensure_ascii=False,
                    )
                )

            if save_steps > 0 and global_step % save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                save_trace_checkpoint(
                    torch,
                    checkpoint_dir,
                    model,
                    tokenizer,
                    build_trace_checkpoint_state(
                        torch,
                        optimizer,
                        scheduler,
                        global_step=global_step,
                        completed_micro_batches=completed_micro_batches,
                        run_config=run_config,
                    ),
                )

            write_json(
                state_output,
                {
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "progress_percent": progress_metrics.get("progress_percent"),
                    "epoch": trace_payload["epoch"],
                    "last_loss": avg_loss,
                    "last_grad_norm": trace_payload["grad_norm"],
                    "last_learning_rate": current_lr,
                    "last_eval_loss": eval_loss,
                    "trace_output": str(trace_output),
                    "metrics_output": str(metrics_output),
                },
            )
            running_loss = 0.0
            trace_buffer = []

        if global_step >= total_steps:
            break

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if swanlab_logger is not None:
        swanlab_logger.finish()
    print(json.dumps({"status": "done", "output_dir": str(output_dir), "trace_output": str(trace_output)}, ensure_ascii=False))


if __name__ == "__main__":
    freeze_support()
    main()
