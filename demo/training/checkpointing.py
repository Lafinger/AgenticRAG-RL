from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Iterable


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
TRACE_CHECKPOINT_STATE_FILE = "trace_checkpoint_state.pt"

TRAINER_REQUIRED_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)

TRACE_REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    TRACE_CHECKPOINT_STATE_FILE,
)


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    step: int


def list_numbered_checkpoints(output_dir: str | Path) -> list[CheckpointInfo]:
    root = Path(output_dir)
    if not root.exists():
        return []

    checkpoints: list[CheckpointInfo] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append(CheckpointInfo(path=path, step=int(match.group(1))))
    return sorted(checkpoints, key=lambda item: item.step)


def latest_numbered_checkpoint(output_dir: str | Path) -> CheckpointInfo | None:
    checkpoints = list_numbered_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def require_latest_checkpoint(output_dir: str | Path) -> CheckpointInfo:
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"严格断点续训失败：训练输出目录不存在：{root}。首次训练请显式使用 --overwrite。"
        )
    if not root.is_dir():
        raise NotADirectoryError(f"严格断点续训失败：训练输出路径不是目录：{root}。")

    checkpoint = latest_numbered_checkpoint(root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"严格断点续训失败：{root} 下没有 checkpoint-* 目录。重新训练请显式使用 --overwrite。"
        )
    return checkpoint


def require_files(path: str | Path, filenames: Iterable[str], *, context: str) -> None:
    root = Path(path)
    missing = [filename for filename in filenames if not (root / filename).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"严格断点续训失败：{context} 缺少必要文件：{joined}。")


def require_rng_state_file(path: str | Path, *, context: str) -> None:
    root = Path(path)
    if not any(root.glob("rng_state*.pth")):
        raise FileNotFoundError(f"严格断点续训失败：{context} 缺少 RNG 状态文件 rng_state*.pth。")


def resolve_trainer_resume_checkpoint(output_dir: str | Path) -> Path:
    checkpoint = require_latest_checkpoint(output_dir)
    require_files(checkpoint.path, TRAINER_REQUIRED_FILES, context=str(checkpoint.path))
    require_rng_state_file(checkpoint.path, context=str(checkpoint.path))
    return checkpoint.path


def resolve_trace_resume_checkpoint(output_dir: str | Path) -> Path:
    checkpoint = require_latest_checkpoint(output_dir)
    require_files(checkpoint.path, TRACE_REQUIRED_FILES, context=str(checkpoint.path))
    return checkpoint.path


def reset_output_dir(output_dir: str | Path) -> None:
    root = Path(output_dir)
    resolved = root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"拒绝覆盖根目录：{root}")
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(f"--overwrite 只能覆盖目录，当前路径不是目录：{root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def validate_trace_checkpoint_state(payload: object, *, checkpoint_path: str | Path) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 trace checkpoint 状态不是 dict。")

    required_keys = {
        "version",
        "global_step",
        "completed_micro_batches",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "run_config",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 trace checkpoint 状态缺少字段：{joined}。")
    if int(payload["global_step"]) < 0:
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 global_step 非法。")
    if int(payload["completed_micro_batches"]) < 0:
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 completed_micro_batches 非法。")
    if not isinstance(payload["rng_state"], dict):
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 rng_state 非法。")
    if not isinstance(payload["run_config"], dict):
        raise ValueError(f"严格断点续训失败：{checkpoint_path} 的 run_config 非法。")
    return payload
