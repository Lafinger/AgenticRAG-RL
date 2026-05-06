from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\r\n")


def normalize_report_to(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "false", "off", "no"}:
            return []
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def is_tensorboard_enabled(report_to: Any) -> bool:
    return any(item.lower() in {"tensorboard", "all"} for item in normalize_report_to(report_to))


def create_jsonl_metrics_callback(path: str | Path, reset: bool = True) -> Any:
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise SystemExit("当前环境未安装 transformers，无法启用 JSONL Trainer 指标回调。") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and output_path.exists():
        output_path.unlink()

    class JsonlMetricsCallback(TrainerCallback):
        def _safe_append(self, payload: dict[str, Any]) -> None:
            try:
                append_jsonl(output_path, payload)
            except OSError as exc:
                print(f"JSONL metrics write failed: {exc}")

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            self._safe_append(
                {
                    "event": "train_begin",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": int(getattr(state, "global_step", 0) or 0),
                    "epoch": getattr(state, "epoch", None),
                },
            )

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            del args, control, kwargs
            logs = logs or {}
            payload: dict[str, Any] = {
                "event": "log",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": int(getattr(state, "global_step", 0) or 0),
                "epoch": getattr(state, "epoch", None),
            }
            for key, value in logs.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    payload[key] = value
            self._safe_append(payload)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            self._safe_append(
                {
                    "event": "train_end",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": int(getattr(state, "global_step", 0) or 0),
                    "epoch": getattr(state, "epoch", None),
                },
            )

    return JsonlMetricsCallback()


def create_summary_writer(log_dir: str | Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "当前环境未安装 TensorBoard。请先执行 `uv pip install tensorboard` 或重新安装 requirements.txt。"
        ) from exc
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(output_dir))


def write_tensorboard_scalars(writer: Any, metrics: dict[str, Any], step: int) -> None:
    scalar_map = {
        "loss": "train/loss",
        "eval_loss": "eval/loss",
        "grad_norm": "train/grad_norm",
        "learning_rate": "train/learning_rate",
    }
    for key, tag in scalar_map.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(tag, float(value), step)
    writer.flush()
