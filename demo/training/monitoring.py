from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SWANLAB_MODES = ("cloud", "local", "offline", "disabled")


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


def is_swanlab_enabled(report_to: Any) -> bool:
    return any(item.lower() in {"swanlab", "all"} for item in normalize_report_to(report_to))


def normalize_swanlab_mode(mode: Any) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if not normalized:
        return None
    if normalized not in SWANLAB_MODES:
        raise SystemExit(
            "SwanLab mode 只能是 cloud、local、offline 或 disabled，"
            f"当前值：{mode!r}。"
        )
    return normalized


def configure_swanlab_environment(
    *,
    project: str | None = None,
    workspace: str | None = None,
    mode: str | None = None,
    logdir: str | Path | None = None,
    experiment_name: str | None = None,
) -> None:
    if project:
        os.environ["SWANLAB_PROJ_NAME"] = project
        os.environ["SWANLAB_PROJECT"] = project
    if workspace:
        os.environ["SWANLAB_WORKSPACE"] = workspace
    if mode:
        normalized_mode = normalize_swanlab_mode(mode)
        if normalized_mode:
            os.environ["SWANLAB_MODE"] = normalized_mode
    if logdir:
        os.environ["SWANLAB_LOG_DIR"] = str(logdir)
    if experiment_name:
        os.environ["SWANLAB_EXP_NAME"] = experiment_name


def require_swanlab(report_to: Any) -> None:
    if not is_swanlab_enabled(report_to):
        return
    try:
        importlib.import_module("swanlab")
    except ImportError as exc:
        raise SystemExit("当前环境未安装 SwanLab。请先执行 `uv pip install swanlab>=0.7.12`。") from exc


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


@dataclass(slots=True)
class SwanLabScalarLogger:
    project: str
    workspace: str | None
    experiment_name: str
    mode: str
    logdir: str | Path
    config: dict[str, Any]
    _swanlab: Any = field(init=False, repr=False)
    _run: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mode = normalize_swanlab_mode(self.mode) or "cloud"
        try:
            self._swanlab = importlib.import_module("swanlab")
        except ImportError as exc:
            raise SystemExit("当前环境未安装 SwanLab。请先执行 `uv pip install swanlab>=0.7.12`。") from exc

        self._run = self._swanlab.init(
            project=self.project,
            workspace=self.workspace,
            experiment_name=self.experiment_name,
            mode=mode,
            logdir=str(self.logdir),
            config=self.config,
        )

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        scalar_map = {
            "loss": "train/loss",
            "grad_norm": "train/grad_norm",
            "learning_rate": "train/learning_rate",
            "eval_loss": "eval/loss",
        }
        payload: dict[str, float] = {}
        for key, target in scalar_map.items():
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                payload[target] = float(value)
        if payload:
            self._swanlab.log(payload, step=step)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.log_scalars(metrics, step)

    def finish(self) -> None:
        finish = getattr(self._swanlab, "finish", None)
        if callable(finish):
            finish()
            return
        run_finish = getattr(self._run, "finish", None)
        if callable(run_finish):
            run_finish()
