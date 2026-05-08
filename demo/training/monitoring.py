from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import uuid


SWANLAB_MODES = ("cloud", "local", "offline", "disabled")
SWANLAB_RESUME_MODES = ("allow", "must", "never")
SWANLAB_RUN_ID_FILE = "swanlab_run_id.txt"
SWANLAB_HISTORY_REPLAY_STATE_FILE = "swanlab_history_replay_state.json"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
SWANLAB_SCALAR_MAP = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "eval_loss": "eval/loss",
    "progress_percent": "train/progress_percent",
    "epoch": "train/epoch",
}


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


def normalize_swanlab_resume(resume: Any) -> str | None:
    if resume is None:
        return None
    if isinstance(resume, bool):
        return "allow" if resume else "never"
    normalized = str(resume).strip().lower()
    if not normalized:
        return None
    if normalized not in SWANLAB_RESUME_MODES:
        raise SystemExit(
            "SwanLab resume 只能是 allow、must 或 never，"
            f"当前值：{resume!r}。"
        )
    return normalized


def normalize_swanlab_replay_history(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled", "enable"}:
        return "enabled"
    if normalized in {"0", "false", "no", "off", "disabled", "disable"}:
        return "disabled"
    if normalized == "auto":
        return "auto"
    raise SystemExit(
        "swanlab_replay_history 只能是 auto、true 或 false，"
        f"当前值：{value!r}。"
    )


def should_replay_swanlab_history(value: Any, *, resume_checkpoint: str | Path | None) -> bool:
    normalized = normalize_swanlab_replay_history(value)
    if normalized == "enabled":
        return True
    if normalized == "disabled":
        return False
    return resume_checkpoint is not None


def build_training_progress_metrics(step: int, total_steps: int | None) -> dict[str, float]:
    if total_steps is None:
        return {}
    total = int(total_steps)
    if total <= 0:
        return {}

    current = max(0, min(int(step), total))
    return {
        "progress_percent": current / total * 100.0,
    }


def configure_swanlab_environment(
    *,
    project: str | None = None,
    workspace: str | None = None,
    mode: str | None = None,
    logdir: str | Path | None = None,
    experiment_name: str | None = None,
    run_id: str | None = None,
    resume: str | None = None,
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
    if run_id:
        os.environ["SWANLAB_RUN_ID"] = run_id
    if resume:
        normalized_resume = normalize_swanlab_resume(resume)
        if normalized_resume:
            os.environ["SWANLAB_RESUME"] = normalized_resume


def require_swanlab(report_to: Any) -> None:
    if not is_swanlab_enabled(report_to):
        return
    try:
        importlib.import_module("swanlab")
    except ImportError as exc:
        raise SystemExit("当前环境未安装 SwanLab。请先执行 `uv pip install swanlab>=0.7.12`。") from exc


def checkpoint_step_from_path(path: str | Path | None) -> int | None:
    if path is None:
        return None
    match = CHECKPOINT_PATTERN.match(Path(path).name)
    return int(match.group(1)) if match else None


def generate_swanlab_run_id() -> str:
    return uuid.uuid4().hex


def resolve_swanlab_run_id(
    output_dir: str | Path,
    *,
    explicit_run_id: str | None = None,
    reset: bool = False,
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / SWANLAB_RUN_ID_FILE
    if explicit_run_id:
        run_id = explicit_run_id.strip()
        if not run_id:
            raise ValueError("--swanlab-run-id 不能为空。")
        state_path.write_text(run_id + "\r\n", encoding="utf-8")
        return run_id
    if not reset and state_path.exists():
        run_id = state_path.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    run_id = generate_swanlab_run_id()
    state_path.write_text(run_id + "\r\n", encoding="utf-8")
    return run_id


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return []
    return records


def _numeric_step(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def build_swanlab_scalar_payload(metrics: dict[str, Any], *, step: int) -> dict[str, float]:
    payload: dict[str, float] = {"train/global_step": float(step)}
    for source_key, target_key in SWANLAB_SCALAR_MAP.items():
        scalar = _finite_scalar(metrics.get(source_key))
        if scalar is not None:
            payload[target_key] = scalar
    return payload


def _normalize_history_records(
    records: list[dict[str, Any]],
    *,
    max_step: int | None = None,
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, float]] = {}
    for record in records:
        event = record.get("event")
        if event is not None and event != "log":
            continue
        step = _numeric_step(record.get("step")) or _numeric_step(record.get("global_step"))
        if step is None or step <= 0:
            continue
        if max_step is not None and step > max_step:
            continue
        payload = build_swanlab_scalar_payload(record, step=step)
        if len(payload) <= 1:
            continue
        by_step.setdefault(step, {}).update(payload)
    return [{"step": step, "payload": by_step[step]} for step in sorted(by_step)]


def _latest_trainer_state_path(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = checkpoint_step_from_path(path)
        state_path = path / "trainer_state.json"
        if step is not None and state_path.is_file():
            candidates.append((step, state_path))
    direct_state = output_dir / "trainer_state.json"
    if direct_state.is_file():
        candidates.append((10**18, direct_state))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def load_swanlab_history_records(
    output_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    max_step: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    root = Path(output_dir)
    candidates: list[tuple[str, Path, str]] = []
    if metrics_path:
        candidates.append(("jsonl", Path(metrics_path), str(metrics_path)))
    candidates.extend(
        [
            ("jsonl", root / "metrics.jsonl", str(root / "metrics.jsonl")),
            ("jsonl", root / "trace_metrics.jsonl", str(root / "trace_metrics.jsonl")),
        ]
    )
    seen: set[Path] = set()
    for source_type, path, source_name in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records = _read_jsonl_records(path) if source_type == "jsonl" else []
        normalized = _normalize_history_records(records, max_step=max_step)
        if normalized:
            return normalized, source_name

    trainer_state_path = Path(checkpoint_path) / "trainer_state.json" if checkpoint_path else _latest_trainer_state_path(root)
    if trainer_state_path and trainer_state_path.is_file():
        state = _safe_read_json(trainer_state_path) or {}
        history = state.get("log_history", [])
        records = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
        normalized = _normalize_history_records(records, max_step=max_step)
        if normalized:
            return normalized, str(trainer_state_path)
    return [], None


def _read_replay_state(path: Path) -> dict[str, Any] | None:
    return _safe_read_json(path)


def _write_replay_state(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def replay_swanlab_history(
    swanlab_module: Any,
    output_dir: str | Path,
    *,
    run_id: str,
    max_step: int | None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    replay_state_path = root / SWANLAB_HISTORY_REPLAY_STATE_FILE
    target_step = int(max_step or 0)
    if target_step <= 0:
        return {
            "run_id": run_id,
            "replayed": False,
            "reason": "no_resume_step",
            "replayed_until_step": target_step,
            "count": 0,
        }

    previous_state = _read_replay_state(replay_state_path)
    previous_until_step = (
        int(previous_state.get("replayed_until_step") or 0)
        if previous_state and previous_state.get("run_id") == run_id
        else 0
    )
    if not dry_run and previous_until_step >= target_step:
        return {
            "run_id": run_id,
            "replayed": False,
            "reason": "already_replayed",
            "replayed_until_step": target_step,
            "count": 0,
            "state_path": str(replay_state_path),
        }

    records, source = load_swanlab_history_records(
        root,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        max_step=target_step,
    )
    if previous_until_step > 0:
        records = [record for record in records if int(record["step"]) > previous_until_step]
    for record in records:
        if not dry_run:
            swanlab_module.log(record["payload"], step=int(record["step"]))

    first_step = records[0]["step"] if records else None
    last_step = records[-1]["step"] if records else None
    summary = {
        "run_id": run_id,
        "replayed": not dry_run,
        "dry_run": dry_run,
        "source": source,
        "previous_replayed_until_step": previous_until_step,
        "replayed_until_step": target_step,
        "count": len(records),
        "first_step": first_step,
        "last_step": last_step,
        "state_path": str(replay_state_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not dry_run:
        _write_replay_state(replay_state_path, summary)
    return summary


def create_swanlab_history_replay_callback(
    *,
    output_dir: str | Path,
    run_id: str,
    max_step: int | None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> Any:
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise SystemExit("当前环境未安装 transformers，无法启用 SwanLab 历史回放回调。") from exc

    class SwanLabHistoryReplayCallback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if not getattr(state, "is_world_process_zero", True):
                return
            swanlab = importlib.import_module("swanlab")
            get_run = getattr(swanlab, "get_run", None)
            if callable(get_run) and get_run() is None:
                swanlab.init()
            replay_swanlab_history(
                swanlab,
                output_dir,
                run_id=run_id,
                max_step=max_step,
                metrics_path=metrics_path,
                checkpoint_path=checkpoint_path,
            )

    return SwanLabHistoryReplayCallback()


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
            step = int(getattr(state, "global_step", 0) or 0)
            payload: dict[str, Any] = {
                "event": "log",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "epoch": getattr(state, "epoch", None),
            }
            payload.update(build_training_progress_metrics(step, getattr(state, "max_steps", None)))
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
    run_id: str | None = None
    resume: str | None = None
    _swanlab: Any = field(init=False, repr=False)
    _run: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mode = normalize_swanlab_mode(self.mode) or "cloud"
        try:
            self._swanlab = importlib.import_module("swanlab")
        except ImportError as exc:
            raise SystemExit("当前环境未安装 SwanLab。请先执行 `uv pip install swanlab>=0.7.12`。") from exc

        init_args: dict[str, Any] = {
            "project": self.project,
            "workspace": self.workspace,
            "experiment_name": self.experiment_name,
            "mode": mode,
            "logdir": str(self.logdir),
            "config": self.config,
        }
        if self.run_id:
            init_args["id"] = self.run_id
        resume = normalize_swanlab_resume(self.resume)
        if resume:
            init_args["resume"] = resume
        self._run = self._swanlab.init(**init_args)

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        payload = build_swanlab_scalar_payload(metrics, step=step)
        if len(payload) > 1:
            self._swanlab.log(payload, step=step)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.log_scalars(metrics, step)

    def replay_history(
        self,
        output_dir: str | Path,
        *,
        max_step: int | None,
        metrics_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return replay_swanlab_history(
            self._swanlab,
            output_dir,
            run_id=self.run_id or "",
            max_step=max_step,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            dry_run=dry_run,
        )

    def finish(self) -> None:
        finish = getattr(self._swanlab, "finish", None)
        if callable(finish):
            finish()
            return
        run_finish = getattr(self._run, "finish", None)
        if callable(run_finish):
            run_finish()
