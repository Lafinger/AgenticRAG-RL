from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
STANDARD_SCRIPT = ROOT / "scripts" / "train_sft_unsloth.py"
TRACE_SCRIPT = ROOT / "scripts" / "train_sft_unsloth_trace.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.checkpointing import (
    TRACE_CHECKPOINT_STATE_FILE,
    reset_output_dir,
    resolve_trace_resume_checkpoint,
    resolve_trainer_resume_checkpoint,
)


def load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_train_mode_defaults_to_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_resume_default")
    trace = load_script(TRACE_SCRIPT, "train_sft_unsloth_trace_resume_default")

    monkeypatch.setattr(sys, "argv", ["train_sft_unsloth.py"])
    assert standard.parse_args().train_mode == "resume"

    monkeypatch.setattr(sys, "argv", ["train_sft_unsloth_trace.py"])
    assert trace.parse_args().train_mode == "resume"


def test_train_mode_resume_and_overwrite_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_resume_conflict")

    monkeypatch.setattr(sys, "argv", ["train_sft_unsloth.py", "--resume", "--overwrite"])
    with pytest.raises(SystemExit):
        standard.parse_args()


def test_standard_trainer_accepts_eval_data_path(monkeypatch: pytest.MonkeyPatch) -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_eval_arg")

    monkeypatch.setattr(sys, "argv", ["train_sft_unsloth.py", "--eval-data-path", "eval.jsonl"])

    assert standard.parse_args().eval_data_path == "eval.jsonl"


def test_trainers_accept_swanlab_resume_options(monkeypatch: pytest.MonkeyPatch) -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_swanlab_resume_args")
    trace = load_script(TRACE_SCRIPT, "train_sft_unsloth_trace_swanlab_resume_args")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft_unsloth.py",
            "--swanlab-run-id",
            "run-abc",
            "--swanlab-resume",
            "allow",
            "--no-swanlab-replay-history",
        ],
    )
    standard_args = standard.parse_args()
    assert standard_args.swanlab_run_id == "run-abc"
    assert standard_args.swanlab_resume == "allow"
    assert standard_args.swanlab_replay_history is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft_unsloth_trace.py",
            "--swanlab-run-id",
            "run-trace",
            "--swanlab-resume",
            "must",
            "--swanlab-replay-history",
        ],
    )
    trace_args = trace.parse_args()
    assert trace_args.swanlab_run_id == "run-trace"
    assert trace_args.swanlab_resume == "must"
    assert trace_args.swanlab_replay_history is True


def test_standard_trainer_adds_eval_strategy_for_current_sft_config() -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_eval_strategy_current")

    class FakeSFTConfig:
        def __init__(self, eval_strategy: str | None = None, eval_steps: int | None = None, do_eval: bool = False) -> None:
            pass

    training_config: dict[str, Any] = {}
    standard.add_eval_training_args(training_config, {"eval_steps": 17}, FakeSFTConfig)

    assert training_config == {"eval_strategy": "steps", "eval_steps": 17, "do_eval": True}


def test_standard_trainer_adds_eval_strategy_for_legacy_sft_config() -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_eval_strategy_legacy")

    class FakeSFTConfig:
        def __init__(self, evaluation_strategy: str | None = None, eval_steps: int | None = None) -> None:
            pass

    training_config: dict[str, Any] = {}
    standard.add_eval_training_args(training_config, {"logging_steps": 9}, FakeSFTConfig)

    assert training_config == {"evaluation_strategy": "steps", "eval_steps": 9}


def test_trainer_resume_requires_complete_trainer_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    checkpoint = output_dir / "checkpoint-9"
    checkpoint.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        resolve_trainer_resume_checkpoint(output_dir)

    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        touch(checkpoint / filename)

    assert resolve_trainer_resume_checkpoint(output_dir) == checkpoint


def test_trace_resume_requires_complete_trace_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    checkpoint = output_dir / "checkpoint-45"
    checkpoint.mkdir(parents=True)

    for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json"):
        touch(checkpoint / filename)

    with pytest.raises(FileNotFoundError, match=TRACE_CHECKPOINT_STATE_FILE):
        resolve_trace_resume_checkpoint(output_dir)

    touch(checkpoint / TRACE_CHECKPOINT_STATE_FILE)
    assert resolve_trace_resume_checkpoint(output_dir) == checkpoint


def test_resume_fails_when_output_dir_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="输出目录不存在"):
        resolve_trace_resume_checkpoint(tmp_path / "missing")


def test_overwrite_resets_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    touch(output_dir / "checkpoint-1" / "adapter_model.safetensors")

    reset_output_dir(output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_standard_trainer_receives_resume_checkpoint(tmp_path: Path) -> None:
    standard = load_script(STANDARD_SCRIPT, "train_sft_unsloth_train_call")
    checkpoint = tmp_path / "checkpoint-1"

    class FakeTrainer:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def train(self, **kwargs: str) -> str:
            self.calls.append(kwargs)
            return "ok"

    trainer = FakeTrainer()
    assert standard.train_with_resume_mode(trainer, checkpoint) == "ok"
    assert trainer.calls == [{"resume_from_checkpoint": str(checkpoint)}]


def test_trace_checkpoint_state_build_and_restore() -> None:
    trace = load_script(TRACE_SCRIPT, "train_sft_unsloth_trace_state")

    class FakeCuda:
        def __init__(self) -> None:
            self.restored: list[str] | None = None

        def is_available(self) -> bool:
            return True

        def get_rng_state_all(self) -> list[str]:
            return ["cuda-rng"]

        def set_rng_state_all(self, value: list[str]) -> None:
            self.restored = value

    class FakeTorch:
        cuda = FakeCuda()
        restored_rng: str | None = None

        @staticmethod
        def get_rng_state() -> str:
            return "torch-rng"

        @classmethod
        def set_rng_state(cls, value: str) -> None:
            cls.restored_rng = value

    class FakeStateful:
        def __init__(self, state: dict[str, str]) -> None:
            self.state = state
            self.loaded: dict[str, str] | None = None

        def state_dict(self) -> dict[str, str]:
            return self.state

        def load_state_dict(self, state: dict[str, str]) -> None:
            self.loaded = state

    optimizer = FakeStateful({"optimizer": "state"})
    scheduler = FakeStateful({"scheduler": "state"})

    state = trace.build_trace_checkpoint_state(
        FakeTorch,
        optimizer,
        scheduler,
        global_step=45,
        completed_micro_batches=360,
        run_config={"total_steps": 525},
    )
    assert state["global_step"] == 45
    assert state["completed_micro_batches"] == 360
    assert state["rng_state"] == {"torch": "torch-rng", "cuda": ["cuda-rng"]}

    trace.restore_trace_checkpoint_state(FakeTorch, optimizer, scheduler, state)
    assert optimizer.loaded == {"optimizer": "state"}
    assert scheduler.loaded == {"scheduler": "state"}
    assert FakeTorch.restored_rng == "torch-rng"
    assert FakeTorch.cuda.restored == ["cuda-rng"]


def test_trace_resume_config_mismatch_raises() -> None:
    trace = load_script(TRACE_SCRIPT, "train_sft_unsloth_trace_config_mismatch")
    saved = {key: "same" for key in trace.TRACE_RESUME_CONFIG_KEYS}
    current = dict(saved)
    current["total_steps"] = "different"

    with pytest.raises(ValueError, match="当前训练配置与 checkpoint 不一致"):
        trace.assert_trace_resume_config_compatible(saved, current)
