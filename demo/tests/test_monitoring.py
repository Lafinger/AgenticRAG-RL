from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import monitoring
from training.monitoring import (
    SWANLAB_HISTORY_REPLAY_STATE_FILE,
    SWANLAB_RUN_ID_FILE,
    SwanLabScalarLogger,
    build_training_progress_metrics,
    configure_swanlab_environment,
    load_swanlab_history_records,
    normalize_swanlab_mode,
    normalize_swanlab_resume,
    replay_swanlab_history,
    require_swanlab,
    resolve_swanlab_run_id,
    should_replay_swanlab_history,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def test_configure_swanlab_environment_sets_expected_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "SWANLAB_PROJ_NAME",
        "SWANLAB_PROJECT",
        "SWANLAB_WORKSPACE",
        "SWANLAB_MODE",
        "SWANLAB_LOG_DIR",
        "SWANLAB_EXP_NAME",
        "SWANLAB_RUN_ID",
        "SWANLAB_RESUME",
    ):
        monkeypatch.delenv(key, raising=False)

    configure_swanlab_environment(
        project="agentic-rag-rl",
        workspace="team-a",
        mode="cloud",
        logdir=tmp_path / "swanlab",
        experiment_name="sft-test",
        run_id="run-123",
        resume="allow",
    )

    assert monitoring.os.environ["SWANLAB_PROJ_NAME"] == "agentic-rag-rl"
    assert monitoring.os.environ["SWANLAB_PROJECT"] == "agentic-rag-rl"
    assert monitoring.os.environ["SWANLAB_WORKSPACE"] == "team-a"
    assert monitoring.os.environ["SWANLAB_MODE"] == "cloud"
    assert monitoring.os.environ["SWANLAB_LOG_DIR"] == str(tmp_path / "swanlab")
    assert monitoring.os.environ["SWANLAB_EXP_NAME"] == "sft-test"
    assert monitoring.os.environ["SWANLAB_RUN_ID"] == "run-123"
    assert monitoring.os.environ["SWANLAB_RESUME"] == "allow"


def test_configure_swanlab_environment_supports_local_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in ("SWANLAB_MODE", "SWANLAB_LOG_DIR"):
        monkeypatch.delenv(key, raising=False)

    configure_swanlab_environment(
        mode="LOCAL",
        logdir=tmp_path / "swanlab",
    )

    assert monitoring.os.environ["SWANLAB_MODE"] == "local"
    assert monitoring.os.environ["SWANLAB_LOG_DIR"] == str(tmp_path / "swanlab")


def test_normalize_swanlab_mode_rejects_unknown_mode() -> None:
    assert normalize_swanlab_mode(" cloud ") == "cloud"
    assert normalize_swanlab_mode(None) is None
    with pytest.raises(SystemExit, match="SwanLab mode 只能是"):
        normalize_swanlab_mode("wrong")


def test_normalize_swanlab_resume_and_replay_policy() -> None:
    assert normalize_swanlab_resume(" ALLOW ") == "allow"
    assert normalize_swanlab_resume(True) == "allow"
    assert normalize_swanlab_resume(False) == "never"
    with pytest.raises(SystemExit, match="SwanLab resume 只能是"):
        normalize_swanlab_resume("bad")

    assert should_replay_swanlab_history("auto", resume_checkpoint="checkpoint-1") is True
    assert should_replay_swanlab_history("auto", resume_checkpoint=None) is False
    assert should_replay_swanlab_history(True, resume_checkpoint=None) is True
    assert should_replay_swanlab_history(False, resume_checkpoint="checkpoint-1") is False


def test_resolve_swanlab_run_id_reuses_and_resets(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    first = resolve_swanlab_run_id(output_dir)
    assert (output_dir / SWANLAB_RUN_ID_FILE).read_text(encoding="utf-8").strip() == first
    assert resolve_swanlab_run_id(output_dir) == first

    explicit = resolve_swanlab_run_id(output_dir, explicit_run_id="manual-run")
    assert explicit == "manual-run"
    assert resolve_swanlab_run_id(output_dir) == "manual-run"

    reset = resolve_swanlab_run_id(output_dir, reset=True)
    assert reset != "manual-run"
    assert (output_dir / SWANLAB_RUN_ID_FILE).read_text(encoding="utf-8").strip() == reset


def test_build_training_progress_metrics() -> None:
    assert build_training_progress_metrics(step=200, total_steps=525) == {
        "progress_percent": 200 / 525 * 100,
    }
    assert build_training_progress_metrics(step=600, total_steps=525) == {
        "progress_percent": 100.0,
    }
    assert build_training_progress_metrics(step=1, total_steps=0) == {}


def test_swanlab_scalar_logger_init_log_and_finish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {"logs": [], "finished": 0}

    def fake_init(**kwargs: Any) -> SimpleNamespace:
        calls["init"] = kwargs
        return SimpleNamespace(finish=lambda: calls.__setitem__("run_finished", True))

    def fake_log(payload: dict[str, float], step: int) -> None:
        calls["logs"].append((payload, step))

    def fake_finish() -> None:
        calls["finished"] += 1

    fake_swanlab = SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    logger = SwanLabScalarLogger(
        project="agentic-rag-rl",
        workspace="team-a",
        experiment_name="trace-test",
        mode="LOCAL",
        logdir=tmp_path / "swanlab",
        config={"seed": 3407},
    )
    logger.log_scalars(
        {
            "loss": 1.25,
            "grad_norm": 3,
            "learning_rate": 0.0001,
            "eval_loss": None,
            "progress_percent": 200 / 525 * 100,
            "sample_ids": ["sft_000001"],
            "source_lines": [1],
            "question": "不要上传",
            "answer": "不要上传",
            "gold_chunks": ["chunk-1"],
        },
        step=7,
    )
    logger.log_scalars({"eval_loss": 0.8}, step=8)
    logger.finish()

    assert calls["init"] == {
        "project": "agentic-rag-rl",
        "workspace": "team-a",
        "experiment_name": "trace-test",
        "mode": "local",
        "logdir": str(tmp_path / "swanlab"),
        "config": {"seed": 3407},
    }
    assert calls["logs"] == [
        (
            {
                "train/global_step": 7.0,
                "train/loss": 1.25,
                "train/grad_norm": 3.0,
                "train/learning_rate": 0.0001,
                "train/progress_percent": 200 / 525 * 100,
            },
            7,
        ),
        ({"train/global_step": 8.0, "eval/loss": 0.8}, 8),
    ]
    assert calls["finished"] == 1


def test_swanlab_scalar_logger_passes_run_id_and_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> SimpleNamespace:
        calls["init"] = kwargs
        return SimpleNamespace()

    fake_swanlab = SimpleNamespace(init=fake_init, log=lambda payload, step: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    SwanLabScalarLogger(
        project="agentic-rag-rl",
        workspace=None,
        experiment_name="resume-test",
        mode="cloud",
        logdir=tmp_path / "swanlab",
        config={},
        run_id="run-abc",
        resume="allow",
    )

    assert calls["init"]["id"] == "run-abc"
    assert calls["init"]["resume"] == "allow"


def test_load_swanlab_history_records_filters_and_maps_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    metrics = output_dir / "metrics.jsonl"
    write_jsonl(
        metrics,
        [
            {"event": "train_begin", "step": 0},
            {"event": "log", "step": 5, "loss": 1.2, "grad_norm": 3, "learning_rate": 1e-4, "epoch": 0.1},
            {"event": "log", "step": 10, "eval_loss": 0.8, "question": "不要上传", "source_lines": [1]},
            {"event": "log", "step": 15, "loss": 0.7},
        ],
    )

    records, source = load_swanlab_history_records(output_dir, metrics_path=metrics, max_step=10)

    assert source == str(metrics)
    assert records == [
        {
            "step": 5,
            "payload": {
                "train/global_step": 5.0,
                "train/loss": 1.2,
                "train/grad_norm": 3.0,
                "train/learning_rate": 1e-4,
                "train/epoch": 0.1,
            },
        },
        {"step": 10, "payload": {"train/global_step": 10.0, "eval/loss": 0.8}},
    ]


def test_load_swanlab_history_records_falls_back_to_trainer_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    checkpoint = output_dir / "checkpoint-45"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 5, "loss": 1.0},
                    {"step": 50, "loss": 0.5},
                ]
            }
        ),
        encoding="utf-8",
    )

    records, source = load_swanlab_history_records(output_dir, checkpoint_path=checkpoint, max_step=45)

    assert source == str(checkpoint / "trainer_state.json")
    assert records == [{"step": 5, "payload": {"train/global_step": 5.0, "train/loss": 1.0}}]


def test_replay_swanlab_history_is_idempotent(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    metrics = output_dir / "metrics.jsonl"
    write_jsonl(metrics, [{"event": "log", "step": 5, "loss": 1.2}, {"event": "log", "step": 10, "loss": 0.8}])
    calls: list[tuple[dict[str, float], int]] = []
    fake_swanlab = SimpleNamespace(log=lambda payload, step: calls.append((payload, step)))

    first = replay_swanlab_history(fake_swanlab, output_dir, run_id="run-1", max_step=10, metrics_path=metrics)
    second = replay_swanlab_history(fake_swanlab, output_dir, run_id="run-1", max_step=10, metrics_path=metrics)

    assert first["count"] == 2
    assert second["reason"] == "already_replayed"
    assert calls == [
        ({"train/global_step": 5.0, "train/loss": 1.2}, 5),
        ({"train/global_step": 10.0, "train/loss": 0.8}, 10),
    ]
    state = json.loads((output_dir / SWANLAB_HISTORY_REPLAY_STATE_FILE).read_text(encoding="utf-8"))
    assert state["run_id"] == "run-1"
    assert state["replayed_until_step"] == 10


def test_replay_swanlab_history_only_uploads_incremental_steps(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    metrics = output_dir / "metrics.jsonl"
    write_jsonl(
        metrics,
        [
            {"event": "log", "step": 5, "loss": 1.2},
            {"event": "log", "step": 10, "loss": 0.8},
            {"event": "log", "step": 15, "loss": 0.6},
        ],
    )
    fake_swanlab = SimpleNamespace(log=lambda payload, step: None)
    replay_swanlab_history(fake_swanlab, output_dir, run_id="run-1", max_step=10, metrics_path=metrics)
    calls: list[tuple[dict[str, float], int]] = []
    fake_swanlab = SimpleNamespace(log=lambda payload, step: calls.append((payload, step)))

    summary = replay_swanlab_history(fake_swanlab, output_dir, run_id="run-1", max_step=15, metrics_path=metrics)

    assert summary["previous_replayed_until_step"] == 10
    assert summary["count"] == 1
    assert calls == [({"train/global_step": 15.0, "train/loss": 0.6}, 15)]


def test_require_swanlab_reports_clear_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "swanlab":
            raise ImportError("missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(monitoring.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match="当前环境未安装 SwanLab"):
        require_swanlab(["swanlab"])
