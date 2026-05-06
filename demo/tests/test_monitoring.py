from __future__ import annotations

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
    SwanLabScalarLogger,
    configure_swanlab_environment,
    normalize_swanlab_mode,
    require_swanlab,
)


def test_configure_swanlab_environment_sets_expected_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in ("SWANLAB_PROJ_NAME", "SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_MODE", "SWANLAB_LOG_DIR", "SWANLAB_EXP_NAME"):
        monkeypatch.delenv(key, raising=False)

    configure_swanlab_environment(
        project="agentic-rag-rl",
        workspace="team-a",
        mode="cloud",
        logdir=tmp_path / "swanlab",
        experiment_name="sft-test",
    )

    assert monitoring.os.environ["SWANLAB_PROJ_NAME"] == "agentic-rag-rl"
    assert monitoring.os.environ["SWANLAB_PROJECT"] == "agentic-rag-rl"
    assert monitoring.os.environ["SWANLAB_WORKSPACE"] == "team-a"
    assert monitoring.os.environ["SWANLAB_MODE"] == "cloud"
    assert monitoring.os.environ["SWANLAB_LOG_DIR"] == str(tmp_path / "swanlab")
    assert monitoring.os.environ["SWANLAB_EXP_NAME"] == "sft-test"


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
                "train/loss": 1.25,
                "train/grad_norm": 3.0,
                "train/learning_rate": 0.0001,
            },
            7,
        ),
        ({"eval/loss": 0.8}, 8),
    ]
    assert calls["finished"] == 1


def test_require_swanlab_reports_clear_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "swanlab":
            raise ImportError("missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(monitoring.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match="当前环境未安装 SwanLab"):
        require_swanlab(["swanlab"])
