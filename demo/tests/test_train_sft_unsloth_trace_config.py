from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE_SCRIPT = ROOT / "scripts" / "train_sft_unsloth_trace.py"


def load_trace_module() -> Any:
    spec = importlib.util.spec_from_file_location("train_sft_unsloth_trace", TRACE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {TRACE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_args(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "model_name_or_path": None,
        "data_path": None,
        "num_train_epochs": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "learning_rate": None,
        "warmup_steps": None,
        "save_steps": None,
        "logging_steps": None,
        "eval_steps": None,
        "max_grad_norm": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_trace_sft_config_contains_yaml_managed_training_controls() -> None:
    module = load_trace_module()
    config = module.load_config(ROOT / "training" / "unsloth_sft.yaml")

    module.apply_cli_config_overrides(config, make_args())

    assert config["per_device_train_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 8
    assert config["learning_rate"] == 1.0e-4
    assert config["eval_steps"] == 105
    assert config["max_grad_norm"] == 1.0


def test_trace_sft_cli_still_overrides_yaml_training_controls() -> None:
    module = load_trace_module()
    config = module.load_config(ROOT / "training" / "unsloth_sft.yaml")

    module.apply_cli_config_overrides(
        config,
        make_args(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            eval_steps=12,
            max_grad_norm=0.5,
        ),
    )

    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 2
    assert config["learning_rate"] == 5e-5
    assert config["eval_steps"] == 12
    assert config["max_grad_norm"] == 0.5
