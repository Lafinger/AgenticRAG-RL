from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


def run_help(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def load_multihop_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "domain_multihop_synthesis.py"
    spec = importlib.util.spec_from_file_location("domain_multihop_synthesis_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_business_scripts_accept_newapi_and_rightcode_providers() -> None:
    scripts = [
        "scripts/build_index.py",
        "scripts/gen_seed_qa.py",
        "scripts/domain_multihop_synthesis.py",
        "scripts/run_llm_judge.py",
    ]

    for script in scripts:
        result = run_help(script)
        assert result.returncode == 0
        assert "doubao,newapi,rightcode" in result.stdout


def test_build_index_rejects_newapi_batch_inference() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_index.py",
            "--corpus",
            "missing.jsonl",
            "--index-dir",
            "indexes",
            "--embedding-model",
            "BAAI/bge-m3",
            "--llm-provider",
            "newapi",
            "--use-batch-inference",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode != 0
    assert "--use-batch-inference only supports --llm-provider doubao" in result.stderr


def test_build_index_rejects_rightcode_batch_inference() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_index.py",
            "--corpus",
            "missing.jsonl",
            "--index-dir",
            "indexes",
            "--embedding-model",
            "BAAI/bge-m3",
            "--llm-provider",
            "rightcode",
            "--use-batch-inference",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode != 0
    assert "--use-batch-inference only supports --llm-provider doubao" in result.stderr


def test_multihop_batch_rejects_unlimited_candidate_multiplier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/domain_multihop_synthesis.py",
            "--use-batch-inference",
            "--candidate-multiplier",
            "-1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode != 0
    assert "--use-batch-inference does not support --candidate-multiplier -1" in result.stderr


def test_multihop_batch_candidate_multiplier_groups_candidates() -> None:
    module = load_multihop_script_module()
    chains = [[{"doc_chunk_id": f"c{index}", "question": f"q{index}"}] for index in range(11)]

    assert module._batch_candidate_limit(50, 5) == 250
    assert module._batch_candidate_limit(50, 5, 12) == 12
    groups = module._candidate_groups(chains, 5)
    assert [len(group) for group in groups] == [5, 5, 1]
