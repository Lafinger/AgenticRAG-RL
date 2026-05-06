from __future__ import annotations

from pathlib import Path
import subprocess
import sys


DEMO_ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=DEMO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_llm_business_scripts_only_accept_newapi_and_rightcode_providers() -> None:
    removed_provider = "dou" + "bao"
    removed_batch_arg = "--use-batch" + "-inference"
    scripts = [
        "scripts/build_index.py",
        "scripts/gen_seed_qa.py",
        "scripts/domain_multihop_synthesis.py",
        "scripts/run_llm_judge.py",
    ]

    for script in scripts:
        result = run_script(script, "--help")
        assert result.returncode == 0
        assert "newapi,rightcode" in result.stdout
        assert removed_provider not in result.stdout.lower()
        assert removed_batch_arg not in result.stdout


def test_build_index_rejects_removed_legacy_provider() -> None:
    removed_provider = "dou" + "bao"
    result = run_script(
        "scripts/build_index.py",
        "--corpus",
        "missing.jsonl",
        "--index-dir",
        "indexes",
        "--embedding-model",
        "BAAI/bge-m3",
        "--llm-provider",
        removed_provider,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "newapi" in result.stderr
    assert "rightcode" in result.stderr


def test_removed_legacy_argument_is_unrecognized() -> None:
    removed_batch_arg = "--use-batch" + "-inference"
    result = run_script(
        "scripts/domain_multihop_synthesis.py",
        removed_batch_arg,
    )

    assert result.returncode != 0
    assert f"unrecognized arguments: {removed_batch_arg}" in result.stderr
