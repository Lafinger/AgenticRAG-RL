from __future__ import annotations

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


def test_llm_business_scripts_accept_xingjianya_provider() -> None:
    scripts = [
        "scripts/build_index.py",
        "scripts/gen_seed_qa.py",
        "scripts/domain_multihop_synthesis.py",
        "scripts/run_llm_judge.py",
    ]

    for script in scripts:
        result = run_help(script)
        assert result.returncode == 0
        assert "doubao,xingjianya" in result.stdout


def test_build_index_rejects_xingjianya_batch_inference() -> None:
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
            "xingjianya",
            "--use-batch-inference",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode != 0
    assert "--use-batch-inference only supports --llm-provider doubao" in result.stderr
