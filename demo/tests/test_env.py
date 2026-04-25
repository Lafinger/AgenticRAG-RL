from __future__ import annotations

import os
from pathlib import Path

from agentic_rag_rl.env import load_env_file


def test_load_env_file_sets_environment_variables(tmp_path: Path, monkeypatch) -> None:
    keys = ["ARK_API_KEY", "DOUBAO_MODEL", "DOUBAO_BASE_URL"]
    original_values = {key: os.environ.get(key) for key in keys}
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\r\n".join(
            [
                "ARK_API_KEY=test-key",
                "DOUBAO_MODEL='doubao-seed-1-6-flash-250828'",
                'DOUBAO_BASE_URL="https://example.test/api/v3"',
            ]
        ),
        encoding="utf-8",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    try:
        loaded = load_env_file(env_file)

        assert loaded["ARK_API_KEY"] == "test-key"
        assert os.environ["DOUBAO_MODEL"] == "doubao-seed-1-6-flash-250828"
        assert os.environ["DOUBAO_BASE_URL"] == "https://example.test/api/v3"
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
