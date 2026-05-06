from __future__ import annotations

import os
from pathlib import Path

from agentic_rag_rl.env import load_env_file


def test_load_env_file_sets_environment_variables(tmp_path: Path, monkeypatch) -> None:
    keys = ["NEWAPI_API_KEY", "NEWAPI_MODEL", "NEWAPI_BASE_URL"]
    original_values = {key: os.environ.get(key) for key in keys}
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\r\n".join(
            [
                "NEWAPI_API_KEY=test-key",
                "NEWAPI_MODEL='gpt-5.5'",
                'NEWAPI_BASE_URL="https://example.test/v1"',
            ]
        ),
        encoding="utf-8",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    try:
        loaded = load_env_file(env_file)

        assert loaded["NEWAPI_API_KEY"] == "test-key"
        assert os.environ["NEWAPI_MODEL"] == "gpt-5.5"
        assert os.environ["NEWAPI_BASE_URL"] == "https://example.test/v1"
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
