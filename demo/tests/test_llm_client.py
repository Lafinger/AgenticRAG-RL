from __future__ import annotations

import pytest

from agentic_rag_rl.llm_client import (
    DoubaoLLMClient,
    create_llm_client,
    get_doubao_base_url,
    get_doubao_model,
    get_doubao_thinking_model,
)


def test_doubao_llm_client_uses_transport() -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_transport(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "ok"

    client = DoubaoLLMClient(api_key="test-key", transport=fake_transport)
    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls == [[{"role": "user", "content": "你好"}]]


def test_create_llm_client_returns_doubao_client() -> None:
    client = create_llm_client("doubao", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, DoubaoLLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("unknown", api_key="test-key")


def test_doubao_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ARK_API_KEY"):
        DoubaoLLMClient(api_key="")


def test_doubao_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOUBAO_MODEL", "env-model")
    monkeypatch.setenv("DOUBAO_THINKING_MODEL", "env-thinking-model")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://env.example/api/v3")

    assert get_doubao_model() == "env-model"
    assert get_doubao_thinking_model() == "env-thinking-model"
    assert get_doubao_base_url() == "https://env.example/api/v3"
