from __future__ import annotations

import httpx
import pytest

from agentic_rag_rl.llm_client import (
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_CHOICES,
    NewAPILLMClient,
    RightCodeLLMClient,
    create_llm_client,
    get_newapi_base_url,
    get_newapi_model,
    get_rightcode_base_url,
    get_rightcode_model,
    resolve_judge_model,
    resolve_kg_model,
    resolve_llm_base_url,
    resolve_llm_model,
    resolve_thinking_model,
)


def test_default_provider_is_newapi() -> None:
    assert DEFAULT_LLM_PROVIDER == "newapi"
    assert LLM_PROVIDER_CHOICES == ("newapi", "rightcode")


def test_create_llm_client_returns_newapi_client_by_default() -> None:
    client = create_llm_client(api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, NewAPILLMClient)


def test_create_llm_client_returns_newapi_client() -> None:
    client = create_llm_client("newapi", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, NewAPILLMClient)


def test_create_llm_client_returns_rightcode_client() -> None:
    client = create_llm_client("rightcode", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, RightCodeLLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("unknown", api_key="test-key")


def test_newapi_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("NEWAPI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NEWAPI_API_KEY"):
        NewAPILLMClient(api_key="")


def test_rightcode_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("RIGHTCODE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RIGHTCODE_API_KEY"):
        RightCodeLLMClient(api_key="")


def test_newapi_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEWAPI_MODEL", "env-newapi-model")
    monkeypatch.setenv("NEWAPI_KG_MODEL", "env-newapi-kg")
    monkeypatch.setenv("NEWAPI_THINKING_MODEL", "env-newapi-thinking")
    monkeypatch.setenv("NEWAPI_JUDGE_MODEL", "env-newapi-judge")
    monkeypatch.setenv("NEWAPI_BASE_URL", "https://newapi.example/v1")

    assert get_newapi_model() == "env-newapi-model"
    assert get_newapi_base_url() == "https://newapi.example/v1"
    assert resolve_llm_model("newapi") == "env-newapi-model"
    assert resolve_kg_model("newapi") == "env-newapi-kg"
    assert resolve_thinking_model("newapi") == "env-newapi-thinking"
    assert resolve_judge_model("newapi") == "env-newapi-judge"
    assert resolve_llm_base_url("newapi") == "https://newapi.example/v1"


def test_newapi_builtin_defaults_use_gpt_5_5(monkeypatch) -> None:
    monkeypatch.delenv("NEWAPI_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_KG_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_THINKING_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_JUDGE_MODEL", raising=False)

    assert get_newapi_model() == "gpt-5.5"
    assert resolve_llm_model("newapi") == "gpt-5.5"
    assert resolve_kg_model("newapi") == "gpt-5.5"
    assert resolve_thinking_model("newapi") == "gpt-5.5"
    assert resolve_judge_model("newapi") == "gpt-5.5"


def test_rightcode_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RIGHTCODE_MODEL", "env-rc-model")
    monkeypatch.setenv("RIGHTCODE_KG_MODEL", "env-rc-kg")
    monkeypatch.setenv("RIGHTCODE_THINKING_MODEL", "env-rc-thinking")
    monkeypatch.setenv("RIGHTCODE_JUDGE_MODEL", "env-rc-judge")
    monkeypatch.setenv("RIGHTCODE_BASE_URL", "https://rc.example/v1")

    assert get_rightcode_model() == "env-rc-model"
    assert get_rightcode_base_url() == "https://rc.example/v1"
    assert resolve_llm_model("rightcode") == "env-rc-model"
    assert resolve_kg_model("rightcode") == "env-rc-kg"
    assert resolve_thinking_model("rightcode") == "env-rc-thinking"
    assert resolve_judge_model("rightcode") == "env-rc-judge"
    assert resolve_llm_base_url("rightcode") == "https://rc.example/v1"


def test_newapi_online_inference_uses_openai_compatible_chat_path(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = NewAPILLMClient(
        api_key="sk-test",
        model="gpt-5.5",
        base_url="https://api.6i2.com/v1",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "https://api.6i2.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"] == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout"] == 123
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_rightcode_online_inference_uses_openai_compatible_chat_path(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = RightCodeLLMClient(
        api_key="sk-test",
        model="gpt-5.5",
        base_url="https://api.right.codes/v1",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "https://api.right.codes/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"] == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout"] == 123
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
