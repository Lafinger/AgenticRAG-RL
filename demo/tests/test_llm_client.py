from __future__ import annotations

import httpx
import pytest

from agentic_rag_rl.llm_client import (
    CommonLLMClient,
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_CHOICES,
    RightCodeLLMClient,
    create_llm_client,
    get_common_base_url,
    get_common_model,
    get_rightcode_base_url,
    get_rightcode_model,
    resolve_judge_model,
    resolve_kg_model,
    resolve_llm_base_url,
    resolve_llm_model,
    resolve_thinking_model,
)


def test_default_provider_is_common() -> None:
    assert DEFAULT_LLM_PROVIDER == "common"
    assert LLM_PROVIDER_CHOICES == ("common", "rightcode")


def test_create_llm_client_returns_common_client_by_default() -> None:
    client = create_llm_client(api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, CommonLLMClient)


def test_create_llm_client_returns_common_client() -> None:
    client = create_llm_client("common", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, CommonLLMClient)


def test_create_llm_client_returns_rightcode_client() -> None:
    client = create_llm_client("rightcode", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, RightCodeLLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("unknown", api_key="test-key")


def test_create_llm_client_rejects_previous_common_provider_name() -> None:
    removed_provider = "new" + "api"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(removed_provider, api_key="test-key")


def test_common_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("COMMON_API_KEY", raising=False)

    with pytest.raises(ValueError, match="COMMON_API_KEY"):
        CommonLLMClient(api_key="")


def test_rightcode_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("RIGHTCODE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RIGHTCODE_API_KEY"):
        RightCodeLLMClient(api_key="")


def test_common_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("COMMON_MODEL", "env-common-model")
    monkeypatch.setenv("COMMON_KG_MODEL", "env-common-kg")
    monkeypatch.setenv("COMMON_THINKING_MODEL", "env-common-thinking")
    monkeypatch.setenv("COMMON_JUDGE_MODEL", "env-common-judge")
    monkeypatch.setenv("COMMON_BASE_URL", "https://common.example/v1")

    assert get_common_model() == "env-common-model"
    assert get_common_base_url() == "https://common.example/v1"
    assert resolve_llm_model("common") == "env-common-model"
    assert resolve_kg_model("common") == "env-common-kg"
    assert resolve_thinking_model("common") == "env-common-thinking"
    assert resolve_judge_model("common") == "env-common-judge"
    assert resolve_llm_base_url("common") == "https://common.example/v1"


def test_common_builtin_defaults_use_gpt_5_5_and_cockpit_base_url(monkeypatch) -> None:
    monkeypatch.delenv("COMMON_MODEL", raising=False)
    monkeypatch.delenv("COMMON_KG_MODEL", raising=False)
    monkeypatch.delenv("COMMON_THINKING_MODEL", raising=False)
    monkeypatch.delenv("COMMON_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("COMMON_BASE_URL", raising=False)

    assert get_common_model() == "gpt-5.5"
    assert get_common_base_url() == "http://127.0.0.1:53389/v1"
    assert resolve_llm_model("common") == "gpt-5.5"
    assert resolve_kg_model("common") == "gpt-5.5"
    assert resolve_thinking_model("common") == "gpt-5.5"
    assert resolve_judge_model("common") == "gpt-5.5"
    assert resolve_llm_base_url("common") == "http://127.0.0.1:53389/v1"


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


def test_common_online_inference_uses_openai_compatible_chat_path(monkeypatch) -> None:
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
    client = CommonLLMClient(
        api_key="sk-test",
        model="gpt-5.5",
        base_url="http://127.0.0.1:53389/v1",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "http://127.0.0.1:53389/v1/chat/completions"
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
