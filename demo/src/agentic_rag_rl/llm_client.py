from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any, Protocol, TypedDict

import httpx


DEFAULT_LLM_PROVIDER = "common"
LLM_PROVIDER_CHOICES = ("common", "rightcode")
DEFAULT_COMMON_MODEL = "gpt-5.5"
DEFAULT_COMMON_BASE_URL = "http://127.0.0.1:53389/v1"
DEFAULT_RIGHTCODE_MODEL = "gpt-5.5"
DEFAULT_RIGHTCODE_BASE_URL = "https://api.right.codes/v1"
OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        ...


def create_llm_client(
    provider: str = DEFAULT_LLM_PROVIDER,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    transport: Callable[[list[ChatMessage]], str] | None = None,
) -> LLMClient:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return CommonLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    if normalized_provider == "rightcode":
        return RightCodeLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_common_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("COMMON_MODEL") or DEFAULT_COMMON_MODEL


def get_common_kg_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("COMMON_KG_MODEL") or os.getenv("COMMON_MODEL") or DEFAULT_COMMON_MODEL


def get_common_thinking_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("COMMON_THINKING_MODEL")
        or os.getenv("COMMON_MODEL")
        or DEFAULT_COMMON_MODEL
    )


def get_common_judge_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("COMMON_JUDGE_MODEL")
        or os.getenv("COMMON_MODEL")
        or DEFAULT_COMMON_MODEL
    )


def get_common_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("COMMON_BASE_URL") or DEFAULT_COMMON_BASE_URL


def get_rightcode_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("RIGHTCODE_MODEL") or DEFAULT_RIGHTCODE_MODEL


def get_rightcode_kg_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("RIGHTCODE_KG_MODEL") or os.getenv("RIGHTCODE_MODEL") or DEFAULT_RIGHTCODE_MODEL


def get_rightcode_thinking_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("RIGHTCODE_THINKING_MODEL")
        or os.getenv("RIGHTCODE_MODEL")
        or DEFAULT_RIGHTCODE_MODEL
    )


def get_rightcode_judge_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("RIGHTCODE_JUDGE_MODEL")
        or os.getenv("RIGHTCODE_MODEL")
        or DEFAULT_RIGHTCODE_MODEL
    )


def get_rightcode_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("RIGHTCODE_BASE_URL") or DEFAULT_RIGHTCODE_BASE_URL


def normalize_llm_provider(provider: str | None) -> str:
    normalized = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if normalized not in LLM_PROVIDER_CHOICES:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: {', '.join(LLM_PROVIDER_CHOICES)}")
    return normalized


def resolve_llm_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_model(explicit_model)
    return get_rightcode_model(explicit_model)


def resolve_kg_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_kg_model(explicit_model)
    return get_rightcode_kg_model(explicit_model)


def resolve_thinking_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_thinking_model(explicit_model)
    return get_rightcode_thinking_model(explicit_model)


def resolve_judge_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_judge_model(explicit_model)
    return get_rightcode_judge_model(explicit_model)


def resolve_llm_base_url(provider: str, explicit_base_url: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_base_url(explicit_base_url)
    return get_rightcode_base_url(explicit_base_url)


def get_common_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("COMMON_TIMEOUT_SECONDS", "60"))


def get_rightcode_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("RIGHTCODE_TIMEOUT_SECONDS", "60"))


def resolve_llm_timeout(provider: str, explicit_timeout: float | None = None) -> float:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "common":
        return get_common_timeout(explicit_timeout)
    return get_rightcode_timeout(explicit_timeout)


def _read_common_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("COMMON_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("Common API key is required. Set COMMON_API_KEY or pass --api-key.")
    return api_key.strip()


def _read_rightcode_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("RIGHTCODE_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("RightCode API key is required. Set RIGHTCODE_API_KEY or pass --api-key.")
    return api_key.strip()


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        completions_path: str = OPENAI_CHAT_COMPLETIONS_PATH,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        self.provider = normalize_llm_provider(provider)
        if self.provider == "common":
            self.api_key = _read_common_api_key(api_key) if transport is None else (api_key or "test-key")
        else:
            self.api_key = _read_rightcode_api_key(api_key) if transport is None else (api_key or "test-key")
        self.model = resolve_llm_model(self.provider, model)
        self.base_url = resolve_llm_base_url(self.provider, base_url).rstrip("/")
        self.completions_path = completions_path
        self.timeout_seconds = resolve_llm_timeout(self.provider, timeout_seconds)
        self.last_usage: dict[str, Any] | None = None
        self._transport = transport

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        start_time = time.perf_counter()
        logger.info(
            "llm.chat.start provider=%s mode=online model=%s base_url=%s path=%s message_count=%s",
            self.provider,
            self.model,
            self.base_url,
            self.completions_path,
            len(messages),
        )
        try:
            self.last_usage = None
            content = self._transport(messages) if self._transport else self._chat_completions(messages, temperature)
        except Exception:
            logger.exception(
                "llm.chat.failed provider=%s mode=online elapsed_seconds=%.2f",
                self.provider,
                time.perf_counter() - start_time,
            )
            raise
        logger.info(
            "llm.chat.done provider=%s mode=online elapsed_seconds=%.2f response_chars=%s",
            self.provider,
            time.perf_counter() - start_time,
            len(content),
        )
        return content

    def _chat_completions(self, messages: list[ChatMessage], temperature: float) -> str:
        response = httpx.post(
            f"{self.base_url}{self.completions_path}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
        usage = payload.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else None
        if self.last_usage:
            logger.info("llm.chat.usage provider=%s model=%s usage=%s", self.provider, self.model, self.last_usage)
        return payload["choices"][0]["message"]["content"]

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            raise RuntimeError(
                f"{self.provider} request failed: status={response.status_code}, model={self.model}, "
                f"base_url={self.base_url}, response={body}."
            ) from exc


class CommonLLMClient(OpenAICompatibleLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(
            provider="common",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            completions_path=OPENAI_CHAT_COMPLETIONS_PATH,
            transport=transport,
        )


class RightCodeLLMClient(OpenAICompatibleLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(
            provider="rightcode",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            completions_path=OPENAI_CHAT_COMPLETIONS_PATH,
            transport=transport,
        )
