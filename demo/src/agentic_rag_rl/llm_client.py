from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Protocol, TypedDict

import httpx


DEFAULT_LLM_PROVIDER = "doubao"
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DOUBAO_THINKING_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DOUBAO_JUDGE_MODEL = DEFAULT_DOUBAO_THINKING_MODEL
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
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
    if provider != "doubao":
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return DoubaoLLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def get_doubao_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_MODEL") or DEFAULT_DOUBAO_MODEL


def get_doubao_thinking_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_THINKING_MODEL") or DEFAULT_DOUBAO_THINKING_MODEL


def get_doubao_judge_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_JUDGE_MODEL") or os.getenv("DOUBAO_THINKING_MODEL") or DEFAULT_DOUBAO_JUDGE_MODEL


def get_doubao_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("DOUBAO_BASE_URL") or DEFAULT_DOUBAO_BASE_URL


def get_doubao_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("DOUBAO_TIMEOUT_SECONDS", "60"))


def _read_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("ARK_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("Doubao API key is required. Set ARK_API_KEY.")
    return api_key.strip()


class DoubaoLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        self.api_key = _read_api_key(api_key) if transport is None else (api_key or "test-key")
        self.model = get_doubao_model(model)
        self.base_url = get_doubao_base_url(base_url).rstrip("/")
        self.timeout_seconds = get_doubao_timeout(timeout_seconds)
        self._transport = transport

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        start_time = time.perf_counter()
        logger.info(
            "llm.chat.start provider=doubao model=%s base_url=%s message_count=%s",
            self.model,
            self.base_url,
            len(messages),
        )
        try:
            content = self._transport(messages) if self._transport else self._chat_completions(messages, temperature)
        except Exception:
            logger.exception("llm.chat.failed provider=doubao elapsed_seconds=%.2f", time.perf_counter() - start_time)
            raise
        logger.info(
            "llm.chat.done provider=doubao elapsed_seconds=%.2f response_chars=%s",
            time.perf_counter() - start_time,
            len(content),
        )
        return content

    def _chat_completions(self, messages: list[ChatMessage], temperature: float) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            hint = ""
            if "InvalidEndpointOrModel.NotFound" in body:
                hint = (
                    " Hint: the configured Doubao model or Ark endpoint is not available for this account. "
                    "Set DOUBAO_THINKING_MODEL to an enabled Ark model/endpoint, pass --merge-model, "
                    "or use --disable-llm-merge for offline synthesis."
                )
            raise RuntimeError(
                f"Doubao request failed: status={response.status_code}, model={self.model}, "
                f"base_url={self.base_url}, response={body}.{hint}"
            ) from exc
