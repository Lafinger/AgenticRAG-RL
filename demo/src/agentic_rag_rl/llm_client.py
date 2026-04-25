from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

import httpx


DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    payload = json.loads(cleaned)
    if not isinstance(payload, list):
        raise ValueError("LLM seed QA response must be a JSON array.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Every seed QA item must be a JSON object.")
    return payload


def _read_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("Doubao API key is required. Set ARK_API_KEY or DOUBAO_API_KEY.")
    return api_key.strip()


def get_doubao_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_MODEL") or DEFAULT_DOUBAO_MODEL


def get_doubao_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("DOUBAO_BASE_URL") or DEFAULT_DOUBAO_BASE_URL


def get_doubao_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    value = os.getenv("DOUBAO_TIMEOUT_SECONDS", "60")
    return float(value)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


class DoubaoSeedQAClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.api_key = _read_api_key(api_key) if transport is None else (api_key or "test-key")
        self.model = get_doubao_model(model)
        self.base_url = _normalize_base_url(get_doubao_base_url(base_url))
        self.timeout_seconds = get_doubao_timeout(timeout_seconds)
        self._transport = transport

    def generate_seed_qa(self, chunk_text: str, *, max_items: int) -> list[dict[str, Any]]:
        messages = self._build_messages(chunk_text, max_items=max_items)
        content = self._transport(messages) if self._transport else self._chat_completions(messages)
        records = extract_json_array(content)
        return [self._normalize_record(record) for record in records[:max_items] if self._is_usable_record(record)]

    def _build_messages(self, chunk_text: str, *, max_items: int) -> list[dict[str, str]]:
        system_prompt = (
            "你是中文小说阅读问答数据构造专家。"
            "请只基于给定片段生成可由该片段直接回答的 seed QA。"
            "问题类型应覆盖人物身份、人物关系、地点归属、事件原因、事件结果、人物行为。"
            "只能输出 JSON 数组，不要输出解释。"
        )
        user_prompt = f"""请为下面《平凡的世界》片段生成最多 {max_items} 条 seed QA。

要求：
1. 每条必须能从片段中直接找到证据。
2. answer 要短，避免长段摘抄。
3. qa_type 只能取 character_identity、character_relation、place_origin、event_cause、event_result、character_behavior、inference。
4. entities 写出问题涉及的人物、地点或事件关键词。
5. 输出 JSON 数组，字段为 question、answer、qa_type、entities。

片段：
{chunk_text}
"""
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _chat_completions(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _is_usable_record(record: dict[str, Any]) -> bool:
        return bool(str(record.get("question", "")).strip()) and bool(str(record.get("answer", "")).strip())

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        entities = record.get("entities", [])
        if isinstance(entities, str):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []
        return {
            "question": str(record.get("question", "")).strip(),
            "answer": str(record.get("answer", "")).strip(),
            "qa_type": str(record.get("qa_type", "inference")).strip() or "inference",
            "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
        }
