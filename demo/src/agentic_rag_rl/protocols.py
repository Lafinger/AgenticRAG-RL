from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_AGENT_NAME = "tool_agent"

SYSTEM_PROMPT_ZH = (
    "你是一个中文小说阅读问答 Agent。"
    "你需要通过文本检索工具逐步搜索人物、地点、事件和关系证据，最后用 <answer>...</answer> 输出最终答案。"
)
SYSTEM_PROMPT_EN = (
    "You are a Chinese novel reading QA agent. "
    "Use text search tools step by step and end with <answer>...</answer>."
)

TOOL_SCHEMAS = [
    {
        "name": "keyword_search",
        "description": "Use BM25 style keyword retrieval for precise character, place, and event mentions.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "dense_search",
        "description": "Use dense retrieval for semantic matching in novel passages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "hybrid_search",
        "description": "Use keyword and dense retrieval together with fusion and rerank over novel chunks.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]


def normalize_tool_spec(tool: str | list[str], fallback_tools: list[str] | None = None) -> tuple[list[str], bool]:
    if isinstance(tool, list):
        return tool, len(tool) > 1
    if tool == "hybrid_search":
        return list(fallback_tools or []), True
    return [tool], False


def make_tool_call(name: str, query: str) -> str:
    payload = {"name": name, "arguments": {"query": query}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def format_tool_response(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(f"[{record['chunk_id']}] {record['text']}")
    return "<tool_response>" + "\n".join(lines) + "</tool_response>"


def extract_answer_tag(text: str) -> str:
    matched = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matched.group(1).strip() if matched else ""
