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
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "使用关键词匹配检索中文小说段落，适合精确查找人物、地点、事件和关系线索。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索查询关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dense_search",
            "description": "使用向量语义检索中文小说段落，适合查找表达不同但语义相关的证据。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "语义搜索查询"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "融合关键词和向量检索结果并重排，适合需要多证据召回的小说问答。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索查询"}},
                "required": ["query"],
            },
        },
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
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n</tool_call>"


def format_tool_response(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(f"[{record['chunk_id']}] {record['text']}")
    return "\n".join(lines)


def extract_answer_tag(text: str) -> str:
    matched = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matched.group(1).strip() if matched else ""
