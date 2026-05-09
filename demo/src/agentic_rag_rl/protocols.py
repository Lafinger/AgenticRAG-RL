from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_AGENT_NAME = "tool_agent"
IM_START_MARKER = "<|im_start|>"
IM_END_MARKER = "<|im_end|>"
ASSISTANT_START_MARKER = f"{IM_START_MARKER}assistant\n"
MAX_TOOL_RESPONSE_TEXT_CHARS = 420

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


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    tools_json = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return (
        "\n\n"
        "# Tools\n\n"
        "You may call one function per assistant turn to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "For each function call, first write exactly one short search intent within <think></think> XML tags. "
        "Then return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<think>要回答最终问题，先查：<query></think>\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n\n"
        "Every assistant turn must start with either <think> or <answer>. "
        "Never start a turn with <tool_call> or </tool_call>, and never emit a closing tag unless you opened the same tag in that turn.\n\n"
        "When enough evidence has been retrieved, stop calling tools and answer with <answer>...</answer>."
    )


def render_tools_system_prompt(system_prompt: str, tools: list[dict[str, Any]] | None = None) -> str:
    if not tools or "# Tools" in system_prompt:
        return system_prompt
    return f"{system_prompt}{render_tools_block(tools)}"


def render_tool_response_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("<tool_response>") and stripped.endswith("</tool_response>"):
        return stripped
    return f"<tool_response>\n{content}\n</tool_response>"


def render_canonical_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "system":
            content = render_tools_system_prompt(content, tools)
        elif role == "tool":
            role = "user"
            content = render_tool_response_content(content)
        elif role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role!r}.")
        rendered.append(f"{IM_START_MARKER}{role}\n{content}{IM_END_MARKER}\n")
    if add_generation_prompt:
        rendered.append(f"{IM_START_MARKER}assistant\n")
    return "".join(rendered)


def format_tool_response(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(f"[{record['chunk_id']}] {truncate_tool_response_text(str(record['text']))}")
    return "\n".join(lines)


def extract_answer_tag(text: str) -> str:
    matched = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matched.group(1).strip() if matched else ""


def truncate_tool_response_text(text: str, max_chars: int = MAX_TOOL_RESPONSE_TEXT_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}..."
