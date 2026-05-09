from __future__ import annotations

from agentic_rag_rl.protocols import (
    TOOL_SCHEMAS,
    extract_answer_tag,
    make_tool_call,
    normalize_tool_spec,
    render_canonical_chat,
    render_tools_system_prompt,
)


def test_normalize_tool_spec_handles_legacy_and_hybrid_cases() -> None:
    assert normalize_tool_spec("keyword_search") == (["keyword_search"], False)
    assert normalize_tool_spec(["keyword_search", "dense_search"]) == (["keyword_search", "dense_search"], True)
    assert normalize_tool_spec("hybrid_search", fallback_tools=["keyword_search", "dense_search"]) == (
        ["keyword_search", "dense_search"],
        True,
    )


def test_extract_answer_tag() -> None:
    text = "前置文本<answer>双水村</answer>"
    assert extract_answer_tag(text) == "双水村"


def test_qwen3_tool_schema_and_tool_call_format() -> None:
    assert TOOL_SCHEMAS[0]["type"] == "function"
    assert TOOL_SCHEMAS[0]["function"]["name"] == "keyword_search"
    assert make_tool_call("keyword_search", "问题").startswith("<tool_call>\n")


def test_canonical_renderer_injects_react_tool_contract_and_tool_response() -> None:
    rendered = render_canonical_chat(
        [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "问题"},
            {"role": "tool", "content": "[chunk-a] 证据"},
            {"role": "assistant", "content": "<answer>答案</answer>"},
        ],
        tools=TOOL_SCHEMAS,
    )

    assert "# Tools" in rendered
    assert "first write exactly one short search intent" in rendered
    assert "Never start a turn with <tool_call> or </tool_call>" in rendered
    assert "<think>要回答最终问题，先查：<query></think>" in rendered
    assert "<tool_response>\n[chunk-a] 证据\n</tool_response>" in rendered
    assert "<think>\n\n</think>\n\n<answer>" not in rendered
    assert "<|im_start|>assistant\n<answer>答案</answer><|im_end|>" in rendered


def test_render_tools_system_prompt_is_idempotent() -> None:
    once = render_tools_system_prompt("系统提示", TOOL_SCHEMAS)
    twice = render_tools_system_prompt(once, TOOL_SCHEMAS)

    assert once == twice
    assert once.count("# Tools") == 1
