from __future__ import annotations

from agentic_rag_rl.protocols import TOOL_SCHEMAS, extract_answer_tag, make_tool_call, normalize_tool_spec


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
