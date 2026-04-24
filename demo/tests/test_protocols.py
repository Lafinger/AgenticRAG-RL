from __future__ import annotations

from agentic_rag_rl.protocols import extract_answer_tag, normalize_tool_spec


def test_normalize_tool_spec_handles_legacy_and_hybrid_cases() -> None:
    assert normalize_tool_spec("keyword_search") == (["keyword_search"], False)
    assert normalize_tool_spec(["keyword_search", "dense_search"]) == (["keyword_search", "dense_search"], True)
    assert normalize_tool_spec("hybrid_search", fallback_tools=["keyword_search", "dense_search"]) == (
        ["keyword_search", "dense_search"],
        True,
    )


def test_extract_answer_tag() -> None:
    text = "<think>...</think><answer>双水村</answer>"
    assert extract_answer_tag(text) == "双水村"
