from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_convert_module() -> Any:
    spec = importlib.util.spec_from_file_location("convert_sft_to_unsloth_for_test", ROOT / "scripts" / "convert_sft_to_unsloth.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_record(content: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": content},
            {"role": "tool", "content": "[chunk] 证据"},
            {"role": "assistant", "content": "<answer>答案</answer>"},
        ],
        "tools": [],
        "metadata": {"sample_type": "full_trace"},
    }


def test_validate_record_accepts_react_tool_turn() -> None:
    module = load_convert_module()
    content = (
        "<think>要回答最终问题，先查：问题</think>\n"
        '<tool_call>\n{"name":"keyword_search","arguments":{"query":"问题"}}\n</tool_call>'
    )

    assert module.validate_record(valid_record(content), 1)["messages"][2]["content"] == content


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('<tool_call>\n{"name":"keyword_search","arguments":{"query":"问题"}}\n</tool_call>', "must be exactly"),
        ("<think>要回答最终问题，先查：问题</think>\n<tool_call>问题</tool_call>", "JSON is invalid"),
        (
            '<think>{"query":"问题"}</think>\n<tool_call>\n{"name":"keyword_search","arguments":{"query":"问题"}}\n</tool_call>',
            "must not contain JSON",
        ),
    ],
)
def test_validate_record_rejects_invalid_react_tool_turns(content: str, message: str) -> None:
    module = load_convert_module()

    with pytest.raises(ValueError, match=message):
        module.validate_record(valid_record(content), 1)

