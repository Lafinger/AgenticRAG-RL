from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sft_label_mask import (
    ACTION_START_TAG_WEIGHT,
    ANSWER_END_TAG_WEIGHT,
    ASSISTANT_END_WEIGHT,
    ASSISTANT_START_TOKEN_WEIGHT,
    DEFAULT_LABEL_WEIGHT,
    IGNORE_INDEX,
    IM_END_MARKER,
    THINK_END_TAG_WEIGHT,
    TOOL_CALL_END_TAG_WEIGHT,
    TOOL_CALL_START_TAG_WEIGHT,
    find_assistant_spans,
    tokenize_chat_with_assistant_labels,
)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.last_tools: list[dict[str, Any]] | None = None

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
        if add_special_tokens:
            raise AssertionError("This fake tokenizer does not add special tokens.")
        payload: dict[str, Any] = {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload


def load_trace_module() -> Any:
    spec = importlib.util.spec_from_file_location("train_sft_unsloth_trace_for_test", ROOT / "scripts" / "train_sft_unsloth_trace.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assistant_only_labels_mask_non_assistant_turns() -> None:
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
        {"role": "user", "content": "<tool_response>证据</tool_response>"},
        {"role": "assistant", "content": "<answer>答案</answer>"},
    ]

    sample = tokenize_chat_with_assistant_labels(FakeTokenizer(), messages)
    spans = find_assistant_spans(sample.text)
    assert spans
    assert len(sample.loss_weights) == len(sample.labels)

    for index, label in enumerate(sample.labels):
        inside_assistant = any(index >= start and index < end for start, end in spans)
        if inside_assistant:
            assert label == sample.input_ids[index]
            assert sample.loss_weights[index] > 0.0
        else:
            assert label == IGNORE_INDEX
            assert sample.loss_weights[index] == 0.0
    supervised_text = "".join(chr(label) for label in sample.labels if label != IGNORE_INDEX)
    assert supervised_text.count(IM_END_MARKER) == 2


def test_multiple_assistant_turns_are_supervised() -> None:
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "第一次回答"},
        {"role": "user", "content": "<tool_response>证据</tool_response>"},
        {"role": "assistant", "content": "第二次回答"},
    ]

    sample = tokenize_chat_with_assistant_labels(FakeTokenizer(), messages)
    supervised_text = "".join(chr(label) for label in sample.labels if label != IGNORE_INDEX)

    assert "第一次回答" in supervised_text
    assert "第二次回答" in supervised_text
    assert supervised_text.count(IM_END_MARKER) == 2
    assert "<tool_response>证据</tool_response>" not in supervised_text


def test_loss_false_assistant_turn_is_context_only() -> None:
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
        {
            "role": "assistant",
            "content": '<think>要回答最终问题，先查：第一跳</think>\n<tool_call>\n{"name":"keyword_search","arguments":{"query":"第一跳"}}\n</tool_call>',
            "loss": False,
        },
        {"role": "tool", "content": "[chunk-a] 证据"},
        {
            "role": "assistant",
            "content": '<think>已获得上一跳线索“线索”，继续查：第二跳</think>\n<tool_call>\n{"name":"keyword_search","arguments":{"query":"第二跳"}}\n</tool_call>',
        },
    ]

    sample = tokenize_chat_with_assistant_labels(FakeTokenizer(), messages)
    supervised_text = "".join(chr(label) for label in sample.labels if label != IGNORE_INDEX)

    assert "第一跳" not in supervised_text
    assert "第二跳" in supervised_text
    assert supervised_text.count(IM_END_MARKER) == 1
    assert "[chunk-a] 证据" not in supervised_text


def test_tools_rendering_masks_tool_role_response() -> None:
    tokenizer = FakeTokenizer()
    tools = [{"type": "function", "function": {"name": "keyword_search", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
        {
            "role": "assistant",
            "content": '<think>要回答最终问题，先查：问题</think>\n<tool_call>\n{"name":"keyword_search","arguments":{"query":"问题"}}\n</tool_call>',
        },
        {"role": "tool", "content": "[chunk-a] 证据"},
        {"role": "assistant", "content": "<answer>答案</answer>"},
    ]

    sample = tokenize_chat_with_assistant_labels(tokenizer, messages, tools=tools)
    supervised_text = "".join(chr(label) for label in sample.labels if label != IGNORE_INDEX)

    assert "# Tools" in sample.text
    assert "first write exactly one short search intent" in sample.text
    assert "<think>要回答最终问题，先查：问题</think>" in supervised_text
    assert "<tool_call>" in supervised_text
    assert "<answer>答案</answer>" in supervised_text
    assert supervised_text.count(IM_END_MARKER) == 2
    assert "<think>\n\n</think>\n\n<answer>" not in supervised_text
    assert "[chunk-a] 证据" not in supervised_text
    assert "<tool_response>" not in supervised_text


def test_protocol_boundary_tokens_receive_higher_loss_weights() -> None:
    content = (
        '<think>要回答最终问题，先查：问题</think>\n'
        '<tool_call>\n{"name":"keyword_search","arguments":{"query":"问题"}}\n</tool_call>'
    )
    sample = tokenize_chat_with_assistant_labels(
        FakeTokenizer(),
        [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": content},
        ],
    )

    def weights_for(fragment: str, *, last: bool = False) -> list[float]:
        start = sample.text.rindex(fragment) if last else sample.text.index(fragment)
        return sample.loss_weights[start : start + len(fragment)]

    assert weights_for("<think>")[0] == ASSISTANT_START_TOKEN_WEIGHT
    assert min(weights_for("<think>")) >= ACTION_START_TAG_WEIGHT
    assert min(weights_for("</think>")) >= THINK_END_TAG_WEIGHT
    assert min(weights_for("<tool_call>")) >= TOOL_CALL_START_TAG_WEIGHT
    assert max(weights_for("</tool_call>")) == TOOL_CALL_END_TAG_WEIGHT
    assert min(weights_for(IM_END_MARKER, last=True)) >= ASSISTANT_END_WEIGHT


def test_answer_boundary_tokens_receive_higher_loss_weights() -> None:
    content = "<answer>答案</answer>"
    sample = tokenize_chat_with_assistant_labels(
        FakeTokenizer(),
        [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": content},
        ],
    )

    def weights_for(fragment: str, *, last: bool = False) -> list[float]:
        start = sample.text.rindex(fragment) if last else sample.text.index(fragment)
        return sample.loss_weights[start : start + len(fragment)]

    assert weights_for("<answer>")[0] == ASSISTANT_START_TOKEN_WEIGHT
    assert min(weights_for("<answer>")) >= ACTION_START_TAG_WEIGHT
    assert min(weights_for("</answer>")) >= ANSWER_END_TAG_WEIGHT
    assert min(weights_for(IM_END_MARKER, last=True)) >= ASSISTANT_END_WEIGHT


def test_truncation_without_assistant_label_raises_clear_error() -> None:
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]

    with pytest.raises(ValueError, match="no assistant labels after truncation"):
        tokenize_chat_with_assistant_labels(FakeTokenizer(), messages, max_length=5)


def test_trace_collate_keeps_label_shape_and_masks_padding() -> None:
    trace_module = load_trace_module()

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values: list[list[int]], dtype: Any) -> list[list[int]]:
            return values

    batch = trace_module.collate_trace_features(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [IGNORE_INDEX, 2, 3],
                "sample_id": "a",
                "source_line": 1,
                "token_length": 3,
                "truncated": False,
                "supervised_token_count": 2,
                "question": "q1",
                "answer": "a1",
                "hop_count": 1,
                "gold_chunks": ["c1"],
            },
            {
                "input_ids": [4],
                "attention_mask": [1],
                "labels": [4],
                "sample_id": "b",
                "source_line": 2,
                "token_length": 1,
                "truncated": False,
                "supervised_token_count": 1,
                "question": "q2",
                "answer": "a2",
                "hop_count": 1,
                "gold_chunks": ["c2"],
            },
        ],
        FakeTokenizer(),
        FakeTorch,
    )

    assert len(batch["input_ids"][0]) == len(batch["labels"][0])
    assert len(batch["input_ids"][1]) == len(batch["labels"][1])
    assert batch["input_ids"][1] == [4, 0, 0]
    assert batch["labels"][1] == [4, IGNORE_INDEX, IGNORE_INDEX]
    assert batch["loss_weights"][1] == [1.0, 0.0, 0.0]
