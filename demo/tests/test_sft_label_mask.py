from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sft_label_mask import IGNORE_INDEX, find_assistant_spans, tokenize_chat_with_assistant_labels


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize or add_generation_prompt:
            raise AssertionError("This fake tokenizer only supports rendered training chats.")
        return "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)

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

    for index, label in enumerate(sample.labels):
        inside_assistant = any(index >= start and index < end for start, end in spans)
        if inside_assistant:
            assert label == sample.input_ids[index]
        else:
            assert label == IGNORE_INDEX


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
    assert "<tool_response>证据</tool_response>" not in supervised_text


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
