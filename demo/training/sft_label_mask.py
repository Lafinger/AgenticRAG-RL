from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


IGNORE_INDEX = -100
ASSISTANT_START_MARKER = "<|im_start|>assistant\n"
IM_END_MARKER = "<|im_end|>"


@dataclass(frozen=True)
class MaskedChatSample:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_length: int
    supervised_token_count: int


def find_assistant_spans(rendered_text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        marker_index = rendered_text.find(ASSISTANT_START_MARKER, cursor)
        if marker_index < 0:
            break

        content_start = marker_index + len(ASSISTANT_START_MARKER)
        content_end = rendered_text.find(IM_END_MARKER, content_start)
        if content_end < 0:
            raise ValueError("Rendered chat contains an assistant turn without <|im_end|>.")

        span_end = content_end + len(IM_END_MARKER)
        if span_end > content_start:
            spans.append((content_start, span_end))
        cursor = content_end + len(IM_END_MARKER)
    return spans


def _token_overlaps_spans(token_start: int, token_end: int, spans: Sequence[tuple[int, int]]) -> bool:
    if token_end <= token_start:
        return False
    return any(token_start < span_end and token_end > span_start for span_start, span_end in spans)


def tokenize_chat_with_assistant_labels(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_length: int | None = None,
) -> MaskedChatSample:
    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    assistant_spans = find_assistant_spans(rendered_text)
    if not assistant_spans:
        raise ValueError("Rendered chat has no assistant turn to supervise.")

    encoded = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    offsets = list(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned offset_mapping with a different length from input_ids.")

    labels = [
        token_id if _token_overlaps_spans(int(start), int(end), assistant_spans) else IGNORE_INDEX
        for token_id, (start, end) in zip(input_ids, offsets, strict=True)
    ]
    token_length = len(input_ids)

    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    supervised_token_count = sum(1 for label in labels if label != IGNORE_INDEX)
    if supervised_token_count == 0:
        raise ValueError("Rendered chat has no assistant labels after truncation; increase max_seq_length or inspect the sample.")

    return MaskedChatSample(
        text=rendered_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        token_length=token_length,
        supervised_token_count=supervised_token_count,
    )
