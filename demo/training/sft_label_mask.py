from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag_rl.protocols import ASSISTANT_START_MARKER, IM_END_MARKER, render_canonical_chat


IGNORE_INDEX = -100
DEFAULT_LABEL_WEIGHT = 1.0
ASSISTANT_START_TOKEN_WEIGHT = 12.0
ACTION_START_TAG_WEIGHT = 12.0
TOOL_CALL_START_TAG_WEIGHT = 4.0
ASSISTANT_END_WEIGHT = 4.0


@dataclass(frozen=True)
class MaskedChatSample:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    loss_weights: list[float]
    token_length: int
    supervised_token_count: int


def find_assistant_spans(
    rendered_text: str,
    supervise_assistant_turns: Sequence[bool] | None = None,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    assistant_turn_index = 0
    while True:
        marker_index = rendered_text.find(ASSISTANT_START_MARKER, cursor)
        if marker_index < 0:
            break

        content_start = marker_index + len(ASSISTANT_START_MARKER)
        content_end = rendered_text.find(IM_END_MARKER, content_start)
        if content_end < 0:
            raise ValueError("Rendered chat contains an assistant turn without <|im_end|>.")

        span_end = content_end + len(IM_END_MARKER)
        if supervise_assistant_turns is not None and assistant_turn_index >= len(supervise_assistant_turns):
            raise ValueError("Rendered assistant turn count does not match message loss metadata.")
        supervise = True if supervise_assistant_turns is None else supervise_assistant_turns[assistant_turn_index]
        if supervise and span_end > content_start:
            spans.append((content_start, span_end))
        assistant_turn_index += 1
        cursor = content_end + len(IM_END_MARKER)
    if supervise_assistant_turns is not None and assistant_turn_index != len(supervise_assistant_turns):
        raise ValueError("Rendered assistant turn count does not match message loss metadata.")
    return spans


def _token_overlaps_spans(token_start: int, token_end: int, spans: Sequence[tuple[int, int]]) -> bool:
    if token_end <= token_start:
        return False
    return any(token_start < span_end and token_end > span_start for span_start, span_end in spans)


def _find_literal_spans(text: str, literal: str, assistant_spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = text.find(literal, cursor)
        if index < 0:
            break
        span = (index, index + len(literal))
        if _token_overlaps_spans(span[0], span[1], assistant_spans):
            spans.append(span)
        cursor = index + len(literal)
    return spans


def _build_loss_weights(
    rendered_text: str,
    offsets: Sequence[tuple[int, int]],
    labels: Sequence[int],
    assistant_spans: Sequence[tuple[int, int]],
) -> list[float]:
    weights = [DEFAULT_LABEL_WEIGHT if label != IGNORE_INDEX else 0.0 for label in labels]

    high_weight_spans = _find_literal_spans(rendered_text, "<think>", assistant_spans)
    high_weight_spans.extend(_find_literal_spans(rendered_text, "<answer>", assistant_spans))
    tool_call_start_spans = _find_literal_spans(rendered_text, "<tool_call>", assistant_spans)
    assistant_end_spans = _find_literal_spans(rendered_text, IM_END_MARKER, assistant_spans)

    for span_start, span_end in high_weight_spans:
        for index, (token_start, token_end) in enumerate(offsets):
            if labels[index] != IGNORE_INDEX and token_start < span_end and token_end > span_start:
                weights[index] = max(weights[index], ACTION_START_TAG_WEIGHT)

    for span_start, span_end in tool_call_start_spans:
        for index, (token_start, token_end) in enumerate(offsets):
            if labels[index] != IGNORE_INDEX and token_start < span_end and token_end > span_start:
                weights[index] = max(weights[index], TOOL_CALL_START_TAG_WEIGHT)

    for span_start, span_end in assistant_end_spans:
        for index, (token_start, token_end) in enumerate(offsets):
            if labels[index] != IGNORE_INDEX and token_start < span_end and token_end > span_start:
                weights[index] = max(weights[index], ASSISTANT_END_WEIGHT)

    for span_start, span_end in assistant_spans:
        for index, (token_start, token_end) in enumerate(offsets):
            if labels[index] != IGNORE_INDEX and token_start < span_end and token_end > span_start:
                weights[index] = max(weights[index], ASSISTANT_START_TOKEN_WEIGHT)
                break

    return weights


def _assistant_loss_enabled(message: dict[str, Any]) -> bool:
    if message.get("loss") is False:
        return False
    if message.get("train") is False:
        return False
    if message.get("trainable") is False:
        return False
    return True


def _assistant_loss_flags(messages: list[dict[str, Any]]) -> list[bool]:
    return [_assistant_loss_enabled(message) for message in messages if message.get("role") == "assistant"]


def tokenize_chat_with_assistant_labels(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_length: int | None = None,
) -> MaskedChatSample:
    rendered_text = render_canonical_chat(messages, tools=tools, add_generation_prompt=False)
    assistant_spans = find_assistant_spans(rendered_text, _assistant_loss_flags(messages))
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
    loss_weights = _build_loss_weights(
        rendered_text,
        [(int(start), int(end)) for start, end in offsets],
        labels,
        assistant_spans,
    )
    token_length = len(input_ids)

    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        loss_weights = loss_weights[:max_length]

    supervised_token_count = sum(1 for label in labels if label != IGNORE_INDEX)
    if supervised_token_count == 0:
        raise ValueError("Rendered chat has no assistant labels after truncation; increase max_seq_length or inspect the sample.")

    return MaskedChatSample(
        text=rendered_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
        token_length=token_length,
        supervised_token_count=supervised_token_count,
    )
