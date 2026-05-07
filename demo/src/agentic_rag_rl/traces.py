from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .io import chunk_map
from .protocols import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH, TOOL_SCHEMAS, format_tool_response, make_tool_call
from .types import Chunk, MultiHopExample


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"\s*<tool_response>\s*(.*?)\s*</tool_response>\s*", re.DOTALL)


def _trace_system_prompt(use_zh: bool) -> str:
    return SYSTEM_PROMPT_ZH if use_zh else SYSTEM_PROMPT_EN


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(chunk.metadata)
    if chunk.company:
        metadata["company"] = chunk.company
    return metadata


def build_oracle_traces(examples: Iterable[MultiHopExample], chunks: list[Chunk], use_zh: bool = True) -> list[dict[str, Any]]:
    corpus = chunk_map(chunks)
    traces: list[dict[str, Any]] = []
    for example in examples:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _trace_system_prompt(use_zh)},
            {"role": "user", "content": example.final_question},
        ]
        gold_chunks: list[str] = []
        for hop in example.hops:
            chunk = corpus[hop.doc_chunk_id]
            records = [
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source": "oracle",
                    "metadata": _chunk_metadata(chunk),
                }
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{hop.question}</think>\n{make_tool_call('keyword_search', hop.question)}",
                }
            )
            messages.append({"role": "tool", "content": format_tool_response(records)})
            gold_chunks.append(hop.doc_chunk_id)
        messages.append({"role": "assistant", "content": f"<answer>{example.final_answer}</answer>"})
        traces.append(
            {
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "metadata": {
                    "final_question": example.final_question,
                    "final_answer": example.final_answer,
                    "hop_count": example.hop_count,
                    "subset": example.subset,
                    "qa_type": example.qa_type,
                    "gold_chunks": gold_chunks,
                    "answer_aliases": example.answer_aliases or [example.final_answer],
                },
            }
        )
    return traces


def _strip_tool_response_tags(content: str) -> str:
    matched = TOOL_RESPONSE_RE.fullmatch(content)
    return matched.group(1).strip() if matched else content


def _normalize_assistant_content(content: str) -> str:
    tool_match = TOOL_CALL_RE.search(content)
    if tool_match is None:
        return content

    think_match = THINK_RE.search(content)
    think = think_match.group(1).strip() if think_match else ""
    raw_payload = tool_match.group(1).strip()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        payload_text = raw_payload
    else:
        payload_text = json.dumps(payload, ensure_ascii=False)
    return f"<think>{think}</think>\n<tool_call>\n{payload_text}\n</tool_call>"


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    content = str(message.get("content", ""))
    if role == "assistant":
        content = _normalize_assistant_content(content)
    elif role == "tool":
        content = _strip_tool_response_tags(content)
    return {"role": role, "content": content}


def _metadata_with_sample_type(metadata: dict[str, Any], sample_type: str) -> dict[str, Any]:
    return {**metadata, "sample_type": sample_type}


def _build_finalization_record(messages: list[dict[str, Any]], tools: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if len(messages) < 3:
        raise ValueError("SFT trace is too short to build finalization sample.")
    system_message = messages[0]
    user_message = messages[1]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    final_answer = str(metadata.get("final_answer", "")).strip()
    if not final_answer:
        raise ValueError("SFT trace metadata is missing final_answer.")
    finalization_messages = [
        {"role": "system", "content": system_message["content"]},
        {"role": "user", "content": user_message["content"]},
        *tool_messages,
        {"role": "assistant", "content": f"<answer>{final_answer}</answer>"},
    ]
    return {
        "messages": finalization_messages,
        "tools": tools,
        "metadata": {
            **_metadata_with_sample_type(metadata, "finalization_only"),
            "tool_response_count": len(tool_messages),
        },
    }


def convert_traces_to_sft_records(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        messages = [_normalize_message(message) for message in trace["messages"]]
        tools = trace.get("tools") if isinstance(trace.get("tools"), list) else TOOL_SCHEMAS
        metadata = dict(trace["metadata"])
        records.append(
            {
                "messages": messages,
                "tools": tools,
                "metadata": {
                    **_metadata_with_sample_type(metadata, "full_trace"),
                    "tool_turn_count": sum(1 for message in messages if message["role"] == "assistant" and "<tool_call>" in message["content"]),
                },
            }
        )
        records.append(_build_finalization_record(messages, tools, metadata))
    return records


def convert_traces_to_sharegpt(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "messages": record["messages"],
            "tools": record.get("tools", TOOL_SCHEMAS),
            "metadata": record["metadata"],
        }
        for record in records
    ]
