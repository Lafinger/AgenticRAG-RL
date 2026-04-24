from __future__ import annotations

from typing import Any, Iterable

from .io import chunk_map
from .protocols import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH, format_tool_response, make_tool_call
from .types import Chunk, MultiHopExample


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
                    "content": f"<think>{hop.question}</think>{make_tool_call('keyword_search', hop.question)}",
                }
            )
            messages.append({"role": "tool", "content": format_tool_response(records)})
            gold_chunks.append(hop.doc_chunk_id)
        messages.append({"role": "assistant", "content": f"<answer>{example.final_answer}</answer>"})
        traces.append(
            {
                "messages": messages,
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


def convert_traces_to_sft_records(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        messages: list[dict[str, Any]] = []
        for message in trace["messages"]:
            if message["role"] == "tool":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append(message)
        records.append({"messages": messages, "metadata": trace["metadata"]})
    return records


def convert_traces_to_sharegpt(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": record["messages"], "metadata": record["metadata"]} for record in records]
