from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .io import chunk_map
from .protocols import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH, TOOL_SCHEMAS, format_tool_response, make_tool_call
from .types import Chunk, MultiHopExample


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"\s*<tool_response>\s*(.*?)\s*</tool_response>\s*", re.DOTALL)
TOOL_SCHEMA_NAMES = {str(tool["function"]["name"]) for tool in TOOL_SCHEMAS}


def _trace_system_prompt(use_zh: bool) -> str:
    return SYSTEM_PROMPT_ZH if use_zh else SYSTEM_PROMPT_EN


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(chunk.metadata)
    if chunk.company:
        metadata["company"] = chunk.company
    return metadata


def _normalize_tool_name(tool_name: str) -> str:
    if tool_name == "semantic_search":
        return "dense_search"
    return tool_name


def select_trace_tool(search_tools: list[str] | None) -> tuple[str, list[str]]:
    unknown_tools: list[str] = []
    normalized_tools: list[str] = []
    for raw_tool in search_tools or ["keyword_search"]:
        tool = _normalize_tool_name(str(raw_tool))
        if tool in TOOL_SCHEMA_NAMES:
            if tool not in normalized_tools:
                normalized_tools.append(tool)
        else:
            unknown_tools.append(str(raw_tool))
    if not normalized_tools:
        return "keyword_search", unknown_tools
    if len(normalized_tools) == 1:
        return normalized_tools[0], unknown_tools
    return "hybrid_search", unknown_tools


def make_short_think(sub_query: str, hop_index: int = 0, previous_answer: str | None = None) -> str:
    query = sub_query.strip()
    if hop_index == 0:
        return f"要回答最终问题，先查：{query}"
    previous = str(previous_answer or "").strip()
    if previous:
        return f"已获得上一跳线索“{previous}”，继续查：{query}"
    return f"已获得上一跳线索，继续查：{query}"


def make_react_tool_action(name: str, query: str, hop_index: int = 0, previous_answer: str | None = None) -> str:
    return f"<think>{make_short_think(query, hop_index, previous_answer)}</think>\n{make_tool_call(name, query)}"


def _evidence_record_for_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "text": chunk.text,
        "source": "oracle",
        "metadata": _chunk_metadata(chunk),
    }


def render_react_messages(
    *,
    system_prompt: str,
    question: str,
    plan: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    final_answer: str,
) -> list[dict[str, Any]]:
    evidence_by_step = {entry.get("step_id"): entry for entry in evidence}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    previous_answer: str | None = None
    for hop_index, step in enumerate(plan):
        sub_query = str(step.get("sub_query", "")).strip()
        tool_name = str(step.get("tool") or "keyword_search")
        messages.append(
            {
                "role": "assistant",
                "content": make_react_tool_action(tool_name, sub_query, hop_index, previous_answer),
            }
        )
        entry = evidence_by_step.get(step.get("id"), {})
        results = entry.get("results", []) if isinstance(entry, dict) else []
        messages.append({"role": "tool", "content": format_tool_response(list(results))})
        previous_answer = str(step.get("gold_answer", "")).strip() or previous_answer
    messages.append({"role": "assistant", "content": f"<answer>{final_answer}</answer>"})
    return messages


def build_oracle_traces(examples: Iterable[MultiHopExample], chunks: list[Chunk], use_zh: bool = True) -> list[dict[str, Any]]:
    corpus = chunk_map(chunks)
    traces: list[dict[str, Any]] = []
    for example in examples:
        gold_chunks: list[str] = []
        plan: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        tool_fallbacks: list[dict[str, Any]] = []
        for hop_index, hop in enumerate(example.hops):
            chunk = corpus[hop.doc_chunk_id]
            tool_name, unknown_tools = select_trace_tool(hop.search_tools)
            if unknown_tools:
                tool_fallbacks.append({"hop_idx": hop.hop_idx, "unknown_tools": unknown_tools, "fallback_tool": tool_name})
            step_id = hop_index + 1
            plan.append(
                {
                    "id": step_id,
                    "sub_query": hop.question,
                    "tool": tool_name,
                    "depends_on": [] if hop_index == 0 else [hop_index],
                    "gold_answer": hop.answer,
                    "gold_chunk_id": hop.doc_chunk_id,
                }
            )
            evidence_records = [_evidence_record_for_chunk(chunk)]
            evidence.append(
                {
                    "step_id": step_id,
                    "sub_query": hop.question,
                    "tool": tool_name,
                    "results": evidence_records,
                }
            )
            tool_calls.append({"step_id": step_id, "tool": tool_name, "query": hop.question, "num_results": len(evidence_records)})
            gold_chunks.append(hop.doc_chunk_id)
        messages = render_react_messages(
            system_prompt=_trace_system_prompt(use_zh),
            question=example.final_question,
            plan=plan,
            evidence=evidence,
            final_answer=example.final_answer,
        )
        traces.append(
            {
                "question": example.final_question,
                "gold": example.final_answer,
                "pred": example.final_answer,
                "plan": plan,
                "tool_calls": tool_calls,
                "evidence": evidence,
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
                    "tool_fallbacks": tool_fallbacks,
                },
            }
        )
    return traces


def _strip_tool_response_tags(content: str) -> str:
    matched = TOOL_RESPONSE_RE.fullmatch(content)
    return matched.group(1).strip() if matched else content


def _normalize_assistant_content(content: str, hop_index: int = 0) -> str:
    tool_match = TOOL_CALL_RE.search(content)
    if tool_match is None:
        return content

    raw_payload = tool_match.group(1).strip()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        payload = {"name": "keyword_search", "arguments": {"query": raw_payload}}
    else:
        if not isinstance(payload, dict):
            payload = {"name": "keyword_search", "arguments": {"query": raw_payload}}
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    arguments = payload.get("arguments") if isinstance(payload, dict) else None
    query = arguments.get("query") if isinstance(arguments, dict) else raw_payload
    think = make_short_think(str(query), hop_index)
    return f"<think>{think}</think>\n<tool_call>\n{payload_text}\n</tool_call>"


def _normalize_message(message: dict[str, Any], hop_index: int = 0) -> dict[str, Any]:
    role = message["role"]
    content = str(message.get("content", ""))
    if role == "assistant":
        content = _normalize_assistant_content(content, hop_index)
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


def _first_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def _trace_question(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    return str(trace.get("question") or metadata.get("final_question") or _first_message_content(trace.get("messages", []), "user"))


def _trace_final_answer(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    return str(metadata.get("final_answer") or trace.get("pred") or trace.get("gold") or "")


def _trace_system_prompt_from_messages(trace: dict[str, Any]) -> str:
    content = _first_message_content(trace.get("messages", []), "system")
    return content or SYSTEM_PROMPT_ZH


def _messages_from_legacy_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    hop_index = 0
    for message in trace.get("messages", []):
        normalized = _normalize_message(message, hop_index)
        if normalized["role"] == "assistant" and "<tool_call>" in normalized["content"]:
            hop_index += 1
        messages.append(normalized)
    return messages


def _messages_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    plan = trace.get("plan")
    evidence = trace.get("evidence")
    if isinstance(plan, list) and isinstance(evidence, list):
        return render_react_messages(
            system_prompt=_trace_system_prompt_from_messages(trace),
            question=_trace_question(trace),
            plan=plan,
            evidence=evidence,
            final_answer=_trace_final_answer(trace),
        )
    return _messages_from_legacy_trace(trace)


def convert_traces_to_sft_records(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        messages = _messages_from_trace(trace)
        tools = trace.get("tools") if isinstance(trace.get("tools"), list) else TOOL_SCHEMAS
        metadata = dict(trace.get("metadata") or {})
        if "final_question" not in metadata:
            metadata["final_question"] = _trace_question(trace)
        if "final_answer" not in metadata:
            metadata["final_answer"] = _trace_final_answer(trace)
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
