from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.evaluation import exact_match, hop_recall, token_f1
from agentic_rag_rl.io import load_chunks, load_jsonl
from agentic_rag_rl.protocols import (
    SYSTEM_PROMPT_ZH,
    TOOL_SCHEMAS,
    render_canonical_chat,
    truncate_tool_response_text,
)
from agentic_rag_rl.retrieval import HybridRetriever


AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT_ZH
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<answer>.*?</answer>|<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
COMPLETE_REACT_TOOL_ACTION_RE = re.compile(
    r"<think>.+?</think>\s*<tool_call>.*?</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
THINK_TAG_FRAGMENT_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
TOOL_OPEN_RE = re.compile(r"<tool_call\b[^>]*>", re.IGNORECASE)
TOOL_CLOSE_RE = re.compile(r"</tool_call>", re.IGNORECASE)
TOOL_TAG_FRAGMENT_RE = re.compile(r"</?tool_call\b[^>]*>", re.IGNORECASE)


def assistant_start_anchor_text(anchor: str) -> str:
    if anchor == "none":
        return ""
    if anchor == "think":
        return "<think>"
    raise ValueError(f"Unsupported assistant start anchor: {anchor!r}.")


def build_inputs_for_generation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    template_name: str,
    assistant_start_anchor: str = "none",
) -> Any:
    del template_name
    rendered_text = render_canonical_chat(messages, tools=TOOL_SCHEMAS, add_generation_prompt=True)
    rendered_text += assistant_start_anchor_text(assistant_start_anchor)
    return tokenizer(rendered_text, return_tensors="pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model in a local retrieval agent loop.")
    parser.add_argument("--model", required=True, help="Base or merged model path/name.")
    parser.add_argument("--adapter", help="Optional LoRA adapter path.")
    parser.add_argument("--data", default=str(ROOT / "data" / "novel_eval" / "test.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", required=True, help="Eval JSON or JSONL output path.")
    parser.add_argument("--template", default="qwen3_react", help="Recorded in output metadata.")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--per-turn-max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--assistant-start-anchor",
        choices=["none", "think"],
        default="none",
        help="Optional diagnostic generation prefix. Main baseline uses none.",
    )
    parser.add_argument(
        "--protocol-constraints",
        choices=["none", "strict"],
        default="none",
        help="Optional generation guard. Main baseline uses none.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--system-prompt", default=AGENT_SYSTEM_PROMPT)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_local_path_or_repo(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if not path.is_absolute():
        repo_relative = ROOT / path
        if repo_relative.exists():
            return str(repo_relative.resolve())
    return value


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = resolve_local_path_or_repo(args.model)
    adapter_path = resolve_local_path_or_repo(args.adapter) if args.adapter else None
    args.model = model_name_or_path
    args.adapter = adapter_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("使用 --adapter 时需要安装 peft。") from exc

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> Any:
    embedding = model.get_input_embeddings()
    if embedding is not None and hasattr(embedding, "weight"):
        device = embedding.weight.device
        if device.type != "meta":
            return device

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError("没有找到可用的模型设备。")


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_tool_payload(raw_payload: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        return None, f"invalid_tool_call_json: {exc.msg}"

    if not isinstance(payload, dict):
        return None, "tool_call_must_be_object"

    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return None, "tool_call_missing_name"
    if not isinstance(arguments, dict):
        return None, "tool_call_arguments_must_be_object"
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, "tool_call_missing_query"

    return {"name": name.strip(), "arguments": {"query": query.strip()}}, None


def format_tool_call(tool_call: dict[str, Any]) -> str:
    return f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False, separators=(',', ':'))}\n</tool_call>"


def make_short_think(query: str, turn_index: int = 1) -> str:
    if turn_index <= 1:
        return f"要回答最终问题，先查：{query.strip()}"
    return f"已获得上一跳线索，继续查：{query.strip()}"


def format_react_tool_action(tool_call: dict[str, Any], think: str | None = None, turn_index: int = 1) -> str:
    query = str(tool_call["arguments"]["query"])
    think_text = think.strip() if think and think.strip() else make_short_think(query, turn_index)
    return f"<think>{think_text}</think>\n{format_tool_call(tool_call)}"


def _extract_valid_think_prefix(text: str, action_start: int) -> tuple[str | None, bool]:
    prefix = text[:action_start].strip()
    if not prefix:
        return None, False
    match = THINK_RE.fullmatch(prefix)
    if match is None:
        return None, False
    think = match.group(1).strip()
    return (think or None), bool(think)


def analyze_turn_output(text: str) -> dict[str, Any]:
    tool_payloads = TOOL_CALL_RE.findall(text)
    invalid_tool_payload_present = False
    for payload in tool_payloads:
        tool_call, error = _parse_tool_payload(payload.strip())
        if tool_call is None or error is not None:
            invalid_tool_payload_present = True
            break

    tool_open_count = len(TOOL_OPEN_RE.findall(text))
    tool_close_count = len(TOOL_CLOSE_RE.findall(text))
    action_matches = list(ACTION_RE.finditer(text))
    action_count = len(action_matches)
    tool_fragment_present = TOOL_TAG_FRAGMENT_RE.search(text) is not None
    starts_with_closing_tool = text.lstrip().lower().startswith("</tool_call>")
    empty_tool_payload_present = any(not payload.strip() for payload in tool_payloads)
    think_open_count = len(THINK_OPEN_RE.findall(text))
    think_close_count = len(THINK_CLOSE_RE.findall(text))
    first_action = action_matches[0] if action_matches else None
    first_action_is_tool = bool(first_action and first_action.group(0).lstrip().lower().startswith("<tool_call"))
    first_action_is_answer = bool(first_action and first_action.group(0).lstrip().lower().startswith("<answer"))
    _, valid_think_prefix = _extract_valid_think_prefix(text, first_action.start()) if first_action else (None, False)
    missing_think_tool_call_present = first_action_is_tool and not valid_think_prefix
    empty_answer_think_present = bool(first_action_is_answer and re.fullmatch(r"\s*<think>\s*</think>\s*", text[: first_action.start()], flags=re.DOTALL | re.IGNORECASE))
    malformed_think_present = THINK_TAG_FRAGMENT_RE.search(text) is not None and (
        think_open_count != think_close_count
        or (first_action_is_tool and not valid_think_prefix)
        or think_open_count != 1
        or (first_action is not None and THINK_TAG_FRAGMENT_RE.search(text[first_action.end() :]) is not None)
    )
    malformed_tool_fragment_present = tool_fragment_present and (
        starts_with_closing_tool
        or tool_open_count != tool_close_count
        or empty_tool_payload_present
        or invalid_tool_payload_present
        or missing_think_tool_call_present
    )
    return {
        "tool_fragment_present": tool_fragment_present,
        "malformed_tool_fragment_present": malformed_tool_fragment_present,
        "multi_action_present": action_count > 1,
        "think_tag_present": first_action_is_tool and valid_think_prefix,
        "missing_think_tool_call_present": missing_think_tool_call_present,
        "empty_answer_think_present": empty_answer_think_present,
        "malformed_think_present": malformed_think_present,
        "starts_with_closing_tool": starts_with_closing_tool,
        "action_count": action_count,
        "think_open_count": think_open_count,
        "think_close_count": think_close_count,
        "tool_open_count": tool_open_count,
        "tool_close_count": tool_close_count,
    }


def parse_first_action(text: str, turn_index: int = 1) -> tuple[dict[str, Any] | None, str | None]:
    first_error: str | None = None
    for match in ACTION_RE.finditer(text):
        action_text = match.group(0)
        answer = extract_answer(action_text)
        if answer is not None:
            return {"kind": "answer", "answer": answer, "normalized_text": f"<answer>{answer}</answer>"}, None

        tool_match = TOOL_CALL_RE.fullmatch(action_text)
        if tool_match is None:
            continue
        tool_call, error = _parse_tool_payload(tool_match.group(1).strip())
        if tool_call is None:
            if first_error is None:
                first_error = error or "invalid_tool_call"
            continue
        think, _ = _extract_valid_think_prefix(text, match.start())
        if think is None:
            if first_error is None:
                first_error = "missing_think_for_tool_call"
            continue
        return {
            "kind": "tool_call",
            "tool_call": tool_call,
            "normalized_text": format_react_tool_action(tool_call, think, turn_index),
        }, None

    if first_error:
        return None, first_error
    if TOOL_TAG_FRAGMENT_RE.search(text):
        return None, "invalid_tool_call_fragment"
    return None, "missing_tool_call"


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    action, error = parse_first_action(text)
    if action is None:
        return None, error
    if action["kind"] != "tool_call":
        return None, "missing_tool_call"
    return action["tool_call"], None


def format_tool_response(results: list[Any]) -> str:
    blocks = []
    for result in results:
        blocks.append(f"[{result.chunk_id}] {result.title}\n{truncate_tool_response_text(str(result.text))}")
    return "\n\n".join(blocks)


def unique_append(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value not in seen:
            target.append(value)
            seen.add(value)


class StopOnActionCriteria:
    def __init__(self, tokenizer: Any, prompt_length: int, generated_prefix: str = "") -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.generated_prefix = generated_prefix

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        del scores, kwargs
        generated_ids = input_ids[0][self.prompt_length :]
        if hasattr(generated_ids, "tolist"):
            generated_ids = generated_ids.tolist()
        generated_text = self.generated_prefix + self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return ANSWER_RE.search(generated_text) is not None or COMPLETE_REACT_TOOL_ACTION_RE.search(generated_text) is not None


class ProtocolStartLogitsProcessor:
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.blocked_token_ids: list[int] = []
        for literal in ("</tool_call>", "<tool_call>", "</think>", "<|im_end|>"):
            token_ids = tokenizer.encode(literal, add_special_tokens=False)
            if len(token_ids) == 1:
                self.blocked_token_ids.append(int(token_ids[0]))

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[-1]) == self.prompt_length:
            for token_id in self.blocked_token_ids:
                scores[:, token_id] = float("-inf")
        return scores


def best_scores(prediction: str, gold: str, aliases: list[str]) -> tuple[float, float]:
    candidates = [gold, *aliases]
    return (
        max(exact_match(prediction, candidate) for candidate in candidates),
        max(token_f1(prediction, candidate) for candidate in candidates),
    )


def run_agentic_loop(
    question: str,
    retriever: Any,
    generate_turn: Callable[[list[dict[str, str]]], str],
    *,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
    max_turns: int = 5,
    top_k: int = 3,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    raw_turns: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    evidence: list[dict[str, Any]] = []

    for turn_index in range(1, max_turns + 1):
        assistant_text = generate_turn(messages).strip()
        diagnostics = analyze_turn_output(assistant_text)
        action, error = parse_first_action(assistant_text, turn_index)

        if action is not None and action["kind"] == "answer":
            history_assistant = action["normalized_text"]
            messages.append({"role": "assistant", "content": history_assistant})
            raw_turns.append(
                {
                    "turn": turn_index,
                    "assistant": assistant_text,
                    "status": "answered",
                    **diagnostics,
                    "normalized_action": action["normalized_text"],
                    "history_assistant": history_assistant,
                    "truncated_to_first_action": history_assistant != assistant_text,
                }
            )
            return {
                "prediction": action["answer"],
                "status": "answered",
                "raw_turns": raw_turns,
                "tool_calls": tool_calls,
                "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evidence": evidence,
            }

        if action is None:
            messages.append({"role": "assistant", "content": assistant_text})
            raw_turns.append(
                {
                    "turn": turn_index,
                    "assistant": assistant_text,
                    "status": "failed",
                    "error": error,
                    **diagnostics,
                }
            )
            return {
                "prediction": "",
                "status": error or "invalid_tool_call",
                "raw_turns": raw_turns,
                "tool_calls": tool_calls,
                "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evidence": evidence,
            }

        tool_call = action["tool_call"]
        history_assistant = action["normalized_text"]
        messages.append({"role": "assistant", "content": history_assistant})
        name = tool_call["name"]
        query = tool_call["arguments"]["query"]
        call_record = {"turn": turn_index, "name": name, "arguments": {"query": query}, "valid": True}
        try:
            results = list(retriever.dispatch(name, query, top_k=top_k))
        except Exception as exc:
            call_record["valid"] = False
            call_record["error"] = str(exc)
            tool_calls.append(call_record)
            raw_turns.append(
                {
                    "turn": turn_index,
                    "assistant": assistant_text,
                    "status": "failed",
                    "error": str(exc),
                    **diagnostics,
                    "normalized_action": action["normalized_text"],
                    "history_assistant": history_assistant,
                    "truncated_to_first_action": history_assistant != assistant_text,
                }
            )
            return {
                "prediction": "",
                "status": "tool_dispatch_failed",
                "raw_turns": raw_turns,
                "tool_calls": tool_calls,
                "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evidence": evidence,
            }

        result_records = [result.to_record() for result in results]
        call_record["retrieved_chunk_ids"] = [str(result.chunk_id) for result in results]
        tool_calls.append(call_record)
        evidence.extend(result_records)
        unique_append(retrieved_chunk_ids, call_record["retrieved_chunk_ids"])

        tool_response = format_tool_response(results)
        messages.append({"role": "tool", "content": tool_response})
        raw_turns.append(
            {
                "turn": turn_index,
                "assistant": assistant_text,
                "status": "tool_called",
                "tool_call": tool_call,
                "retrieved_chunk_ids": call_record["retrieved_chunk_ids"],
                **diagnostics,
                "normalized_action": action["normalized_text"],
                "history_assistant": history_assistant,
                "truncated_to_first_action": history_assistant != assistant_text,
            }
        )

    return {
        "prediction": "",
        "status": "max_turns_exceeded",
        "raw_turns": raw_turns,
        "tool_calls": tool_calls,
        "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "evidence": evidence,
    }


def build_model_generate_turn(model: Any, tokenizer: Any, args: argparse.Namespace) -> Callable[[list[dict[str, str]]], str]:
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    def generate_turn(messages: list[dict[str, str]]) -> str:
        anchor_text = assistant_start_anchor_text(args.assistant_start_anchor)
        model_inputs = build_inputs_for_generation(tokenizer, messages, args.template, args.assistant_start_anchor)
        if hasattr(model_inputs, "keys"):
            model_inputs = dict(model_inputs)
        else:
            model_inputs = {"input_ids": model_inputs}

        device = model_input_device(model)
        model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in model_inputs.items()}
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.per_turn_max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "repetition_penalty": args.repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([StopOnActionCriteria(tokenizer, int(input_ids.shape[-1]), anchor_text)]),
        }
        if args.protocol_constraints == "strict":
            generation_kwargs["logits_processor"] = LogitsProcessorList(
                [ProtocolStartLogitsProcessor(tokenizer, int(input_ids.shape[-1]))]
            )
        if args.temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        return (anchor_text + tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()

    return generate_turn


def evaluate_record(record: dict[str, Any], loop_result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    gold = str(record["final_answer"])
    aliases = [str(alias) for alias in record.get("answer_aliases", [])]
    prediction = str(loop_result["prediction"])
    gold_chunks = [str(hop["doc_chunk_id"]) for hop in record.get("hops", [])]
    em, f1 = best_scores(prediction, gold, aliases)
    return {
        "question": str(record["final_question"]),
        "gold": gold,
        "aliases": aliases,
        "prediction": prediction,
        "status": loop_result["status"],
        "raw_turns": loop_result["raw_turns"],
        "tool_calls": loop_result["tool_calls"],
        "valid_tool_call_count": loop_result["valid_tool_call_count"],
        "retrieved_chunk_ids": loop_result["retrieved_chunk_ids"],
        "gold_chunks": gold_chunks,
        "evidence": loop_result["evidence"],
        "em": em,
        "f1": f1,
        "hop_recall": hop_recall(loop_result["retrieved_chunk_ids"], gold_chunks),
        "metadata": metadata,
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    divisor = max(count, 1)
    raw_turns = [turn for record in records for turn in record.get("raw_turns", [])]
    turn_divisor = max(len(raw_turns), 1)
    tool_turns = [turn for turn in raw_turns if turn.get("status") == "tool_called"]
    tool_turn_divisor = max(len(tool_turns), 1)
    anchors = sorted(
        {
            str(record.get("metadata", {}).get("assistant_start_anchor", "none"))
            for record in records
        }
    )
    constraints = sorted(
        {
            str(record.get("metadata", {}).get("protocol_constraints", "none"))
            for record in records
        }
    )
    return {
        "count": count,
        "assistant_start_anchor": anchors[0] if len(anchors) == 1 else anchors,
        "protocol_constraints": constraints[0] if len(constraints) == 1 else constraints,
        "avg_em": sum(record["em"] for record in records) / divisor,
        "avg_f1": sum(record["f1"] for record in records) / divisor,
        "avg_hop_recall": sum(record["hop_recall"] for record in records) / divisor,
        "answer_tag_rate": sum(1 for record in records if record["status"] == "answered") / divisor,
        "valid_tool_call_rate": sum(1 for record in records if record["valid_tool_call_count"] > 0) / divisor,
        "think_tag_rate": sum(1 for turn in tool_turns if turn.get("think_tag_present")) / tool_turn_divisor,
        "missing_think_tool_rate": sum(1 for turn in raw_turns if turn.get("missing_think_tool_call_present")) / turn_divisor,
        "empty_answer_think_rate": sum(1 for turn in raw_turns if turn.get("empty_answer_think_present")) / turn_divisor,
        "malformed_think_rate": sum(1 for turn in raw_turns if turn.get("malformed_think_present")) / turn_divisor,
        "malformed_tool_fragment_rate": sum(1 for turn in raw_turns if turn.get("malformed_tool_fragment_present")) / turn_divisor,
        "multi_action_turn_rate": sum(1 for turn in raw_turns if turn.get("multi_action_present")) / turn_divisor,
        "starts_with_closing_tool_rate": sum(1 for turn in raw_turns if turn.get("starts_with_closing_tool")) / turn_divisor,
        "max_turns_exceeded_rate": sum(1 for record in records if record["status"] == "max_turns_exceeded") / divisor,
        "avg_valid_tool_calls": sum(record["valid_tool_call_count"] for record in records) / divisor,
        "avg_turns": sum(len(record.get("raw_turns", [])) for record in records) / divisor,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def summary_path_for_jsonl(path: Path) -> Path:
    return path.with_name(f"{path.stem}_summary.json")


def write_eval_output(path: Path, payload: dict[str, Any]) -> Path | None:
    if path.suffix.lower() != ".jsonl":
        write_json(path, payload)
        return None

    records = payload.get("results", [])
    if not isinstance(records, list):
        raise ValueError("JSONL eval output requires payload['results'] to be a list.")
    write_jsonl(path, records)

    summary_payload = {
        **payload.get("summary", {}),
        "results_file": str(path),
        "format": "jsonl_records",
    }
    summary_path = summary_path_for_jsonl(path)
    write_json(summary_path, summary_payload)
    return summary_path


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data)[: args.max_samples]
    retriever = HybridRetriever(load_chunks(args.corpus))
    model, tokenizer = load_model_and_tokenizer(args)
    generate_turn = build_model_generate_turn(model, tokenizer, args)

    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "template": args.template,
        "max_turns": args.max_turns,
        "per_turn_max_new_tokens": args.per_turn_max_new_tokens,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "assistant_start_anchor": args.assistant_start_anchor,
        "protocol_constraints": args.protocol_constraints,
    }

    results: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        loop_result = run_agentic_loop(
            str(example["final_question"]),
            retriever,
            generate_turn,
            system_prompt=args.system_prompt,
            max_turns=args.max_turns,
            top_k=args.top_k,
        )
        result = evaluate_record(example, loop_result, metadata)
        results.append(result)
        print(
            f"[{index}/{len(examples)}] status={result['status']} "
            f"f1={result['f1']:.4f} hop_recall={result['hop_recall']:.4f} question={result['question']}"
        )

    payload = {"summary": build_summary(results), "results": results}
    summary_output = write_eval_output(Path(args.output), payload)
    print(
        json.dumps(
            {"summary": payload["summary"], "output": args.output, "summary_output": str(summary_output) if summary_output else None},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
