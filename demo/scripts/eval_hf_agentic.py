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
from agentic_rag_rl.retrieval import HybridRetriever


AGENT_SYSTEM_PROMPT = (
    "你是一个中文小说阅读问答 Agent。你可以通过检索工具逐步查找证据。"
    "每一轮只能输出一个 <tool_call>{\"name\":\"keyword_search\",\"arguments\":{\"query\":\"...\"}}</tool_call> "
    "或最终 <answer>...</answer>。不要在同一轮同时输出工具调用和最终答案。"
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model in a local retrieval agent loop.")
    parser.add_argument("--model", required=True, help="Base or merged model path/name.")
    parser.add_argument("--adapter", help="Optional LoRA adapter path.")
    parser.add_argument("--data", default=str(ROOT / "data" / "novel_eval" / "test.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--output", required=True, help="Eval JSON output path.")
    parser.add_argument("--template", default="qwen3_nothink", help="Recorded in output metadata.")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--per-turn-max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )

    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("使用 --adapter 时需要安装 peft。") from exc

        model = PeftModel.from_pretrained(model, args.adapter)

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


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None, "missing_tool_call"

    raw_payload = matches[0].strip()
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


def format_tool_response(results: list[Any]) -> str:
    blocks = []
    for result in results:
        blocks.append(f"[{result.chunk_id}] {result.title}\n{result.text}")
    return "<tool_response>" + "\n\n".join(blocks) + "</tool_response>"


def unique_append(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value not in seen:
            target.append(value)
            seen.add(value)


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
        messages.append({"role": "assistant", "content": assistant_text})

        final_answer = extract_answer(assistant_text)
        if final_answer is not None:
            raw_turns.append({"turn": turn_index, "assistant": assistant_text, "status": "answered"})
            return {
                "prediction": final_answer,
                "status": "answered",
                "raw_turns": raw_turns,
                "tool_calls": tool_calls,
                "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evidence": evidence,
            }

        tool_call, error = parse_tool_call(assistant_text)
        if tool_call is None:
            raw_turns.append({"turn": turn_index, "assistant": assistant_text, "status": "failed", "error": error})
            return {
                "prediction": "",
                "status": error or "invalid_tool_call",
                "raw_turns": raw_turns,
                "tool_calls": tool_calls,
                "valid_tool_call_count": sum(1 for item in tool_calls if item.get("valid")),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "evidence": evidence,
            }

        name = tool_call["name"]
        query = tool_call["arguments"]["query"]
        call_record = {"turn": turn_index, "name": name, "arguments": {"query": query}, "valid": True}
        try:
            results = list(retriever.dispatch(name, query, top_k=top_k))
        except Exception as exc:
            call_record["valid"] = False
            call_record["error"] = str(exc)
            tool_calls.append(call_record)
            raw_turns.append({"turn": turn_index, "assistant": assistant_text, "status": "failed", "error": str(exc)})
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
        messages.append({"role": "user", "content": tool_response})
        raw_turns.append(
            {
                "turn": turn_index,
                "assistant": assistant_text,
                "status": "tool_called",
                "tool_call": tool_call,
                "retrieved_chunk_ids": call_record["retrieved_chunk_ids"],
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

    def generate_turn(messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            template_kwargs: dict[str, Any] = {}
            if args.template in {"qwen3_nothink", "qwen3_no_think", "qwen3-notthink"}:
                template_kwargs["enable_thinking"] = False
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
            )
            if hasattr(model_inputs, "keys"):
                model_inputs = dict(model_inputs)
            else:
                model_inputs = {"input_ids": model_inputs}
        else:
            rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant: "
            model_inputs = dict(tokenizer(rendered, return_tensors="pt"))

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
        }
        if args.temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
    return {
        "count": count,
        "avg_em": sum(record["em"] for record in records) / divisor,
        "avg_f1": sum(record["f1"] for record in records) / divisor,
        "avg_hop_recall": sum(record["hop_recall"] for record in records) / divisor,
        "answer_tag_rate": sum(1 for record in records if record["status"] == "answered") / divisor,
        "valid_tool_call_rate": sum(1 for record in records if record["valid_tool_call_count"] > 0) / divisor,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


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
    write_json(Path(args.output), payload)
    print(json.dumps({"summary": payload["summary"], "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
