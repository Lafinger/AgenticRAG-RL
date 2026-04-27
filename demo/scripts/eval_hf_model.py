from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文小说阅读问答 Agent。你需要通过文本检索工具逐步搜索人物、地点、事件和关系证据，"
    "最后用 <answer>...</answer> 输出最终答案。"
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())


def normalize_answer(text: str) -> str:
    return "".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with a HuggingFace causal LM.")
    parser.add_argument("--model", required=True, help="Base or merged model path/name.")
    parser.add_argument("--adapter", help="Optional LoRA adapter path.")
    parser.add_argument("--data", default=str(ROOT / "data" / "novel_eval" / "test.jsonl"))
    parser.add_argument("--output", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--eval-output", help="Optional eval JSON output for run_llm_judge.py.")
    parser.add_argument("--template", default="qwen3_nothink", help="Recorded in output metadata.")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
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


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def parse_tool_calls(text: str) -> tuple[int, int]:
    calls = TOOL_CALL_RE.findall(text)
    valid_count = 0
    for call in calls:
        try:
            payload = json.loads(call.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            valid_count += 1
    return len(calls), valid_count


def best_scores(prediction: str, gold: str, aliases: list[str]) -> tuple[float, float]:
    candidates = [gold, *aliases]
    return (
        max(exact_match(prediction, candidate) for candidate in candidates),
        max(token_f1(prediction, candidate) for candidate in candidates),
    )


def build_prompt(tokenizer: Any, args: argparse.Namespace, question: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        template_kwargs: dict[str, Any] = {}
        if args.template in {"qwen3_nothink", "qwen3_no_think", "qwen3-notthink"}:
            template_kwargs["enable_thinking"] = False
        prompt_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )
        if hasattr(prompt_inputs, "keys"):
            return dict(prompt_inputs)
        return {"input_ids": prompt_inputs}

    fallback_prompt = f"{args.system_prompt}\n\n用户：{question}\n助手："
    return dict(tokenizer(fallback_prompt, return_tensors="pt"))


def generate_one(model: Any, tokenizer: Any, args: argparse.Namespace, question: str) -> str:
    import torch

    model_inputs = build_prompt(tokenizer, args, question)
    device = model_input_device(model)
    model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if args.temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    divisor = max(count, 1)
    return {
        "count": count,
        "avg_em": sum(record["em"] for record in records) / divisor,
        "avg_f1": sum(record["f1"] for record in records) / divisor,
        "answer_tag_rate": sum(1 for record in records if record["answer_tag_present"]) / divisor,
        "tool_call_rate": sum(1 for record in records if record["tool_call_count"] > 0) / divisor,
        "valid_tool_call_rate": sum(1 for record in records if record["valid_tool_call_count"] > 0) / divisor,
        "avg_generation_chars": sum(len(record["raw_prediction"]) for record in records) / divisor,
    }


def write_eval_payload(records: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        results.append(
            {
                "question": record["question"],
                "gold": record["gold"],
                "aliases": record["aliases"],
                "prediction": record["prediction"],
                "raw_prediction": record["raw_prediction"],
                "evidence": [],
                "gold_chunks": record["gold_chunks"],
                "retrieved_chunk_ids": [],
                "em": record["em"],
                "f1": record["f1"],
            }
        )

    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data)[: args.max_samples]
    model, tokenizer = load_model_and_tokenizer(args)

    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        question = str(example["final_question"])
        gold = str(example["final_answer"])
        aliases = [str(alias) for alias in example.get("answer_aliases", [])]
        raw_prediction = generate_one(model, tokenizer, args, question)
        prediction = extract_answer(raw_prediction)
        tool_call_count, valid_tool_call_count = parse_tool_calls(raw_prediction)
        em, f1 = best_scores(prediction, gold, aliases)

        record = {
            "index": index,
            "question": question,
            "gold": gold,
            "aliases": aliases,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "em": em,
            "f1": f1,
            "answer_tag_present": bool(ANSWER_RE.search(raw_prediction)),
            "tool_call_count": tool_call_count,
            "valid_tool_call_count": valid_tool_call_count,
            "gold_chunks": [str(hop["doc_chunk_id"]) for hop in example.get("hops", [])],
            "metadata": {
                "model": args.model,
                "adapter": args.adapter,
                "template": args.template,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }
        records.append(record)
        print(f"[{index}/{len(examples)}] f1={f1:.4f} em={em:.1f} question={question}")

    output_path = Path(args.output)
    write_jsonl(records, output_path)

    summary = build_summary(records)
    eval_output = Path(args.eval_output) if args.eval_output else output_path.with_name(f"{output_path.stem}_as_eval.json")
    write_eval_payload(records, summary, eval_output)
    print(json.dumps({"summary": summary, "predictions": str(output_path), "eval_output": str(eval_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
