from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_rl.protocols import render_canonical_chat
from training.sft_label_mask import IGNORE_INDEX, tokenize_chat_with_assistant_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose canonical ReAct SFT rendering and labels.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--data-path")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--show-rendered", action="store_true")
    parser.add_argument("--show-supervised", action="store_true")
    parser.add_argument("--eval-summary", help="Optional Agent loop summary JSON to include in the report.")
    parser.add_argument("--probe-model", help="Optional HF model path/name for first-token protocol probability probe.")
    parser.add_argument("--probe-adapter", help="Optional LoRA adapter path for the probability probe.")
    parser.add_argument("--probe-max-prompts", type=int, default=5)
    parser.add_argument("--probe-device-map", default="auto")
    parser.add_argument("--probe-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def iter_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def assistant_contents(record: dict[str, Any]) -> list[str]:
    return [str(message.get("content", "")) for message in record.get("messages", []) if message.get("role") == "assistant"]


def assistant_loss_enabled(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and message.get("loss") is not False and message.get("train") is not False and message.get("trainable") is not False


def probe_prompt(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for index, message in enumerate(messages):
        if assistant_loss_enabled(message):
            tools = record.get("tools") if isinstance(record.get("tools"), list) else None
            return render_canonical_chat(messages[:index], tools=tools, add_generation_prompt=True)
    return None


def torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]


def resolve_local_path_or_repo(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if not path.is_absolute():
        repo_relative = ROOT / path
        if repo_relative.exists():
            return str(repo_relative.resolve())
    return value


def load_probe_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_local_path_or_repo(str(args.probe_model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.probe_device_map,
        torch_dtype=torch_dtype(args.probe_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    if args.probe_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolve_local_path_or_repo(str(args.probe_adapter)))
    model.eval()
    return model, tokenizer


def first_model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if getattr(parameter, "device", None) is not None and parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Cannot find model device.")


def sequence_probability(model: Any, tokenizer: Any, prompt_text: str, sequence_text: str) -> float | None:
    import torch

    token_ids = tokenizer.encode(sequence_text, add_special_tokens=False)
    if not token_ids:
        return None
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = first_model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    probability = 1.0
    with torch.no_grad():
        for token_id in token_ids:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
            probability *= float(probs[int(token_id)].detach().cpu())
            next_id = torch.tensor([[int(token_id)]], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
    return probability


def run_probability_probe(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not args.probe_model:
        return None
    model, tokenizer = load_probe_model(args)
    prompts = [prompt for record in records for prompt in [probe_prompt(record)] if prompt][: args.probe_max_prompts]
    if not prompts:
        return {"prompt_count": 0}
    literals = ["<think>", "<answer>", "<tool_call>", "</tool_call>"]
    rows: list[dict[str, Any]] = []
    top1_counts: dict[str, int] = {}
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = first_model_device(model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        import torch

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            values, ids = torch.topk(probs, 5)
        top_tokens = [
            {
                "token_id": int(token_id),
                "text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "prob": float(prob),
            }
            for prob, token_id in zip(values.detach().cpu().tolist(), ids.detach().cpu().tolist(), strict=True)
        ]
        top1 = top_tokens[0]["text"]
        top1_counts[top1] = top1_counts.get(top1, 0) + 1
        rows.append(
            {
                "top_tokens": top_tokens,
                "sequence_probs": {literal: sequence_probability(model, tokenizer, prompt, literal) for literal in literals},
            }
        )
    return {
        "prompt_count": len(prompts),
        "top1_counts": top1_counts,
        "top1_closing_tool_rate": top1_counts.get("</tool_call>", 0) / max(len(prompts), 1),
        "probes": rows,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_path = project_path(args.data_path or config["data_path"])
    model_name = args.model_name_or_path or config["model_name_or_path"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    records = iter_jsonl(data_path, args.max_samples)
    report = {
        "data": str(data_path),
        "model_tokenizer": model_name,
        "sampled_records": len(records),
        "tool_turns": 0,
        "tool_turns_with_think": 0,
        "answer_turns": 0,
        "rendered_empty_answer_think": 0,
        "tool_response_label_leaks": 0,
        "max_token_length": 0,
        "min_supervised_tokens": None,
    }
    first_rendered = ""
    first_supervised = ""

    for record in records:
        tools = record.get("tools") if isinstance(record.get("tools"), list) else None
        rendered = render_canonical_chat(record["messages"], tools=tools, add_generation_prompt=False)
        sample = tokenize_chat_with_assistant_labels(tokenizer, record["messages"], tools=tools)
        labels = [token_id for token_id, label in zip(sample.input_ids, sample.labels, strict=True) if label != IGNORE_INDEX]
        supervised = tokenizer.decode(labels, skip_special_tokens=False)
        report["max_token_length"] = max(int(report["max_token_length"]), sample.token_length)
        current_min = report["min_supervised_tokens"]
        report["min_supervised_tokens"] = sample.supervised_token_count if current_min is None else min(current_min, sample.supervised_token_count)
        if "<tool_response>" in supervised:
            report["tool_response_label_leaks"] += 1
        if "<think>\n\n</think>\n\n<answer>" in rendered or "<think></think>\n<answer>" in rendered:
            report["rendered_empty_answer_think"] += 1
        for content in assistant_contents(record):
            if "<tool_call>" in content:
                report["tool_turns"] += 1
                if "<think>" in content and "</think>" in content:
                    report["tool_turns_with_think"] += 1
            if "<answer>" in content:
                report["answer_turns"] += 1
        if not first_rendered:
            first_rendered = rendered
            first_supervised = supervised

    if report["tool_turns"]:
        report["tool_turn_think_rate"] = report["tool_turns_with_think"] / report["tool_turns"]
    else:
        report["tool_turn_think_rate"] = None

    if args.eval_summary:
        with project_path(args.eval_summary).open("r", encoding="utf-8") as handle:
            report["eval_summary"] = json.load(handle)

    probe = run_probability_probe(args, records)
    if probe is not None:
        report["first_token_probe"] = probe

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show_rendered and first_rendered:
        print("\n--- rendered sample ---")
        print(first_rendered[:3000])
    if args.show_supervised and first_supervised:
        print("\n--- supervised sample ---")
        print(first_supervised[:1500])


if __name__ == "__main__":
    main()
