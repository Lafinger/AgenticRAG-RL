from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "agentic_rag_rl"


def load_project_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


env_module = load_project_module("agentic_rag_rl_env_for_judge", SRC / "env.py")
llm_client_module = load_project_module("agentic_rag_rl_llm_client_for_judge", SRC / "llm_client.py")

load_env_file = env_module.load_env_file
DEFAULT_LLM_PROVIDER = llm_client_module.DEFAULT_LLM_PROVIDER
LLM_PROVIDER_CHOICES = llm_client_module.LLM_PROVIDER_CHOICES
create_llm_client = llm_client_module.create_llm_client
resolve_judge_model = llm_client_module.resolve_judge_model
resolve_llm_base_url = llm_client_module.resolve_llm_base_url

interrupts_module = load_project_module("agentic_rag_rl_interrupts_for_judge", SRC / "interrupts.py")
force_exit_on_keyboard_interrupt = interrupts_module.force_exit_on_keyboard_interrupt
shutdown_thread_pool = interrupts_module.shutdown_thread_pool


JUDGE_SYSTEM_PROMPT = """你是一个严格的中文阅读理解评测裁判。你需要判断模型回答是否和标准答案语义一致，并只输出 JSON。

评分要求：
- correctness：0 到 1。只看预测答案是否回答了问题，是否与标准答案或可接受别名语义等价。
- faithfulness：0 到 1。如果提供了证据文本，判断预测答案是否被证据支持；如果没有证据文本，则基于标准答案一致性给出保守评分。
- answer_format：0 到 1。判断原始输出是否遵守要求的最终答案格式，尤其是是否使用 <answer>...</answer> 或输出清晰最终答案。
- overall：0 到 1。综合 correctness、faithfulness、answer_format。

只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge on evaluation results.")
    parser.add_argument("results_file", help="Eval JSON with summary/results, or JSONL with one eval result per line.")
    parser.add_argument("--output")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=LLM_PROVIDER_CHOICES)
    parser.add_argument("--judge-model", help="Judge model for the selected provider.")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent LLM judge requests.")
    parser.add_argument("--checkpoint-output", help="JSONL checkpoint file for successful and failed judge tasks.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoint and regenerate judge results.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_eval_results(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object: {input_path}")
                records.append(payload)
        return records

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Eval JSON must contain a results list: {input_path}")
    return payload["results"]


def evidence_text(item: dict[str, Any], *, max_chars: int = 5000) -> str:
    parts: list[str] = []
    for evidence in item.get("evidence", []):
        if isinstance(evidence, dict):
            for result in evidence.get("results", []):
                if isinstance(result, dict) and result.get("text"):
                    parts.append(str(result["text"]))
        elif isinstance(evidence, str):
            parts.append(evidence)
    return "\n\n".join(parts)[:max_chars]


def build_user_prompt(item: dict[str, Any]) -> str:
    aliases = item.get("aliases") or item.get("answer_aliases") or []
    raw_prediction = item.get("raw_prediction", item.get("prediction", ""))
    payload = {
        "question": item.get("question", ""),
        "gold_answer": item.get("gold", ""),
        "answer_aliases": aliases,
        "prediction": item.get("prediction", ""),
        "raw_prediction": raw_prediction,
        "evidence_text": evidence_text(item),
    }
    return (
        "请评测下面这条中文小说问答结果。返回 JSON 字段必须包含："
        "correctness, faithfulness, answer_format, overall, verdict, rationale。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")
    return normalize_judge_payload(payload)


def score(value: Any) -> float:
    number = float(value)
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return round(number, 4)


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "correctness": score(payload.get("correctness", 0.0)),
        "faithfulness": score(payload.get("faithfulness", 0.0)),
        "answer_format": score(payload.get("answer_format", 0.0)),
        "overall": score(payload.get("overall", 0.0)),
        "verdict": str(payload.get("verdict", "")).strip(),
        "rationale": str(payload.get("rationale", "")).strip(),
    }


def judge_one(client: Any, item: dict[str, Any], *, temperature: float, max_attempts: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            response = client.chat(messages, temperature=temperature)
            judge = parse_judge_json(response)
            judge["raw_judge_response"] = response
            return judge
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"LLM judge failed after {max_attempts} attempts: {last_error}") from last_error


def average(results: list[dict[str, Any]], key: str) -> float:
    values = [item["judge"][key] for item in results if isinstance(item.get("judge"), dict) and key in item["judge"]]
    return sum(values) / max(len(values), 1)


def default_output_path(results_file: str) -> Path:
    return Path(results_file).with_name(Path(results_file).stem + "_judged.json")


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.checkpoint.jsonl")


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\r\n")
        handle.flush()


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                index = int(record.get("index"))
            except (TypeError, ValueError):
                continue
            latest[index] = record
    return latest


def write_output(
    output_path: Path,
    *,
    provider: str,
    model: str,
    successes: dict[int, dict[str, Any]],
    failures: dict[int, dict[str, Any]],
) -> None:
    judged_results = [successes[index] for index in sorted(successes)]
    failed_results = [failures[index] for index in sorted(failures) if index not in successes]
    summary = {
        "count": len(judged_results),
        "failed_count": len(failed_results),
        "judge_provider": provider,
        "judge_model": model,
        "avg_correctness": average(judged_results, "correctness"),
        "avg_faithfulness": average(judged_results, "faithfulness"),
        "avg_answer_format": average(judged_results, "answer_format"),
        "avg_overall": average(judged_results, "overall"),
    }
    output_payload = {"summary": summary, "results": judged_results, "failed_results": failed_results}
    temp_output = output_path.with_name(f"{output_path.name}.tmp")
    temp_output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_output.replace(output_path)


def judge_item(
    index: int,
    total_count: int,
    client: Any,
    item: dict[str, Any],
    *,
    temperature: float,
    max_attempts: int,
) -> tuple[int, dict[str, Any], Exception | None]:
    judged_item = dict(item)
    try:
        judged_item["judge"] = judge_one(
            client,
            judged_item,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        print(
            f"[{index}/{total_count}] "
            f"correctness={judged_item['judge']['correctness']:.4f} "
            f"overall={judged_item['judge']['overall']:.4f}"
        )
        return index, judged_item, None
    except Exception as exc:
        judged_item["judge_error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"[{index}/{total_count}] judge_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        return index, judged_item, exc


def main() -> None:
    args = parse_args()
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1.")
    load_env_file(args.env_file)
    input_results = load_eval_results(args.results_file)
    if args.max_samples is not None:
        input_results = input_results[: args.max_samples]
    output_path = Path(args.output) if args.output else default_output_path(args.results_file)
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else default_checkpoint_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    judge_model = resolve_judge_model(args.llm_provider, args.judge_model)
    checkpoint = {} if args.overwrite else load_checkpoint(checkpoint_path)
    successes = {
        index: record["item"]
        for index, record in checkpoint.items()
        if record.get("status") == "ok" and isinstance(record.get("item"), dict)
    }
    failures = {
        index: record["item"]
        for index, record in checkpoint.items()
        if record.get("status") == "failed" and isinstance(record.get("item"), dict)
    }
    pending_items = [(index, item) for index, item in enumerate(input_results, start=1) if index not in successes]
    completed_count = 0
    write_output(
        output_path,
        provider=args.llm_provider,
        model=judge_model,
        successes=successes,
        failures=failures,
    )
    print(
        f"judge_resume success={len(successes)} failed_to_retry={len(failures)} "
        f"pending={len(pending_items)} checkpoint={checkpoint_path}"
    )
    client = create_llm_client(
        args.llm_provider,
        api_key=args.api_key,
        model=judge_model,
        base_url=resolve_llm_base_url(args.llm_provider, args.base_url),
        timeout_seconds=args.timeout_seconds,
    )
    if args.max_concurrency == 1:
        for index, item in pending_items:
            _, judged_item, error = judge_item(
                index,
                len(input_results),
                client,
                item,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
            )
            if error is not None:
                failures[index] = judged_item
                append_checkpoint(checkpoint_path, {"index": index, "status": "failed", "item": judged_item})
                if args.fail_fast:
                    raise error
            else:
                successes[index] = judged_item
                failures.pop(index, None)
                append_checkpoint(checkpoint_path, {"index": index, "status": "ok", "item": judged_item})
            completed_count += 1
            write_output(
                output_path,
                provider=args.llm_provider,
                model=judge_model,
                successes=successes,
                failures=failures,
            )
            print(f"judge_progress completed={completed_count}/{len(pending_items)} success={len(successes)} failed={len(failures)}")
    else:
        executor = ThreadPoolExecutor(max_workers=args.max_concurrency)
        interrupted = False
        try:
            futures = {
                executor.submit(
                    judge_item,
                    index,
                    len(input_results),
                    client,
                    item,
                    temperature=args.temperature,
                    max_attempts=args.max_attempts,
                ): index
                for index, item in pending_items
            }
            for future in as_completed(futures):
                index = futures[future]
                _, judged_item, error = future.result()
                if error is not None:
                    failures[index] = judged_item
                    append_checkpoint(checkpoint_path, {"index": index, "status": "failed", "item": judged_item})
                    if args.fail_fast:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise error
                else:
                    successes[index] = judged_item
                    failures.pop(index, None)
                    append_checkpoint(checkpoint_path, {"index": index, "status": "ok", "item": judged_item})
                completed_count += 1
                write_output(
                    output_path,
                    provider=args.llm_provider,
                    model=judge_model,
                    successes=successes,
                    failures=failures,
                )
                print(f"judge_progress completed={completed_count}/{len(pending_items)} success={len(successes)} failed={len(failures)}")
        except KeyboardInterrupt:
            interrupted = True
            shutdown_thread_pool(executor, futures.keys(), wait=False)
            force_exit_on_keyboard_interrupt(
                "run_llm_judge",
                output_path=output_path,
                checkpoint_path=checkpoint_path,
            )
        finally:
            if not interrupted:
                executor.shutdown(wait=True, cancel_futures=False)

    final_payload = json.loads(output_path.read_text(encoding="utf-8"))
    summary = final_payload["summary"]
    print(json.dumps({"summary": summary, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
