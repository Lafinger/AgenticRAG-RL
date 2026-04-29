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
create_llm_client = llm_client_module.create_llm_client
get_doubao_base_url = llm_client_module.get_doubao_base_url
get_doubao_judge_model = llm_client_module.get_doubao_judge_model


JUDGE_SYSTEM_PROMPT = """你是一个严格的中文阅读理解评测裁判。你需要判断模型回答是否和标准答案语义一致，并只输出 JSON。

评分要求：
- correctness：0 到 1。只看预测答案是否回答了问题，是否与标准答案或可接受别名语义等价。
- faithfulness：0 到 1。如果提供了证据文本，判断预测答案是否被证据支持；如果没有证据文本，则基于标准答案一致性给出保守评分。
- answer_format：0 到 1。判断原始输出是否遵守要求的最终答案格式，尤其是是否使用 <answer>...</answer> 或输出清晰最终答案。
- overall：0 到 1。综合 correctness、faithfulness、answer_format。

只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Doubao LLM-as-Judge on evaluation results.")
    parser.add_argument("results_file", help="Eval JSON file with summary/results.")
    parser.add_argument("--output")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=["doubao"])
    parser.add_argument("--judge-model", help="Defaults to DOUBAO_JUDGE_MODEL or DOUBAO_THINKING_MODEL.")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=5, help="Maximum concurrent LLM judge requests.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


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
    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    input_results = payload["results"]
    if args.max_samples is not None:
        input_results = input_results[: args.max_samples]

    client = create_llm_client(
        args.llm_provider,
        api_key=args.api_key,
        model=get_doubao_judge_model(args.judge_model),
        base_url=get_doubao_base_url(args.base_url),
        timeout_seconds=args.timeout_seconds,
    )

    indexed_results: dict[int, dict[str, Any]] = {}
    failed_indexes: set[int] = set()
    if args.max_concurrency == 1:
        for index, item in enumerate(input_results, start=1):
            _, judged_item, error = judge_item(
                index,
                len(input_results),
                client,
                item,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
            )
            indexed_results[index] = judged_item
            if error is not None:
                failed_indexes.add(index)
                if args.fail_fast:
                    raise error
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
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
                for index, item in enumerate(input_results, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                _, judged_item, error = future.result()
                indexed_results[index] = judged_item
                if error is not None:
                    failed_indexes.add(index)
                    if args.fail_fast:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise error

    ordered_results = [indexed_results[index] for index in sorted(indexed_results)]
    judged_results = [item for index, item in enumerate(ordered_results, start=1) if index not in failed_indexes]
    failed_results = [item for index, item in enumerate(ordered_results, start=1) if index in failed_indexes]

    summary = {
        "count": len(judged_results),
        "failed_count": len(failed_results),
        "judge_provider": args.llm_provider,
        "judge_model": get_doubao_judge_model(args.judge_model),
        "avg_correctness": average(judged_results, "correctness"),
        "avg_faithfulness": average(judged_results, "faithfulness"),
        "avg_answer_format": average(judged_results, "answer_format"),
        "avg_overall": average(judged_results, "overall"),
    }
    output_payload = {"summary": summary, "results": judged_results, "failed_results": failed_results}
    output_path = Path(args.output) if args.output else Path(args.results_file).with_name(Path(args.results_file).stem + "_judged.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
