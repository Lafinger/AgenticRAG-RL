from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect traceable SFT optimizer-step sample mapping.")
    parser.add_argument("--trace", required=True, help="Path to step_sample_trace.jsonl.")
    parser.add_argument("--step", type=int, help="Optimizer step to inspect.")
    parser.add_argument("--top-grad-norm", type=int, help="Show N steps with the largest grad_norm.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def flatten_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for micro_batch in record.get("micro_batches", []):
        for item in micro_batch.get("items", []):
            payload = dict(item)
            payload["micro_batch_index"] = micro_batch.get("micro_batch_index")
            payload["epoch_progress"] = micro_batch.get("epoch_progress")
            items.append(payload)
    return items


def compact_step(record: dict[str, Any]) -> dict[str, Any]:
    items = flatten_items(record)
    return {
        "step": record.get("step"),
        "epoch": record.get("epoch"),
        "loss": record.get("loss"),
        "eval_loss": record.get("eval_loss"),
        "learning_rate": record.get("learning_rate"),
        "grad_norm": record.get("grad_norm"),
        "sample_count": len(items),
        "samples": items,
    }


def write_json(path: str | Path, payload: Any) -> None:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered.replace("\n", "\r\n"))
        handle.write("\r\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(project_path(args.trace))
    if not records:
        raise SystemExit(f"No trace records found: {args.trace}")

    payload: Any
    if args.top_grad_norm:
        payload = [
            compact_step(record)
            for record in sorted(records, key=lambda item: float(item.get("grad_norm") or 0.0), reverse=True)[
                : args.top_grad_norm
            ]
        ]
    elif args.step is not None:
        matched = [record for record in records if int(record.get("step", -1)) == args.step]
        if not matched:
            raise SystemExit(f"Step {args.step} not found in {args.trace}")
        payload = compact_step(matched[0])
    else:
        raise SystemExit("Pass either --step or --top-grad-norm.")

    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
