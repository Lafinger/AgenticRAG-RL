from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_jsonl


def _enrich_aliases(answer: str, aliases: list[str]) -> list[str]:
    candidates = {answer, answer.strip(), answer.replace("股份有限公司", ""), answer.replace("有限公司", "")}
    candidates.update(alias.strip() for alias in aliases if alias.strip())
    return sorted(candidate for candidate in candidates if candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced answer aliases.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    enriched = []
    for record in records:
        aliases = _enrich_aliases(record.get("final_answer", ""), record.get("answer_aliases", []))
        enriched.append({**record, "answer_aliases": aliases})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"enhanced_count={len(enriched)}")


if __name__ == "__main__":
    main()
