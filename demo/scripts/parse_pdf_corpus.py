from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.chunking import chunk_pdf_to_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF into corpus chunks.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    parser.add_argument("--prefix")
    args = parser.parse_args()

    chunk_pdf_to_jsonl(args.pdf, args.output, title=args.title, prefix=args.prefix)
    print(f"chunked_pdf={args.pdf}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
