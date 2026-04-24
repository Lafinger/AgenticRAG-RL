from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.chunking import chunk_text_file_to_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a UTF-8 novel text file into stable corpus chunks.")
    parser.add_argument("--input", default=str(ROOT / "data" / "original_data" / "平凡的世界utf8.txt"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--title", default="平凡的世界")
    parser.add_argument("--prefix", default="corpus_chunkids")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    chunk_text_file_to_jsonl(
        args.input,
        args.output,
        title=args.title,
        prefix=args.prefix,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
