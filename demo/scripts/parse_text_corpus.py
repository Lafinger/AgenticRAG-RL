from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.chunking import chunk_text_dir_to_jsonl, chunk_text_file_to_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse UTF-8 Jin Yong novel text files into paragraph-window corpus chunks.")
    parser.add_argument("--input", help="Parse a single txt file. When set, it takes precedence over --input-dir.")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "original_data"))
    parser.add_argument("--output", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--title", help="Override title when --input parses one file.")
    parser.add_argument("--prefix", help="Override chunk id prefix when --input parses one file.")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--chunk-chars", type=int, default=500)
    parser.add_argument("--overlap-chars", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--author", default="金庸")
    args = parser.parse_args()

    if args.input:
        chunk_text_file_to_jsonl(
            args.input,
            args.output,
            title=args.title,
            prefix=args.prefix,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
            author=args.author,
        )
    else:
        chunk_text_dir_to_jsonl(
            args.input_dir,
            args.output,
            pattern=args.pattern,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
            author=args.author,
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
