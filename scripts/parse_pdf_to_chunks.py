from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_study.chunking import chunk_pdf, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PDF 解析为简化版 chunk JSONL。")
    parser.add_argument("--pdf", required=True, help="输入 PDF 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--title", default=None, help="文档标题")
    parser.add_argument("--prefix", default=None, help="chunk_id 前缀")
    args = parser.parse_args()

    records = chunk_pdf(args.pdf, title=args.title, prefix=args.prefix)
    write_jsonl(records, args.output)
    print(f"已写入 {len(records)} 个 chunks -> {args.output}")


if __name__ == "__main__":
    main()

