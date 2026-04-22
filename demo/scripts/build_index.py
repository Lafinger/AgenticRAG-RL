from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.indexing import build_index_bundle, save_index_bundle
from agentic_rag_rl.io import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight index bundle from corpus chunks.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    bundle = build_index_bundle(chunks)
    save_index_bundle(bundle, args.index_dir)
    print(f"chunk_count={bundle['manifest']['chunk_count']}")
    print(f"index_dir={args.index_dir}")


if __name__ == "__main__":
    main()
