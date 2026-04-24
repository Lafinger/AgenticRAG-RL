from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the retrieval HTTP server.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "novel" / "corpus.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker-model")
    args = parser.parse_args()

    del args.device, args.embedding_model, args.reranker_model
    app = create_app(load_chunks(args.corpus))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
