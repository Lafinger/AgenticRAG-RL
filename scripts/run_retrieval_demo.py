from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_study.demo_data import load_chunks, load_examples
from agentic_rag_study.retrieval import HybridRetriever


def print_block(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    data_dir = ROOT / "data" / "demo_financial"
    chunks = load_chunks(data_dir / "corpus.jsonl")
    examples = load_examples(data_dir / "questions.jsonl")
    retriever = HybridRetriever(chunks)

    for example in examples[:3]:
        query = example.final_question
        print_block(f"问题: {query}")
        for tool_name, runner in (
            ("keyword_search", retriever.keyword_search),
            ("dense_search", retriever.dense_search),
            ("hybrid_search", retriever.hybrid_search),
        ):
            print(f"[{tool_name}]")
            for index, result in enumerate(runner(query, top_k=3), start=1):
                preview = result.text[:70].replace("\n", " ")
                print(f"{index}. {result.chunk_id} | {result.metadata.get('company', '')} | {result.score:.4f}")
                print(f"   {preview}")
        print()


if __name__ == "__main__":
    main()

