from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.synthesis import clean_multihop_examples, generate_seed_questions, synthesize_multihop_examples


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


class FakeSeedQAClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_seed_qa(self, chunk_text: str, *, max_items: int) -> list[dict[str, object]]:
        self.calls.append(chunk_text)
        return [
            {
                "question": "孙少平在学校生活艰难的表现是什么？",
                "answer": "最后去取黑高粱面馍。",
                "qa_type": "character_behavior",
                "entities": ["孙少平"],
            }
        ][:max_items]


def test_generate_seed_questions_uses_llm_client() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    client = FakeSeedQAClient()
    seeds = generate_seed_questions(chunks[:1], client, max_per_chunk=1)

    assert len(client.calls) == 1
    assert seeds == [
        {
            "question": "孙少平在学校生活艰难的表现是什么？",
            "answer": "最后去取黑高粱面馍。",
            "doc_chunk_id": "corpus_chunkids_000001",
            "tool": "keyword_search",
            "entities": ["孙少平"],
            "qa_type": "character_behavior",
        }
    ]


def test_synthesize_and_clean_multihop_examples() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, FakeSeedQAClient(), max_per_chunk=1)
    examples = synthesize_multihop_examples(seeds, chunks, target_count=2)
    cleaned = clean_multihop_examples(examples, {chunk.chunk_id for chunk in chunks})

    assert cleaned
    assert cleaned[0]["hop_count"] >= 2
    assert all(hop["doc_chunk_id"] for hop in cleaned[0]["hops"])
