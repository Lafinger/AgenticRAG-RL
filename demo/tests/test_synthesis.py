from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.synthesis import (
    clean_multihop_examples,
    generate_seed_questions,
    iter_seed_question_batches,
    synthesize_multihop_examples,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del temperature
        self.calls.append(messages)
        return """```json
[
            {
                "question": "孙少平在学校生活艰难的表现是什么？",
                "answer": "最后去取黑高粱面馍。",
                "qa_type": "character_behavior",
                "entities": ["孙少平"]
            }
]
```"""


def test_generate_seed_questions_uses_llm_client() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    client = FakeLLMClient()
    seeds = generate_seed_questions(chunks[:1], client, max_per_chunk=1)

    assert len(client.calls) == 1
    assert "平凡的世界" in client.calls[0][1]["content"]
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


def test_iter_seed_question_batches_yields_per_chunk() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")[:2]
    client = FakeLLMClient()
    batches = list(iter_seed_question_batches(chunks, client, max_per_chunk=1))

    assert len(batches) == 2
    assert [batch[0]["doc_chunk_id"] for batch in batches] == [chunk.chunk_id for chunk in chunks]


def test_synthesize_and_clean_multihop_examples() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, FakeLLMClient(), max_per_chunk=1)
    examples = synthesize_multihop_examples(seeds, chunks, target_count=2)
    cleaned = clean_multihop_examples(examples, {chunk.chunk_id for chunk in chunks})

    assert cleaned
    assert cleaned[0]["hop_count"] >= 2
    assert all(hop["doc_chunk_id"] for hop in cleaned[0]["hops"])
