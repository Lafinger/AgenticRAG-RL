from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    title: str
    text: str
    company: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    score: float
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Hop:
    hop_idx: int
    question: str
    answer: str
    doc_chunk_id: str
    qa_type: str


@dataclass(slots=True)
class MultiHopExample:
    final_question: str
    final_answer: str
    hop_count: int
    qa_type: Literal["comparison", "inference"]
    subset: str
    hops: list[Hop]

