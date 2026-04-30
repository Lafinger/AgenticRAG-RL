from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .retrieval import HybridRetriever
from .types import Chunk


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    tool: Literal["keyword_search", "dense_search", "semantic_search", "graph_search", "hybrid_search"] = "hybrid_search"


def create_app(chunks: list[Chunk] | None = None, *, retriever: Any | None = None) -> FastAPI:
    if retriever is None:
        if chunks is None:
            raise ValueError("Either chunks or retriever must be provided.")
        retriever = HybridRetriever(chunks)
    app = FastAPI(title="Agentic RAG RL Retrieval Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, object]:
        results = retriever.dispatch(request.tool, request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "tool": request.tool,
            "results": [item.to_record() for item in results],
        }

    return app
