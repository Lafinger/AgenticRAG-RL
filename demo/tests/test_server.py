from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.server import create_app


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


def test_retrieval_server_health_and_search() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    app = create_app(chunks)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/search", json={"query": "双水村 东拉河 哭咽河", "top_k": 2, "tool": "hybrid_search"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "corpus_chunkids_000002"

    graph_response = client.post("/search", json={"query": "双水村 东拉河 哭咽河", "top_k": 2, "tool": "graph_search"})
    assert graph_response.status_code == 200
    assert graph_response.json()["results"]
