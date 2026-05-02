from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from agentic_rag_rl.indexing import _parse_triples, build_index_bundle, build_knowledge_graph, save_index_bundle
from agentic_rag_rl.io import load_chunks


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke_novel"


class FakeFaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = vectors


class FakeFaissModule:
    @staticmethod
    def IndexFlatIP(dim: int) -> FakeFaissIndex:
        return FakeFaissIndex(dim)

    @staticmethod
    def write_index(index: FakeFaissIndex, path: str) -> None:
        Path(path).write_bytes(f"fake-faiss:{index.dim}".encode("utf-8"))


class FakeLLMClient:
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        del messages, temperature
        return '[{"head": "孙少平", "relation": "生活地点", "tail": "县立高中"}]'


class RecordingLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        del messages, temperature
        self.calls += 1
        return '[{"head": "补写实体", "relation": "补写关系", "tail": "补写尾实体"}]'


def fake_encoder(texts: Sequence[str], batch_size: int) -> np.ndarray:
    del batch_size
    rows = []
    for text in texts:
        rows.append([float(len(text)), float(text.count("孙少平")), 1.0])
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-6)


def test_build_index_bundle_keeps_chunk_alignment() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    bundle = build_index_bundle(chunks)

    assert bundle["manifest"]["chunk_count"] == len(chunks)
    assert bundle["chunk_ids"][0] == chunks[0].chunk_id
    assert bundle["chunk_store"]["corpus_chunkids_000002"]["title"] == "平凡的世界 段落 2"
    assert bundle["bm25"] is not None


def test_build_and_save_real_index_artifacts_with_fakes(tmp_path: Path) -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")[:2]
    cache_path = tmp_path / "triples_cache.jsonl"
    bundle = build_index_bundle(
        chunks,
        embedding_model="fake-bge-m3",
        text_encoder=fake_encoder,
        faiss_module=FakeFaissModule,
        skip_kg=False,
        kg_llm_client=FakeLLMClient(),
        kg_cache_path=cache_path,
        kg_max_concurrency=2,
    )

    save_index_bundle(bundle, tmp_path, faiss_module=FakeFaissModule)

    expected_files = {
        "manifest.json",
        "faiss.index",
        "bm25.pkl",
        "chunk_ids.json",
        "chunk_store.pkl",
        "knowledge_graph.json",
        "entity_embeddings.pkl",
        "triples_cache.jsonl",
    }
    assert expected_files <= {path.name for path in tmp_path.iterdir()}

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["faiss"] is True
    assert manifest["bm25"] is True
    assert manifest["knowledge_graph"] is True
    assert manifest["triple_count"] == 2
    assert manifest["llm_max_concurrency"] == 2


def test_knowledge_graph_retries_failed_cache_and_rewrites_ordered_checkpoint(tmp_path: Path) -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")[:2]
    cache_path = tmp_path / "triples_cache.jsonl"
    cache_path.write_text(
        "\r\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": chunks[0].chunk_id,
                        "status": "failed",
                        "error_type": "RuntimeError",
                        "error": "temporary failure",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"chunk_id": chunks[1].chunk_id, "status": "ok", "triples": []}, ensure_ascii=False),
            ]
        )
        + "\r\n",
        encoding="utf-8",
    )
    client = RecordingLLMClient()

    _, triples, completed_chunk_ids = build_knowledge_graph(
        chunks,
        llm_client=client,
        cache_path=cache_path,
        max_concurrency=1,
    )

    assert client.calls == 1
    assert len(triples) == 1
    assert completed_chunk_ids == [chunk.chunk_id for chunk in chunks]
    checkpoint_records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
    assert [record["chunk_id"] for record in checkpoint_records] == [chunk.chunk_id for chunk in chunks]
    assert [record["status"] for record in checkpoint_records] == ["ok", "ok"]


def test_parse_triples_treats_no_result_reply_as_empty_list() -> None:
    assert _parse_triples("抱歉，没有找到相关的结果。") == []
    assert _parse_triples("抱歉，这个问题未找到相关结果。") == []
    assert _parse_triples("抱歉，我无法回答这个问题。") == []
    assert _parse_triples("你好，这个问题我无法回答，很遗憾不能帮助你。") == []
    assert _parse_triples("你好，我无法给到相关内容。") == []
