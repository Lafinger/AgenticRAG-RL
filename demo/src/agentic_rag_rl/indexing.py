from __future__ import annotations

import json
import logging
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .types import Chunk


logger = logging.getLogger(__name__)


KG_EXTRACTION_PROMPT_ZH = """你是一个知识图谱构建器。从给定中文武侠小说文本中抽取事实性三元组。

每个三元组格式：(头实体, 关系, 尾实体)

规则：
- 抽取具体、明确的实体（人名、地名、门派、武功、物品、事件、身份）
- 关系用简短的动词短语（如"师从"、"属于"、"持有"、"前往"、"击败"、"身份是"）
- 每个三元组必须是文本中明确陈述的事实，不能推断
- 实体名称应使用规范全称，不使用代词
- 每段文本抽取 3-8 个三元组
- 如果文本太短或无实质信息，返回空列表

仅返回 JSON 数组：
[{{"head": "实体1", "relation": "关系类型", "tail": "实体2"}}, ...]

文本：
{text}
"""


TextEncoder = Callable[[Sequence[str], int], np.ndarray]


def _chunk_record(chunk: Chunk) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "text": chunk.text,
        "pages": chunk.pages,
        "section": chunk.section,
        "metadata": chunk.metadata,
    }
    if chunk.company:
        record["company"] = chunk.company
    return record


def _as_float32_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D matrix, got shape={matrix.shape}.")
    return matrix


def encode_texts(
    texts: Sequence[str],
    *,
    embedding_model: str | None = None,
    batch_size: int = 64,
    encoder: TextEncoder | None = None,
) -> np.ndarray:
    if encoder is not None:
        logger.info("index_build.progress stage=embedding_custom_start text_count=%s batch_size=%s", len(texts), batch_size)
        matrix = _as_float32_matrix(encoder(texts, batch_size))
        logger.info("index_build.progress stage=embedding_custom_done shape=%s", matrix.shape)
        return matrix
    if not embedding_model:
        raise ValueError("embedding_model is required when encoder is not provided.")

    from sentence_transformers import SentenceTransformer

    logger.info(
        "index_build.progress stage=embedding_model_load_start model=%s text_count=%s batch_size=%s",
        embedding_model,
        len(texts),
        batch_size,
    )
    model = SentenceTransformer(embedding_model)
    logger.info("index_build.progress stage=embedding_encode_start model=%s text_count=%s", embedding_model, len(texts))
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    matrix = _as_float32_matrix(embeddings)
    logger.info("index_build.progress stage=embedding_encode_done shape=%s", matrix.shape)
    return matrix


def _load_faiss(faiss_module: Any | None = None) -> Any:
    if faiss_module is not None:
        return faiss_module
    import faiss

    return faiss


def build_faiss_index(embeddings: np.ndarray, *, faiss_module: Any | None = None) -> Any:
    faiss = _load_faiss(faiss_module)
    logger.info("index_build.progress stage=faiss_build_start vector_count=%s dim=%s", embeddings.shape[0], embeddings.shape[1])
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    logger.info("index_build.progress stage=faiss_build_done ntotal=%s", getattr(index, "ntotal", embeddings.shape[0]))
    return index


def _tokenized_corpus(chunks: Sequence[Chunk]) -> list[list[str]]:
    from .retrieval import tokenize

    total = len(chunks)
    logger.info("index_build.progress stage=bm25_tokenize_start chunk_count=%s", total)
    tokenized: list[list[str]] = []
    checkpoint = max(1, total // 10) if total else 1
    for index, chunk in enumerate(chunks, start=1):
        tokenized.append(tokenize(chunk.text))
        if index == total or index % checkpoint == 0:
            logger.info("index_build.progress stage=bm25_tokenize_progress completed=%s/%s", index, total)
    logger.info("index_build.progress stage=bm25_tokenize_done chunk_count=%s", total)
    return tokenized


def _parse_triples(content: str) -> list[dict[str, str]]:
    stripped = content.strip()
    if not stripped:
        return []
    no_result_markers = (
        "没有找到相关",
        "未找到相关",
        "无相关结果",
        "没有相关结果",
        "无法回答这个问题",
        "无法回答该问题",
        "不能帮助你",
        "无法给到相关内容",
        "无法给到相关",
    )
    if any(marker in stripped for marker in no_result_markers):
        return []
    if not stripped.startswith("["):
        match = re.search(r"\[[\s\S]*\]", stripped)
        if not match:
            raise ValueError("LLM triple response does not contain a JSON array.")
        stripped = match.group(0)
    payload = json.loads(stripped)
    if not isinstance(payload, list):
        raise ValueError("LLM triple response must be a JSON array.")

    triples: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail = str(item.get("tail", "")).strip()
        if head and relation and tail:
            triples.append({"head": head, "relation": relation, "tail": tail})
    return triples


def _load_triples_cache(cache_path: str | Path | None) -> dict[str, list[dict[str, str]]]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}

    cached: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id", "")).strip()
            if record.get("status") == "failed":
                continue
            if chunk_id and isinstance(record.get("triples"), list):
                triples: list[dict[str, str]] = []
                for item in record["triples"]:
                    if not isinstance(item, dict):
                        continue
                    head = str(item.get("head", "")).strip()
                    relation = str(item.get("relation", "")).strip()
                    tail = str(item.get("tail", "")).strip()
                    if head and relation and tail:
                        triples.append({"head": head, "relation": relation, "tail": tail})
                cached[chunk_id] = triples
                continue
            head = str(record.get("head", "")).strip()
            relation = str(record.get("relation", "")).strip()
            tail = str(record.get("tail", "")).strip()
            if chunk_id and head and relation and tail:
                cached.setdefault(chunk_id, []).append({"head": head, "relation": relation, "tail": tail})
    return cached


def _normalize_cache_triples(triples: Sequence[dict[str, str]], chunk_id: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for triple in triples:
        if str(triple.get("chunk_id", chunk_id)).strip() != chunk_id:
            continue
        head = str(triple.get("head", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        if head and relation and tail:
            normalized.append({"head": head, "relation": relation, "tail": tail})
    return normalized


def _append_chunk_triples_cache(cache_path: str | Path | None, chunk_id: str, triples: Sequence[dict[str, str]]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"chunk_id": chunk_id, "status": "ok", "triples": _normalize_cache_triples(triples, chunk_id)}
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\r\n")
        handle.flush()


def _append_chunk_triples_failure(cache_path: str | Path | None, chunk_id: str, exc: Exception) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "chunk_id": chunk_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\r\n")
        handle.flush()


def _write_triples_cache(
    cache_path: str | Path | None,
    triples: Sequence[dict[str, str]],
    *,
    completed_chunk_ids: Sequence[str] | None = None,
) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = {}
    for triple in triples:
        chunk_id = str(triple.get("chunk_id", "")).strip()
        if chunk_id:
            grouped.setdefault(chunk_id, []).append(triple)
    ordered_chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for chunk_id in completed_chunk_ids or []:
        normalized_chunk_id = str(chunk_id)
        if normalized_chunk_id not in seen_chunk_ids:
            ordered_chunk_ids.append(normalized_chunk_id)
            seen_chunk_ids.add(normalized_chunk_id)
        grouped.setdefault(normalized_chunk_id, [])
    for chunk_id in sorted(grouped):
        if chunk_id not in seen_chunk_ids:
            ordered_chunk_ids.append(chunk_id)
            seen_chunk_ids.add(chunk_id)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for chunk_id in ordered_chunk_ids:
            record = {
                "chunk_id": chunk_id,
                "status": "ok",
                "triples": _normalize_cache_triples(grouped[chunk_id], chunk_id),
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def _extract_chunk_triples(llm_client: Any, chunk: Chunk) -> list[dict[str, str]]:
    prompt = KG_EXTRACTION_PROMPT_ZH.format(text=chunk.text[:2000])
    content = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    triples = _parse_triples(content)
    return [{**triple, "chunk_id": chunk.chunk_id} for triple in triples]


def build_knowledge_graph(
    chunks: Sequence[Chunk],
    *,
    llm_client: Any,
    cache_path: str | Path | None = None,
    max_concurrency: int = 5,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    import networkx as nx

    cached = _load_triples_cache(cache_path)
    all_triples: list[dict[str, str]] = []
    missing_chunks: list[Chunk] = []
    completed_chunk_ids: list[str] = []
    for chunk in chunks:
        cached_triples = cached.get(chunk.chunk_id)
        if cached_triples is None:
            missing_chunks.append(chunk)
            continue
        completed_chunk_ids.append(chunk.chunk_id)
        all_triples.extend({**triple, "chunk_id": chunk.chunk_id} for triple in cached_triples)

    logger.info(
        "kg_extraction.progress total_chunks=%s cached_chunks=%s pending_chunks=%s max_concurrency=%s",
        len(chunks),
        len(chunks) - len(missing_chunks),
        len(missing_chunks),
        max_concurrency,
    )
    if missing_chunks:
        completed_count = 0
        failed_count = 0
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
            futures = {executor.submit(_extract_chunk_triples, llm_client, chunk): chunk for chunk in missing_chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    chunk_triples = future.result()
                except Exception as exc:
                    failed_count += 1
                    _append_chunk_triples_failure(cache_path, chunk.chunk_id, exc)
                    logger.exception(
                        "kg_extraction.progress completed=%s/%s failed=%s chunk_id=%s status=failed",
                        completed_count,
                        len(missing_chunks),
                        failed_count,
                        chunk.chunk_id,
                    )
                    raise
                completed_count += 1
                all_triples.extend(chunk_triples)
                completed_chunk_ids.append(chunk.chunk_id)
                _append_chunk_triples_cache(cache_path, chunk.chunk_id, chunk_triples)
                logger.info(
                    "kg_extraction.progress completed=%s/%s failed=%s chunk_id=%s triples=%s total_triples=%s",
                    completed_count,
                    len(missing_chunks),
                    failed_count,
                    chunk.chunk_id,
                    len(chunk_triples),
                    len(all_triples),
                )

    graph = nx.MultiDiGraph()

    def add_mention(entity: str, chunk_id: str) -> None:
        if not graph.has_node(entity):
            graph.add_node(entity, mentions=[])
        mentions = graph.nodes[entity].setdefault("mentions", [])
        if chunk_id not in mentions:
            mentions.append(chunk_id)

    for triple in all_triples:
        head = triple["head"]
        tail = triple["tail"]
        relation = triple["relation"]
        chunk_id = triple["chunk_id"]
        add_mention(head, chunk_id)
        add_mention(tail, chunk_id)
        graph.add_edge(head, tail, relation=relation, chunk_id=chunk_id)

    completed_chunk_id_set = set(completed_chunk_ids)
    ordered_completed_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_id in completed_chunk_id_set]
    _write_triples_cache(cache_path, all_triples, completed_chunk_ids=ordered_completed_chunk_ids)
    return nx.node_link_data(graph), all_triples, ordered_completed_chunk_ids


def build_entity_embeddings(
    graph_data: dict[str, Any],
    *,
    embedding_model: str | None = None,
    encoder: TextEncoder | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    nodes = graph_data.get("nodes", [])
    entities = [str(node.get("id", "")) for node in nodes if str(node.get("id", "")).strip()]
    entity_to_idx = {entity: index for index, entity in enumerate(entities)}
    if entities:
        embeddings = encode_texts(entities, embedding_model=embedding_model, batch_size=batch_size, encoder=encoder)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    return {"entities": entities, "entity_to_idx": entity_to_idx, "embeddings": embeddings}


def build_index_bundle(
    chunks: Sequence[Chunk],
    *,
    embedding_model: str | None = None,
    batch_size: int = 64,
    text_encoder: TextEncoder | None = None,
    faiss_module: Any | None = None,
    skip_kg: bool = True,
    kg_llm_client: Any | None = None,
    kg_cache_path: str | Path | None = None,
    kg_max_concurrency: int = 5,
) -> dict[str, Any]:
    logger.info("index_build.progress stage=prepare_chunks_start")
    chunk_list = list(chunks)
    chunk_ids = [chunk.chunk_id for chunk in chunk_list]
    chunk_store = {chunk.chunk_id: _chunk_record(chunk) for chunk in chunk_list}
    logger.info("index_build.progress stage=prepare_chunks_done chunk_count=%s", len(chunk_list))
    tokenized = _tokenized_corpus(chunk_list)
    logger.info("index_build.progress stage=bm25_build_start document_count=%s", len(tokenized))
    bm25 = BM25Okapi(tokenized)
    logger.info("index_build.progress stage=bm25_build_done document_count=%s", len(tokenized))

    faiss_index = None
    embedding_dim = None
    if embedding_model or text_encoder is not None:
        logger.info("index_build.progress stage=semantic_index_start")
        embeddings = encode_texts(
            [chunk.text for chunk in chunk_list],
            embedding_model=embedding_model,
            batch_size=batch_size,
            encoder=text_encoder,
        )
        embedding_dim = int(embeddings.shape[1]) if embeddings.size else 0
        faiss_index = build_faiss_index(embeddings, faiss_module=faiss_module)
        logger.info("index_build.progress stage=semantic_index_done embedding_dim=%s", embedding_dim)

    knowledge_graph = None
    entity_embeddings = None
    triples: list[dict[str, str]] = []
    if not skip_kg:
        if kg_llm_client is None:
            raise ValueError("kg_llm_client is required when skip_kg=False.")
        logger.info("index_build.progress stage=kg_build_start cache_path=%s", kg_cache_path)
        knowledge_graph, triples, completed_chunk_ids = build_knowledge_graph(
            chunk_list,
            llm_client=kg_llm_client,
            cache_path=kg_cache_path,
            max_concurrency=kg_max_concurrency,
        )
        logger.info("index_build.progress stage=kg_build_done triple_count=%s", len(triples))
        logger.info("index_build.progress stage=entity_embedding_start")
        entity_embeddings = build_entity_embeddings(
            knowledge_graph,
            embedding_model=embedding_model,
            encoder=text_encoder,
            batch_size=256,
        )
        logger.info(
            "index_build.progress stage=entity_embedding_done entity_count=%s",
            len(entity_embeddings.get("entities", [])),
        )

    logger.info("index_build.progress stage=bundle_done")
    return {
        "manifest": {
            "chunk_count": len(chunk_list),
            "index_type": "faiss_bge_m3+bm25+kg",
            "embedding_model": embedding_model or "custom_encoder",
            "embedding_dim": embedding_dim,
            "faiss": faiss_index is not None,
            "bm25": True,
            "knowledge_graph": knowledge_graph is not None,
            "triple_count": len(triples),
            "llm_max_concurrency": kg_max_concurrency if not skip_kg else None,
        },
        "chunk_ids": chunk_ids,
        "chunk_store": chunk_store,
        "bm25": bm25,
        "faiss_index": faiss_index,
        "knowledge_graph": knowledge_graph,
        "entity_embeddings": entity_embeddings,
        "triples": triples,
        "kg_completed_chunk_ids": completed_chunk_ids if not skip_kg else [],
    }


def save_index_bundle(bundle: dict[str, Any], output_dir: str | Path, *, faiss_module: Any | None = None) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    logger.info("index_save.progress stage=manifest_start output_dir=%s", target)
    (target / "manifest.json").write_text(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "chunk_ids.json").write_text(json.dumps(bundle["chunk_ids"], ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("index_save.progress stage=manifest_done chunk_count=%s", len(bundle["chunk_ids"]))

    logger.info("index_save.progress stage=chunk_store_start")
    with (target / "chunk_store.pkl").open("wb") as handle:
        pickle.dump(bundle["chunk_store"], handle)
    logger.info("index_save.progress stage=chunk_store_done")

    logger.info("index_save.progress stage=bm25_start")
    with (target / "bm25.pkl").open("wb") as handle:
        pickle.dump(bundle["bm25"], handle)
    logger.info("index_save.progress stage=bm25_done")

    faiss_index = bundle.get("faiss_index")
    if faiss_index is not None:
        faiss = _load_faiss(faiss_module)
        logger.info("index_save.progress stage=faiss_start")
        faiss.write_index(faiss_index, str(target / "faiss.index"))
        logger.info("index_save.progress stage=faiss_done")

    if bundle.get("knowledge_graph") is not None:
        logger.info("index_save.progress stage=knowledge_graph_start")
        (target / "knowledge_graph.json").write_text(
            json.dumps(bundle["knowledge_graph"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("index_save.progress stage=knowledge_graph_done")
    if bundle.get("entity_embeddings") is not None:
        logger.info("index_save.progress stage=entity_embeddings_start")
        with (target / "entity_embeddings.pkl").open("wb") as handle:
            pickle.dump(bundle["entity_embeddings"], handle)
        logger.info("index_save.progress stage=entity_embeddings_done")
    if bundle.get("triples") or bundle.get("kg_completed_chunk_ids"):
        logger.info("index_save.progress stage=triples_cache_start")
        _write_triples_cache(
            target / "triples_cache.jsonl",
            bundle["triples"],
            completed_chunk_ids=bundle.get("kg_completed_chunk_ids"),
        )
        logger.info("index_save.progress stage=triples_cache_done")
    logger.info("index_save.progress stage=done output_dir=%s", target)
