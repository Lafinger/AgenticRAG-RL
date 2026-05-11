from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from training.tools.novel_search_tool import HybridSearchTool, KeywordSearchTool


def test_novel_search_tool_calls_retrieval_server_and_returns_tool_response() -> None:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append(payload)
            body = json.dumps(
                {
                    "query": payload["query"],
                    "tool": payload["tool"],
                    "results": [
                        {"chunk_id": "chunk-a", "text": "第一段很长的证据文本", "score": 1.0},
                        {"chunk_id": "chunk-b", "text": "第二段证据", "score": 0.8},
                    ],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        tool = KeywordSearchTool({"retrieval_server_url": url, "top_k": 2, "max_text_len": 6}, {})

        async def run_tool() -> tuple[object, float, dict]:
            instance_id, _ = await tool.create()
            return await tool.execute(instance_id, {"query": "侯监集"})

        response, reward, extra_info = asyncio.run(run_tool())
    finally:
        server.shutdown()
        server.server_close()

    assert requests == [{"query": "侯监集", "tool": "keyword_search", "top_k": 2}]
    assert reward == 0.0
    assert response.text == "[chunk-a] 第一段很长的...\n[chunk-b] 第二段证据"
    assert extra_info["tool"] == "keyword_search"
    assert extra_info["retrieved_chunk_ids"] == ["chunk-a", "chunk-b"]


def test_hybrid_tool_uses_canonical_query_only_schema() -> None:
    tool = HybridSearchTool({"retrieval_server_url": "http://127.0.0.1:8790"}, {})

    assert tool.tool_name == "hybrid_search"
    assert "tools" not in tool.tool_schema.get("function", {}).get("parameters", {}).get("properties", {})


def test_novel_tool_config_uses_verl_base_tool_classes() -> None:
    config = (Path(__file__).resolve().parents[1] / "training" / "config" / "novel_tool_config.yaml").read_text(
        encoding="utf-8"
    )

    assert "training.tools.novel_search_tool.KeywordSearchTool" in config
    assert "training.tools.novel_search_tool.DenseSearchTool" in config
    assert "training.tools.novel_search_tool.HybridSearchTool" in config
    assert "graph_search" not in config
    assert "type: native" in config
