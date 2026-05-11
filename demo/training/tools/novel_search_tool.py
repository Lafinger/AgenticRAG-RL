from __future__ import annotations

import json
import uuid
from typing import Any
from urllib import request

try:
    from verl.tools.base_tool import BaseTool, ToolResponse
except Exception:  # pragma: no cover - local unit tests may run without verl installed.
    class ToolResponse:  # type: ignore[no-redef]
        def __init__(self, text: str = "") -> None:
            self.text = text

    class BaseTool:  # type: ignore[no-redef]
        def __init__(self, config: dict[str, Any], tool_schema: dict[str, Any]) -> None:
            self.config = config
            self.tool_schema = tool_schema


class _BaseNovelSearchTool(BaseTool):
    tool_name = "hybrid_search"

    def __init__(self, config: dict[str, Any], tool_schema: dict[str, Any]):
        super().__init__(config, tool_schema)
        self.top_k = int(config.get("top_k", 3))
        self.max_text_len = int(config.get("max_text_len", 420))
        self.retrieval_server_url = str(config.get("retrieval_server_url", "http://127.0.0.1:8790")).rstrip("/")
        self.timeout = float(config.get("timeout", 30))
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: str | None = None, **kwargs: Any):
        del kwargs
        instance_id = instance_id or str(uuid.uuid4())
        self._instance_dict[instance_id] = {"responses": [], "retrieved_chunk_ids": []}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any):
        del kwargs
        query = str(parameters.get("query", "")).strip()
        if not query:
            return ToolResponse(text="(empty query)"), 0.0, {"query": query, "tool": self.tool_name, "num_results": 0}

        try:
            results = self._search(query)
            text = self._format_results(results)
            retrieved_chunk_ids = [str(item.get("chunk_id", "")) for item in results if item.get("chunk_id")]
            state = self._instance_dict.setdefault(instance_id, {"responses": [], "retrieved_chunk_ids": []})
            state["responses"].append(text)
            state["retrieved_chunk_ids"].extend(retrieved_chunk_ids)
            return (
                ToolResponse(text=text),
                0.0,
                {
                    "query": query,
                    "tool": self.tool_name,
                    "num_results": len(results),
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                },
            )
        except Exception as exc:  # pragma: no cover - depends on external retrieval server.
            return (
                ToolResponse(text=f"(search error: {exc})"),
                0.0,
                {"query": query, "tool": self.tool_name, "num_results": 0, "error": str(exc)},
            )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        del instance_id, kwargs
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)

    def _search(self, query: str) -> list[dict[str, Any]]:
        payload = json.dumps({"query": query, "tool": self.tool_name, "top_k": self.top_k}, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.retrieval_server_url}/search",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return list(data.get("results", []))

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "(no results)"
        lines: list[str] = []
        for item in results:
            chunk_id = str(item.get("chunk_id", ""))
            text = str(item.get("text", "")).strip()
            if len(text) > self.max_text_len:
                text = f"{text[:self.max_text_len].rstrip()}..."
            lines.append(f"[{chunk_id}] {text}")
        return "\n".join(lines)


class KeywordSearchTool(_BaseNovelSearchTool):
    tool_name = "keyword_search"


class DenseSearchTool(_BaseNovelSearchTool):
    tool_name = "dense_search"


class HybridSearchTool(_BaseNovelSearchTool):
    tool_name = "hybrid_search"
