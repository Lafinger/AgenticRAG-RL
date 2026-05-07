from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_agentic_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_hf_agentic_for_test", ROOT / "scripts" / "eval_hf_agentic.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeResult:
    chunk_id: str
    title: str = "标题"
    text: str = "证据"
    score: float = 1.0
    source: str = "keyword"

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": {},
        }


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def dispatch(self, tool_name: str, query: str, top_k: int = 5) -> list[FakeResult]:
        self.calls.append((tool_name, query, top_k))
        return [FakeResult("chunk-a"), FakeResult("chunk-b")]


def test_valid_tool_call_invokes_retriever_and_answer_finishes_loop() -> None:
    module = load_agentic_eval_module()
    outputs = iter(
        [
            '<tool_call>{"name":"keyword_search","arguments":{"query":"卖饼老者 集市"}}</tool_call>',
            "<answer>侯赢</answer>",
        ]
    )
    retriever = FakeRetriever()

    result = module.run_agentic_loop("问题", retriever, lambda messages: next(outputs), max_turns=2, top_k=3)

    assert retriever.calls == [("keyword_search", "卖饼老者 集市", 3)]
    assert result["prediction"] == "侯赢"
    assert result["status"] == "answered"
    assert result["valid_tool_call_count"] == 1
    assert result["retrieved_chunk_ids"] == ["chunk-a", "chunk-b"]


def test_invalid_json_tool_call_records_failed_turn_without_answer() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()

    result = module.run_agentic_loop("问题", retriever, lambda messages: "<tool_call>{bad json}</tool_call>", max_turns=1)

    assert retriever.calls == []
    assert result["prediction"] == ""
    assert result["status"].startswith("invalid_tool_call_json")
    assert result["valid_tool_call_count"] == 0
    assert result["raw_turns"][0]["status"] == "failed"


def test_answer_tag_ends_loop_without_retrieval() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()

    result = module.run_agentic_loop("问题", retriever, lambda messages: "<answer>段誉</answer>", max_turns=3)

    assert retriever.calls == []
    assert result["prediction"] == "段誉"
    assert result["status"] == "answered"
    assert result["valid_tool_call_count"] == 0


def test_max_turns_without_answer_is_marked_unfinished() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()

    result = module.run_agentic_loop(
        "问题",
        retriever,
        lambda messages: '<tool_call>{"name":"keyword_search","arguments":{"query":"继续找"}}</tool_call>',
        max_turns=2,
        top_k=1,
    )

    assert result["prediction"] == ""
    assert result["status"] == "max_turns_exceeded"
    assert result["valid_tool_call_count"] == 2
    assert retriever.calls == [("keyword_search", "继续找", 1), ("keyword_search", "继续找", 1)]


def test_evaluate_record_includes_hop_recall_and_gold_chunks() -> None:
    module = load_agentic_eval_module()
    record = {
        "final_question": "问题",
        "final_answer": "侯赢",
        "answer_aliases": [],
        "hops": [{"doc_chunk_id": "chunk-a"}, {"doc_chunk_id": "chunk-c"}],
    }
    loop_result = {
        "prediction": "侯赢",
        "status": "answered",
        "raw_turns": [],
        "tool_calls": [],
        "valid_tool_call_count": 0,
        "retrieved_chunk_ids": ["chunk-a", "chunk-b"],
        "evidence": [],
    }

    result = module.evaluate_record(record, loop_result, {"model": "fake"})

    assert result["em"] == 1.0
    assert result["f1"] == 1.0
    assert result["hop_recall"] == 0.5
    assert result["gold_chunks"] == ["chunk-a", "chunk-c"]


def test_write_eval_output_jsonl_writes_records_and_sidecar_summary(tmp_path: Path) -> None:
    module = load_agentic_eval_module()
    output = tmp_path / "sft_agentic_eval.jsonl"
    payload = {
        "summary": {
            "count": 1,
            "avg_em": 1.0,
            "avg_f1": 1.0,
            "avg_hop_recall": 0.5,
            "answer_tag_rate": 1.0,
            "valid_tool_call_rate": 1.0,
        },
        "results": [{"question": "问题", "prediction": "侯赢"}],
    }

    summary_path = module.write_eval_output(output, payload)

    assert summary_path == tmp_path / "sft_agentic_eval_summary.json"
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"question": "问题", "prediction": "侯赢"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["avg_em"] == 1.0
    assert summary["avg_f1"] == 1.0
    assert summary["avg_hop_recall"] == 0.5
    assert summary["answer_tag_rate"] == 1.0
    assert summary["valid_tool_call_rate"] == 1.0
    assert summary["format"] == "jsonl_records"


def test_write_eval_output_json_keeps_legacy_summary_results_shape(tmp_path: Path) -> None:
    module = load_agentic_eval_module()
    output = tmp_path / "sft_agentic_eval.json"
    payload = {"summary": {"count": 1}, "results": [{"question": "问题"}]}

    summary_path = module.write_eval_output(output, payload)

    assert summary_path is None
    assert json.loads(output.read_text(encoding="utf-8")) == payload
