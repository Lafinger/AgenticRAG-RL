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


class FakeTemplateTokenizer:
    chat_template = "fake-template"

    def __init__(self, *, fail_tools: bool = False) -> None:
        self.fail_tools = fail_tools
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": [dict(message) for message in messages], "kwargs": dict(kwargs)})
        if self.fail_tools and "tools" in kwargs:
            raise TypeError("tools are not supported by this fake template")
        return {"messages": messages, "tools": kwargs.get("tools"), "kwargs": kwargs}


def test_apply_chat_template_for_generation_passes_qwen3_tools() -> None:
    module = load_agentic_eval_module()
    tokenizer = FakeTemplateTokenizer()
    messages = [
        {"role": "system", "content": module.AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "问题"},
        {"role": "tool", "content": "[chunk-a] 证据"},
    ]

    rendered = module.apply_chat_template_for_generation(tokenizer, messages, "qwen3_react")

    assert rendered["tools"] == module.TOOL_SCHEMAS
    assert tokenizer.calls[0]["messages"][2]["role"] == "tool"
    assert "enable_thinking" not in tokenizer.calls[0]["kwargs"]


def test_apply_chat_template_for_generation_falls_back_to_manual_tools_prompt() -> None:
    module = load_agentic_eval_module()
    tokenizer = FakeTemplateTokenizer(fail_tools=True)
    messages = [
        {"role": "system", "content": module.AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "问题"},
        {
            "role": "assistant",
            "content": '<think>要回答最终问题，先查：问题</think>\n<tool_call>{"name":"keyword_search","arguments":{"query":"问题"}}</tool_call>',
        },
        {"role": "tool", "content": "[chunk-a] 证据"},
    ]

    rendered = module.apply_chat_template_for_generation(tokenizer, messages, "qwen3_react")
    fallback_messages = tokenizer.calls[1]["messages"]

    assert rendered["tools"] is None
    assert "# Tools" in fallback_messages[0]["content"]
    assert fallback_messages[3]["role"] == "user"
    assert fallback_messages[3]["content"] == "<tool_response>\n[chunk-a] 证据\n</tool_response>"


def test_valid_tool_call_invokes_retriever_and_answer_finishes_loop() -> None:
    module = load_agentic_eval_module()
    outputs = iter(
        [
            '<think>要回答最终问题，先查：卖饼老者 集市</think>\n<tool_call>{"name":"keyword_search","arguments":{"query":"卖饼老者 集市"}}</tool_call>',
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
    assert result["raw_turns"][0]["think_tag_present"] is True
    assert result["raw_turns"][0]["history_assistant"] == (
        '<think>要回答最终问题，先查：卖饼老者 集市</think>\n'
        '<tool_call>\n{"name":"keyword_search","arguments":{"query":"卖饼老者 集市"}}\n</tool_call>'
    )


def test_invalid_json_tool_call_records_failed_turn_without_answer() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()

    result = module.run_agentic_loop("问题", retriever, lambda messages: "<tool_call>{bad json}</tool_call>", max_turns=1)

    assert retriever.calls == []
    assert result["prediction"] == ""
    assert result["status"].startswith("invalid_tool_call_json")
    assert result["valid_tool_call_count"] == 0
    assert result["raw_turns"][0]["status"] == "failed"


def test_query_only_tool_call_remains_malformed() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()

    result = module.run_agentic_loop("问题", retriever, lambda messages: "<tool_call>卖饼老者 集市</tool_call>", max_turns=1)

    assert retriever.calls == []
    assert result["prediction"] == ""
    assert result["status"].startswith("invalid_tool_call_json")
    assert result["valid_tool_call_count"] == 0
    assert result["raw_turns"][0]["malformed_tool_fragment_present"] is True


def test_resolve_local_path_or_repo_keeps_repo_ids_and_resolves_local_paths(tmp_path: Path) -> None:
    module = load_agentic_eval_module()
    local_dir = tmp_path / "checkpoint-1"
    local_dir.mkdir()

    assert module.resolve_local_path_or_repo(str(local_dir)) == str(local_dir.resolve())
    assert module.resolve_local_path_or_repo("unsloth/Qwen3-4B-Instruct-2507") == "unsloth/Qwen3-4B-Instruct-2507"


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


def test_multiple_tool_calls_uses_first_valid_action_and_records_diagnostics() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()
    text = (
        '<tool_call>{"name":"keyword_search","arguments":{"query":"第一跳"}}</tool_call>\n'
        '<tool_call>{"name":"keyword_search","arguments":{"query":"第二跳"}}</tool_call>'
    )

    result = module.run_agentic_loop("问题", retriever, lambda messages: text, max_turns=1, top_k=2)

    assert retriever.calls == [("keyword_search", "第一跳", 2)]
    assert result["status"] == "max_turns_exceeded"
    assert result["raw_turns"][0]["multi_action_present"] is True
    assert result["raw_turns"][0]["normalized_action"] == (
        '<think>要回答最终问题，先查：第一跳</think>\n'
        '<tool_call>\n{"name":"keyword_search","arguments":{"query":"第一跳"}}\n</tool_call>'
    )
    assert result["raw_turns"][0]["think_tag_present"] is False


def test_starts_with_closing_tool_tag_is_recorded_but_valid_action_can_run() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()
    text = '</tool_call>\n<tool_call>{"name":"keyword_search","arguments":{"query":"可解析查询"}}</tool_call>'

    result = module.run_agentic_loop("问题", retriever, lambda messages: text, max_turns=1)

    assert retriever.calls == [("keyword_search", "可解析查询", 3)]
    assert result["raw_turns"][0]["starts_with_closing_tool"] is True
    assert result["raw_turns"][0]["malformed_tool_fragment_present"] is True


def test_agent_loop_writes_normalized_action_to_next_turn_history() -> None:
    module = load_agentic_eval_module()
    retriever = FakeRetriever()
    seen_histories: list[list[dict[str, str]]] = []
    outputs = iter(
        [
            '</tool_call>\n<tool_call>{"name":"keyword_search","arguments":{"query":"第一跳"}}</tool_call>\n'
            '<tool_call>{"name":"keyword_search","arguments":{"query":"第二跳"}}</tool_call>',
            "<answer>侯赢</answer>",
        ]
    )

    def generate(messages: list[dict[str, str]]) -> str:
        seen_histories.append([dict(message) for message in messages])
        return next(outputs)

    result = module.run_agentic_loop("问题", retriever, generate, max_turns=2)

    assert result["prediction"] == "侯赢"
    assert result["raw_turns"][0]["assistant"].startswith("</tool_call>")
    assert result["raw_turns"][0]["history_assistant"] == (
        '<think>要回答最终问题，先查：第一跳</think>\n'
        '<tool_call>\n{"name":"keyword_search","arguments":{"query":"第一跳"}}\n</tool_call>'
    )
    assert result["raw_turns"][0]["truncated_to_first_action"] is True
    assert seen_histories[1][2]["role"] == "assistant"
    assert seen_histories[1][2]["content"] == result["raw_turns"][0]["history_assistant"]
    assert seen_histories[1][3]["role"] == "tool"
    assert seen_histories[1][3]["content"].startswith("[chunk-a]")
    assert "<tool_response>" not in seen_histories[1][3]["content"]


def test_stop_on_action_criteria_stops_after_complete_action_tag() -> None:
    module = load_agentic_eval_module()

    class DecodeTokenizer:
        def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return "".join(chr(token_id) for token_id in ids)

    criteria = module.StopOnActionCriteria(DecodeTokenizer(), prompt_length=2)

    assert criteria([[1, 2, *map(ord, "<think>要回答最终问题，先查：问题</think>")]]) is False
    assert criteria([[1, 2, *map(ord, "<tool_call>{}")]]) is False
    assert criteria([[1, 2, *map(ord, "<tool_call>{}</tool_call>")]]) is True
    assert criteria([[1, 2, *map(ord, "<answer>侯赢</answer>")]]) is True


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


def test_build_summary_includes_agent_loop_diagnostic_rates() -> None:
    module = load_agentic_eval_module()
    records = [
        {
            "em": 0.0,
            "f1": 0.0,
            "hop_recall": 0.5,
            "status": "max_turns_exceeded",
            "valid_tool_call_count": 2,
            "raw_turns": [
                {
                    "status": "tool_called",
                    "malformed_tool_fragment_present": True,
                    "malformed_think_present": True,
                    "think_tag_present": False,
                    "multi_action_present": True,
                    "starts_with_closing_tool": True,
                },
                {
                    "status": "tool_called",
                    "malformed_tool_fragment_present": False,
                    "malformed_think_present": False,
                    "think_tag_present": True,
                    "multi_action_present": False,
                    "starts_with_closing_tool": False,
                },
            ],
        },
        {
            "em": 1.0,
            "f1": 1.0,
            "hop_recall": 1.0,
            "status": "answered",
            "valid_tool_call_count": 1,
            "raw_turns": [
                {
                    "status": "answered",
                    "malformed_tool_fragment_present": False,
                    "malformed_think_present": False,
                    "think_tag_present": False,
                    "multi_action_present": False,
                    "starts_with_closing_tool": False,
                }
            ],
        },
    ]

    summary = module.build_summary(records)

    assert summary["max_turns_exceeded_rate"] == 0.5
    assert summary["avg_valid_tool_calls"] == 1.5
    assert summary["avg_turns"] == 1.5
    assert summary["think_tag_rate"] == 0.5
    assert summary["malformed_think_rate"] == 1 / 3
    assert summary["malformed_tool_fragment_rate"] == 1 / 3
    assert summary["multi_action_turn_rate"] == 1 / 3
    assert summary["starts_with_closing_tool_rate"] == 1 / 3


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
