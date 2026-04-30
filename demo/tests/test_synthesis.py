from __future__ import annotations

import threading
import time
from pathlib import Path

from agentic_rag_rl.io import load_chunks
from agentic_rag_rl.synthesis import (
    _build_seed_qa_messages,
    clean_seed_qa_records,
    clean_multihop_examples,
    generate_seed_qa,
    generate_seed_questions,
    iter_synthesize_multihop_examples,
    iter_seed_question_batches,
    multihop_chain_key,
    synthesize_multihop_examples,
)
from agentic_rag_rl.types import Chunk


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
                "qa_type": "action_result",
                "entities": ["孙少平"]
            }
]
```"""


class FakeMergeLLMClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del temperature
        self.calls.append(messages)
        return """{
  "final_question": "孙少平的学校处境相关线索最终指向什么表现？",
  "final_answer": "最后去取黑高粱面馍。",
  "qa_type": "inference",
  "answer_aliases": ["最后去取黑高粱面馍"]
}"""


class SlowMergeLLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del messages, temperature
        with self.lock:
            self.calls += 1
            call_id = self.calls
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
            return f"""{{
  "final_question": "并发合并问题{call_id}？",
  "final_answer": "学校",
  "qa_type": "inference",
  "answer_aliases": ["学校"]
}}"""
        finally:
            with self.lock:
                self.active_calls -= 1


class BrokenMergeLLMClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del temperature
        self.calls.append(messages)
        raise TimeoutError("merge timeout")


class FlakySeedLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del messages, temperature
        self.calls += 1
        if self.calls == 1:
            return '[{"question": "坏 JSON", "answer" "缺少冒号"}]'
        return '[{"question": "孙少平取了什么？", "answer": "黑高粱面馍", "qa_type": "object", "entities": ["孙少平"]}]'


class BrokenSeedLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del messages, temperature
        self.calls += 1
        return '[{"question": "坏 JSON", "answer" "缺少冒号"}]'


class SlowSeedLLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        del messages, temperature
        with self.lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
            return (
                '[{"question": "孙少平在哪里生活艰难？", "answer": "学校", '
                '"qa_type": "place", "entities": ["孙少平", "学校"]}]'
            )
        finally:
            with self.lock:
                self.active_calls -= 1


def test_generate_seed_questions_uses_llm_client() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    client = FakeLLMClient()
    seeds = generate_seed_questions(chunks[:1], client, max_per_chunk=1)

    assert len(client.calls) == 1
    assert "中文武侠小说" in client.calls[0][1]["content"]
    assert seeds == [
            {
                "question": "孙少平在学校生活艰难的表现是什么？",
                "answer": "最后去取黑高粱面馍",
                "doc_chunk_id": "corpus_chunkids_000001",
                "tool": "keyword_search",
            "entities": ["孙少平"],
            "qa_type": "action_result",
        }
    ]


def test_iter_seed_question_batches_yields_per_chunk() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")[:2]
    client = FakeLLMClient()
    batches = list(iter_seed_question_batches(chunks, client, max_per_chunk=1))

    assert len(batches) == 2
    assert [batch[0]["doc_chunk_id"] for batch in batches] == [chunk.chunk_id for chunk in chunks]


def test_iter_seed_question_batches_respects_max_concurrency() -> None:
    chunks = [
        Chunk(chunk_id=f"c{index}", title=str(index), text="孙少平在学校生活艰难。")
        for index in range(4)
    ]
    client = SlowSeedLLMClient()

    batches = list(iter_seed_question_batches(chunks, client, max_per_chunk=1, max_concurrency=2))

    assert len(batches) == 4
    assert client.calls == 4
    assert client.max_active_calls == 2
    assert sorted(batch[0]["doc_chunk_id"] for batch in batches) == ["c0", "c1", "c2", "c3"]


def test_generate_seed_qa_retries_invalid_json_response() -> None:
    client = FlakySeedLLMClient()

    records = generate_seed_qa(client, "孙少平取走黑高粱面馍。", max_items=1, max_attempts=2)

    assert client.calls == 2
    assert records == [
        {
            "question": "孙少平取了什么？",
            "answer": "黑高粱面馍",
            "qa_type": "object",
            "entities": ["孙少平"],
        }
    ]


def test_clean_seed_qa_records_filters_low_quality_items() -> None:
    records = [
        {
            "question": "郝红梅的名字叫什么？",
            "answer": "郝红梅",
            "qa_type": "character",
            "entities": "郝红梅,名字",
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
        },
        {
            "question": "孙少平取走了什么主食？",
            "answer": "黑高粱面馍。",
            "qa_type": "object",
            "entities": "孙少平,黑高粱面馍",
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
        },
        {
            "question": "孙少平取走了什么主食？",
            "answer": "黑高粱面馍。",
            "qa_type": "object",
            "entities": ["孙少平"],
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
        },
    ]

    cleaned, dropped = clean_seed_qa_records(records, valid_chunk_ids={"c1"})

    assert cleaned == [
        {
            "question": "孙少平取走了什么主食？",
            "answer": "黑高粱面馍",
            "qa_type": "object",
            "entities": ["孙少平", "黑高粱面馍"],
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
        }
    ]
    assert len(dropped) == 2
    assert "answer_leaked_in_question" in dropped[0]["drop_reasons"]
    assert "duplicate_question" in dropped[1]["drop_reasons"]


def test_iter_seed_question_batches_can_continue_after_invalid_json() -> None:
    chunks = [Chunk(chunk_id="c1", title="1", text="孙少平取走黑高粱面馍。")]
    client = BrokenSeedLLMClient()
    failed_chunk_ids: list[str] = []

    batches = list(
        iter_seed_question_batches(
            chunks,
            client,
            max_per_chunk=1,
            max_attempts=1,
            continue_on_error=True,
            on_chunk_failed=lambda chunk, exc: failed_chunk_ids.append(chunk.chunk_id),
        )
    )

    assert batches == []
    assert failed_chunk_ids == ["c1"]
    assert client.calls == 1


def test_seed_qa_prompt_uses_novel_domain_constraints() -> None:
    messages = _build_seed_qa_messages("孙少平取走两个高粱面馍。", max_items=1)
    prompt = messages[1]["content"]

    assert all(term in prompt for term in ["人物名", "地点名", "物品名", "明确关系", "明确行为结果"])
    assert all(qa_type in prompt for qa_type in ["character", "place", "object", "relation", "action_result"])
    assert "character_identity" not in prompt
    assert "如果片段中出现明确时间、年代、季节、上学阶段、事件阶段" in prompt
    assert "不能编造时间" in prompt
    assert "唯一答案" in prompt


def test_synthesize_and_clean_multihop_examples() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, FakeLLMClient(), max_per_chunk=1)
    merge_client = FakeMergeLLMClient()
    examples = synthesize_multihop_examples(seeds, chunks, target_count=2, merge_llm_client=merge_client)
    cleaned = clean_multihop_examples(examples, {chunk.chunk_id for chunk in chunks})

    assert merge_client.calls
    assert cleaned
    assert cleaned[0]["final_question"] == "孙少平的学校处境相关线索最终指向什么表现？"
    assert cleaned[0]["hop_count"] >= 2
    assert all(hop["doc_chunk_id"] for hop in cleaned[0]["hops"])
    assert all(hop["qa_type"] == "action_result" for hop in cleaned[0]["hops"])


def test_iter_synthesize_multihop_examples_respects_merge_concurrency() -> None:
    chunks = [
        Chunk(chunk_id="c1", title="1", text="孙少平在学校取黑高粱面馍。"),
        Chunk(chunk_id="c2", title="2", text="黑高粱面馍让孙少平想起贫困。"),
        Chunk(chunk_id="c3", title="3", text="孙少平在学校生活艰难。"),
    ]
    seeds = [
        {
            "question": "孙少平取了什么主食？",
            "answer": "黑高粱面馍",
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
            "qa_type": "object",
        },
        {
            "question": "黑高粱面馍让孙少平想起什么？",
            "answer": "贫困",
            "doc_chunk_id": "c2",
            "tool": "keyword_search",
            "qa_type": "action_result",
        },
        {
            "question": "孙少平在哪里生活艰难？",
            "answer": "学校",
            "doc_chunk_id": "c3",
            "tool": "keyword_search",
            "qa_type": "place",
        },
    ]
    merge_client = SlowMergeLLMClient()

    examples = list(
        iter_synthesize_multihop_examples(
            seeds,
            chunks,
            target_count=2,
            merge_llm_client=merge_client,
            max_concurrency=2,
        )
    )

    assert len(examples) == 2
    assert merge_client.max_active_calls == 2


def test_multihop_merge_failure_falls_back_to_rule_template() -> None:
    chunks = load_chunks(DATA_DIR / "corpus.jsonl")
    seeds = generate_seed_questions(chunks, FakeLLMClient(), max_per_chunk=1)
    merge_client = BrokenMergeLLMClient()

    examples = synthesize_multihop_examples(seeds, chunks, target_count=1, merge_llm_client=merge_client)

    assert merge_client.calls
    assert len(examples) == 1
    assert examples[0]["final_question"].startswith("第1步回答")
    assert examples[0]["final_answer"]


def test_multihop_resume_skips_existing_chain_before_llm_call() -> None:
    chunks = [
        Chunk(chunk_id="c1", title="1", text="孙少平在学校取黑高粱面馍。"),
        Chunk(chunk_id="c2", title="2", text="孙少平在学校生活艰难。"),
    ]
    seeds = [
        {
            "question": "孙少平取了什么？",
            "answer": "黑高粱面馍",
            "doc_chunk_id": "c1",
            "tool": "keyword_search",
            "qa_type": "object",
        },
        {
            "question": "孙少平在哪里生活艰难？",
            "answer": "学校",
            "doc_chunk_id": "c2",
            "tool": "keyword_search",
            "qa_type": "place",
        },
    ]
    skip_chain_keys = {
        multihop_chain_key(
            [
                {"doc_chunk_id": "c1", "question": "孙少平取了什么？"},
                {"doc_chunk_id": "c2", "question": "孙少平在哪里生活艰难？"},
            ]
        ),
        multihop_chain_key(
            [
                {"doc_chunk_id": "c2", "question": "孙少平在哪里生活艰难？"},
                {"doc_chunk_id": "c1", "question": "孙少平取了什么？"},
            ]
        ),
    }
    merge_client = FakeMergeLLMClient()

    examples = list(
        iter_synthesize_multihop_examples(
            seeds,
            chunks,
            target_count=1,
            merge_llm_client=merge_client,
            skip_chain_keys=skip_chain_keys,
        )
    )

    assert examples == []
    assert merge_client.calls == []
