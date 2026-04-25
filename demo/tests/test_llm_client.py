from __future__ import annotations

import pytest

from agentic_rag_rl.llm_client import DoubaoSeedQAClient, extract_json_array, get_doubao_base_url, get_doubao_model


def test_extract_json_array_handles_markdown_fenced_output() -> None:
    text = """```json
[
  {"question": "孙少平在学校生活艰难的表现是什么？", "answer": "最后去取黑高粱面馍。", "qa_type": "character_behavior", "entities": ["孙少平"]}
]
```"""

    records = extract_json_array(text)

    assert records[0]["question"].startswith("孙少平")


def test_doubao_seed_qa_client_normalizes_records() -> None:
    client = DoubaoSeedQAClient(api_key="test-key", transport=lambda _: """[
      {"question": "双水村为什么叫双水村？", "answer": "因为有东拉河和哭咽河。", "qa_type": "place_origin", "entities": ["双水村"]}
    ]""")

    records = client.generate_seed_qa("双水村有东拉河和哭咽河。", max_items=1)

    assert records == [
        {
            "question": "双水村为什么叫双水村？",
            "answer": "因为有东拉河和哭咽河。",
            "qa_type": "place_origin",
            "entities": ["双水村"],
        }
    ]


def test_doubao_seed_qa_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ARK_API_KEY"):
        DoubaoSeedQAClient(api_key="")


def test_doubao_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOUBAO_MODEL", "env-model")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://env.example/api/v3")

    assert get_doubao_model() == "env-model"
    assert get_doubao_base_url() == "https://env.example/api/v3"
