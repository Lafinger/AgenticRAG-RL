from __future__ import annotations

import json

import pytest
import httpx

from agentic_rag_rl.llm_client import (
    BatchChatResult,
    DoubaoBatchJobClient,
    DoubaoBatchJobConfig,
    DoubaoLLMClient,
    NewAPILLMClient,
    RightCodeLLMClient,
    create_llm_client,
    get_doubao_batch_job_config,
    get_doubao_base_url,
    get_doubao_model,
    get_doubao_thinking_model,
    get_doubao_use_batch_inference,
    get_newapi_base_url,
    get_newapi_model,
    get_rightcode_base_url,
    get_rightcode_model,
    resolve_judge_model,
    resolve_kg_model,
    resolve_thinking_model,
    split_doubao_model_version,
)


def test_doubao_llm_client_uses_transport() -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_transport(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "ok"

    client = DoubaoLLMClient(api_key="test-key", transport=fake_transport)
    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls == [[{"role": "user", "content": "你好"}]]


def test_create_llm_client_returns_doubao_client() -> None:
    client = create_llm_client("doubao", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, DoubaoLLMClient)


def test_create_llm_client_returns_newapi_client() -> None:
    client = create_llm_client("newapi", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, NewAPILLMClient)


def test_create_llm_client_returns_rightcode_client() -> None:
    client = create_llm_client("rightcode", api_key="test-key", transport=lambda _: "ok")

    assert isinstance(client, RightCodeLLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("unknown", api_key="test-key")


def test_doubao_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ARK_API_KEY"):
        DoubaoLLMClient(api_key="")


def test_newapi_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("NEWAPI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NEWAPI_API_KEY"):
        NewAPILLMClient(api_key="")


def test_rightcode_llm_client_requires_api_key_for_real_transport(monkeypatch) -> None:
    monkeypatch.delenv("RIGHTCODE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RIGHTCODE_API_KEY"):
        RightCodeLLMClient(api_key="")


def test_doubao_llm_client_reports_model_not_found() -> None:
    client = DoubaoLLMClient(api_key="test-key", transport=lambda _: "ok")
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(
        404,
        request=request,
        text='{"error":{"code":"InvalidEndpointOrModel.NotFound","message":"not found"}}',
    )

    with pytest.raises(RuntimeError, match="DOUBAO_THINKING_MODEL|--merge-model|--disable-llm-merge"):
        client._raise_for_status(response)


def test_doubao_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOUBAO_MODEL", "env-model")
    monkeypatch.setenv("DOUBAO_THINKING_MODEL", "env-thinking-model")
    monkeypatch.setenv("DOUBAO_BASE_URL", "https://env.example/api/v3")
    monkeypatch.setenv("DOUBAO_USE_BATCH_INFERENCE", "true")

    assert get_doubao_model() == "env-model"
    assert get_doubao_thinking_model() == "env-thinking-model"
    assert get_doubao_base_url() == "https://env.example/api/v3"
    assert get_doubao_use_batch_inference() is True


def test_newapi_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEWAPI_MODEL", "env-newapi-model")
    monkeypatch.setenv("NEWAPI_KG_MODEL", "env-newapi-kg")
    monkeypatch.setenv("NEWAPI_THINKING_MODEL", "env-newapi-thinking")
    monkeypatch.setenv("NEWAPI_JUDGE_MODEL", "env-newapi-judge")
    monkeypatch.setenv("NEWAPI_BASE_URL", "https://newapi.example/v1")

    assert get_newapi_model() == "env-newapi-model"
    assert get_newapi_base_url() == "https://newapi.example/v1"
    assert resolve_kg_model("newapi") == "env-newapi-kg"
    assert resolve_thinking_model("newapi") == "env-newapi-thinking"
    assert resolve_judge_model("newapi") == "env-newapi-judge"


def test_newapi_builtin_defaults_use_gpt_5_5(monkeypatch) -> None:
    monkeypatch.delenv("NEWAPI_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_KG_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_THINKING_MODEL", raising=False)
    monkeypatch.delenv("NEWAPI_JUDGE_MODEL", raising=False)

    assert get_newapi_model() == "gpt-5.5"
    assert resolve_kg_model("newapi") == "gpt-5.5"
    assert resolve_thinking_model("newapi") == "gpt-5.5"
    assert resolve_judge_model("newapi") == "gpt-5.5"


def test_rightcode_defaults_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RIGHTCODE_MODEL", "env-rc-model")
    monkeypatch.setenv("RIGHTCODE_KG_MODEL", "env-rc-kg")
    monkeypatch.setenv("RIGHTCODE_THINKING_MODEL", "env-rc-thinking")
    monkeypatch.setenv("RIGHTCODE_JUDGE_MODEL", "env-rc-judge")
    monkeypatch.setenv("RIGHTCODE_BASE_URL", "https://rc.example/v1")

    assert get_rightcode_model() == "env-rc-model"
    assert get_rightcode_base_url() == "https://rc.example/v1"
    assert resolve_kg_model("rightcode") == "env-rc-kg"
    assert resolve_thinking_model("rightcode") == "env-rc-thinking"
    assert resolve_judge_model("rightcode") == "env-rc-judge"


def test_doubao_online_inference_uses_chat_path(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = DoubaoLLMClient(
        api_key="test-key",
        model="doubao-test",
        base_url="https://online.example/api/v3",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "https://online.example/api/v3/chat/completions"
    assert calls[0]["json"] == {
        "model": "doubao-test",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout"] == 123
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_newapi_online_inference_uses_openai_compatible_chat_path(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = NewAPILLMClient(
        api_key="sk-test",
        model="gpt-5.5",
        base_url="https://api.6i2.com/v1",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "https://api.6i2.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"] == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout"] == 123
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_rightcode_online_inference_uses_openai_compatible_chat_path(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> httpx.Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = RightCodeLLMClient(
        api_key="sk-test",
        model="gpt-5.5",
        base_url="https://api.right.codes/v1",
        timeout_seconds=123,
    )

    content = client.chat([{"role": "user", "content": "你好"}])

    assert content == "ok"
    assert calls[0]["url"] == "https://api.right.codes/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"] == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout"] == 123
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_split_doubao_model_version() -> None:
    assert split_doubao_model_version("doubao-seed-2-0-pro-260215") == ("doubao-seed-2-0-pro", "260215")


def test_get_doubao_batch_job_config_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOLC_ACCESSKEY", "ak")
    monkeypatch.setenv("VOLC_SECRETKEY", "sk")
    monkeypatch.setenv("TOS_BUCKET", "bucket")
    monkeypatch.setenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
    monkeypatch.setenv("DOUBAO_BATCH_INPUT_PREFIX", "input")
    monkeypatch.setenv("DOUBAO_BATCH_OUTPUT_PREFIX", "output")

    config = get_doubao_batch_job_config()

    assert config.access_key == "ak"
    assert config.secret_key == "sk"
    assert config.tos_bucket == "bucket"
    assert config.input_key_prefix == "input/"
    assert config.output_key_prefix == "output/"
    assert config.foundation_model_name == "doubao-seed-2-0-pro"
    assert config.foundation_model_version == "260215"


def test_batch_job_client_writes_chat_requests(tmp_path) -> None:
    config = DoubaoBatchJobConfig(
        access_key="ak",
        secret_key="sk",
        tos_endpoint="tos.example",
        tos_region="cn-beijing",
        tos_bucket="bucket",
        input_key_prefix="input/",
        output_key_prefix="output/",
        ark_region="cn-beijing",
        ark_host="ark.example",
        project_name="default",
        completion_window="1d",
        foundation_model_name="doubao-seed-2-0-pro",
        foundation_model_version="260215",
    )
    client = DoubaoBatchJobClient(config)
    target = tmp_path / "requests.jsonl"

    client.write_chat_requests(
        [{"custom_id": "task-1", "messages": [{"role": "user", "content": "你好"}], "temperature": 0.0}],
        target,
        model="doubao-seed-2-0-pro-260215",
    )

    record = json.loads(target.read_text(encoding="utf-8").strip())
    assert record == {
        "custom_id": "task-1",
        "body": {
            "model": "doubao-seed-2-0-pro-260215",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.0,
        },
    }


def test_batch_job_client_reads_openai_style_results(tmp_path) -> None:
    config = DoubaoBatchJobConfig(
        access_key="ak",
        secret_key="sk",
        tos_endpoint="tos.example",
        tos_region="cn-beijing",
        tos_bucket="bucket",
        input_key_prefix="input/",
        output_key_prefix="output/",
        ark_region="cn-beijing",
        ark_host="ark.example",
        project_name="default",
        completion_window="1d",
        foundation_model_name="doubao-seed-2-0-pro",
        foundation_model_version="260215",
    )
    client = DoubaoBatchJobClient(config)
    target = tmp_path / "results.jsonl"
    target.write_text(
        json.dumps(
            {
                "custom_id": "task-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"total_tokens": 5},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = client.read_result_records(target)

    assert results["task-1"] == BatchChatResult(
        custom_id="task-1",
        content="ok",
        usage={"total_tokens": 5},
        raw=json.loads(target.read_text(encoding="utf-8")),
    )


def test_batch_job_client_creates_output_prefix_placeholder() -> None:
    class FakeTosClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bytes]] = []

        def put_object(self, bucket: str, key: str, content: bytes) -> None:
            self.calls.append((bucket, key, content))

    config = DoubaoBatchJobConfig(
        access_key="ak",
        secret_key="sk",
        tos_endpoint="tos.example",
        tos_region="cn-beijing",
        tos_bucket="bucket",
        input_key_prefix="input/",
        output_key_prefix="output/",
        ark_region="cn-beijing",
        ark_host="ark.example",
        project_name="default",
        completion_window="1d",
        foundation_model_name="doubao-seed-2-0-pro",
        foundation_model_version="260215",
    )
    fake_tos = FakeTosClient()
    client = DoubaoBatchJobClient(config, tos_client=fake_tos)

    client.ensure_output_prefix("output/job-1")

    assert fake_tos.calls == [("bucket", "output/job-1/", b"")]


def test_batch_job_download_outputs_merges_existing_results(tmp_path) -> None:
    class FakeTosClient:
        def get_object_to_file(self, bucket: str, key: str, file_path: str) -> None:
            records = {
                "output/run-2/bi-2/output/results.jsonl": [
                    {"custom_id": "task-2", "response": {"body": {"choices": [{"message": {"content": "new"}}]}}},
                    {"custom_id": "task-4", "response": {"body": {"choices": [{"message": {"content": "extra"}}]}}},
                ],
                "output/run-2/bi-2/error/errors.jsonl": [],
            }[key]
            with open(file_path, "w", encoding="utf-8", newline="") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False))
                    handle.write("\r\n")

    config = DoubaoBatchJobConfig(
        access_key="ak",
        secret_key="sk",
        tos_endpoint="tos.example",
        tos_region="cn-beijing",
        tos_bucket="bucket",
        input_key_prefix="input/",
        output_key_prefix="output/",
        ark_region="cn-beijing",
        ark_host="ark.example",
        project_name="default",
        completion_window="1d",
        foundation_model_name="doubao-seed-2-0-pro",
        foundation_model_version="260215",
    )
    client = DoubaoBatchJobClient(config, tos_client=FakeTosClient())
    existing = [
        {"custom_id": "task-1", "response": {"body": {"choices": [{"message": {"content": "old-1"}}]}}},
        {"custom_id": "task-2", "response": {"body": {"choices": [{"message": {"content": "old-2"}}]}}},
        {"custom_id": "task-3", "response": {"body": {"choices": [{"message": {"content": "old-3"}}]}}},
    ]
    results_path = tmp_path / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        for record in existing:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")

    client.download_outputs("bi-2", "output/run-2/", tmp_path, request_order=["task-2", "task-4"])

    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert [record["custom_id"] for record in records] == ["task-1", "task-2", "task-3", "task-4"]
    assert records[1]["response"]["body"]["choices"][0]["message"]["content"] == "new"
    assert (tmp_path / "results.bi-2.jsonl").exists()
