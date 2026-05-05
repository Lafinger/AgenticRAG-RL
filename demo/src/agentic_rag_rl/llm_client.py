from __future__ import annotations

import logging
import os
import time
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import httpx


DEFAULT_LLM_PROVIDER = "doubao"
LLM_PROVIDER_CHOICES = ("doubao", "newapi", "rightcode")
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DOUBAO_THINKING_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_DOUBAO_JUDGE_MODEL = DEFAULT_DOUBAO_THINKING_MODEL
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_NEWAPI_MODEL = "gpt-5.5"
DEFAULT_NEWAPI_BASE_URL = "https://api.6i2.com/v1"
DEFAULT_RIGHTCODE_MODEL = "gpt-5.5"
DEFAULT_RIGHTCODE_BASE_URL = "https://api.right.codes/v1"
OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
DEFAULT_DOUBAO_BATCH_REGION = "cn-beijing"
DEFAULT_DOUBAO_BATCH_ARK_HOST = "ark.cn-beijing.volcengineapi.com"
DEFAULT_DOUBAO_BATCH_TOS_ENDPOINT = "tos-cn-beijing.volces.com"
DEFAULT_DOUBAO_BATCH_PROJECT_NAME = "default"
DEFAULT_DOUBAO_BATCH_COMPLETION_WINDOW = "1d"
logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        ...


class BatchChatRequest(TypedDict):
    custom_id: str
    messages: list[ChatMessage]
    temperature: float


@dataclass(frozen=True)
class BatchChatResult:
    custom_id: str
    content: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class DoubaoBatchJobConfig:
    access_key: str
    secret_key: str
    tos_endpoint: str
    tos_region: str
    tos_bucket: str
    input_key_prefix: str
    output_key_prefix: str
    ark_region: str
    ark_host: str
    project_name: str
    completion_window: str
    foundation_model_name: str
    foundation_model_version: str
    custom_model_id: str | None = None


def create_llm_client(
    provider: str = DEFAULT_LLM_PROVIDER,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    transport: Callable[[list[ChatMessage]], str] | None = None,
) -> LLMClient:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return DoubaoLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    if normalized_provider == "newapi":
        return NewAPILLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    if normalized_provider == "rightcode":
        return RightCodeLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_batch_job_client(config: DoubaoBatchJobConfig | None = None) -> "DoubaoBatchJobClient":
    return DoubaoBatchJobClient(config or get_doubao_batch_job_config())


def get_doubao_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_MODEL") or DEFAULT_DOUBAO_MODEL


def get_doubao_thinking_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_THINKING_MODEL") or DEFAULT_DOUBAO_THINKING_MODEL


def get_doubao_judge_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("DOUBAO_JUDGE_MODEL") or os.getenv("DOUBAO_THINKING_MODEL") or DEFAULT_DOUBAO_JUDGE_MODEL


def get_doubao_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("DOUBAO_BASE_URL") or DEFAULT_DOUBAO_BASE_URL


def get_newapi_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("NEWAPI_MODEL") or DEFAULT_NEWAPI_MODEL


def get_newapi_kg_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("NEWAPI_KG_MODEL") or os.getenv("NEWAPI_MODEL") or DEFAULT_NEWAPI_MODEL


def get_newapi_thinking_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("NEWAPI_THINKING_MODEL")
        or os.getenv("NEWAPI_MODEL")
        or DEFAULT_NEWAPI_MODEL
    )


def get_newapi_judge_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("NEWAPI_JUDGE_MODEL")
        or os.getenv("NEWAPI_MODEL")
        or DEFAULT_NEWAPI_MODEL
    )


def get_newapi_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("NEWAPI_BASE_URL") or DEFAULT_NEWAPI_BASE_URL


def get_rightcode_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("RIGHTCODE_MODEL") or DEFAULT_RIGHTCODE_MODEL


def get_rightcode_kg_model(explicit_model: str | None = None) -> str:
    return explicit_model or os.getenv("RIGHTCODE_KG_MODEL") or os.getenv("RIGHTCODE_MODEL") or DEFAULT_RIGHTCODE_MODEL


def get_rightcode_thinking_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("RIGHTCODE_THINKING_MODEL")
        or os.getenv("RIGHTCODE_MODEL")
        or DEFAULT_RIGHTCODE_MODEL
    )


def get_rightcode_judge_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.getenv("RIGHTCODE_JUDGE_MODEL")
        or os.getenv("RIGHTCODE_MODEL")
        or DEFAULT_RIGHTCODE_MODEL
    )


def get_rightcode_base_url(explicit_base_url: str | None = None) -> str:
    return explicit_base_url or os.getenv("RIGHTCODE_BASE_URL") or DEFAULT_RIGHTCODE_BASE_URL


def normalize_llm_provider(provider: str | None) -> str:
    normalized = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if normalized not in LLM_PROVIDER_CHOICES:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: {', '.join(LLM_PROVIDER_CHOICES)}")
    return normalized


def resolve_llm_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return get_doubao_model(explicit_model)
    if normalized_provider == "newapi":
        return get_newapi_model(explicit_model)
    return get_rightcode_model(explicit_model)


def resolve_kg_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return explicit_model or os.getenv("KG_EXTRACTION_MODEL") or DEFAULT_DOUBAO_MODEL
    if normalized_provider == "newapi":
        return get_newapi_kg_model(explicit_model)
    return get_rightcode_kg_model(explicit_model)


def resolve_thinking_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return get_doubao_thinking_model(explicit_model)
    if normalized_provider == "newapi":
        return get_newapi_thinking_model(explicit_model)
    return get_rightcode_thinking_model(explicit_model)


def resolve_judge_model(provider: str, explicit_model: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return get_doubao_judge_model(explicit_model)
    if normalized_provider == "newapi":
        return get_newapi_judge_model(explicit_model)
    return get_rightcode_judge_model(explicit_model)


def resolve_llm_base_url(provider: str, explicit_base_url: str | None = None) -> str:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return get_doubao_base_url(explicit_base_url)
    if normalized_provider == "newapi":
        return get_newapi_base_url(explicit_base_url)
    return get_rightcode_base_url(explicit_base_url)


def get_doubao_use_batch_inference(explicit_enabled: bool = False) -> bool:
    return (
        explicit_enabled
        or os.getenv("DOUBAO_USE_BATCH_INFERENCE", "").strip().lower() in TRUTHY_ENV_VALUES
        or os.getenv("DOUBAO_BATCH_INFERENCE", "").strip().lower() in TRUTHY_ENV_VALUES
    )


def get_doubao_batch_job_config(
    *,
    foundation_model_name: str | None = None,
    foundation_model_version: str | None = None,
    custom_model_id: str | None = None,
    tos_bucket: str | None = None,
    input_key_prefix: str | None = None,
    output_key_prefix: str | None = None,
    project_name: str | None = None,
    completion_window: str | None = None,
) -> DoubaoBatchJobConfig:
    model_name, model_version = split_doubao_model_version(get_doubao_model())
    return DoubaoBatchJobConfig(
        access_key=_read_required_env("VOLC_ACCESSKEY", aliases=("VOLC_ACCESS_KEY",)),
        secret_key=_read_required_env("VOLC_SECRETKEY", aliases=("VOLC_SECRET_KEY",)),
        tos_endpoint=os.getenv("TOS_ENDPOINT") or DEFAULT_DOUBAO_BATCH_TOS_ENDPOINT,
        tos_region=os.getenv("TOS_REGION") or os.getenv("VOLC_REGION") or DEFAULT_DOUBAO_BATCH_REGION,
        tos_bucket=tos_bucket or os.getenv("TOS_BUCKET") or os.getenv("DOUBAO_BATCH_TOS_BUCKET") or "",
        input_key_prefix=_normalize_object_prefix(
            input_key_prefix or os.getenv("DOUBAO_BATCH_INPUT_PREFIX") or "agentic-rag-rl/batch/input/"
        ),
        output_key_prefix=_normalize_object_prefix(
            output_key_prefix or os.getenv("DOUBAO_BATCH_OUTPUT_PREFIX") or "agentic-rag-rl/batch/output/"
        ),
        ark_region=os.getenv("VOLC_REGION") or DEFAULT_DOUBAO_BATCH_REGION,
        ark_host=os.getenv("DOUBAO_BATCH_ARK_HOST") or DEFAULT_DOUBAO_BATCH_ARK_HOST,
        project_name=project_name or os.getenv("DOUBAO_BATCH_PROJECT_NAME") or DEFAULT_DOUBAO_BATCH_PROJECT_NAME,
        completion_window=completion_window
        or os.getenv("DOUBAO_BATCH_COMPLETION_WINDOW")
        or DEFAULT_DOUBAO_BATCH_COMPLETION_WINDOW,
        foundation_model_name=foundation_model_name or os.getenv("DOUBAO_BATCH_FOUNDATION_MODEL") or model_name,
        foundation_model_version=foundation_model_version
        or os.getenv("DOUBAO_BATCH_MODEL_VERSION")
        or model_version,
        custom_model_id=custom_model_id or os.getenv("DOUBAO_BATCH_CUSTOM_MODEL_ID") or None,
    )


def get_doubao_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("DOUBAO_TIMEOUT_SECONDS", "60"))


def get_newapi_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("NEWAPI_TIMEOUT_SECONDS") or os.getenv("DOUBAO_TIMEOUT_SECONDS", "60"))


def get_rightcode_timeout(explicit_timeout: float | None = None) -> float:
    if explicit_timeout is not None:
        return explicit_timeout
    return float(os.getenv("RIGHTCODE_TIMEOUT_SECONDS") or os.getenv("DOUBAO_TIMEOUT_SECONDS", "60"))


def resolve_llm_timeout(provider: str, explicit_timeout: float | None = None) -> float:
    normalized_provider = normalize_llm_provider(provider)
    if normalized_provider == "doubao":
        return get_doubao_timeout(explicit_timeout)
    if normalized_provider == "newapi":
        return get_newapi_timeout(explicit_timeout)
    return get_rightcode_timeout(explicit_timeout)


def _read_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("ARK_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("Doubao API key is required. Set ARK_API_KEY.")
    return api_key.strip()


def _read_newapi_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("NEWAPI_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("NewAPI API key is required. Set NEWAPI_API_KEY or pass --api-key.")
    return api_key.strip()


def _read_rightcode_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("RIGHTCODE_API_KEY") or ""
    if not api_key.strip():
        raise ValueError("RightCode API key is required. Set RIGHTCODE_API_KEY or pass --api-key.")
    return api_key.strip()


def _read_required_env(name: str, *, aliases: tuple[str, ...] = ()) -> str:
    for key in (name, *aliases):
        value = os.getenv(key, "").strip()
        if value:
            return value
    alias_text = f" or {'/'.join(aliases)}" if aliases else ""
    raise ValueError(f"{name}{alias_text} is required for Doubao batch inference jobs.")


def _normalize_object_prefix(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("/")
    if normalized and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def split_doubao_model_version(model: str) -> tuple[str, str]:
    value = model.strip()
    match = __import__("re").match(r"^(?P<name>.+)-(?P<version>\d{6})$", value)
    if not match:
        raise ValueError(
            "Batch inference job foundation model must include a 6-digit version, "
            "for example doubao-seed-2-0-pro-260215."
        )
    return match.group("name"), match.group("version")


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        completions_path: str = OPENAI_CHAT_COMPLETIONS_PATH,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        self.provider = normalize_llm_provider(provider)
        if self.provider == "doubao":
            self.api_key = _read_api_key(api_key) if transport is None else (api_key or "test-key")
        elif self.provider == "newapi":
            self.api_key = _read_newapi_api_key(api_key) if transport is None else (api_key or "test-key")
        else:
            self.api_key = _read_rightcode_api_key(api_key) if transport is None else (api_key or "test-key")
        self.model = resolve_llm_model(self.provider, model)
        self.base_url = resolve_llm_base_url(self.provider, base_url).rstrip("/")
        self.completions_path = completions_path
        self.timeout_seconds = resolve_llm_timeout(self.provider, timeout_seconds)
        self.last_usage: dict[str, Any] | None = None
        self._transport = transport

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        start_time = time.perf_counter()
        logger.info(
            "llm.chat.start provider=%s mode=online model=%s base_url=%s path=%s message_count=%s",
            self.provider,
            self.model,
            self.base_url,
            self.completions_path,
            len(messages),
        )
        try:
            self.last_usage = None
            content = self._transport(messages) if self._transport else self._chat_completions(messages, temperature)
        except Exception:
            logger.exception(
                "llm.chat.failed provider=%s mode=online elapsed_seconds=%.2f",
                self.provider,
                time.perf_counter() - start_time,
            )
            raise
        logger.info(
            "llm.chat.done provider=%s mode=online elapsed_seconds=%.2f response_chars=%s",
            self.provider,
            time.perf_counter() - start_time,
            len(content),
        )
        return content

    def _chat_completions(self, messages: list[ChatMessage], temperature: float) -> str:
        response = httpx.post(
            f"{self.base_url}{self.completions_path}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
        usage = payload.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else None
        if self.last_usage:
            logger.info("llm.chat.usage provider=%s model=%s usage=%s", self.provider, self.model, self.last_usage)
        return payload["choices"][0]["message"]["content"]

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            hint = ""
            if self.provider == "doubao" and "InvalidEndpointOrModel.NotFound" in body:
                hint = (
                    " Hint: the configured Doubao model or Ark endpoint is not available for this account. "
                    "Set DOUBAO_THINKING_MODEL to an enabled Ark model/endpoint, pass --merge-model, "
                    "or use --disable-llm-merge for offline synthesis."
                )
            raise RuntimeError(
                f"{self.provider} request failed: status={response.status_code}, model={self.model}, "
                f"base_url={self.base_url}, response={body}.{hint}"
            ) from exc


class DoubaoLLMClient(OpenAICompatibleLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(
            provider="doubao",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            completions_path=DOUBAO_CHAT_COMPLETIONS_PATH,
            transport=transport,
        )


class NewAPILLMClient(OpenAICompatibleLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(
            provider="newapi",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            completions_path=OPENAI_CHAT_COMPLETIONS_PATH,
            transport=transport,
        )


class RightCodeLLMClient(OpenAICompatibleLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(
            provider="rightcode",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            completions_path=OPENAI_CHAT_COMPLETIONS_PATH,
            transport=transport,
        )


class DoubaoBatchJobClient:
    def __init__(
        self,
        config: DoubaoBatchJobConfig,
        *,
        tos_client: Any | None = None,
        ark_api: Any | None = None,
    ) -> None:
        if not config.tos_bucket:
            raise ValueError("TOS bucket is required. Set TOS_BUCKET or DOUBAO_BATCH_TOS_BUCKET.")
        self.config = config
        self._tos_client = tos_client
        self._ark_api = ark_api

    def run_chat_batch(
        self,
        requests: list[BatchChatRequest],
        *,
        local_dir: str | Path,
        job_name: str,
        model: str,
        poll_interval_seconds: float = 60.0,
        wait_timeout_seconds: float | None = None,
    ) -> dict[str, BatchChatResult]:
        if not requests:
            return {}
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        request_file = target_dir / "requests.jsonl"
        self.write_chat_requests(requests, request_file, model=model)
        object_base = f"{job_name}-{int(time.time())}"
        input_key = f"{self.config.input_key_prefix}{object_base}.jsonl"
        output_prefix = f"{self.config.output_key_prefix}{object_base}/"
        logger.info(
            "batch_job.start request_count=%s job_name=%s bucket=%s input_key=%s output_prefix=%s",
            len(requests),
            job_name,
            self.config.tos_bucket,
            input_key,
            output_prefix,
        )
        self.upload_input_file(request_file, input_key)
        self.ensure_output_prefix(output_prefix)
        job_id = self.create_job(job_name=job_name, input_key=input_key, output_prefix=output_prefix)
        status = self.wait_for_job(job_id, poll_interval_seconds=poll_interval_seconds, timeout_seconds=wait_timeout_seconds)
        logger.info("batch_job.completed job_id=%s status=%s", job_id, status.get("Status"))
        request_order = [request["custom_id"] for request in requests]
        results_path, errors_path = self.download_outputs(job_id, output_prefix, target_dir, request_order=request_order)
        errors = self.read_error_records(errors_path)
        if errors:
            logger.warning("batch_job.errors job_id=%s error_count=%s errors_file=%s", job_id, len(errors), errors_path)
        results = self.read_result_records(results_path)
        logger.info("batch_job.results_loaded job_id=%s result_count=%s", job_id, len(results))
        return results

    def write_chat_requests(self, requests: list[BatchChatRequest], path: str | Path, *, model: str) -> None:
        seen: set[str] = set()
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            for request in requests:
                custom_id = request["custom_id"]
                if custom_id in seen:
                    raise ValueError(f"Duplicate batch custom_id: {custom_id}")
                seen.add(custom_id)
                record = {
                    "custom_id": custom_id,
                    "body": {
                        "model": model,
                        "messages": request["messages"],
                        "temperature": request.get("temperature", 0.2),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\r\n")

    def upload_input_file(self, input_file: Path, object_key: str) -> None:
        client = self._get_tos_client()
        client.put_object_from_file(self.config.tos_bucket, object_key, str(input_file))

    def ensure_output_prefix(self, output_prefix: str) -> None:
        client = self._get_tos_client()
        prefix_key = output_prefix if output_prefix.endswith("/") else f"{output_prefix}/"
        client.put_object(self.config.tos_bucket, prefix_key, content=b"")

    def create_job(self, *, job_name: str, input_key: str, output_prefix: str) -> str:
        volcenginesdkark = self._import_ark_sdk()
        body = volcenginesdkark.CreateBatchInferenceJobRequest(
            name=job_name,
            project_name=self.config.project_name,
            completion_window=self.config.completion_window,
            input_file_tos_location=volcenginesdkark.InputFileTosLocationForCreateBatchInferenceJobInput(
                bucket_name=self.config.tos_bucket,
                object_key=input_key,
            ),
            output_dir_tos_location=volcenginesdkark.OutputDirTosLocationForCreateBatchInferenceJobInput(
                bucket_name=self.config.tos_bucket,
                object_key=output_prefix,
            ),
            model_reference=self._build_model_reference(volcenginesdkark),
        )
        response = self._get_ark_api().create_batch_inference_job(body)
        job_id = getattr(response, "id", None) or getattr(response, "Id", None)
        if not job_id:
            raise RuntimeError(f"CreateBatchInferenceJob did not return Id: {response}")
        return str(job_id)

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            status = self.get_job(job_id)
            phase = str(((status.get("Status") or {}).get("Phase") or "")).strip()
            counts = status.get("RequestCounts") or {}
            logger.info(
                "batch_job.poll job_id=%s phase=%s counts=%s message=%s",
                job_id,
                phase,
                counts,
                (status.get("Status") or {}).get("Message"),
            )
            if phase.lower() in {"completed", "succeeded", "success"}:
                return status
            if phase.lower() in {"failed", "cancelled", "canceled", "expired", "stopped"}:
                raise RuntimeError(f"Batch inference job failed: job_id={job_id}, status={status}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Batch inference job timed out: job_id={job_id}, last_status={status}")
            time.sleep(max(1.0, poll_interval_seconds))

    def get_job(self, job_id: str) -> dict[str, Any]:
        volcenginesdkark = self._import_ark_sdk()
        request = volcenginesdkark.ListBatchInferenceJobsRequest(
            project_name=self.config.project_name,
            page_number=1,
            page_size=1,
            filter=volcenginesdkark.FilterForListBatchInferenceJobsInput(ids=[job_id]),
        )
        response = self._get_ark_api().list_batch_inference_jobs(request)
        items = getattr(response, "items", None) or getattr(getattr(response, "result", None), "items", None)
        if not items:
            raise RuntimeError(f"Batch inference job not found: {job_id}")
        return _sdk_object_to_dict(items[0])

    def download_outputs(
        self,
        job_id: str,
        output_prefix: str,
        local_dir: Path,
        *,
        request_order: list[str] | None = None,
    ) -> tuple[Path, Path]:
        results_key = f"{output_prefix}{job_id}/output/results.jsonl"
        errors_key = f"{output_prefix}{job_id}/error/errors.jsonl"
        results_path = local_dir / "results.jsonl"
        errors_path = local_dir / "errors.jsonl"
        downloaded_results_path = local_dir / f"results.{job_id}.jsonl"
        downloaded_errors_path = local_dir / f"errors.{job_id}.jsonl"
        client = self._get_tos_client()
        if downloaded_results_path.exists():
            downloaded_results_path.unlink()
        if downloaded_errors_path.exists():
            downloaded_errors_path.unlink()
        client.get_object_to_file(self.config.tos_bucket, results_key, str(downloaded_results_path))
        try:
            client.get_object_to_file(self.config.tos_bucket, errors_key, str(downloaded_errors_path))
        except Exception:
            downloaded_errors_path.write_text("", encoding="utf-8", newline="")
        _merge_jsonl_by_custom_id(results_path, downloaded_results_path, request_order=request_order)
        _merge_jsonl_by_custom_id(errors_path, downloaded_errors_path, request_order=request_order)
        return results_path, errors_path

    def read_result_records(self, path: str | Path) -> dict[str, BatchChatResult]:
        results: dict[str, BatchChatResult] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = str(record.get("custom_id", "")).strip()
                if not custom_id:
                    continue
                content, usage = _extract_batch_chat_content(record)
                results[custom_id] = BatchChatResult(custom_id=custom_id, content=content, usage=usage, raw=record)
        return results

    def read_error_records(self, path: str | Path) -> list[dict[str, Any]]:
        if not Path(path).exists():
            return []
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _build_model_reference(self, volcenginesdkark: Any) -> Any:
        if self.config.custom_model_id:
            return volcenginesdkark.ModelReferenceForCreateBatchInferenceJobInput(
                custom_model_id=self.config.custom_model_id
            )
        return volcenginesdkark.ModelReferenceForCreateBatchInferenceJobInput(
            foundation_model=volcenginesdkark.FoundationModelForCreateBatchInferenceJobInput(
                name=self.config.foundation_model_name,
                model_version=self.config.foundation_model_version,
            )
        )

    def _get_tos_client(self) -> Any:
        if self._tos_client is not None:
            return self._tos_client
        try:
            import tos
        except ImportError as exc:
            raise RuntimeError("Install TOS SDK first: uv pip install tos") from exc
        self._tos_client = tos.TosClientV2(
            self.config.access_key,
            self.config.secret_key,
            self.config.tos_endpoint,
            self.config.tos_region,
        )
        return self._tos_client

    def _get_ark_api(self) -> Any:
        if self._ark_api is not None:
            return self._ark_api
        try:
            import volcenginesdkcore
            import volcenginesdkark
        except ImportError as exc:
            raise RuntimeError('Install Volcengine SDK first: uv pip install "volcengine-python-sdk[ark]"') from exc
        configuration = volcenginesdkcore.Configuration()
        configuration.ak = self.config.access_key
        configuration.sk = self.config.secret_key
        configuration.region = self.config.ark_region
        configuration.host = self.config.ark_host
        configuration.scheme = "https"
        self._ark_api = volcenginesdkark.ARKApi(volcenginesdkcore.ApiClient(configuration))
        return self._ark_api

    def _import_ark_sdk(self) -> Any:
        try:
            import volcenginesdkark
        except ImportError as exc:
            raise RuntimeError('Install Volcengine SDK first: uv pip install "volcengine-python-sdk[ark]"') from exc
        return volcenginesdkark


def _extract_batch_chat_content(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if record.get("error"):
        raise RuntimeError(f"Batch request failed: custom_id={record.get('custom_id')}, error={record['error']}")
    response = record.get("response")
    body = None
    if isinstance(response, dict):
        status_code = int(response.get("status_code", 200))
        if status_code >= 400:
            raise RuntimeError(f"Batch request failed: custom_id={record.get('custom_id')}, response={response}")
        body = response.get("body")
    body = body or record.get("body") or record.get("result") or record.get("output")
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError(f"Batch result body must be an object: custom_id={record.get('custom_id')}")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    return body["choices"][0]["message"]["content"], usage


def _merge_jsonl_by_custom_id(target_path: Path, new_path: Path, *, request_order: list[str] | None = None) -> None:
    existing_records = _read_jsonl_records_by_custom_id(target_path)
    new_records = _read_jsonl_records_by_custom_id(new_path)
    if not new_records:
        if not target_path.exists():
            target_path.write_text("", encoding="utf-8", newline="")
        return

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    for custom_id in existing_records:
        if custom_id not in seen_ids:
            ordered_ids.append(custom_id)
            seen_ids.add(custom_id)
    for custom_id in request_order or []:
        if custom_id in new_records and custom_id not in seen_ids:
            ordered_ids.append(custom_id)
            seen_ids.add(custom_id)
    for custom_id in new_records:
        if custom_id not in seen_ids:
            ordered_ids.append(custom_id)
            seen_ids.add(custom_id)

    merged_records = {**existing_records, **new_records}

    with target_path.open("w", encoding="utf-8", newline="") as handle:
        for custom_id in ordered_ids:
            handle.write(json.dumps(merged_records[custom_id], ensure_ascii=False))
            handle.write("\r\n")


def _read_jsonl_records_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = str(record.get("custom_id", "")).strip()
            if custom_id:
                records[custom_id] = record
    return records


def _sdk_object_to_dict(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_sdk_object_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sdk_object_to_dict(item) for key, item in value.items()}
    payload: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        item = getattr(value, name)
        if callable(item):
            continue
        if item is not None:
            payload[_camel_case(name)] = _sdk_object_to_dict(item)
    return payload


def _camel_case(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_") if part)
