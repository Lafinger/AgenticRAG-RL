from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import modal


APP_NAME = "agentic-rag-rl-grpo"
SECRET_NAME = "agentic-rag-rl-secrets"

LOCAL_DEMO_DIR = Path(__file__).resolve().parents[1]
LOCAL_REPO_DIR = LOCAL_DEMO_DIR.parent

REMOTE_ROOT = PurePosixPath("/workspace/AgenticRAG-RL")
REMOTE_PROJECT_DIR = REMOTE_ROOT / "demo"
REMOTE_VERL_DIR = REMOTE_ROOT / "example" / "verl"

DATA_PATH = PurePosixPath("/vol/data")
MODELS_PATH = PurePosixPath("/vol/models")
OUTPUTS_PATH = PurePosixPath("/vol/outputs")

TRAIN_FILE = DATA_PATH / "novel_eval" / "grpo_agentic_train.parquet"
VAL_FILE = DATA_PATH / "novel_eval" / "grpo_agentic_val.parquet"
INDEX_DIR = DATA_PATH / "novel" / "indexes"
BGE_MODEL = MODELS_PATH / "bge-m3"
RERANKER_MODEL = MODELS_PATH / "bge-reranker-v2-m3"
MERGED_MODEL = MODELS_PATH / "Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged"
OUTPUT_DIR = OUTPUTS_PATH / "grpo_tool_agent_react_v4"
FORMAL_OUTPUT_DIR = OUTPUTS_PATH / "grpo_tool_agent_react_v4_h100x2"
LOG_DIR = OUTPUTS_PATH / "logs"
LOG_FILE = LOG_DIR / "grpo_tool_agent_react_v4.log"

RETRIEVAL_HOST = "127.0.0.1"
RETRIEVAL_PORT = 8790
RETRIEVAL_URL = f"http://{RETRIEVAL_HOST}:{RETRIEVAL_PORT}"

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("agentic-rag-rl-data", create_if_missing=True)
models_volume = modal.Volume.from_name("agentic-rag-rl-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("agentic-rag-rl-outputs", create_if_missing=True)


def _container_path(path: PurePosixPath) -> Path:
    return Path(path.as_posix())


runtime_packages = [
    "fastapi>=0.115",
    "faiss-cpu>=1.13",
    "httpx>=0.28",
    "jieba>=0.42",
    "networkx>=3.4",
    "numpy<2.0.0",
    "pandas>=2.2",
    "pyarrow>=17.0",
    "pydantic>=2.9",
    "pypdf>=5.0",
    "pyyaml>=6.0",
    "rank-bm25>=0.2.2",
    "requests>=2.32",
    "scikit-learn>=1.5",
    "sentence-transformers==2.7.0",
    "swanlab>=0.7.12",
    "uvicorn>=0.30",
]

image = (
    modal.Image.from_registry("verlai/verl:vllm011.latest")
    .entrypoint([])
    .apt_install("curl", "git", "procps")
    .uv_pip_install(*runtime_packages)
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(MODELS_PATH / ".cache" / "huggingface"),
            "VLLM_USE_V1": "1",
        }
    )
    .add_local_dir(
        LOCAL_DEMO_DIR / "src",
        str(REMOTE_PROJECT_DIR / "src"),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_DEMO_DIR / "training",
        str(REMOTE_PROJECT_DIR / "training"),
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            "outputs/**",
            "swanlab/**",
        ],
    )
    .add_local_dir(
        LOCAL_REPO_DIR / "example" / "verl",
        str(REMOTE_VERL_DIR),
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".git/**",
            ".github/**",
            "docs/**",
            "docker/**",
            "examples/**",
            "tests/**",
        ],
    )
)

volumes = {
    str(DATA_PATH): data_volume,
    str(MODELS_PATH): models_volume,
    str(OUTPUTS_PATH): outputs_volume,
}


def _pythonpath() -> str:
    parts = [
        str(REMOTE_PROJECT_DIR),
        str(REMOTE_PROJECT_DIR / "src"),
        str(REMOTE_VERL_DIR),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["RETRIEVAL_SERVER_URL"] = RETRIEVAL_URL
    env.setdefault("AGENTIC_RAG_REWARD_VERSION", "v9a")
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _require_paths() -> dict[str, bool]:
    checks = {
        "train_file": _container_path(TRAIN_FILE).is_file(),
        "val_file": _container_path(VAL_FILE).is_file(),
        "index_dir": _container_path(INDEX_DIR).is_dir(),
        "bge_model": _container_path(BGE_MODEL).is_dir(),
        "reranker_model": _container_path(RERANKER_MODEL).is_dir(),
        "merged_model": _container_path(MERGED_MODEL).is_dir(),
        "project_src": _container_path(REMOTE_PROJECT_DIR / "src" / "agentic_rag_rl").is_dir(),
        "project_training": _container_path(REMOTE_PROJECT_DIR / "training").is_dir(),
        "verl_package": _container_path(REMOTE_VERL_DIR / "verl").is_dir(),
    }
    missing = [name for name, ok in checks.items() if not ok]
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if missing:
        raise FileNotFoundError(f"Missing Modal inputs: {', '.join(missing)}")
    return checks


def _wait_for_retrieval_server(process: subprocess.Popen[bytes], timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{RETRIEVAL_URL}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"retrieval server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                body = response.read().decode("utf-8")
            if response.status == 200 and "ok" in body:
                print(f"retrieval server ready at {health_url}")
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"retrieval server did not become ready within {timeout_seconds}s")


def _start_retrieval_server() -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        str(REMOTE_PROJECT_DIR / "training" / "tools" / "retrieval_server.py"),
        "--index-dir",
        str(INDEX_DIR),
        "--embedding-model",
        str(BGE_MODEL),
        "--reranker-model",
        str(RERANKER_MODEL),
        "--host",
        RETRIEVAL_HOST,
        "--port",
        str(RETRIEVAL_PORT),
    ]
    print("starting retrieval server:")
    print(" ".join(cmd))
    process = subprocess.Popen(cmd, cwd=str(REMOTE_PROJECT_DIR), env=_base_env())
    _wait_for_retrieval_server(process)
    return process


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _run(cmd: list[str], *, cwd: PurePosixPath, env: dict[str, str]) -> None:
    log_dir = _container_path(LOG_DIR)
    log_file = _container_path(LOG_FILE)
    log_dir.mkdir(parents=True, exist_ok=True)
    print("running command:")
    print(" ".join(cmd))
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n\n=== modal verl command ===\n")
        handle.write(" ".join(cmd))
        handle.write("\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    outputs_volume.commit()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _verl_command(
    *,
    n_gpus: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
    total_epochs: int,
    save_freq: int,
    test_freq: int,
    output_dir: PurePosixPath,
    extra_args: tuple[str, ...],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={TRAIN_FILE}",
        f"data.val_files={VAL_FILE}",
        f"data.train_batch_size={train_batch_size}",
        "data.max_prompt_length=1024",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={MERGED_MODEL}",
        "actor_rollout_ref.hybrid_engine=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=5e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.05",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={REMOTE_PROJECT_DIR / 'training' / 'config' / 'novel_tool_config.yaml'}",
        "actor_rollout_ref.rollout.multi_turn.format=hermes",
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"reward.custom_reward_function.path={REMOTE_PROJECT_DIR / 'training' / 'reward_agentic_rag.py'}",
        "reward.custom_reward_function.name=compute_score",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=[console]",
        "trainer.project_name=agentic-rag-rl",
        "trainer.experiment_name=qwen3_4b_grpo_tool_agent_react_v4_modal",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_epochs={total_epochs}",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={output_dir}",
    ]
    cmd.extend(extra_args)
    return cmd


def _train(
    *,
    n_gpus: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
    total_epochs: int,
    save_freq: int,
    test_freq: int,
    output_dir: PurePosixPath,
    extra_args: tuple[str, ...],
) -> None:
    data_volume.reload()
    models_volume.reload()
    outputs_volume.reload()
    _require_paths()
    retrieval_process: subprocess.Popen[bytes] | None = None
    try:
        retrieval_process = _start_retrieval_server()
        cmd = _verl_command(
            n_gpus=n_gpus,
            train_batch_size=train_batch_size,
            ppo_mini_batch_size=ppo_mini_batch_size,
            rollout_n=rollout_n,
            total_epochs=total_epochs,
            save_freq=save_freq,
            test_freq=test_freq,
            output_dir=output_dir,
            extra_args=extra_args,
        )
        _run(cmd, cwd=REMOTE_VERL_DIR, env=_base_env())
    finally:
        _stop_process(retrieval_process)
        outputs_volume.commit()


@app.function(image=image, volumes=volumes, timeout=10 * MINUTES)
def check_inputs() -> dict[str, bool]:
    data_volume.reload()
    models_volume.reload()
    return _require_paths()


@app.function(
    image=image,
    gpu="H100:1",
    volumes=volumes,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=24 * HOURS,
    single_use_containers=True,
)
def train_smoke(*extra_args: str) -> None:
    smoke_args = (
        "trainer.total_training_steps=1",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.20",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        *extra_args,
    )
    _train(
        n_gpus=1,
        train_batch_size=4,
        ppo_mini_batch_size=2,
        rollout_n=2,
        total_epochs=1,
        save_freq=1,
        test_freq=1,
        output_dir=OUTPUT_DIR,
        extra_args=smoke_args,
    )


@app.function(
    image=image,
    gpu="H100:2",
    volumes=volumes,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=24 * HOURS,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=60.0),
    single_use_containers=True,
)
def train(*extra_args: str) -> None:
    formal_args = (
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.25",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        *extra_args,
    )
    _train(
        n_gpus=2,
        train_batch_size=16,
        ppo_mini_batch_size=8,
        rollout_n=4,
        total_epochs=2,
        save_freq=5,
        test_freq=3,
        output_dir=FORMAL_OUTPUT_DIR,
        extra_args=formal_args,
    )
