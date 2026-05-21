from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
REPO_DIR = DEMO_DIR.parent

APP_NAME = "agentic-rag-rl-grpo"
SECRET_NAME = "agentic-rag-rl-secrets"

DATA_VOLUME = "agentic-rag-rl-data"
MODELS_VOLUME = "agentic-rag-rl-models"
OUTPUTS_VOLUME = "agentic-rag-rl-outputs"
REMOTE_OUTPUT_ROOT = "/grpo_tool_agent_react_v4_h100x2"

REQUIRED_CHECKPOINT_FILES = (
    "data.pt",
    "actor/fsdp_config.json",
    "actor/model_world_size_2_rank_0.pt",
    "actor/model_world_size_2_rank_1.pt",
    "actor/optim_world_size_2_rank_0.pt",
    "actor/optim_world_size_2_rank_1.pt",
    "actor/extra_state_world_size_2_rank_0.pt",
    "actor/extra_state_world_size_2_rank_1.pt",
)

ASSET_UPLOADS = (
    (DATA_VOLUME, "data/novel_eval/grpo_agentic_train.parquet", "/novel_eval/grpo_agentic_train.parquet"),
    (DATA_VOLUME, "data/novel_eval/grpo_agentic_val.parquet", "/novel_eval/grpo_agentic_val.parquet"),
    (DATA_VOLUME, "data/novel/indexes", "/novel/indexes"),
    (MODELS_VOLUME, "models/bge-m3", "/bge-m3"),
    (MODELS_VOLUME, "models/bge-reranker-v2-m3", "/bge-reranker-v2-m3"),
    (
        MODELS_VOLUME,
        "models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged",
        "/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged",
    ),
)


def resolve_modal_exe(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate.resolve()
        command = shutil.which(explicit)
        if command:
            return Path(command).resolve()
        raise FileNotFoundError(f"Modal CLI not found: {explicit}")

    candidates = (
        DEMO_DIR / ".venv" / "Scripts" / "modal.exe",
        DEMO_DIR / ".venv" / "bin" / "modal",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    command = shutil.which("modal")
    if command:
        return Path(command).resolve()
    raise FileNotFoundError("Modal CLI not found. Run `python -m pip install modal` and `modal setup` first.")


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(command)


def run_command(command: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {format_command(command)}")
    result = subprocess.run(
        command,
        cwd=REPO_DIR,
        env=build_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )
    if capture:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result


def normalize_remote_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def remote_join(*parts: str) -> str:
    joined = "/".join(part.strip("/") for part in parts if part)
    return normalize_remote_path(joined)


def parse_step(text: str) -> str:
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        raise ValueError(f"Could not parse checkpoint step from: {text!r}")
    return match.group(1)


def resolve_project_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = DEMO_DIR / candidate
    return candidate.resolve()


def find_latest_checkpoint_dir() -> Path:
    outputs_dir = DEMO_DIR / "training" / "outputs"
    candidates: list[tuple[int, Path]] = []
    for path in outputs_dir.glob("modal_resume_global_step_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"modal_resume_global_step_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No local checkpoint found under {outputs_dir}")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def resolve_checkpoint_dir(explicit: str | None) -> tuple[Path, str]:
    if explicit:
        checkpoint_root = Path(explicit).expanduser()
        if not checkpoint_root.is_absolute():
            checkpoint_root = REPO_DIR / checkpoint_root
        checkpoint_root = checkpoint_root.resolve()
        if checkpoint_root.name.startswith("global_step_"):
            checkpoint_root = checkpoint_root.parent
    else:
        checkpoint_root = find_latest_checkpoint_dir()

    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"Missing {latest_file}")
    step = parse_step(latest_file.read_text(encoding="utf-8", errors="replace"))
    checkpoint_dir = checkpoint_root / f"global_step_{step}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing {checkpoint_dir}")
    return checkpoint_root, step


def validate_checkpoint(checkpoint_root: Path, step: str) -> None:
    failures: list[str] = []
    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_step = parse_step(latest_file.read_text(encoding="utf-8", errors="replace"))
    if latest_step != step:
        failures.append(f"{latest_file} contains {latest_step}, expected {step}")

    checkpoint_dir = checkpoint_root / f"global_step_{step}"
    for relative in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint_dir / Path(*relative.split("/"))
        if not path.exists():
            failures.append(f"Missing {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"Zero-byte required file: {path}")

    zero_files = [path for path in checkpoint_root.rglob("*") if path.is_file() and path.stat().st_size == 0]
    failures.extend(f"Zero-byte file: {path}" for path in zero_files)

    if failures:
        raise RuntimeError("Checkpoint validation failed:\n" + "\n".join(failures))

    files = [path for path in checkpoint_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print("Local checkpoint validation passed.")
    print(f"CHECKPOINT_ROOT={checkpoint_root}")
    print(f"LATEST_STEP={step}")
    print(f"FILE_COUNT={len(files)}")
    print(f"TOTAL_GIB={total_bytes / (1024 ** 3):.2f}")


def create_volume(modal_exe: Path, name: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"Would ensure Modal Volume: {name}")
        return
    result = run_command([str(modal_exe), "volume", "create", name], capture=True, check=False)
    if result.returncode != 0:
        print(f"Warning: `modal volume create {name}` returned {result.returncode}; continuing so existing volumes can be used.")


def upload_path(modal_exe: Path, volume: str, local_path: Path, remote_path: str, *, dry_run: bool) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")
    if dry_run:
        print(f"Would upload {local_path} -> {volume}:{remote_path}")
        return
    run_command([str(modal_exe), "volume", "put", "--force", volume, str(local_path), normalize_remote_path(remote_path)])


def show_account(modal_exe: Path, *, dry_run: bool) -> None:
    run_command([str(modal_exe), "profile", "current"])
    run_command([str(modal_exe), "token", "info"])
    if dry_run:
        print("Dry run: account display only; no remote resources will be changed.")


def create_placeholder_secret(modal_exe: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"Would create placeholder Secret: {SECRET_NAME}")
        return
    run_command([str(modal_exe), "secret", "create", "--force", SECRET_NAME, "PLACEHOLDER=1"])


def upload_assets(modal_exe: Path, *, dry_run: bool) -> None:
    for volume in (DATA_VOLUME, MODELS_VOLUME, OUTPUTS_VOLUME):
        create_volume(modal_exe, volume, dry_run=dry_run)
    for volume, local_relative, remote_path in ASSET_UPLOADS:
        upload_path(modal_exe, volume, resolve_project_path(local_relative), remote_path, dry_run=dry_run)


def upload_checkpoint(modal_exe: Path, checkpoint_root: Path, step: str, *, dry_run: bool) -> None:
    create_volume(modal_exe, OUTPUTS_VOLUME, dry_run=dry_run)
    upload_path(
        modal_exe,
        OUTPUTS_VOLUME,
        checkpoint_root / "latest_checkpointed_iteration.txt",
        remote_join(REMOTE_OUTPUT_ROOT, "latest_checkpointed_iteration.txt"),
        dry_run=dry_run,
    )
    upload_path(
        modal_exe,
        OUTPUTS_VOLUME,
        checkpoint_root / f"global_step_{step}",
        REMOTE_OUTPUT_ROOT + "/",
        dry_run=dry_run,
    )


def verify_remote_checkpoint(modal_exe: Path, step: str, *, dry_run: bool) -> None:
    paths = (
        REMOTE_OUTPUT_ROOT,
        remote_join(REMOTE_OUTPUT_ROOT, f"global_step_{step}"),
        remote_join(REMOTE_OUTPUT_ROOT, f"global_step_{step}", "actor"),
    )
    for path in paths:
        if dry_run:
            print(f"Would verify {OUTPUTS_VOLUME}:{path}")
            continue
        run_command([str(modal_exe), "volume", "ls", OUTPUTS_VOLUME, path])


def run_check_inputs(modal_exe: Path, *, dry_run: bool) -> None:
    target = str(SCRIPT_DIR / "modal_grpo_tool_agent.py") + "::check_inputs"
    if dry_run:
        print(f"Would run check_inputs: {target}")
        return
    run_command([str(modal_exe), "run", target])


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def get_field(item: dict[str, Any], *names: str) -> Any:
    normalized = {normalize_key(str(key)): value for key, value in item.items()}
    for name in names:
        key = normalize_key(name)
        if key in normalized:
            return normalized[key]
    return None


def active_training_apps(modal_exe: Path) -> list[dict[str, Any]]:
    result = run_command([str(modal_exe), "app", "list", "--json"], capture=True)
    apps = json.loads(result.stdout.strip() or "[]")
    if not isinstance(apps, list):
        raise ValueError(f"Unexpected app list JSON: {apps!r}")

    active: list[dict[str, Any]] = []
    for app in apps:
        if not isinstance(app, dict):
            continue
        description = str(get_field(app, "Description", "description", "Name", "name") or "")
        state = str(get_field(app, "State", "state") or "").lower()
        tasks = str(get_field(app, "Tasks", "tasks") or "")
        try:
            task_count = int(tasks)
        except ValueError:
            task_count = 0
        relevant = APP_NAME in description or description.startswith("agentic-rag")
        if state:
            is_active = state != "stopped"
        else:
            is_active = task_count > 0
        if relevant and is_active:
            active.append(app)
    return active


def ensure_no_active_training(modal_exe: Path, *, dry_run: bool) -> None:
    if dry_run:
        print("Would check `modal app list --json` before starting training.")
        return
    active = active_training_apps(modal_exe)
    if active:
        formatted = json.dumps(active, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Refusing to start training because an active training app exists:\n{formatted}")


def validate_train_extra_args(values: list[str]) -> None:
    invalid = [value for value in values if "=" not in value]
    if invalid:
        raise ValueError(f"--train-extra-arg values must look like KEY=VALUE: {invalid}")


def start_training(modal_exe: Path, train_extra_args: list[str], *, dry_run: bool) -> None:
    validate_train_extra_args(train_extra_args)
    ensure_no_active_training(modal_exe, dry_run=dry_run)
    target = str(SCRIPT_DIR / "modal_grpo_tool_agent.py") + "::train"
    command = [
        str(modal_exe),
        "run",
        "--detach",
        target,
        "--",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.max_critic_ckpt_to_keep=2",
        *train_extra_args,
    ]
    if dry_run:
        print(f"Would start training: {format_command(command)}")
        return
    result = run_command(command, capture=True)
    combined = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"\b(ap-[A-Za-z0-9]+)\b", combined)
    if match:
        app_id = match.group(1)
        print(f"APP_ID={app_id}")
        print(f"Follow logs: {modal_exe} app logs {app_id} -f --timestamps")
    else:
        print("Could not parse App ID from Modal output. Use `modal app list` to find the new app.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a new Modal account/workspace for checkpoint resume after account switch.",
    )
    parser.add_argument("--modal-exe", help="Path to modal.exe or modal command name.")
    parser.add_argument("--checkpoint-dir", help="Local modal_resume_global_step_<STEP> directory.")
    parser.add_argument("--create-placeholder-secret", action="store_true")
    parser.add_argument("--skip-assets", action="store_true", help="Do not upload data, indexes, or base models.")
    parser.add_argument("--skip-checkpoint-upload", action="store_true", help="Do not upload the local checkpoint.")
    parser.add_argument("--start-training", action="store_true", help="Start the detached 2xH100 training run.")
    parser.add_argument(
        "--train-extra-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override appended after the default checkpoint retention overrides.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating or uploading resources.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modal_exe = resolve_modal_exe(args.modal_exe)
    checkpoint_root, step = resolve_checkpoint_dir(args.checkpoint_dir)
    validate_checkpoint(checkpoint_root, step)

    show_account(modal_exe, dry_run=args.dry_run)

    if args.create_placeholder_secret:
        create_placeholder_secret(modal_exe, dry_run=args.dry_run)
    else:
        print(f"Secret is not modified. Ensure `{SECRET_NAME}` exists in the active Modal workspace.")

    if args.skip_assets:
        print("Skipping asset upload.")
    else:
        upload_assets(modal_exe, dry_run=args.dry_run)

    if args.skip_checkpoint_upload:
        print("Skipping checkpoint upload.")
    else:
        upload_checkpoint(modal_exe, checkpoint_root, step, dry_run=args.dry_run)

    verify_remote_checkpoint(modal_exe, step, dry_run=args.dry_run)
    run_check_inputs(modal_exe, dry_run=args.dry_run)

    if args.start_training:
        start_training(modal_exe, args.train_extra_arg, dry_run=args.dry_run)
    else:
        print("Training was not started. Add --start-training when you are ready to resume.")


if __name__ == "__main__":
    main()
