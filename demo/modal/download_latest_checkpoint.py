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

DEFAULT_OUTPUTS_VOLUME = "agentic-rag-rl-outputs"
DEFAULT_REMOTE_ROOT = "/grpo_tool_agent_react_v4_h100x2"
DEFAULT_LATEST_FILE = "latest_checkpointed_iteration.txt"
DEFAULT_LOCAL_BASE_DIR = DEMO_DIR / "training" / "outputs"

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


def read_latest_step(modal_exe: Path, outputs_volume: str, remote_root: str, latest_file: str) -> str:
    remote_latest = remote_join(remote_root, latest_file)
    result = run_command(
        [str(modal_exe), "volume", "get", "--force", outputs_volume, remote_latest, "-"],
        capture=True,
    )
    return parse_step(result.stdout)


def list_volume_entries(modal_exe: Path, outputs_volume: str, remote_path: str) -> list[dict[str, Any]]:
    result = run_command(
        [str(modal_exe), "volume", "ls", "--json", outputs_volume, normalize_remote_path(remote_path)],
        capture=True,
    )
    text = result.stdout.strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected volume ls JSON: {data!r}")
    return data


def iter_volume_files(modal_exe: Path, outputs_volume: str, remote_path: str) -> list[str]:
    files: list[str] = []
    for entry in list_volume_entries(modal_exe, outputs_volume, remote_path):
        filename = str(entry.get("Filename") or entry.get("filename") or "").replace("\\", "/")
        entry_type = str(entry.get("Type") or entry.get("type") or "").lower()
        if not filename:
            continue
        if entry_type == "dir":
            files.extend(iter_volume_files(modal_exe, outputs_volume, filename))
        elif entry_type == "file":
            files.append(filename)
    return files


def should_skip_file(remote_file: str, *, skip_hf_weights: bool) -> bool:
    normalized = remote_file.replace("\\", "/")
    return skip_hf_weights and "/huggingface/model-" in normalized and normalized.endswith(".safetensors")


def local_path_for_remote_file(remote_file: str, remote_checkpoint_root: str, local_checkpoint_dir: Path) -> Path:
    normalized_file = remote_file.replace("\\", "/").lstrip("/")
    normalized_root = normalize_remote_path(remote_checkpoint_root).lstrip("/")
    if normalized_file == normalized_root:
        relative = Path(normalized_file).name
    elif normalized_file.startswith(normalized_root + "/"):
        relative = normalized_file[len(normalized_root) + 1 :]
    else:
        relative = Path(normalized_file).name
    return local_checkpoint_dir / Path(*relative.split("/"))


def download_file(
    modal_exe: Path,
    outputs_volume: str,
    remote_file: str,
    local_file: Path,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    if local_file.exists() and local_file.stat().st_size > 0 and not force:
        print(f"Already present: {local_file}")
        return
    if dry_run:
        print(f"Would download {remote_file} -> {local_file}")
        return
    local_file.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [str(modal_exe), "volume", "get", "--force", outputs_volume, normalize_remote_path(remote_file), str(local_file)],
    )


def validate_checkpoint(local_root: Path, step: str) -> None:
    latest_file = local_root / DEFAULT_LATEST_FILE
    checkpoint_dir = local_root / f"global_step_{step}"
    failures: list[str] = []

    if not latest_file.exists():
        failures.append(f"Missing {latest_file}")
    else:
        latest_step = parse_step(latest_file.read_text(encoding="utf-8", errors="replace"))
        if latest_step != step:
            failures.append(f"{latest_file} contains {latest_step}, expected {step}")

    for relative in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint_dir / Path(*relative.split("/"))
        if not path.exists():
            failures.append(f"Missing {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"Zero-byte required file: {path}")

    zero_files = [path for path in local_root.rglob("*") if path.is_file() and path.stat().st_size == 0]
    failures.extend(f"Zero-byte file: {path}" for path in zero_files)

    if failures:
        raise RuntimeError("Checkpoint validation failed:\n" + "\n".join(failures))

    files = [path for path in local_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print("Checkpoint validation passed.")
    print(f"ROOT={local_root}")
    print(f"LATEST_STEP={step}")
    print(f"FILE_COUNT={len(files)}")
    print(f"TOTAL_GIB={total_bytes / (1024 ** 3):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest Modal GRPO checkpoint before switching accounts.",
    )
    parser.add_argument("--modal-exe", help="Path to modal.exe or modal command name.")
    parser.add_argument("--outputs-volume", default=DEFAULT_OUTPUTS_VOLUME)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--latest-file", default=DEFAULT_LATEST_FILE)
    parser.add_argument("--local-base-dir", default=str(DEFAULT_LOCAL_BASE_DIR))
    parser.add_argument("--skip-hf-weights", action="store_true", help="Skip actor/huggingface/model-*.safetensors.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when a non-empty local file exists.")
    parser.add_argument("--dry-run", action="store_true", help="List the files that would be downloaded without writing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modal_exe = resolve_modal_exe(args.modal_exe)
    remote_root = normalize_remote_path(args.remote_root)
    step = read_latest_step(modal_exe, args.outputs_volume, remote_root, args.latest_file)
    local_root = Path(args.local_base_dir).expanduser()
    if not local_root.is_absolute():
        local_root = REPO_DIR / local_root
    local_root = local_root.resolve() / f"modal_resume_global_step_{step}"
    local_checkpoint_dir = local_root / f"global_step_{step}"
    remote_checkpoint_root = remote_join(remote_root, f"global_step_{step}")

    print(f"Latest checkpoint step: {step}")
    print(f"Remote checkpoint: {remote_checkpoint_root}")
    print(f"Local checkpoint root: {local_root}")

    remote_files = iter_volume_files(modal_exe, args.outputs_volume, remote_checkpoint_root)
    selected_files = [path for path in remote_files if not should_skip_file(path, skip_hf_weights=args.skip_hf_weights)]
    skipped_count = len(remote_files) - len(selected_files)
    print(f"Remote files: {len(remote_files)}")
    print(f"Selected files: {len(selected_files)}")
    if skipped_count:
        print(f"Skipped HF weight files: {skipped_count}")

    if args.dry_run:
        print(f"Would download {args.latest_file} -> {local_root / args.latest_file}")
    else:
        local_root.mkdir(parents=True, exist_ok=True)
        download_file(
            modal_exe,
            args.outputs_volume,
            remote_join(remote_root, args.latest_file),
            local_root / args.latest_file,
            dry_run=False,
            force=args.force,
        )

    for remote_file in selected_files:
        local_file = local_path_for_remote_file(remote_file, remote_checkpoint_root, local_checkpoint_dir)
        download_file(
            modal_exe,
            args.outputs_volume,
            remote_file,
            local_file,
            dry_run=args.dry_run,
            force=args.force,
        )

    if args.dry_run:
        print("Dry run finished. No checkpoint files were written.")
        return

    validate_checkpoint(local_root, step)


if __name__ == "__main__":
    main()
