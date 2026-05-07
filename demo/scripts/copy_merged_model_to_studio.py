from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_STUDIO_MODELS_DIR = Path.home() / ".unsloth" / "studio" / "models"
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def validate_model_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"模型目录不存在: {path}")

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    safetensors = list(path.glob("*.safetensors"))
    if missing or not safetensors:
        details = []
        if missing:
            details.append("缺少文件: " + ", ".join(missing))
        if not safetensors:
            details.append("缺少 *.safetensors 权重文件")
        raise SystemExit(
            f"不是完整的 HF merged model 目录: {path}\n" + "\n".join(details)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 merged model 复制到本机 Unsloth Studio models 目录。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="要复制的 merged model 目录",
    )
    parser.add_argument(
        "--studio-models-dir",
        type=Path,
        default=DEFAULT_STUDIO_MODELS_DIR,
        help="Unsloth Studio models 目录",
    )
    parser.add_argument(
        "--name",
        help="复制到 Studio 后的目录名；默认使用源目录名",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    name = args.name or source.name
    destination = (args.studio_models_dir / name).resolve()

    validate_model_dir(source)

    if destination.exists():
        raise SystemExit(f"目标目录已存在，避免覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    print(f"source: {source}")
    print(f"destination: {destination}")


if __name__ == "__main__":
    main()
