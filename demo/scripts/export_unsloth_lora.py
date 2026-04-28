from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Unsloth LoRA adapter as a merged HuggingFace model.")
    parser.add_argument("--config", default=str(ROOT / "training" / "unsloth_sft.yaml"))
    parser.add_argument("--adapter-path")
    parser.add_argument("--export-dir")
    parser.add_argument("--save-method", default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"])
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def project_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter_path = args.adapter_path or config.get("output_dir")
    export_dir = args.export_dir or config.get("merged_export_dir")
    if not adapter_path or not export_dir:
        raise SystemExit("请通过配置或命令行提供 adapter-path 和 export-dir。")

    try:
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise SystemExit("当前环境未安装 Unsloth。请先安装 unsloth 后再导出 merged model。") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=project_path(adapter_path),
        max_seq_length=int(config.get("max_seq_length", 2048)),
        load_in_4bit=False,
    )
    model.save_pretrained_merged(project_path(export_dir), tokenizer, save_method=args.save_method)
    print(f"merged_model_dir={project_path(export_dir)}")


if __name__ == "__main__":
    main()
