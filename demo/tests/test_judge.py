from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from agentic_rag_rl.judge import judge_agentic_answer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_run_llm_judge_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_llm_judge_for_test", ROOT / "scripts" / "run_llm_judge.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_judge_agentic_answer_scores_grounded_predictions() -> None:
    result = judge_agentic_answer(
        question="双水村为什么叫双水村？",
        prediction="因为有东拉河和哭咽河。",
        gold="因为有东拉河和哭咽河。",
        aliases=["东拉河和哭咽河", "因东拉河和哭咽河得名"],
        evidence_text="村边有东拉河和哭咽河，因此这个村子取名叫双水村。",
        gold_chunks=["corpus_chunkids_000002"],
        retrieved_chunk_ids=["corpus_chunkids_000002"],
    )

    assert result["correctness"] == 1.0
    assert result["faithfulness"] > 0.5
    assert result["context_precision"] == 1.0


def test_run_llm_judge_loads_eval_json_results(tmp_path: Path) -> None:
    module = load_run_llm_judge_module()
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"summary": {"count": 1}, "results": [{"question": "问题"}]}, ensure_ascii=False), encoding="utf-8")

    assert module.load_eval_results(path) == [{"question": "问题"}]


def test_run_llm_judge_loads_eval_jsonl_results(tmp_path: Path) -> None:
    module = load_run_llm_judge_module()
    path = tmp_path / "sft_agentic_eval.jsonl"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"question": "问题1"}, ensure_ascii=False))
        handle.write("\r\n")
        handle.write(json.dumps({"question": "问题2"}, ensure_ascii=False))
        handle.write("\r\n")

    assert module.load_eval_results(path) == [{"question": "问题1"}, {"question": "问题2"}]
