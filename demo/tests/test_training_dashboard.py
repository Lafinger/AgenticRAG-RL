from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.watch_unsloth_training import TrainingReader, create_app


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\r\n")


def test_training_reader_prefers_trace_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    write_jsonl(
        output_dir / "trace_metrics.jsonl",
        [
            {"step": 1, "loss": 1.2, "grad_norm": 0.5, "learning_rate": 1e-4, "eval_loss": None},
            {"step": 2, "loss": 0.9, "grad_norm": 3.0, "learning_rate": 9e-5, "eval_loss": 1.1},
        ],
    )
    write_jsonl(
        output_dir / "step_sample_trace.jsonl",
        [
            {
                "step": 2,
                "loss": 0.9,
                "grad_norm": 3.0,
                "micro_batches": [
                    {
                        "micro_batch_index": 7,
                        "items": [
                            {
                                "sample_id": "sft_000007",
                                "source_line": 7,
                                "question": "问题",
                                "answer": "答案",
                                "token_length": 128,
                                "truncated": False,
                            }
                        ],
                    }
                ],
            }
        ],
    )

    reader = TrainingReader(output_dir)

    assert [record["step"] for record in reader.metrics()] == [1, 2]
    assert reader.spikes(top=1)[0]["step"] == 2
    step = reader.step(2)
    assert step["found"] is True
    assert step["samples"][0]["source_line"] == 7


def test_training_reader_falls_back_to_metrics_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    write_jsonl(
        output_dir / "metrics.jsonl",
        [
            {"event": "train_begin", "step": 0},
            {"event": "log", "step": 1, "loss": 1.0, "learning_rate": 1e-4},
        ],
    )

    reader = TrainingReader(output_dir)

    assert reader.metrics() == [{"step": 1, "loss": 1.0, "learning_rate": 0.0001, "event": "log"}]
    assert reader.state()["last_loss"] == 1.0


def test_dashboard_api_routes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    write_jsonl(output_dir / "metrics.jsonl", [{"event": "log", "step": 3, "loss": 0.8, "grad_norm": 1.4}])

    client = TestClient(create_app(output_dir))

    assert client.get("/").status_code == 200
    assert client.get("/api/state").json()["last_loss"] == 0.8
    assert client.get("/api/metrics").json()["records"][0]["step"] == 3
    assert client.get("/api/spikes").json()["records"][0]["grad_norm"] == 1.4
