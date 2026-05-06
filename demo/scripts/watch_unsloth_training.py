from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch Unsloth training metrics and trace files in a local dashboard.")
    parser.add_argument("--output-dir", required=True, help="Training output directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--refresh-seconds", type=float, default=3.0)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return []
    if limit is not None and limit > 0:
        return records[-limit:]
    return records


def numeric_step(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def compact_metric(record: dict[str, Any], index: int) -> dict[str, Any]:
    step = numeric_step(record.get("step")) or numeric_step(record.get("global_step")) or index
    metric = {
        "step": step,
        "epoch": record.get("epoch"),
        "loss": record.get("loss"),
        "eval_loss": record.get("eval_loss"),
        "grad_norm": record.get("grad_norm"),
        "learning_rate": record.get("learning_rate"),
        "reward": record.get("reward"),
        "event": record.get("event"),
    }
    return {key: value for key, value in metric.items() if value is not None}


def latest_checkpoint_state(output_dir: Path) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = CHECKPOINT_PATTERN.search(path.name)
        if not match:
            continue
        state_path = path / "trainer_state.json"
        if state_path.exists():
            candidates.append((int(match.group(1)), state_path))
    direct_state = output_dir / "trainer_state.json"
    if direct_state.exists():
        candidates.append((10**18, direct_state))
    if not candidates:
        return None
    _, path = sorted(candidates, key=lambda item: item[0])[-1]
    return read_json(path)


class TrainingReader:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.trace_metrics_path = output_dir / "trace_metrics.jsonl"
        self.metrics_path = output_dir / "metrics.jsonl"
        self.state_path = output_dir / "trace_training_state.json"
        self.trace_path = output_dir / "step_sample_trace.jsonl"
        self.reward_metrics_path = output_dir / "reward_metrics.jsonl"

    def files(self) -> dict[str, Any]:
        paths = {
            "output_dir": self.output_dir,
            "trace_metrics": self.trace_metrics_path,
            "metrics": self.metrics_path,
            "trace_state": self.state_path,
            "step_sample_trace": self.trace_path,
            "reward_metrics": self.reward_metrics_path,
        }
        payload: dict[str, Any] = {}
        for name, path in paths.items():
            item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
            if path.exists():
                stat = path.stat()
                item["size"] = stat.st_size
                item["mtime"] = stat.st_mtime
            payload[name] = item
        return payload

    def metrics(self, limit: int | None = None) -> list[dict[str, Any]]:
        if self.trace_metrics_path.exists():
            records = read_jsonl(self.trace_metrics_path, limit=limit)
            return [compact_metric(record, index + 1) for index, record in enumerate(records)]
        if self.metrics_path.exists():
            records = [record for record in read_jsonl(self.metrics_path, limit=limit) if record.get("event") == "log"]
            return [compact_metric(record, index + 1) for index, record in enumerate(records)]
        state = latest_checkpoint_state(self.output_dir)
        if not state:
            return []
        history = state.get("log_history", [])
        if not isinstance(history, list):
            return []
        records = [record for record in history if isinstance(record, dict)]
        if limit is not None and limit > 0:
            records = records[-limit:]
        return [compact_metric(record, index + 1) for index, record in enumerate(records)]

    def state(self) -> dict[str, Any]:
        trace_state = read_json(self.state_path)
        metrics = self.metrics(limit=1)
        latest = metrics[-1] if metrics else {}
        files = self.files()
        if trace_state:
            state = dict(trace_state)
        else:
            trainer_state = latest_checkpoint_state(self.output_dir) or {}
            state = {
                "global_step": trainer_state.get("global_step", latest.get("step")),
                "total_steps": trainer_state.get("max_steps"),
                "epoch": trainer_state.get("epoch", latest.get("epoch")),
                "last_loss": latest.get("loss"),
                "last_grad_norm": latest.get("grad_norm"),
                "last_learning_rate": latest.get("learning_rate"),
                "last_eval_loss": latest.get("eval_loss"),
            }
        state["output_dir"] = str(self.output_dir)
        state["files"] = files
        return state

    def spikes(self, top: int = 20) -> list[dict[str, Any]]:
        records = [record for record in self.metrics() if finite_number(record.get("grad_norm")) is not None]
        records.sort(key=lambda item: float(item.get("grad_norm") or 0.0), reverse=True)
        return records[:top]

    def step(self, step: int) -> dict[str, Any]:
        if not self.trace_path.exists():
            return {"step": step, "found": False, "reason": "step_sample_trace.jsonl not found."}
        for record in read_jsonl(self.trace_path):
            if numeric_step(record.get("step")) != step:
                continue
            items: list[dict[str, Any]] = []
            for micro_batch in record.get("micro_batches", []):
                if not isinstance(micro_batch, dict):
                    continue
                for item in micro_batch.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    payload = dict(item)
                    payload["micro_batch_index"] = micro_batch.get("micro_batch_index")
                    payload["epoch_progress"] = micro_batch.get("epoch_progress")
                    items.append(payload)
            return {
                "found": True,
                "step": step,
                "epoch": record.get("epoch"),
                "loss": record.get("loss"),
                "eval_loss": record.get("eval_loss"),
                "grad_norm": record.get("grad_norm"),
                "learning_rate": record.get("learning_rate"),
                "sample_count": len(items),
                "samples": items,
            }
        return {"step": step, "found": False, "reason": f"Step {step} not found."}


def html_page(refresh_seconds: float) -> str:
    refresh_ms = max(int(refresh_seconds * 1000), 1000)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unsloth Training Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f7f7f4; color: #202124; }}
    header {{ padding: 16px 24px; border-bottom: 1px solid #d9d9d2; background: #ffffff; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 700; }}
    main {{ padding: 20px 24px 28px; display: grid; gap: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .panel {{ background: #ffffff; border: 1px solid #deded8; border-radius: 8px; padding: 14px; }}
    .metric-label {{ color: #62625c; font-size: 12px; }}
    .metric-value {{ font-size: 22px; margin-top: 4px; font-weight: 700; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .chart {{ height: 230px; }}
    .chart-title {{ font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
    svg {{ width: 100%; height: 190px; overflow: visible; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ecece7; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: #62625c; font-weight: 700; }}
    tr.clickable {{ cursor: pointer; }}
    tr.clickable:hover {{ background: #f3f6f8; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f5f5f1; border: 1px solid #e3e3dc; border-radius: 6px; padding: 10px; max-height: 420px; overflow: auto; }}
    .muted {{ color: #74746d; }}
    .status {{ font-size: 13px; color: #62625c; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Unsloth Training Dashboard</h1>
    <div class="status" id="status">正在读取训练输出...</div>
  </header>
  <main>
    <section class="grid" id="cards"></section>
    <section class="charts">
      <div class="panel chart"><div class="chart-title">Training Loss</div><svg id="loss"></svg></div>
      <div class="panel chart"><div class="chart-title">Eval Loss</div><svg id="eval_loss"></svg></div>
      <div class="panel chart"><div class="chart-title">Gradient Norm</div><svg id="grad_norm"></svg></div>
      <div class="panel chart"><div class="chart-title">Learning Rate</div><svg id="learning_rate"></svg></div>
    </section>
    <section class="panel">
      <h2 style="font-size:16px;margin:0 0 10px;">Top Gradient Norm Steps</h2>
      <table>
        <thead><tr><th>Step</th><th>Epoch</th><th>Loss</th><th>Grad Norm</th><th>LR</th></tr></thead>
        <tbody id="spikes"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2 style="font-size:16px;margin:0 0 10px;">Step Samples</h2>
      <div class="muted" id="stepHint">点击上方 step 查看样本追溯。</div>
      <pre id="stepDetail"></pre>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_ms};
    function fmt(value) {{
      if (value === null || value === undefined) return "-";
      if (typeof value === "number") {{
        if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
        return Number(value.toFixed(6)).toString();
      }}
      return value;
    }}
    function card(label, value) {{
      return `<div class="panel"><div class="metric-label">${{label}}</div><div class="metric-value">${{fmt(value)}}</div></div>`;
    }}
    function pointsFor(records, key) {{
      return records.filter(row => Number.isFinite(Number(row[key])) && Number.isFinite(Number(row.step)));
    }}
    function drawChart(id, records, key, color) {{
      const svg = document.getElementById(id);
      const points = pointsFor(records, key);
      svg.innerHTML = "";
      if (!points.length) {{
        svg.innerHTML = '<text x="12" y="28" fill="#74746d">暂无数据</text>';
        return;
      }}
      const width = 640, height = 180, pad = 28;
      const xs = points.map(p => Number(p.step));
      const ys = points.map(p => Number(p[key]));
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      const ymin = Math.min(...ys), ymax = Math.max(...ys);
      const sx = value => pad + (xmax === xmin ? 0 : (value - xmin) / (xmax - xmin)) * (width - pad * 2);
      const sy = value => height - pad - (ymax === ymin ? 0.5 : (value - ymin) / (ymax - ymin)) * (height - pad * 2);
      const d = points.map((p, index) => `${{index ? "L" : "M"}}${{sx(Number(p.step)).toFixed(2)}},${{sy(Number(p[key])).toFixed(2)}}`).join(" ");
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="#d8d8d0" />
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="#d8d8d0" />
        <path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2.5" />
        <text x="${{pad}}" y="16" fill="#74746d">${{fmt(ymax)}}</text>
        <text x="${{pad}}" y="${{height - 5}}" fill="#74746d">${{fmt(ymin)}}</text>
        <text x="${{width - pad - 60}}" y="${{height - 5}}" fill="#74746d">step ${{xmax}}</text>`;
    }}
    async function loadStep(step) {{
      const response = await fetch(`/api/step/${{step}}`);
      const payload = await response.json();
      document.getElementById("stepHint").textContent = payload.found ? `step ${{step}} 样本数：${{payload.sample_count}}` : payload.reason;
      document.getElementById("stepDetail").textContent = JSON.stringify(payload, null, 2);
    }}
    async function refresh() {{
      const [state, metrics, spikes] = await Promise.all([
        fetch("/api/state").then(r => r.json()),
        fetch("/api/metrics?limit=2000").then(r => r.json()),
        fetch("/api/spikes?top=20").then(r => r.json())
      ]);
      document.getElementById("status").textContent = state.output_dir || "";
      document.getElementById("cards").innerHTML = [
        card("Step", `${{fmt(state.global_step)}} / ${{fmt(state.total_steps)}}`),
        card("Epoch", state.epoch),
        card("Loss", state.last_loss),
        card("Eval Loss", state.last_eval_loss),
        card("Grad Norm", state.last_grad_norm),
        card("Learning Rate", state.last_learning_rate)
      ].join("");
      const records = metrics.records || [];
      drawChart("loss", records, "loss", "#1f6feb");
      drawChart("eval_loss", records, "eval_loss", "#9a6700");
      drawChart("grad_norm", records, "grad_norm", "#cf222e");
      drawChart("learning_rate", records, "learning_rate", "#116329");
      document.getElementById("spikes").innerHTML = (spikes.records || []).map(row => `
        <tr class="clickable" onclick="loadStep(${{row.step}})">
          <td>${{fmt(row.step)}}</td><td>${{fmt(row.epoch)}}</td><td>${{fmt(row.loss)}}</td>
          <td>${{fmt(row.grad_norm)}}</td><td>${{fmt(row.learning_rate)}}</td>
        </tr>`).join("");
    }}
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def create_app(output_dir: str | Path, refresh_seconds: float = 3.0) -> FastAPI:
    reader = TrainingReader(project_path(output_dir))
    app = FastAPI(title="Unsloth Training Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return html_page(refresh_seconds)

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        return reader.state()

    @app.get("/api/metrics")
    def api_metrics(limit: int = 2000) -> dict[str, Any]:
        return {"records": reader.metrics(limit=limit)}

    @app.get("/api/spikes")
    def api_spikes(top: int = 20) -> dict[str, Any]:
        return {"records": reader.spikes(top=top)}

    @app.get("/api/step/{step}")
    def api_step(step: int) -> dict[str, Any]:
        return reader.step(step)

    @app.get("/api/files")
    def api_files() -> dict[str, Any]:
        return reader.files()

    return app


def main() -> None:
    args = parse_args()
    import uvicorn

    app = create_app(args.output_dir, refresh_seconds=args.refresh_seconds)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
