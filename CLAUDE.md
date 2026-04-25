# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

- The actual Python project lives under `demo/`.
- Repository root mainly contains the source PDF/notes; day-to-day development commands should usually be run from `demo/`.
- This repo targets Windows 11 local development for data prep, retrieval service, and smoke evaluation; full SFT/GRPO training is intended for Linux / WSL2 / remote GPU.

## Environment and toolchain

- Python: `>=3.11` in `demo/pyproject.toml`, with README examples using Python 3.12 via `uv`.
- Environment manager: `uv`.
- Local dependencies: `demo/requirements.txt`.
- Remote RL/training stack additions: `demo/requirements-verl.txt` plus external `verl`, `vLLM`, and `flash-attn` on Linux/WSL GPU environments.

## Common commands

Run these from `demo/` unless noted otherwise.

### Setup

```powershell
Set-Location C:\Workspace\AI\Learning\AgenticRAG-RL\demo
uv venv .venv --python 3.12
uv pip install -r .\requirements.txt
```

### Run tests

```powershell
uv run python -m pytest
```

Run a single test file:

```powershell
uv run python -m pytest .\tests\test_retrieval.py
```

Run a single test function:

```powershell
uv run python -m pytest .\tests\test_retrieval.py -k test_hybrid_search_prefers_novel_chunk
```

### Build core data artifacts

Parse the UTF-8 novel text into the canonical corpus format:

```powershell
uv run python .\scripts\parse_text_corpus.py --input .\data\original_data\平凡的世界utf8.txt --output .\data\novel\corpus.jsonl
```

Build the lightweight retrieval index bundle:

```powershell
uv run python .\scripts\build_index.py --corpus .\data\novel\corpus.jsonl --index-dir .\data\novel\indexes
```

Generate seed QA:

```powershell
uv run python .\scripts\gen_seed_qa.py --corpus .\data\novel\corpus.jsonl --output .\data\novel_eval\seeds.jsonl
```

Synthesize multi-hop QA:

```powershell
uv run python .\scripts\domain_multihop_synthesis.py --seeds .\data\novel_eval\seeds.jsonl --corpus .\data\novel\corpus.jsonl --output .\data\novel_eval\qa_pairs.jsonl --target-count 50
```

Build oracle traces:

```powershell
uv run python .\scripts\build_oracle_traces.py --qa .\data\novel_eval\qa_pairs.jsonl --corpus .\data\novel\corpus.jsonl --output .\data\novel_eval\traces_oracle_zh.jsonl --use-zh
```

Convert traces to SFT data:

```powershell
uv run python .\scripts\trace_to_sft.py --input .\data\novel_eval\traces_oracle_zh.jsonl --output-dir .\data\novel_eval\sft --lang zh
uv run python .\scripts\convert_sft_to_llamafactory.py --input-dir .\data\novel_eval\sft --output-dir .\data\novel_eval\sft_zh_llamafactory
```

Prepare GRPO parquet data:

```powershell
uv run python .\scripts\prepare_agentic_grpo_data.py --input .\data\novel_eval\qa_pairs.jsonl --train-output .\data\novel_eval\grpo_agentic_train.parquet --val-output .\data\novel_eval\grpo_agentic_val.parquet
```

### Run retrieval service

```powershell
uv run python .\training\tools\retrieval_server.py --port 8790 --corpus .\data\novel\corpus.jsonl
```

Health check endpoint: `http://127.0.0.1:8790/health`

### Smoke evaluation

```powershell
uv run python .\scripts\eval_agentic.py --data .\data\novel_eval\qa_pairs.jsonl --corpus .\data\novel\corpus.jsonl --max-samples 2
```

Write evaluation output to JSON:

```powershell
uv run python .\scripts\eval_agentic.py --data .\data\novel_eval\qa_pairs.jsonl --corpus .\data\novel\corpus.jsonl --max-samples 20 --output .\results\agentic_eval.json
```

### SFT / GRPO entry points

LLaMA-Factory SFT config:

```powershell
llamafactory-cli train .\training\sft_zh_react.yaml
llamafactory-cli export .\training\export_sft.yaml
```

Remote GRPO launcher scripts usually start from:

```powershell
.\training\run_stage1_v11e.ps1 -PrintOnly
.\training\run_stage2_v14e.ps1 -PrintOnly
.\training\run_stage3_v15e.ps1 -PrintOnly
```

Use `-PrintOnly` first to inspect resolved paths and commands before running on the actual GPU environment.

## High-level architecture

The repository implements a compact end-to-end reproduction of a vertical-domain multi-hop Agentic RAG + RL workflow over a Chinese novel corpus. The important boundary is that most scripts are thin orchestration layers around reusable library code in `demo/src/agentic_rag_rl/`.

### Core data flow

1. Raw novel text is chunked into the canonical corpus format.
2. Corpus chunks feed retrieval, seed QA generation, multi-hop synthesis, oracle trace generation, and evaluation.
3. Multi-hop QA becomes the shared contract for both SFT supervision and GRPO reward ground truth.
4. Oracle traces teach the model the tool-calling protocol during SFT.
5. GRPO data wraps the same questions with `prompt`, `agent_name`, and `reward_model.ground_truth` so `verl` can run tool-using rollouts.
6. Retrieval service and reward logic stay aligned through shared chunk IDs, answer aliases, and hop counts.

### Main library modules

- `demo/src/agentic_rag_rl/chunking.py`: text/PDF corpus chunking into the shared chunk contract.
- `demo/src/agentic_rag_rl/io.py`: JSONL/parquet-ish loading and writing helpers used by scripts.
- `demo/src/agentic_rag_rl/types.py`: the core typed contracts: `Chunk`, `RetrievalResult`, `Hop`, `MultiHopExample`.
- `demo/src/agentic_rag_rl/retrieval.py`: local retrieval stack.
  - `KeywordRetriever`: BM25 over tokenized chunks.
  - `DenseRetriever`: character-ngram TF-IDF retrieval.
  - `HybridRetriever`: keyword + dense fusion via RRF, then light rerank.
- `demo/src/agentic_rag_rl/synthesis.py`: seed QA generation and rule-based multi-hop synthesis over the novel domain.
- `demo/src/agentic_rag_rl/protocols.py`: the tool-calling protocol contract.
  - Defines the Chinese/English system prompts.
  - Defines the allowed tool names.
  - Encodes `<tool_call>...</tool_call>`, `<tool_response>...</tool_response>`, and `<answer>...</answer>` formatting.
- `demo/src/agentic_rag_rl/traces.py`: converts `MultiHopExample` records into oracle trajectories and then into SFT/ShareGPT-style records.
- `demo/src/agentic_rag_rl/grpo_data.py`: converts examples into GRPO rows with `prompt`, `agent_name`, and `reward_model.ground_truth`.
- `demo/src/agentic_rag_rl/agentic.py`: local rule-based agentic rollout used for smoke evaluation. It is not the real training-time model loop; it exists to validate contracts and retrieval behavior locally.
- `demo/src/agentic_rag_rl/evaluation.py`: answer metrics and hop-aware retrieval metrics.
- `demo/src/agentic_rag_rl/judge.py` and `demo/src/agentic_rag_rl/rewards.py`: scoring helpers used by training reward logic.
- `demo/src/agentic_rag_rl/server.py`: FastAPI app factory for retrieval HTTP serving.

### Script layer vs library layer

Most files in `demo/scripts/` are entry points that wire command-line args to library functions. When changing behavior, prefer modifying the underlying library module instead of only patching the script wrapper.

The major script-to-library mapping is:

- `parse_text_corpus.py` / `parse_pdf_corpus.py` -> chunking + IO layer
- `build_index.py` -> indexing/retrieval prep
- `gen_seed_qa.py`, `domain_multihop_synthesis.py`, `clean_synthesis.py`, `split_train_test.py`, `gen_enhanced_aliases.py` -> synthesis pipeline
- `build_oracle_traces.py`, `trace_to_sft.py`, `convert_sft_to_llamafactory.py` -> SFT data pipeline
- `prepare_agentic_grpo_data.py` -> RL data pipeline
- `eval_agentic.py`, `run_pipeline.py`, `run_cloud_eval.py`, `run_llm_judge.py` -> evaluation paths

### Retrieval architecture

Local retrieval is intentionally lightweight and fully in-process:

- BM25 keyword retrieval over tokenized chunk text.
- Dense retrieval implemented as TF-IDF char n-grams, not a neural embedding model.
- Fusion through reciprocal rank fusion (`rrf_fuse`).
- Final light reranking using token overlap.

The `build_index.py` output is mostly a persistence bundle (`manifest.json`, `chunk_ids.json`, `chunk_store.pkl`) for fast reload and contract checking. The runtime retrieval logic still lives in `HybridRetriever`.

### Training protocol contract

The training/eval stack relies on a stable protocol across traces, retrieval service, and reward:

- Assistant tool requests are serialized as `<tool_call>{json}</tool_call>`.
- Retrieved evidence is serialized as `<tool_response>...</tool_response>`.
- Final answers are extracted from `<answer>...</answer>` when present.
- `DEFAULT_AGENT_NAME` is `tool_agent`.
- Reward ground truth stores `target`, `question`, `answer_aliases`, `gold_chunks`, and `hop_count`.

If you change any of those fields or tags, you must verify the whole chain: trace generation, SFT conversion, retrieval tool config, rollout parsing, and reward computation.

### Retrieval service and RL integration

There are two service layers to keep distinct:

- `demo/src/agentic_rag_rl/server.py` defines the FastAPI app contract.
- `demo/training/tools/retrieval_server.py` is the runnable server entry point used by training/evaluation workflows.

`demo/training/config/novel_tool_config.yaml` binds `keyword_search`, `dense_search`, and `hybrid_search` to the single `/search` endpoint on port `8790`. GRPO scripts and any external rollout environment depend on that HTTP contract remaining stable.

### Evaluation model

There are two different evaluation modes in this repo:

- Local smoke evaluation (`scripts/eval_agentic.py`) validates that questions, chunk IDs, retrieval, evidence, and metrics all stay internally consistent.
- Full training/compare workflows evaluate model checkpoints and may add judge-based scoring.

Do not treat the local `run_agentic_episode` heuristic agent as production inference logic; it is a deterministic contract test harness.

## Testing guidance specific to this repo

- `demo/tests/test_pipeline_integration.py` is the best quick contract test because it touches chunks, retrieval, traces, and GRPO row generation together.
- `demo/tests/test_retrieval.py` validates the hybrid retrieval behavior on the smoke dataset.
- `demo/tests/test_protocols.py`, `demo/tests/test_traces.py`, and `demo/tests/test_rewards.py` are the key tests after changing the tool-call / answer-tag protocol.
- The smoke dataset under `demo/data/smoke_novel/` is intended for fast local verification without regenerating the full novel pipeline.

## Important constraints from existing project docs

- Default corpus/task domain is Chinese novel QA over `平凡的世界utf8.txt`.
- Windows local work is expected for preprocessing, retrieval service, and smoke tests.
- Complete SFT/GRPO runs are expected to move to Linux / WSL2 / A100-class environments.
- `training/sft_zh_react.yaml` currently assumes dataset name `novel_agent_zh_react`, template `qwen3_nothink`, and base model path `./models/Qwen3-4B`.
- Reward behavior is versioned through the `AGENTIC_RAG_REWARD_VERSION` environment variable, with stage scripts selecting versions such as `v6a`, `v5a`, and `v9a`.
