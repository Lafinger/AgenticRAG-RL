# 架构说明

## 数据链路

默认链路：

`平凡的世界utf8.txt -> novel/corpus.jsonl -> index bundle -> seeds -> multihop qa -> oracle traces -> sft sharegpt -> grpo parquet`

默认产物：

- `data/novel/corpus.jsonl`
- `data/novel/indexes/`
- `data/novel_eval/seeds.jsonl`
- `data/novel_eval/qa_pairs.jsonl`
- `data/novel_eval/traces_oracle_zh.jsonl`
- `data/novel_eval/sft/`
- `data/novel_eval/sft_zh_llamafactory/`
- `data/novel_eval/grpo_agentic_train.parquet`
- `data/novel_eval/grpo_agentic_val.parquet`

## 训练链路

- Stage 1 / `v11e`
  - 目标：grounding-first
  - reward：`v6a`
- Stage 2 / `v14e`
  - 目标：correctness-heavy
  - reward：`v5a`
- Stage 3 / `v15e`
  - 目标：对照实验
  - reward：`v9a`

## 本机与远端职责

- 本机 Windows：小说文本解析、数据处理、SFT 数据转换、CPU retrieval server、smoke evaluation
- 远端 Linux / WSL：LLaMA-Factory SFT、verl GRPO、vLLM Judge / rollout
