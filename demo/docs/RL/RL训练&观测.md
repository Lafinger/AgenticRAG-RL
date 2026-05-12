# RL 训练与观测

本文记录当前严格 RL 主线：`SFT v4 merged model -> verl main_ppo -> multi-turn tool_agent -> retrieval server -> reward_agentic_rag.compute_score`。旧的 `scripts/train_grpo_unsloth.py` 只保留为历史 TRL/Unsloth 单轮实验入口，不代表当前主线。

## 前置条件

正式 GRPO 建议在 Linux/WSL/远程 GPU 环境运行。Windows 本机负责数据准备、检索服务、单测和小样本 smoke。

必须准备：

| 项 | 当前路径或参数 |
| --- | --- |
| 初始模型 | `models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged` |
| train parquet | `data/novel_eval/grpo_agentic_train.parquet` |
| val parquet | `data/novel_eval/grpo_agentic_val.parquet` |
| 工具配置 | `training/config/novel_tool_config.yaml` |
| reward 函数 | `training/reward_agentic_rag.py` |
| verl 目录 | 通过环境变量 `VERL_DIR` 指定 |
| 检索服务 | 默认 `http://127.0.0.1:8790` |

## 启动 retrieval server

正式 GRPO 前必须启动检索服务，并使用已构建索引：

```powershell
uv run python `
  ./training/tools/retrieval_server.py `
  --corpus ./data/novel/corpus.jsonl `
  --index-dir ./data/novel/indexes `
  --embedding-model ./models/bge-m3 `
  --reranker-model ./models/bge-reranker-v2-m3 `
  --host 127.0.0.1 `
  --port 8790
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8790/health -UseBasicParsing
```

搜索 smoke：

```powershell
Invoke-WebRequest http://127.0.0.1:8790/search `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"query":"卖饼老者自称是哪个集市的人？","tool":"keyword_search","top_k":3}' `
  -UseBasicParsing
```

如果不传 `--index-dir`，server 会退化到 corpus-only 检索，只能用于功能 smoke，不代表正式训练检索质量。

## 正式训练入口

入口脚本：

```text
training/start_grpo_tool_agent.sh
```

Linux/WSL/远程 GPU 环境运行：

```bash
export VERL_DIR=/path/to/verl
export PROJECT_DIR=/path/to/AgenticRAG-RL/demo
export RETRIEVAL_SERVER_URL=http://127.0.0.1:8790
bash ./training/start_grpo_tool_agent.sh
```

脚本启动前会检查 retrieval server `/health`，并清理残留 Ray、vLLM 和 verl 训练进程。默认输出：

| 项 | 路径 |
| --- | --- |
| checkpoint | `training/outputs/grpo_tool_agent_react_v4` |
| 日志 | `logs/grpo_tool_agent_react_v4.log` |
| reward version | `AGENTIC_RAG_REWARD_VERSION=v9a` |

## 核心 verl 参数

当前脚本对齐 example 的 verl `main_ppo` / `tool_agent` 路线：

```text
algorithm.adv_estimator=grpo
data.return_raw_chat=True
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.multi_turn.tool_config_path=training/config/novel_tool_config.yaml
reward.custom_reward_function.path=training/reward_agentic_rag.py
reward.custom_reward_function.name=compute_score
```

默认训练规模参数：

| 参数 | 当前值 |
| --- | ---: |
| `data.train_batch_size` | 32 |
| `actor_rollout_ref.rollout.n` | 4 |
| `data.max_prompt_length` | 1024 |
| `data.max_response_length` | 1024 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 7 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | 1024 |
| `actor_rollout_ref.rollout.max_model_len` | 4096 |
| `trainer.total_epochs` | 2 |
| `trainer.n_gpus_per_node` | 4 |

## 小样本 smoke

完整训练前可以用环境变量缩小 rollout：

```bash
ROLLOUT_N=2 TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
OUTPUT_DIR=./training/outputs/grpo_tool_agent_react_v4_smoke \
LOG=./logs/grpo_tool_agent_react_v4_smoke.log \
bash ./training/start_grpo_tool_agent.sh
```

smoke 只验证 multi-turn tool_agent、retrieval server、reward 入口和 checkpoint 写入是否闭环。它不能替代正式 GRPO 结果。

## 观测方式

当前严格 verl 主线通过 console logger 和 `tee` 写入日志：

```text
logs/grpo_tool_agent_react_v4.log
```

优先观察：

1. retrieval server 是否持续响应 `/health` 和 `/search`。
2. verl 日志中是否出现 tool_agent 多轮调用。
3. rollout 是否频繁触发 max turns 或 response truncation。
4. reward 是否存在长期为 0 或 NaN。
5. checkpoint 是否按 `SAVE_FREQ` 写入 `OUTPUT_DIR`。

SwanLab 和 `metrics.jsonl` 当前主要服务于 Unsloth SFT 入口；不要把历史 `scripts/train_grpo_unsloth.py` 的 JSONL 指标当作当前严格 verl GRPO 主线结论。

## 常见失败定位

| 现象 | 优先检查 |
| --- | --- |
| 启动即退出 | `VERL_DIR`、retrieval server `/health`、`CUDA_VISIBLE_DEVICES` |
| 工具不可用 | `training/config/novel_tool_config.yaml` 的 class path 和 `PYTHONPATH` |
| 搜索全空 | retrieval server 是否带 `--index-dir`，模型路径是否正确 |
| 协议退化 | reward 是否给坏格式正反馈，tool response 是否过长 |
| 显存不足 | 降 `TRAIN_BATCH_SIZE`、`ROLLOUT_N`、`max_response_length` 或换多卡环境 |
