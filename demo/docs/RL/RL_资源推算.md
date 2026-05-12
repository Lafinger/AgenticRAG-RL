# RL 资源推算

本文只估算当前 verl multi-turn tool-agent GRPO 主线的资源需求。SFT LoRA / QLoRA 资源见 `demo/docs/SFT_LORA/SFT_LORA资源推算.md`。

## 结论

当前 GRPO 训练同时涉及 actor 训练、reference log prob、vLLM rollout、多轮工具响应和 reward 评分。它比 SFT LoRA 更吃显存和调度稳定性，不适合在 RTX 4070 Ti SUPER 16GB 上完整运行。

| 配置 | 适用性 |
| --- | --- |
| 4×A100 80GB | 最稳，推荐完整复现 |
| 4×A100 40GB / 4×L40S 48GB | 可尝试，必要时降低 batch、response length 或 rollout.n |
| 单卡 24GB | 只适合小规模 smoke，不适合当前默认完整 GRPO |
| 单卡 16GB | 只适合数据流程、retrieval server 和极小 smoke |

Windows 本机更适合承担数据准备、检索服务、单测和小样本 smoke；完整 `Qwen3-4B + verl tool-agent GRPO` 建议放到 Linux/WSL/远程多卡 GPU 环境。

## 当前默认参数

核心参数来自 `training/start_grpo_tool_agent.sh`：

```text
data.max_prompt_length = 1024
data.max_response_length = 1024
data.train_batch_size = 32
actor_rollout_ref.rollout.n = 4
actor_rollout_ref.rollout.max_model_len = 4096
actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 7
actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 1024
actor_rollout_ref.rollout.gpu_memory_utilization = 0.35
trainer.total_epochs = 2
trainer.n_gpus_per_node = 4
```

当前数据规模：

```text
grpo_agentic_train.parquet = 2700
grpo_agentic_val.parquet = 300
```

## 为什么需要多卡

每个训练 step 的 rollout 条数：

```text
train_batch_size × rollout.n = 32 × 4 = 128
```

每条 response 最大长度：

```text
max_response_length = 1024 tokens
```

单 step 最大 response token：

```text
128 × 1024 = 131,072 tokens
```

这还没有计算：

1. 初始 system + user prompt。
2. 最多 7 轮 assistant turn。
3. 每轮最多 1024 tokens 的 tool response。
4. vLLM KV cache。
5. actor 训练显存。
6. reference model log prob。
7. Ray / vLLM / FSDP 框架开销。

GRPO 的瓶颈不是单纯模型权重，而是训练、采样和多轮上下文缓存同时存在。

## 训练步数估算

当前 train parquet 为 2700 条：

```text
steps_per_epoch = ceil(2700 / 32) = 85
total_epochs = 2
total_steps ≈ 170
```

最大 rollout 条数：

```text
170 × 128 = 21,760 条 response
```

最大 response token 粗估：

```text
21,760 × 1024 ≈ 22.3M tokens
```

真实训练耗时通常更高，因为还包括工具调用、检索服务延迟、reward judge、验证 rollout 和 checkpoint 保存。

## Retrieval Server 资源

retrieval server 默认运行：

```text
training/tools/retrieval_server.py
```

正式训练应带：

```text
--index-dir ./data/novel/indexes
--embedding-model ./models/bge-m3
--reranker-model ./models/bge-reranker-v2-m3
```

小说域当前数据量不大，retrieval server 可在 CPU 上运行。它的主要压力来自并发 `/search` 请求和 reranker 推理延迟。若和训练机同机运行，需要避免它抢占过多 GPU 显存；CPU 检索更稳但更慢。

当前 tool config 控制了单次响应长度：

```text
top_k = 3
max_text_len = 420
timeout = 30
```

这些参数直接影响 multi-turn 上下文长度、检索延迟和 reward 中的 evidence 文本。

## 降资源参数

小显存 smoke 优先调这些环境变量或 hydra 参数：

| 参数 | 建议 | 影响 |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | 32 -> 8 或 4 | 降低每 step rollout 和训练显存 |
| `ROLLOUT_N` | 4 -> 2 | 降低每问题采样数和 KV cache |
| `PPO_MINI_BATCH_SIZE` | 16 -> 4 或 8 | 降低 PPO 更新显存 |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | 保持 1 | 避免单卡 micro batch 过大 |
| `MAX_ASSISTANT_TURNS` | 7 -> 4 或 5 | 降低多轮上下文，但可能伤害 3-hop |
| `data.max_response_length` | 1024 -> 512 | 降低生成上限，但可能截断答案 |
| `max_tool_response_length` | 1024 -> 512 | 降低证据长度，但可能损失召回 |

示例 smoke：

```bash
ROLLOUT_N=2 TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
OUTPUT_DIR=./training/outputs/grpo_tool_agent_react_v4_smoke \
bash ./training/start_grpo_tool_agent.sh
```

## 本机和外部 GPU 分工

RTX 4070 Ti SUPER 16GB 适合：

```text
GRPO parquet 构造
retrieval server
reward 单测
tool-agent schema 单测
eval_hf_agentic 小样本 smoke
```

不适合：

```text
Qwen3-4B + verl tool-agent GRPO + 多轮 rollout 的完整训练
```

完整训练推荐：

```text
Linux / WSL / 远程 GPU
4×A100 80GB 最稳
至少保证 verl / vLLM / Ray 环境稳定
```

如果只有单卡 24GB，应把目标限定为功能 smoke，而不是可报告的完整 RL 实验。
