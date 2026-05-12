# RL 训练样本长度计算

RL 的长度预算和 SFT 不同。SFT 统计的是一条固定 messages 样本被 tokenizer 渲染后的 token 数；GRPO 需要同时考虑 prompt、每轮 assistant response、tool response、多条 rollout 采样和 vLLM KV cache。

## 当前默认长度参数

当前严格 GRPO 主线来自 `training/start_grpo_tool_agent.sh`：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `data.max_prompt_length` | 1024 | 初始 system + user prompt 上限 |
| `data.max_response_length` | 1024 | 单条 rollout response 上限 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | 1024 | 单次工具返回给模型的文本上限 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 7 | 最多 assistant 工具/答案轮数 |
| `actor_rollout_ref.rollout.n` | 4 | 每个问题采样 response 条数 |
| `actor_rollout_ref.rollout.max_model_len` | 4096 | vLLM 单序列上下文上限 |
| `data.train_batch_size` | 32 | 每个训练 step 的问题数 |

`max_model_len=4096` 是实际上下文硬边界之一。即使多轮理论预算更大，单条序列也不能无限累积 prompt、assistant 输出和 tool response。

## 单 step rollout 规模

每个训练 step 的 rollout 条数：

```text
rollout 条数 = train_batch_size × rollout.n
            = 32 × 4
            = 128
```

按 `max_response_length=1024` 粗算，单 step 最大 response token 数：

```text
128 × 1024 = 131,072 response tokens
```

这还没有包含初始 prompt、多轮工具返回、reference log prob、actor 训练反向传播和 vLLM KV cache。

## 多轮工具上下文预算

单条样本理论上最多有：

```text
max_assistant_turns = 7
max_tool_response_length = 1024
```

如果每轮都调用工具，工具响应理论上最多：

```text
7 × 1024 = 7168 tool response tokens
```

但当前 `rollout.max_model_len=4096` 会限制真实上下文，因此训练时更关键的是：

1. 单次 tool response 不要过长。
2. `top_k` 和 `max_text_len` 不要让证据文本挤占全部上下文。
3. 模型应在证据足够后及时输出 `<answer>`，避免无效多轮搜索。

当前 `training/config/novel_tool_config.yaml` 中每个工具默认：

```text
top_k = 3
max_text_len = 420
```

这能把单次检索返回控制在较小范围内，降低超过 `max_model_len` 的风险。

## 训练步数估算

设多跳 QA 总数为 `T`，验证集比例为 `val_ratio=0.1`：

```text
N_train = T × 0.9
steps_per_epoch = ceil(N_train / train_batch_size)
```

当前 parquet 为：

```text
N_train = 2700
train_batch_size = 32
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
21,760 × 1024 ≈ 22.3M response tokens
```

真实 token 和耗时会受提前停止、工具轮数、检索返回长度、失败重试和验证频率影响。

## 调参影响

| 调小参数 | 直接影响 | 代价 |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | 降低每 step 问题数和显存压力 | step 数增加 |
| `ROLLOUT_N` | 降低每问题采样条数和 KV cache 压力 | GRPO 组内比较信号变弱 |
| `max_response_length` | 降低 response 上限和生成耗时 | 过小会截断答案或工具调用 |
| `max_tool_response_length` | 降低多轮上下文压力 | 过小会丢证据 |
| `max_assistant_turns` | 降低无效搜索上限 | 过小会伤害 3-hop 样本 |
| `top_k/max_text_len` | 降低工具响应长度 | 过小会伤害 hop recall |

## 推荐工作流

1. 用正式 retrieval server 跑 `/search` smoke，确认返回文本长度和 chunk id 正常。
2. 用小样本 smoke 跑 `ROLLOUT_N=2 TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1`。
3. 检查日志中是否频繁出现 truncation、max turns 或 empty tool response。
4. 用 5 条 `eval_hf_agentic.py` smoke 验证 GRPO checkpoint 协议没有退化。
5. 再扩大到正式 `TRAIN_BATCH_SIZE=32 ROLLOUT_N=4 TOTAL_EPOCHS=2`。

如果单卡或小显存环境无法承受，优先降 `TRAIN_BATCH_SIZE` 和 `ROLLOUT_N`，不要先改工具协议。
