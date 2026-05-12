# SFT_LORA 资源推算

本文只估算当前 SFT LoRA / QLoRA 主线的资源需求。GRPO、multi-turn rollout、vLLM KV cache 和 retrieval server 资源见 `demo/docs/RL/RL_资源推算.md`。

## 结论

当前 `training/unsloth_sft_v4.yaml` 使用 `Qwen3-4B + load_in_4bit: true + LoRA rank 64 + max_seq_length 2048 + batch 2 x grad_accum 8`。完整 SFT 仍建议使用 24GB 级 GPU；16GB 本机可以做更小 batch 的 smoke，但不建议把它作为稳定完整训练环境。

| 配置 | 适用性 |
| --- | --- |
| 1×A10 24GB / 1×L4 24GB / 1×RTX 3090 24GB | 可跑当前 4B QLoRA SFT 主线 |
| RTX 4070 Ti SUPER 16GB | 适合 smoke；完整训练建议降 batch、rank 或长度 |
| 4×A100 / L40S | 可跑 SFT，并作为后续 GRPO 环境 |

## 当前默认参数

配置来自 `training/unsloth_sft_v4.yaml`：

```yaml
model_name_or_path: unsloth/Qwen3-4B-Instruct-2507
data_path: ./data/novel_eval/sft_zh_unsloth_react_v4/train_cli.jsonl
eval_data_path: ./data/novel_eval/sft_zh_unsloth_react_v4/eval.jsonl
max_seq_length: 2048
load_in_4bit: true
lora_rank: 64
lora_alpha: 64
lora_dropout: 0
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
```

当前 v4 全量导出为 `19561` 条 records，命令行训练集 `19361` 条，eval 集 `200` 条。最近一次长度统计：

```text
samples: 19361
max_seq_length: 2048
min: 507 p50: 1292 avg: 1136.7 p90: 1773 p95: 1799 p99: 1840 max: 1885
> 2048: 0 (0.0%)
```

## 训练步数怎么算

SFT 有效 batch：

```text
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × GPU 数
```

单卡当前默认：

```text
effective_batch = 2 × 8 × 1 = 16
```

训练 step 数：

```text
sft_steps = ceil(N_sft / effective_batch) × epochs
```

当前主线：

```text
N_sft = 19361
steps_per_epoch = ceil(19361 / 16) = 1211
epochs = 3
total_steps ≈ 3633
```

这和当前有效 checkpoint `checkpoint-3633` 对齐。

## 显存怎么算

QLoRA / LoRA SFT 不能按全参训练估算。基座模型冻结，optimizer state 主要作用于 LoRA adapter 参数。

粗略公式：

```text
SFT 总显存 ≈ 量化基座权重 + LoRA 参数训练态显存 + activation + 临时 buffer / 框架开销
```

### 1. 基座模型权重

如果 bf16 加载 `Qwen3-4B`：

```text
4B × 2 bytes = 8GB
```

当前配置 `load_in_4bit: true`，基座权重显存通常会下降到约 `2GB - 3GB`，但量化 metadata、反量化 buffer、attention kernel 和框架开销仍然存在。

### 2. LoRA 可训练参数

对一个线性层 `W ∈ R(out × in)`，LoRA 训练两个低秩矩阵：

```text
A ∈ R(r × in)
B ∈ R(out × r)
LoRA params = r × (in + out)
```

当前 rank 为 `64`，目标模块为：

```text
q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
```

对 4B 级模型，LoRA 参数通常是几十 M 到一两百 M 级，远小于 4B 全参。

### 3. LoRA 参数训练态显存

按保守口径：

```text
LoRA bf16 weights = 2 bytes / trainable param
LoRA bf16 gradients = 2 bytes / trainable param
LoRA fp32 master weights = 4 bytes / trainable param
LoRA fp32 Adam m/v = 8 bytes / trainable param
合计 = 16 bytes / trainable param
```

如果 LoRA 参数量约为 `50M`：

```text
50M × 16 bytes = 0.8GB
```

如果约为 `100M`：

```text
100M × 16 bytes = 1.6GB
```

### 4. Activation 显存

SFT 最大不确定项通常是 activation：

```text
activation ∝ per_device_train_batch_size × sequence_length × hidden_size × layer_count
```

当前主线：

```text
per_device_train_batch_size = 2
max_seq_length = 2048
gradient_accumulation_steps = 8
```

`gradient_accumulation_steps` 不会把单步显存乘 8；它表示连续 8 次小 batch 后再更新参数。单步显存主要看 `batch=2` 和 `seq=2048`。开启 gradient checkpointing 可以显著降低 activation，但训练会变慢。

### 5. 汇总估算

按当前 `Qwen3-4B + 4bit base + LoRA rank 64 + batch=2 + seq=2048` 粗略估算：

| 项 | 粗略显存 |
| --- | --- |
| 4bit 基座模型权重 | 2GB - 3GB |
| LoRA 参数 / 梯度 / optimizer | 0.8GB - 2GB |
| activation | 6GB - 14GB，取决于 checkpointing / attention 实现 |
| 临时 buffer / CUDA / dataloader / 框架开销 | 2GB - 5GB |
| 合计 | 约 11GB - 24GB |

因此 24GB GPU 更稳；16GB 可能能跑 smoke，但容易受具体 CUDA kernel、驱动、batch 和后台进程影响。

## 16GB 本机建议

如果要在 RTX 4070 Ti SUPER 16GB 上做 SFT smoke，优先缩小这些参数：

```yaml
per_device_train_batch_size: 1
max_seq_length: 1024
lora_rank: 16 或 32
load_in_4bit: true
```

并确保关闭不必要的 GPU 进程。若 smoke 目标只是验证协议边界和训练入口，不需要使用全量 `19361` 条训练样本。

## 本机和外部 GPU 分工

Windows 本机适合：

```text
文本解析
corpus / index
Seed QA 生成
多跳 QA 合成
Oracle trace
SFT 数据构造
SFT 小样本 smoke
Agent loop smoke eval
```

外部 GPU 更适合：

```text
完整 Qwen3-4B QLoRA SFT
LoRA adapter 合并
后续 verl tool-agent GRPO
```
