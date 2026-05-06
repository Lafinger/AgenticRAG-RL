# Unsloth Studio 训练说明

本文记录如何把本工程的 SFT 数据放到 Unsloth Studio 中进行 `Qwen3-4B + QLoRA 4-bit` 冷启动训练，以及如何理解 Studio 中的 loss、grad norm、learning rate 和验证集。

## 核心结论

- `training\unsloth_sft.yaml` 是本工程 `scripts\train_sft_unsloth.py` 使用的自定义配置，不是 Unsloth Studio 配置文件，Studio 不能直接读取它是正常现象。
- Studio 需要在 UI 中手动填写模型、数据集、LoRA、batch、学习率、epoch 等字段。
- Studio 的训练集和验证集都必须是 SFT `messages` 格式，不能直接使用 `data\novel_eval\test.jsonl`。
- 当前已切分出互不重叠的 Studio 数据：

```text
data\novel_eval\sft_zh_unsloth\train_studio.jsonl
data\novel_eval\sft_zh_unsloth\eval.jsonl
```

Studio 中应这样选择：

```text
Train dataset: E:\AI\AgenticRAG-RL\demo\data\novel_eval\sft_zh_unsloth\train_studio.jsonl
Eval dataset:  E:\AI\AgenticRAG-RL\demo\data\novel_eval\sft_zh_unsloth\eval.jsonl
```

## 数据文件区别

| 文件 | 格式 | 用途 |
| --- | --- | --- |
| `data\novel_eval\sft_zh_unsloth\train.jsonl` | JSONL，每行一条 `messages` 样本 | 全量 SFT 导出数据，项目脚本默认使用 |
| `data\novel_eval\sft_zh_unsloth\train_studio.jsonl` | JSONL，每行一条 `messages` 样本 | Studio 训练集，已移除验证样本 |
| `data\novel_eval\sft_zh_unsloth\eval.jsonl` | JSONL，每行一条 `messages` 样本 | Studio 验证集，与 `train_studio.jsonl` 不重叠 |
| `data\novel_eval\test.jsonl` | QA 评测格式，字段为 `final_question/final_answer/hops` | 训练完成后的最终测试集，不适合作为 Studio Eval dataset |

验证集和测试集的区别：

- 验证集用于训练过程中监控 eval loss、选择 checkpoint 和判断是否过拟合。
- 测试集用于训练完成后的最终效果评估，不应参与训练，也不应反复用来调参。
- 当前 `eval.jsonl` 是 Studio 验证集；`test.jsonl` 是最终 QA 测试集。

## 切分 Studio 训练集和验证集

下面命令会保留全量 `train.jsonl` 不变，从中等间隔切出 200 条作为独立验证集，并生成 `train_studio.jsonl`。两者样本不重叠。

```powershell
Set-Location E:\AI\AgenticRAG-RL\demo

uv run python .\scripts\split_unsloth_studio_data.py `
  --input .\data\novel_eval\sft_zh_unsloth\train.jsonl `
  --train-output .\data\novel_eval\sft_zh_unsloth\train_studio.jsonl `
  --eval-output .\data\novel_eval\sft_zh_unsloth\eval.jsonl `
  --manifest .\data\novel_eval\sft_zh_unsloth\manifest.json `
  --eval-count 200
```

当前切分结果：

```text
train_studio.jsonl: 2800 条
eval.jsonl: 200 条
overlap_count: 0
```

## Studio UI 配置

### Model

| Studio 字段 | 推荐填写 |
| --- | --- |
| Hugging Face Model | `unsloth/Qwen3-4B-Instruct-2507` |
| 如果只显示 Unsloth 模型 | `unsloth/qwen3-4b-instruct-2507` |
| Method | `QLoRA (4-bit)` |
| Hugging Face Token | 只有拉取私有模型或受限模型时才需要 |

### Dataset

| Studio 字段 | 推荐填写 |
| --- | --- |
| Train dataset | `E:\AI\AgenticRAG-RL\demo\data\novel_eval\sft_zh_unsloth\train_studio.jsonl` |
| Eval dataset | `E:\AI\AgenticRAG-RL\demo\data\novel_eval\sft_zh_unsloth\eval.jsonl` |
| Format | `Auto`、`messages` 或 `ShareGPT`，以 Studio 实际可选项为准 |

不要把下面文件填到 Studio Eval dataset：

```text
E:\AI\AgenticRAG-RL\demo\data\novel_eval\test.jsonl
```

它是 QA 测试集，不是 SFT messages 数据。

### Parameters

项目 YAML 到 Studio 字段的推荐映射如下：

| Studio 字段 | 推荐值 | 对应 `unsloth_sft.yaml` |
| --- | --- | --- |
| Context Length | `4096`，显存紧张先用 `2048` | `max_seq_length: 4096` |
| Learning Rate | `0.0001` | `learning_rate: 1.0e-4` |
| Use Epochs | `3` | `num_train_epochs: 3` |
| Batch Size | `2` | `per_device_train_batch_size: 2` |
| Grad Accum | `8` | `gradient_accumulation_steps: 8` |
| Warmup Steps | `4` | `warmup_steps: 4` |
| Save Steps | `45` | `save_steps: 45` |
| Optimizer | `AdamW 8-bit` | Studio 默认可用 |
| LR Scheduler | `Linear` | Studio 默认可用 |
| Weight Decay | `0` 或 Studio 默认 `0.001` | 项目脚本未显式配置 |
| Random Seed | `3407` | `seed: 3407` |
| Packing | 关闭 | `packing: false` |

如果 Studio 只能填 `Max Steps`，按下面公式换算：

```text
steps_per_epoch = ceil(train_count / (batch_size * grad_accum))
max_steps = steps_per_epoch * epochs
```

当前 `train_studio.jsonl` 为 2800 条：

| Batch Size | Grad Accum | Epochs | Max Steps |
| --- | ---: | ---: | ---: |
| 2 | 8 | 3 | 525 |
| 2 | 4 | 3 | 1050 |

如果你看到 Studio 显示约 `1125` steps，通常说明使用了全量 3000 条训练样本且 `grad_accum=4`：

```text
ceil(3000 / (2 * 4)) * 3 = 1125
```

### LoRA Settings

| Studio 字段 | 推荐值 |
| --- | --- |
| Enable LoRA | 开启 |
| Rank | `64` |
| Alpha | `64` |
| Dropout | `0.00` |
| RS-LoRA | 关闭 |
| LoftQ | 关闭 |

Target Modules 选择：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

如果 24GB 显存接近打满，可以优先降低：

```text
Context Length: 4096 -> 2048
Rank: 64 -> 32
Batch Size: 2 -> 1
```

## 图表怎么看

Studio 的 `Training Loss`、`Gradient Norm`、`Learning Rate` 三张图横坐标都是 training step，也就是优化器更新步数，不是 epoch，也不是样本编号。

如果配置为：

```text
batch_size = 2
grad_accum = 8
```

那么：

```text
1 个 training step = 2 * 8 = 16 条样本累计后执行 1 次参数更新
```

### Training Loss

训练初期 loss 小幅上升或抖动是正常的，常见原因包括：

- LoRA adapter 随机初始化。
- learning rate warmup 正在把 LR 从低值升到目标值。
- 多轮 tool-call 样本长度和难度差异大。
- QLoRA 4-bit 初期比普通 LoRA 更容易有轻微抖动。

正常现象：

```text
前几步可能上升或震荡
随后整体下降
下降到较低位置后小幅波动
```

需要警惕：

```text
loss 连续 50-100 step 明显上升
loss 变成 NaN 或 inf
loss 长时间不降且输出格式开始崩坏
```

### Gradient Norm

训练初期 grad norm 先升高再下降是正常的。健康走势通常是：

```text
开头较高
warmup 后快速下降
后续在较小范围内抖动
```

需要警惕：

```text
grad norm 长期 > 10
grad norm 持续暴涨到 > 20
grad norm 和 loss 同时失控
```

### Learning Rate

learning rate 图通常先 warmup 上升，然后按 scheduler 缓慢下降。你看到的 LR 先升到 `1e-4` 附近再下降，是正常的 linear schedule 行为。

### Eval Loss

如果 Studio 显示：

```text
Evaluation not configured
```

说明没有配置验证集或 `eval_steps`。要看 eval loss，需要：

```text
Eval dataset = E:\AI\AgenticRAG-RL\demo\data\novel_eval\sft_zh_unsloth\eval.jsonl
eval_steps = 50 或 100
```

训练 loss 只说明模型对训练样本拟合得更好；eval loss 才能更早发现过拟合。

## 当前收敛判断标准

从前面 Studio 截图看，以下情况属于正常收敛：

```text
Training Loss 从 2.5-3.0 快速降到 0.3 左右
Gradient Norm 从 5-6 降到 0.2-0.5 区间
Learning Rate warmup 后缓慢下降
没有 NaN / inf
```

如果 loss 继续降到 `<0.1`，但没有 eval loss，需要警惕模型只是记住训练轨迹。建议始终配置独立 `eval.jsonl` 并观察 eval loss 是否同步下降。

## 训练输出和后续评测

Studio 默认输出目录通常在：

```text
C:\Users\cjh13\.unsloth\studio\outputs\
```

每次训练会生成类似目录：

```text
C:\Users\cjh13\.unsloth\studio\outputs\unsloth_qwen3-4b-instruct-2507_数字后缀\
```

其中通常包含：

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
checkpoint-*\trainer_state.json
```

训练完成后，仍应使用 held-out QA 测试集做最终评测，而不是只看 Studio training/eval loss。项目内推荐流程是：

```powershell
Set-Location E:\AI\AgenticRAG-RL\demo

uv run --no-sync python .\scripts\eval_hf_model.py `
  --model unsloth/Qwen3-4B-Instruct-2507 `
  --data .\data\novel_eval\test.jsonl `
  --output .\results\sft_compare\base_predictions.jsonl `
  --template qwen3_nothink `
  --max-samples 50 `
  --max-new-tokens 512 `
  --temperature 0

uv run --no-sync python .\scripts\eval_hf_model.py `
  --model unsloth/Qwen3-4B-Instruct-2507 `
  --adapter <Studio输出adapter目录> `
  --data .\data\novel_eval\test.jsonl `
  --output .\results\sft_compare\studio_sft_predictions.jsonl `
  --template qwen3_nothink `
  --max-samples 50 `
  --max-new-tokens 512 `
  --temperature 0

uv run python .\scripts\compare_predictions.py `
  --base .\results\sft_compare\base_predictions.jsonl `
  --sft .\results\sft_compare\studio_sft_predictions.jsonl `
  --output .\results\sft_compare\studio_summary.json
```

重点看：

```text
avg_em
avg_f1
answer_tag_rate
tool_call_rate
valid_tool_call_rate
avg_generation_chars
```

SFT 冷启动的第一目标不是只提高 F1，而是让模型更稳定遵守工具调用和 `<answer>` 输出协议。

## 常见问题

### Studio 不能读取 unsloth_sft.yaml

正常。`unsloth_sft.yaml` 是本工程脚本配置，不是 Studio 配置格式。按本文表格手动填 UI 字段。

### Eval dataset 是否可以用 test.jsonl

不可以。`test.jsonl` 是 QA 评测格式，不是 SFT messages 格式。Studio Eval dataset 使用：

```text
data\novel_eval\sft_zh_unsloth\eval.jsonl
```

### 训练集是否包含验证集

如果使用：

```text
train_studio.jsonl + eval.jsonl
```

则不包含。当前校验结果：

```text
overlap_count = 0
```

如果使用旧方式：

```text
train.jsonl + eval.jsonl
```

则会包含，因为 `eval.jsonl` 是从全量 `train.jsonl` 中切出来的。因此 Studio 训练时不要再使用全量 `train.jsonl`。

### 训练 loss 很低是否代表模型好了

不一定。训练 loss 只代表训练集拟合程度。最终效果要看：

```text
held-out test.jsonl 上的 EM/F1
<answer> 标签命中率
<tool_call> 合法率
LLM Judge 可选评分
```

### 显存接近满载怎么办

优先降低：

```text
Context Length
LoRA Rank
Batch Size
```

不要先降低 `grad_accum`。降低 `grad_accum` 会减少有效 batch，可能让训练更抖。
