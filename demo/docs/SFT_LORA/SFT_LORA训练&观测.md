# 训练与观测

本文记录 demo 当前保留的主线训练流程：canonical ReAct SFT v4 -> Unsloth LoRA -> merged model -> verl multi-turn tool-agent GRPO -> Agent loop 测评。所有训练入口都通过命令行执行。

训练、长度统计和 Agent loop 测评都使用项目 canonical Qwen3 ReAct renderer。该 renderer 统一注入 `# Tools`、包装 `<tool_response>`，并强制主线工具轮是短 `<think>` + JSON `<tool_call>`；不直接使用 tokenizer 原生 `tools=` 模板。

## 主线数据准备

v4 数据的目标是修复 v3 主测评中 50/50 首轮直接生成 `</tool_call>` 的协议边界崩溃。工具轮仍来自 Oracle plan：

```text
<think>要回答最终问题，先查：卖饼老者自称是哪个集市的人？</think>
<tool_call>
{"name":"keyword_search","arguments":{"query":"卖饼老者自称是哪个集市的人？"}}
</tool_call>
```

最终答案仍使用：

```text
<answer>最终答案</answer>
```

重新生成数据时按下面顺序执行：

```powershell
uv run python `
  ./scripts/build_oracle_traces.py `
  --qa ./data/novel_eval/qa_pairs.jsonl `
  --corpus ./data/novel/corpus.jsonl `
  --output ./data/novel_eval/traces_oracle_zh.jsonl `
  --use-zh

uv run python `
  ./scripts/trace_to_sft.py `
  --input ./data/novel_eval/traces_oracle_zh.jsonl `
  --output-dir ./data/novel_eval/sft_react_v4 `
  --lang zh

uv run python `
  ./scripts/convert_sft_to_unsloth.py `
  --input-dir ./data/novel_eval/sft_react_v4 `
  --output-dir ./data/novel_eval/sft_zh_unsloth_react_v4

uv run python `
  ./scripts/split_sft_train_eval.py `
  --input ./data/novel_eval/sft_zh_unsloth_react_v4/train.jsonl `
  --train-output ./data/novel_eval/sft_zh_unsloth_react_v4/train_cli.jsonl `
  --eval-output ./data/novel_eval/sft_zh_unsloth_react_v4/eval.jsonl `
  --manifest ./data/novel_eval/sft_zh_unsloth_react_v4/manifest.json `
  --eval-count 200
```

当前 v4 主线数据规模：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data\novel_eval\sft_zh_unsloth_react_v4\train.jsonl` | 19561 | 全量 SFT 导出 |
| `data\novel_eval\sft_zh_unsloth_react_v4\train_cli.jsonl` | 19361 | 命令行训练输入 |
| `data\novel_eval\sft_zh_unsloth_react_v4\eval.jsonl` | 200 | 训练过程 eval loss 验证集 |

样本类型：

| sample_type | 全量数量 | 目的 |
| --- | ---: | --- |
| `full_trace` | 3000 | 完整 Oracle Agent 轨迹 |
| `first_action_only` | 6000 | 2 倍加权首轮 `<think> + <tool_call>` 起始状态 |
| `next_action_only` | 4561 | 带自然历史，只监督下一轮工具动作 |
| `final_answer_only` | 6000 | 2 倍强化完整自然历史下的最终 `<answer>` |

`train_cli.jsonl` 和 `eval.jsonl` 由同一个全量导出切分得到，二者不重叠。`data\novel_eval\test.jsonl` 是最终 held-out Agent loop 测评集，不进入 SFT 训练。

GRPO parquet 也从同一份 `qa_pairs.jsonl` 生成，但它不是 SFT messages 数据，而是 example 风格的 verl tool-agent rollout 输入：

```powershell
uv run python `
  ./scripts/prepare_agentic_grpo_data.py `
  --input ./data/novel_eval/qa_pairs.jsonl `
  --train-output ./data/novel_eval/grpo_agentic_train.parquet `
  --val-output ./data/novel_eval/grpo_agentic_val.parquet `
  --val-ratio 0.1 `
  --seed 42
```

当前 GRPO parquet 数量：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data\novel_eval\grpo_agentic_train.parquet` | 2700 | verl GRPO train rollout |
| `data\novel_eval\grpo_agentic_val.parquet` | 300 | verl GRPO validation rollout |

每行字段固定为 `data_source/prompt/ability/reward_model/extra_info/metadata`。工具 schema 不写入 parquet，由 `training\config\novel_tool_config.yaml` 在 verl multi-turn rollout 中加载。

## 长度统计

训练前必须用同一 tokenizer 和同一 canonical Qwen3 ReAct renderer 统计长度：

```powershell
uv run python `
  ./scripts/calc_sample_lengths.py `
  --config ./training/unsloth_sft_v4.yaml `
  --limits 1024 2048 4096
```

最近一次 v4 主线数据统计结果：

```text
samples: 19361
max_seq_length: 2048
min: 507 p50: 1292 avg: 1136.7 p90: 1773 p95: 1799 p99: 1840 max: 1885
> 2048: 0 (0.0%)
> 4096: 0 (0.0%)
```

因此当前 `training\unsloth_sft_v4.yaml` 保持 `max_seq_length: 2048`。如果未来新增样本导致超过 2048，优先截断 tool response 文本，不提高 max length。

## 协议诊断

训练前建议先检查 canonical renderer 和 assistant-only labels 是否符合主线协议：

```powershell
uv run python `
  ./scripts/diagnose_sft_protocol.py `
  --config ./training/unsloth_sft_v4.yaml `
  --max-samples 200
```

重点看：

```text
tool_turn_think_rate=1.0
rendered_empty_answer_think=0
tool_response_label_leaks=0
```

`loss: false` 的历史 assistant turn 只作为上下文，不参与 loss；目标 assistant turn 和该 turn 的 `<|im_end|>` 会被监督。
v4 训练入口还会派生 `loss_weights`：assistant 首 token、`<think>`、`</think>`、`<tool_call>`、`<answer>`、`</answer>` 和 `<|im_end|>` 会被加权监督，`</tool_call>` 会降权，避免继续放大 closing-tag 先验。当前完整训练推荐基线为 `checkpoint-3633`，50 条主测评中 `starts_with_closing_tool_rate=0.0`，说明首轮 `</tool_call>` 崩溃已被修复。若后续新实验 smoke 又出现该问题，不要进入完整训练，先重新跑首 token probe 并检查是否使用当前 v4 数据和边界权重。

训练或 smoke checkpoint 产出后，可以用同一个诊断脚本探测首 token 概率，确认模型是否真的把 assistant 起始边界学到参数里：

```powershell
uv run python `
  ./scripts/diagnose_sft_protocol.py `
  --config ./training/unsloth_sft_v4.yaml `
  --max-samples 50 `
  --probe-model unsloth/Qwen3-4B-Instruct-2507 `
  --probe-adapter ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4/checkpoint-3633 `
  --probe-max-prompts 20
```

重点看 `first_token_probe` 中 `<think>`、`</think>`、`<answer>`、`<tool_call>`、`</tool_call>` 的概率和 `top1_closing_tool_rate`。主线期望无锚点 prompt 下 `</tool_call>` 不再成为高概率首 token。

## 标准 LoRA 训练

主线版本配置文件为：

```text
training\unsloth_sft_v4.yaml
```

当前 `training\unsloth_sft.yaml` 也指向同一套 v4 数据和输出目录，保留它是为了兼容已有命令；新实验建议显式使用 `unsloth_sft_v4.yaml`。

v4 weighted protocol loss 需要读取模型 `outputs.logits`。新版 Unsloth 默认会隐藏 logits，训练脚本会在导入 Unsloth 前自动设置：

```text
UNSLOTH_RETURN_LOGITS=1
```

因此不需要手动在 PowerShell 里额外设置该环境变量。

核心训练参数：

| 参数 | 当前值 |
| --- | --- |
| base model | `unsloth/Qwen3-4B-Instruct-2507` |
| data path | `data/novel_eval/sft_zh_unsloth_react_v4/train_cli.jsonl` |
| eval path | `data/novel_eval/sft_zh_unsloth_react_v4/eval.jsonl` |
| output dir | `training/outputs/unsloth_sft_qwen3_4b_lora_react_v4` |
| versioned config | `training/unsloth_sft_v4.yaml` |
| max seq length | `2048` |
| LoRA rank | `64` |
| learning rate | `5e-5` |
| epoch | `3` |
| batch / grad accum | `2 x 8` |
| packing | `false` |

从头训练：

```powershell
uv run --no-sync python `
  ./scripts/train_sft_unsloth.py `
  --config ./training/unsloth_sft_v4.yaml `
  --overwrite
```

当前主线完整训练已完成到 `checkpoint-3633`。新实验完整训练前仍建议先跑 v4 smoke 训练，确认协议边界加权 loss 能让模型在训练样本 prompt 上稳定生成 `<think>`：

```powershell
uv run --no-sync python `
  ./scripts/train_sft_unsloth.py `
  --config ./training/unsloth_sft_v4.yaml `
  --output-dir ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4_smoke `
  --max-samples 512 `
  --max-steps 200 `
  --overwrite
```

smoke checkpoint 先用 `docs\训练测评.md` 的 5 条 Agent loop 命令检查：`starts_with_closing_tool_rate <= 0.05`、`think_tag_rate >= 0.90`、`valid_tool_call_rate >= 0.80` 后，再跑完整训练。已有主线结果不需要重复执行 smoke：`checkpoint-3633` 的 50 条无锚点主测评已通过验收。

继续训练或从最新 checkpoint 恢复时，默认使用 `--resume`；输出目录保持在配置中的 `training\outputs\unsloth_sft_qwen3_4b_lora_react_v4`：

```powershell
uv run --no-sync python `
  ./scripts/train_sft_unsloth.py `
  --config ./training/unsloth_sft_v4.yaml `
  --resume
```

续训时脚本会为当前 `output_dir` 复用或创建 `swanlab_run_id.txt`，设置 `SWANLAB_RUN_ID` 和 `SWANLAB_RESUME=allow`，并先把本地 `metrics.jsonl` 或 checkpoint `trainer_state.json` 中 `step <= checkpoint_step` 的历史标量回放到同一个 SwanLab run，然后再继续上传新日志。回放状态写入：

```text
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4\swanlab_history_replay_state.json
```

手动补传历史指标但不重新训练：

```powershell
uv run python `
  ./scripts/replay_swanlab_history.py `
  --output-dir ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4 `
  --checkpoint ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4/checkpoint-495 `
  --dry-run
```

确认 `count/first_step/last_step` 正常后去掉 `--dry-run`。如果要接到已有 SwanLab run，额外传：

```text
--swanlab-run-id <已有 run id>
```

## 可追溯训练

需要定位 loss 或 grad norm 异常样本时，使用 trace 训练入口：

```powershell
uv run --no-sync python `
  ./scripts/train_sft_unsloth_trace.py `
  --config ./training/unsloth_sft_v4.yaml `
  --output-dir ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4_trace
```

追溯输出：

```text
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4_trace\step_sample_trace.jsonl
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4_trace\sample_loss_trace.jsonl
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4_trace\trace_summary.json
```

按 step 查看样本：

```powershell
uv run python `
  ./scripts/inspect_sft_trace.py `
  --trace ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4_trace `
  --step 735
```

按最高 grad norm 查看样本：

```powershell
uv run python `
  ./scripts/inspect_sft_trace.py `
  --trace ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4_trace `
  --top-grad 20
```

## 训练观测

训练脚本会记录：

| 指标 | 用途 |
| --- | --- |
| `train/loss` | 判断 SFT 是否收敛 |
| `eval/loss` | 判断 held-out train/eval split 上是否过拟合 |
| `train/grad_norm` | 观察梯度突刺 |
| `train/learning_rate` | 检查调度是否按预期衰减 |
| `train/epoch` | 确认训练进度 |
| `train/global_step` | 对齐 checkpoint 和 trace |

本地轮询输出目录：

```powershell
uv run python `
  ./scripts/watch_unsloth_training.py `
  --output-dir ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4 `
  --interval 30
```

SwanLab 页面重点看三点：

- loss 应先快速下降，再进入缓慢收敛区间。
- eval loss 不应持续发散。
- 断点续训后的 SwanLab 曲线应从早期历史 step 开始连续显示，而不是只显示续训后的新 step。

## 当前有效基线

当前推荐 SFT v4 checkpoint：

```text
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4\checkpoint-3633
```

50 条主测评结果：

```text
results\sft_compare\react_v4_full_ckpt3633_50.jsonl
results\sft_compare\react_v4_full_ckpt3633_50_summary.json
```

主测评口径固定为 `--assistant-start-anchor none --protocol-constraints none`。当前 summary 指标为 `avg_em=0.84`、`avg_f1=0.8433`、`avg_hop_recall=0.75`、`answer_tag_rate=1.0`、`valid_tool_call_rate=1.0`、`think_tag_rate=1.0`、`starts_with_closing_tool_rate=0.0`、`malformed_tool_fragment_rate=0.0`。该结果说明 v4 已解决协议边界崩溃，后续优化应聚焦低分样本的检索和答案抽取。

## verl tool-agent GRPO

严格复刻 example 的 GRPO 主入口是：

```text
training\start_grpo_tool_agent.sh
```

该入口需要 Linux/WSL/远程 GPU 环境，并要求 `VERL_DIR` 指向 verl 安装目录。它会调用 `python -m verl.trainer.main_ppo`，启用 `tool_agent`、`multi_turn.enable=True` 和 `multi_turn.format=hermes`，工具配置读取：

```text
training\config\novel_tool_config.yaml
```

训练前先在 demo 目录准备数据并启动检索服务：

```powershell
uv run python `
  ./scripts/prepare_agentic_grpo_data.py `
  --input ./data/novel_eval/qa_pairs.jsonl `
  --train-output ./data/novel_eval/grpo_agentic_train.parquet `
  --val-output ./data/novel_eval/grpo_agentic_val.parquet `
  --val-ratio 0.1 `
  --seed 42

uv run python `
  ./training/tools/retrieval_server.py `
  --corpus ./data/novel/corpus.jsonl `
  --index-dir ./data/novel/indexes `
  --embedding-model ./models/bge-m3 `
  --reranker-model ./models/bge-reranker-v2-m3 `
  --host 127.0.0.1 `
  --port 8790
```

Linux/WSL 上运行：

```bash
export VERL_DIR=/path/to/verl
export PROJECT_DIR=/path/to/AgenticRAG-RL/demo
bash ./training/start_grpo_tool_agent.sh
```

小样本 smoke：

```bash
ROLLOUT_N=2 TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
OUTPUT_DIR=./training/outputs/grpo_tool_agent_react_v4_smoke \
bash ./training/start_grpo_tool_agent.sh
```

旧 `scripts\train_grpo_unsloth.py` 只保留为 TRL/Unsloth 历史实验入口，不代表当前 example 风格 multi-turn tool-agent GRPO 主线。

GRPO 入口、数据 schema、BaseTool、reward 和验收口径的完整变更记录见 `docs\GRPO训练改动.md`。

## 导出 merged model

LoRA 训练完成后，导出主线 merged model：

```powershell
uv run --no-sync python `
  ./scripts/export_unsloth_lora.py `
  --config ./training/unsloth_sft_v4.yaml `
  --adapter-path ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4 `
  --export-dir ./models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged
```

也可以导出某个 checkpoint：

```powershell
uv run --no-sync python `
  ./scripts/export_unsloth_lora.py `
  --config ./training/unsloth_sft_v4.yaml `
  --adapter-path ./training/outputs/unsloth_sft_qwen3_4b_lora_react_v4/checkpoint-3633 `
  --export-dir ./models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-ckpt3633-merged
```

## 训练后检查

训练完成后按下面顺序判断是否进入下一步：

1. `eval/loss` 没有明显发散。
2. 指定 checkpoint 能被 `eval_hf_agentic.py --adapter` 正常加载。
3. 5 条 smoke 样本中工具轮为短 `<think>` 加 JSON `<tool_call>`，不能大面积出现首轮 `</tool_call>`。
4. 50 条 Agent loop 中 `think_tag_rate >= 0.95`、`valid_tool_call_rate >= 0.95`、`answer_tag_rate >= 0.80`、`starts_with_closing_tool_rate <= 0.05`。当前 `checkpoint-3633` 已满足这些门槛。

Agent loop 命令和 checkpoint 选择方法见 `docs\训练测评.md`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `max_seq_length` 截断样本 | 先重新跑长度统计；当前 v4 主线数据不应超过 2048 |
| checkpoint 路径被当成 Hugging Face repo id | 使用当前 `eval_hf_agentic.py`，它会自动把存在的本地 `--model` / `--adapter` 路径解析为绝对路径 |
| 模型首轮输出 `</tool_call>` | 说明起始状态仍不稳，优先检查是否使用 v4 数据、v4 输出目录和 weighted protocol loss 重新训练 |
| 模型输出 `<tool_call>查询文本</tool_call>` | 这是格式错误，应通过主线 ReAct SFT 数据和重新训练修复 |
| 模型只调用工具不回答 | 检查 `final_answer_only` 样本和 assistant turn 结束 token 是否参与 loss |
| Agent loop 达到最大轮次 | 先查看 `raw_turns`，确认是不会停、格式错，还是检索证据不足 |
