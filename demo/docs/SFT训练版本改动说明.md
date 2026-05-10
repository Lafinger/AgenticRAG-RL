# SFT 训练版本改动说明

本文只记录 demo 工程 SFT 训练协议版本的演进，目标是避免把旧数据、旧 checkpoint 或旧测评结果误当成当前有效基线。这里的 V1/V2/V3/V4 指 SFT 主线协议版本，不指 QA 合成流程中的其他 `v1` 口径。

## 版本总览

| 版本 | 主协议 | 主要数据形态 | 暴露的问题 | 处理结论 |
| --- | --- | --- | --- | --- |
| V1 | 普通 messages，工具响应以手写消息混入 | 非严格 tool-aware ShareGPT records | 能检索但不会稳定停止回答，`answer_tag_rate=0` | 废弃为有效基线，转向 canonical ReAct |
| V2 | 短 `<think>` + JSON `<tool_call>`，最终 `<answer>` | `full_trace + finalization_only` | 50 条测评 49 条失败，首轮大量直接输出 `</tool_call>` | 废弃为有效基线，转向 V3 数据拆分 |
| V3 | canonical Qwen3 ReAct renderer + assistant-only loss | `full_trace / first_action_only / next_action_only / final_answer_only` | 无锚点主测评 50/50 首轮以 `</tool_call>` 开头；`think` 锚点能恢复工具调用但仍不会稳定回答 | 废弃为有效主基线，转向 V4 协议边界加权 loss |
| V4 | canonical ReAct + weighted protocol loss | 四类主线样本，`final_answer_only` 2 倍强化，训练时派生 `loss_weights` | 初始 smoke 暴露边界权重不足；完整训练后 `checkpoint-3633` 通过无锚点主测评 | 当前唯一有效主线 |

## V1：普通 messages / 非严格 tool-aware 阶段

### 当时做法

V1 的 SFT 数据更接近普通 ShareGPT messages。工具响应没有作为严格的 `tool` role 保留下来，训练、Agent loop 推理和后续 GRPO prompt 也没有完全共享同一套 tool schema、`# Tools` 注入和 Qwen3 ReAct 渲染协议。

典型特征：

- 数据更像普通 `messages`，不是 record 级 `tools + messages + metadata` 的 canonical record。
- 工具响应容易提前被写成普通消息内容，而不是由 renderer 统一包装成 `<tool_response>`。
- 训练和评测之间的 `<tool_call>`、`<tool_response>`、`<answer>` 协议边界不够一致。

### 暴露的问题

V1 的核心问题不是完全不会检索，而是不会稳定结束并回答。典型症状是：

```text
valid_tool_call_rate 较高
answer_tag_rate = 0
```

也就是说，模型能学到“调用工具”这个动作，但没有可靠学到“证据足够后输出 `<answer>...</answer>` 并停止”的协议。

### 解决方案

V1 之后的修复方向是切换到 Oracle trace 驱动的 canonical ReAct SFT：

- 从 `qa_pairs + corpus/indexes` 生成 Oracle ReAct traces。
- SFT record 保留真实 `tool` role。
- 每条 record 写入同一套 Qwen3 `tools` schema。
- 训练、长度统计和 Agent loop 评测共用项目 canonical Qwen3 ReAct renderer。
- assistant-only loss 只监督 assistant 输出和对应 turn 的 `<|im_end|>`，system/user/tool response/padding 不参与 loss。

### 结论

V1 不是当前有效基线。V1 的经验结论是：Agentic SFT 不能只做普通 messages 微调，必须让训练、评测和后续 RL 共享同一套工具协议和渲染路径。

## V2：canonical ReAct 初版

### 当时做法

V2 已经切换到 canonical ReAct 主线，工具轮采用 example 风格短 `<think>` 加 JSON `<tool_call>`，最终答案轮使用 `<answer>`：

```text
<think>要回答最终问题，先查：卖饼老者自称是哪个集市的人？</think>
<tool_call>
{"name":"keyword_search","arguments":{"query":"卖饼老者自称是哪个集市的人？"}}
</tool_call>
```

V2 的 SFT 数据主要由两类样本组成：

| sample_type | 目的 |
| --- | --- |
| `full_trace` | 学习完整 Oracle Agent 轨迹 |
| `finalization_only` | 学习已有证据后输出 `<answer>` |

当时的默认训练产物包括：

```text
training\outputs\unsloth_sft_qwen3_4b_lora
models\Qwen3-4B-Instruct-2507-Unsloth-SFT-merged
```

### 暴露的问题

V2 重新训练后，50 条 Agent loop 测评不满足验收门槛。关键指标如下：

| 指标 | V2 测评值 | 目标 |
| --- | ---: | ---: |
| `avg_f1` | 0.02 | 明显高于当前 |
| `avg_hop_recall` | 0.0567 | 明显高于当前 |
| `answer_tag_rate` | 0.02 | >= 0.80 |
| `valid_tool_call_rate` | 0.12 | >= 0.95 |
| `malformed_tool_fragment_rate` | 0.8448 | <= 0.20 |
| `starts_with_closing_tool_rate` | 0.8448 | <= 0.10 |
| `think_tag_rate` | 1.0 | >= 0.95 |

50 条样本中 49 条失败，最典型的首轮输出是：

```text
</tool_call>
```

### 根因判断

V2 的主要根因有两点：

1. `finalization_only` 历史不自然：该类样本使用 `system + user + tool response + answer`，缺少前序 assistant tool call 历史。
2. 首轮 action 起始状态监督不足：模型学到了大量 `</tool_call>` 结束标签，却没有足够稳定地学会新一轮 assistant 的起始边界。

同时，旧评测 stop criteria 在看到孤立 `</tool_call>` 时会过早停止，进一步放大了这个失败形态。

### 解决方案

V2 之后的修复方向不是新增独立训练路线，而是把修复并入主线 SFT：

- 删除非自然的 `finalization_only` 结构。
- 增加首轮工具动作校准样本。
- 允许历史 assistant 进入上下文但不参与 loss。
- 修正评测 stop criteria，只在完整 `<answer>` 或完整 `<think> + <tool_call>` 后停止。

### 结论

V2 checkpoint 630/720/765/1089 已确认为协议失败基线，不再作为有效主结果。V2 的经验结论是：canonical ReAct 协议本身可行，但训练样本必须覆盖真实 Agent loop 的起始状态、下一跳动作和最终回答状态。

## V3：自然历史数据拆分阶段

### 当时做法

V3 保留 canonical Qwen3 ReAct 协议：

- 工具轮：短 `<think>` + JSON `<tool_call>`。
- 答案轮：`<answer>...</answer>`。
- renderer：项目 canonical Qwen3 ReAct renderer。
- label mask：assistant-only loss。
- tool response：保留 `tool` role，由 renderer 统一包装成 `<tool_response>`。

V3 每条 Oracle trace 生成四类主线样本：

| sample_type | 目的 |
| --- | --- |
| `full_trace` | 完整 Oracle Agent 轨迹 |
| `first_action_only` | 2 倍加权首轮动作，强化 assistant turn 从 `<think>` 开始 |
| `next_action_only` | 带前序自然历史，只监督下一轮工具动作 |
| `final_answer_only` | 带完整自然历史，只监督最终 `<answer>` |

历史 assistant turn 通过：

```json
{"role": "assistant", "content": "...", "loss": false}
```

保留在上下文中，但不参与 loss。

### 暴露的问题

V3 训练后，主测评 `--assistant-start-anchor none` 下仍然失败：

```text
starts_with_closing_tool_rate = 1.0
think_tag_rate = 0.0
valid_tool_call_rate = 0.0
answer_tag_rate = 0.0
avg_f1 = 0.0
avg_hop_recall = 0.0
```

`--assistant-start-anchor think` 诊断对照能让工具调用恢复，例如 `valid_tool_call_rate=1.0`、`avg_hop_recall≈0.6667`，但仍不会稳定输出 `<answer>`。

### 根因判断

V3 的数据结构已经比 V2 更自然，但模型仍没有稳定学会 assistant turn 的首 token 边界。失败不再像是 JSONL 直接损坏，而是训练目标对关键协议边界的惩罚不够强：

- 新 assistant turn 应从 `<think>` 或 `<answer>` 开始。
- 证据足够后的目标应切到 `<answer>`。
- action 或 answer 完成后应输出对应 turn 的 `<|im_end|>`。

普通 assistant-only CE 对所有 assistant token 基本等权，容易让高频 closing tag 先验压过首 token 协议边界。

### 结论

V3 不再作为有效主基线。V3 的经验结论是：只调整样本形态仍不够，必须在训练目标上显式强化协议边界。

## V4：当前主线 weighted protocol loss

### 当前做法

V4 继续只保留 canonical Agentic SFT 主线，不恢复任何独立训练支线或格式修复支线。数据协议仍是：

- 工具轮：短 `<think>` + JSON `<tool_call>`。
- 答案轮：`<answer>...</answer>`。
- tool response：保留 `tool` role，由 canonical renderer 包装成 `<tool_response>`。
- 主评测：`--assistant-start-anchor none --protocol-constraints none`。

当前主线路径：

```text
data\novel_eval\sft_react_v4
data\novel_eval\sft_zh_unsloth_react_v4
training\unsloth_sft_v4.yaml
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4
models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged
```

### 数据和训练目标变化

V4 全量 SFT 数据仍由四类样本组成，但 `final_answer_only` 也做 2 倍强化：

| sample_type | 全量数量 | 目的 |
| --- | ---: | --- |
| `full_trace` | 3000 | 学习完整多轮工具调用和最终回答 |
| `first_action_only` | 6000 | 强化首轮 `<think> + <tool_call>` 起始状态 |
| `next_action_only` | 4561 | 学习带历史证据时继续发起下一跳检索 |
| `final_answer_only` | 6000 | 强化证据足够后稳定输出 `<answer>` |

训练入口会在 tokenization 后生成 `loss_weights`：

| token 区域 | 权重意图 |
| --- | --- |
| assistant 目标首 token | 强化新 assistant turn 的起始边界 |
| `<think>` | 强化工具轮必须先输出短 think |
| `</think>` | 强化 think 正确闭合，避免在 think 结束位置误输出 `</tool_call>` |
| `<tool_call>` | 强化工具动作开始 |
| `<answer>` | 强化最终回答状态切换 |
| `</answer>` | 强化最终回答闭合 |
| `<|im_end|>` | 强化 action/answer 完成后结束当前 turn |
| `</tool_call>` | 降权，避免继续放大 closing tag 先验 |

`loss_weights` 不写入 JSONL 数据文件，而是在 `training/sft_label_mask.py` tokenization 时派生，并由自定义 collator 和 weighted CE Trainer 消费。早期 v4 smoke 曾显示 `</tool_call>` 仍是高概率首 token，因此后续实现把 assistant 首 token、`<think>`、`</think>`、`<tool_call>`、`<answer>`、`</answer>` 和 `<|im_end|>` 的权重进一步提高，并对 `</tool_call>` 降权；这不改变 JSONL 数据，只改变训练目标。完整训练后的 `checkpoint-3633` 已在主测评中验证该策略有效。

### 当前数据验收

当前已生成并验收的 V4 数据：

```text
train.jsonl count = 19561
train_cli.jsonl count = 19361
eval.jsonl count = 200
tool_turn_think_rate = 1.0
rendered_empty_answer_think = 0
tool_response_label_leaks = 0
max token length = 1885
>2048 = 0
```

当前训练配置要点：

```text
learning_rate = 5e-5
max_seq_length = 2048
LoRA rank = 64
batch x grad accum = 2 x 8
packing = false
```

### 训练和测评要求

新实验完整训练前仍需先跑 smoke：

```text
--max-samples 512
--max-steps 200
```

smoke 门槛：

```text
starts_with_closing_tool_rate <= 0.05
think_tag_rate >= 0.90
valid_tool_call_rate >= 0.80
```

当前完整训练已达到 `checkpoint-3633`。完整训练后，主报告必须使用：

```text
--assistant-start-anchor none
--protocol-constraints none
```

50 条 Agent loop 验收门槛：

```text
think_tag_rate >= 0.95
valid_tool_call_rate >= 0.95
answer_tag_rate >= 0.80
malformed_tool_fragment_rate <= 0.20
starts_with_closing_tool_rate <= 0.05
avg_f1 / avg_hop_recall 高于 V3 失败基线
```

`--assistant-start-anchor think` 和 `--protocol-constraints strict` 只能用于诊断或生产保护对照，不能作为主线达标依据。

### 当前 V4 主测评结果

当前推荐 checkpoint：

```text
training\outputs\unsloth_sft_qwen3_4b_lora_react_v4\checkpoint-3633
```

50 条无锚点、无协议约束 Agent loop 主测评结果：

```text
results\sft_compare\react_v4_full_ckpt3633_50.jsonl
results\sft_compare\react_v4_full_ckpt3633_50_summary.json
```

关键指标：

| 指标 | 当前值 | 验收目标 | 结论 |
| --- | ---: | ---: | --- |
| `avg_em` | 0.84 | 高于 V3 失败基线 | 通过 |
| `avg_f1` | 0.8433 | 高于 V3 失败基线 | 通过 |
| `avg_hop_recall` | 0.75 | 高于 V3 失败基线 | 通过 |
| `answer_tag_rate` | 1.0 | >= 0.80 | 通过 |
| `valid_tool_call_rate` | 1.0 | >= 0.95 | 通过 |
| `think_tag_rate` | 1.0 | >= 0.95 | 通过 |
| `malformed_tool_fragment_rate` | 0.0 | <= 0.20 | 通过 |
| `starts_with_closing_tool_rate` | 0.0 | <= 0.05 | 通过 |

这说明 V4 已解决 V2/V3 的首轮 `</tool_call>` 崩溃、缺 `<answer>` 和 malformed tool call 问题。当前剩余误差主要来自检索召回不完整、答案抽取错位或别名覆盖不足，不再是主协议失效。

## 当前主线结论

当前有效 SFT 主线只认 V4。提交、训练和测评时按下面规则执行：

1. 训练数据使用 `data\novel_eval\sft_zh_unsloth_react_v4\train_cli.jsonl`。
2. 训练验证使用 `data\novel_eval\sft_zh_unsloth_react_v4\eval.jsonl`。
3. 配置使用 `training\unsloth_sft_v4.yaml`，`training\unsloth_sft.yaml` 当前也指向同一套 v4 主线。
4. LoRA 输出使用 `training\outputs\unsloth_sft_qwen3_4b_lora_react_v4`，当前推荐 checkpoint 为 `checkpoint-3633`。
5. Agent loop 主测评使用 `--assistant-start-anchor none --protocol-constraints none`。
6. V1/V2/V3 的数据、checkpoint 和 merged model 只能用于问题追溯，不能作为当前有效基线。

如果后续 V4 测评仍不达标，优化应继续在 canonical Agentic SFT 主线内完成，优先检查协议边界权重、assistant 起始状态、最终答案样本比例、stop criteria 和首 token probe，而不是恢复已经废弃的独立训练路线。
