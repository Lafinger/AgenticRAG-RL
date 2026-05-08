# SFT 训练版本改动说明

本文只记录 demo 工程 SFT 训练协议版本的演进，目标是避免把旧数据、旧 checkpoint 或旧测评结果误当成当前有效基线。这里的 V1/V2/V3 指 SFT 主线协议版本，不指 QA 合成流程中的其他 `v1` 口径。

## 版本总览

| 版本 | 主协议 | 主要数据形态 | 暴露的问题 | 处理结论 |
| --- | --- | --- | --- | --- |
| V1 | 普通 messages，工具响应以手写消息混入 | 非严格 tool-aware ShareGPT records | 能检索但不会稳定停止回答，`answer_tag_rate=0` | 废弃为有效基线，转向 canonical ReAct |
| V2 | 短 `<think>` + JSON `<tool_call>`，最终 `<answer>` | `full_trace + finalization_only` | 50 条测评 49 条失败，首轮大量直接输出 `</tool_call>` | 废弃为有效基线，转向 v3 数据拆分 |
| V3 | canonical Qwen3 ReAct renderer + assistant-only loss | `full_trace / first_action_only / next_action_only / final_answer_only` | 当前主线，待重新训练后用 Agent loop 验收 | 当前唯一有效主线 |

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

这说明模型不是不会输出 `<think>`，也不是检索器完全不可用；唯一成功样本证明完整 ReAct 路线可行。但大部分样本在 assistant turn 起始状态上不稳定，生成一开始就像处在已经打开的 `<tool_call>` 内部。

### 根因判断

V2 的主要根因有两点：

1. `finalization_only` 历史不自然
   该类样本使用 `system + user + tool response + answer`，缺少前序 assistant tool call 历史。它能强化回答，但也引入了与真实 Agent loop 不完全一致的上下文形态。

2. 首轮 action 起始状态监督不足
   V2 对“assistant 新 turn 必须从 `<think>` 或 `<answer>` 开始”的约束不够强。模型学到了大量 `</tool_call>` 结束标签，却没有足够稳定地学会新一轮 assistant 的起始边界。

同时，旧评测 stop criteria 在看到孤立 `</tool_call>` 时会过早停止，进一步放大了这个失败形态。

### 解决方案

V2 之后的修复方向不是新增独立训练路线，而是把修复并入主线 SFT：

- 删除非自然的 `finalization_only` 结构。
- 增加首轮工具动作校准样本。
- 允许历史 assistant 进入上下文但不参与 loss。
- 修正评测 stop criteria，只在完整 `<answer>` 或完整 `<think> + <tool_call>` 后停止。

### 结论

V2 checkpoint 630/720/765/1089 已确认为协议失败基线，不再作为有效主结果。V2 的经验结论是：canonical ReAct 协议本身可行，但训练样本必须覆盖真实 Agent loop 的起始状态、下一跳动作和最终回答状态。

## V3：当前主线 ReAct SFT

### 当前做法

V3 仍保留 canonical Qwen3 ReAct 协议：

- 工具轮：短 `<think>` + JSON `<tool_call>`。
- 答案轮：`<answer>...</answer>`。
- renderer：项目 canonical Qwen3 ReAct renderer。
- label mask：assistant-only loss。
- tool response：保留 `tool` role，由 renderer 统一包装成 `<tool_response>`。

当前主线路径：

```text
data\novel_eval\sft_react_v3
data\novel_eval\sft_zh_unsloth_react_v3
training\outputs\unsloth_sft_qwen3_4b_lora_react_v3
models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v3-merged
```

### 数据变化

V3 每条 Oracle trace 不再只生成完整轨迹和答案样本，而是生成四类主线样本：

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

保留在上下文中，但不参与 loss。这样 `next_action_only` 和 `final_answer_only` 都有真实的 assistant/tool 历史，而不是只有孤立的 tool response。

### 关键修复

V3 的核心修复包括：

- `first_action_only` 2 倍加权，直接强化首轮 `<think> + <tool_call>`。
- `loss: false` 支持历史 assistant 作为上下文，不污染监督目标。
- assistant-only loss 仍监督目标 assistant 内容和该 turn 的 `<|im_end|>`。
- tool response 文本按固定长度截断，保证 `max_seq_length: 2048` 下不截断训练样本。
- Agent loop stop criteria 只在完整 `<answer>` 或完整 `<think> + <tool_call>` 后停止，不再因为孤立 `</tool_call>` 提前停止。
- 新增 `--assistant-start-anchor {none,think}`，默认 `none` 作为主基线；`think` 只用于诊断对照。

### 当前数据验收

当前已生成并验收的 v3 数据：

```text
train.jsonl count = 16561
train_cli.jsonl count = 16361
eval.jsonl count = 200
tool_turn_think_rate = 1.0
tool_response_label_leaks = 0
>2048 = 0
max token length = 1882
```

当前训练配置要点：

```text
learning_rate = 5e-5
max_seq_length = 2048
LoRA rank = 64
batch x grad accum = 2 x 8
packing = false
```

### 训练后验收

V3 重新训练后，主报告必须使用：

```text
--assistant-start-anchor none
```

验收门槛：

```text
think_tag_rate >= 0.95
valid_tool_call_rate >= 0.95
answer_tag_rate >= 0.80
malformed_tool_fragment_rate <= 0.20
starts_with_closing_tool_rate <= 0.10
multi_action_turn_rate <= 0.20
avg_f1 / avg_hop_recall 高于 V2 失败基线
```

`--assistant-start-anchor think` 只能用于判断“是否主要是 assistant 起始 token 不稳”，不能作为主线指标。

## 当前主线结论

当前有效 SFT 主线只认 V3。提交、训练和测评时按下面规则执行：

1. 训练数据使用 `data\novel_eval\sft_zh_unsloth_react_v3\train_cli.jsonl`。
2. 训练验证使用 `data\novel_eval\sft_zh_unsloth_react_v3\eval.jsonl`。
3. LoRA 输出使用 `training\outputs\unsloth_sft_qwen3_4b_lora_react_v3`。
4. Agent loop 主测评使用 `--assistant-start-anchor none`。
5. V1/V2 的数据、checkpoint 和 merged model 只能用于问题追溯，不能作为当前有效基线。

如果后续 V3 测评仍不达标，优化应继续在 canonical Agentic SFT 主线内完成，优先检查数据分布、assistant 起始状态、下一跳动作样本、最终答案样本和 stop criteria，而不是恢复已经废弃的独立训练路线。
