# SFT 训练版本改动说明

本文记录当前主线 SFT 版本状态，避免把旧 checkpoint 或旧数据误当成有效基线。

## 当前主线：ReAct v3

v3 用于修复上一轮 50 条 Agent loop 测评中大面积首轮输出 `</tool_call>` 的问题。当前主线数据、训练和评测统一使用：

- 数据目录：`data\novel_eval\sft_react_v3`、`data\novel_eval\sft_zh_unsloth_react_v3`
- 训练输出：`training\outputs\unsloth_sft_qwen3_4b_lora_react_v3`
- 合并模型：`models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v3-merged`
- renderer：项目 canonical Qwen3 ReAct renderer
- 协议：短 `<think>` + JSON `<tool_call>`，最终 `<answer>`

## v3 数据变化

每条 Oracle trace 不再只生成完整轨迹和答案样本，而是生成四类主线样本：

| sample_type | 目的 |
| --- | --- |
| `full_trace` | 完整 Oracle Agent 轨迹 |
| `first_action_only` | 2 倍加权首轮动作，强化 assistant turn 从 `<think>` 开始 |
| `next_action_only` | 带前序自然历史，只监督下一轮工具动作 |
| `final_answer_only` | 带完整自然历史，只监督最终 `<answer>` |

历史 assistant turn 通过 `loss: false` 保留在上下文中，但不参与监督。这样最终答案样本和下一跳动作样本都保持自然 Agent 历史，不再使用非自然的“只有 tool response 没有 assistant tool call 历史”的格式。

## 当前验收状态

已完成的数据和静态验收：

```text
train.jsonl count = 16561
train_cli.jsonl count = 16361
eval.jsonl count = 200
tool_turn_think_rate = 1.0
tool_response_label_leaks = 0
>2048 = 0
```

训练后主报告必须使用：

```text
--assistant-start-anchor none
```

`--assistant-start-anchor think` 只能作为诊断对照，不能作为主线指标。
