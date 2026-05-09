# FQA

本文按主题归类解释本项目 SFT 训练样本结构，重点说明 `input_ids`、`attention_mask`、`labels` 和 assistant-only label mask。

## 一、样本从 messages 到 input_ids

### Q1: messages、chat 文本、input_ids 是什么关系？

答：原始 SFT 样本首先是 tool-aware canonical ReAct 结构，包含 `messages` 和 `tools`：

```python
{
    "messages": [
        {"role": "system", "content": "你是一个中文小说阅读问答 Agent..."},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "<think>要回答最终问题，先查：问题关键词</think>\n<tool_call>\n{\"name\":\"keyword_search\",\"arguments\":{\"query\":\"问题关键词\"}}\n</tool_call>"},
        {"role": "tool", "content": "[chunk_id] 检索证据..."},
        {"role": "assistant", "content": "<answer>答案</answer>"}
    ],
    "tools": [
        {"type": "function", "function": {"name": "keyword_search", "parameters": {"type": "object"}}}
    ]
}
```

项目会先通过 canonical Qwen3 ReAct renderer 把它渲染成模型实际看到的 chat 文本。renderer 保留 Qwen3 `<|im_start|>` / `<|im_end|>` 对话格式，在 system 后注入 `# Tools` / `<tools>`，并把 `tool` role 渲染为用户侧 `<tool_response>`：

```text
<|im_start|>system
你是一个中文小说阅读问答 Agent...
# Tools
<tools>
...
</tools><|im_end|>
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
<think>要回答最终问题，先查：问题关键词</think>
<tool_call>
{"name":"keyword_search","arguments":{"query":"问题关键词"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
[chunk_id] 检索证据...
</tool_response><|im_end|>
<|im_start|>assistant
<answer>答案</answer><|im_end|>
```

然后 tokenizer 再把这段文本编码成整数数组：

```python
input_ids = [151644, ..., 151645, ...]
```

所以严格关系是：

```text
messages + tools --canonical renderer 渲染--> chat 文本 --tokenizer 编码--> input_ids
```

训练时模型真正接收的是 `input_ids`，但 `input_ids` 承载的信息来自完整 chat 文本。

### Q1.1: 为什么工具轮要保留短思考标签？

答：当前主线对齐 example 风格 ReAct。训练样本要求 assistant 工具轮先输出一个短 `<think>`，再输出一个 JSON tool call。`<think>` 来自结构化 Oracle plan：第 1 跳说明“要回答最终问题，先查什么”，后续跳说明“基于上一跳 gold answer 继续查什么”。它只表达检索决策，不训练长推理链；推理端仍严格校验 `<tool_call>` 内必须是 JSON。

标准工具轮：

```text
<think>要回答最终问题，先查：问题关键词</think>
<tool_call>
{"name":"keyword_search","arguments":{"query":"问题关键词"}}
</tool_call>
```

### Q1.2: V3/V4 为什么新增多种 sample_type？

答：V2 训练后，50 条 Agent loop 测评中大量样本首轮直接输出 `</tool_call>`。V3 不新增独立模型路线，而是在主线 SFT 内拆出更自然的训练视角；V4 继续沿用这四类主线样本，并提高 `final_answer_only` 占比，用于修复加 `<think>` 锚点后仍不会稳定输出 `<answer>` 的问题：

| sample_type | 作用 |
| --- | --- |
| `full_trace` | 学习完整 Oracle Agent 轨迹 |
| `first_action_only` | 2 倍加权首轮 `<think> + <tool_call>`，稳定 assistant 起始状态 |
| `next_action_only` | 带历史 assistant/tool 上下文，只监督下一轮工具动作 |
| `final_answer_only` | 带完整历史上下文，只监督最终 `<answer>`；V4 中也做 2 倍强化 |

历史 assistant turn 会写入 `loss: false`。它仍进入模型上下文，但不会参与 loss。

### Q1.3: V4 为什么不是简单继续加数据，而是增加 `loss_weights`？

答：V3 已经把数据拆成更自然的 Agent 历史，但主测评仍出现 50/50 首轮以 `</tool_call>` 开头。这个现象说明问题集中在 assistant turn 的协议边界，而不是 JSONL 整体缺几条样本。V4 因此在 tokenization 后派生 `loss_weights`，提高 assistant 首 token、`<think>`、`</think>`、`<tool_call>`、`<answer>`、`</answer>` 和 `<|im_end|>` 的训练权重；`</tool_call>` 会降权，避免继续强化 closing tag 先验。最近一次 smoke 已证明初始权重仍不足，因此当前实现额外强化了 `</think>`，专门修复“应该闭合 think 却输出 `</tool_call>`”的失败形态。

因为加权 loss 需要访问模型 logits，训练脚本会自动设置 `UNSLOTH_RETURN_LOGITS=1`。如果没有这个环境变量，新版 Unsloth 会隐藏 logits，weighted loss 无法计算。

### Q1.5: 为什么 JSONL 里要保留 `tool` role，而不是提前改成 user？

答：项目 canonical renderer 知道如何处理 `tool` role。原始 JSONL 保留：

```text
role = tool
content = 原始检索证据
```

渲染时 renderer 会转换成：

```text
<|im_start|>user
<tool_response>
原始检索证据
</tool_response><|im_end|>
```

这样 SFT、Agent loop 和后续 GRPO 都共享同一套 `tools` schema 和 renderer 逻辑，不会出现训练时是手写 `<tool_response>`、推理时又是另一种格式的协议漂移。项目不直接使用 tokenizer 原生 `tools=` 模板作为主渲染路径，因为它不会要求短 `<think>`，并可能在最终答案前补空 `<think>`。

### Q2: 151644 对应什么？

答：`151644` 是 tokenizer 词表里的一个 token id。对 Qwen/Qwen3 tokenizer 来说，它通常对应特殊 token：

```text
151644 -> <|im_start|>
151645 -> <|im_end|>
151643 -> <|endoftext|>
```

也就是说：

```text
<|im_start|> 是 token 字符串
151644 是这个 token 字符串对应的 token id
```

它不是普通中文词，而是 chat template 使用的特殊词元，用来标记一条 message 的开始。

可用下面命令验证：

```powershell
$code = @'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('unsloth/Qwen3-4B-Instruct-2507', trust_remote_code=True)
print(t.convert_ids_to_tokens([151644,151645,151643]))
'@
uv run python `
  -c $code
```

## 二、训练样本四项张量

### Q3: attention_mask 有什么用？

答：`attention_mask` 用来告诉模型哪些位置是真实 token，哪些位置是 padding。

一个 batch 里不同样本长度可能不同，短样本会补齐到同一长度：

```python
input_ids = [
    [151644, 100, 101, 151645],
    [151644, 200, 151645, 0]
]
```

第二条最后的 `0` 是 padding，不是真实输入。对应：

```python
attention_mask = [
    [1, 1, 1, 1],
    [1, 1, 1, 0]
]
```

含义是：

```text
1 = 真实 token，模型可以 attention 到
0 = padding token，模型不要关注
```

### Q4: labels 和 input_ids 是怎么一一对应的？

答：`labels` 和 `input_ids` 等长、同下标一一对应：

```text
位置 i: input_ids[i] -> labels[i]
```

核心规则是：

```python
labels[i] = input_ids[i]   # 如果这个 token 属于 assistant 内容或该 assistant turn 的 <|im_end|>
labels[i] = -100           # 如果这个 token 不属于 assistant 输出
```

例如：

```python
input_ids = [
    151644, ...,   # <|im_start|>system ...
    151645,        # <|im_end|>
    151644, ...,   # <|im_start|>user ...
    151645,        # <|im_end|>
    151644, ...,   # <|im_start|>assistant
    123, 456, 789, # assistant 的 <tool_call>...</tool_call>
    151645,        # <|im_end|>
    151644, ...,   # <|im_start|>user 的 <tool_response>...
    151645,
    151644, ...,   # <|im_start|>assistant
    234, 567, 890, # assistant 的 <answer>答案</answer>
    151645
]

labels = [
    -100, ...,      # system 区域，不算 loss
    -100,
    -100, ...,      # user 问题，不算 loss
    -100,
    -100, ...,      # assistant 起始模板 token，不算 loss
    123, 456, 789, # assistant 内容，算 loss
    151645,        # assistant 的 <|im_end|>，算 loss，学习一轮 action 后停止
    -100, ...,      # tool_response 是 user 内容，不算 loss
    -100,
    -100, ...,      # assistant 起始模板 token，不算 loss
    234, 567, 890, # assistant 答案内容，算 loss
    151645
]
```

所以 `labels` 不是另一段独立文本，而是同一条 token 序列上的监督开关。

### Q5: 最终给训练器的一条样本长什么样？

答：最终给训练器的是一条包含四个数组的样本。前三个数组负责常规 causal LM 训练，`loss_weights` 是 V4 增加的协议边界权重：

```text
input_ids        模型看什么
attention_mask   哪些输入是真的，哪些是 padding
labels           哪些位置要计算训练 loss
loss_weights     每个 label token 的 loss 权重
```

它们共同组成：

```python
{
    "input_ids": [...],
    "attention_mask": [1, 1, 1, 1, ...],
    "labels": [
        -100, -100, ...,
        123, 456, 789,
        -100, -100, ...,
        234, 567, 890,
        ...
    ],
    "loss_weights": [
        0.0, 0.0, ...,
        48.0, 32.0, 1.0,
        0.0, 0.0, ...,
        48.0, 1.0, 8.0,
        ...
    ]
}
```

## 三、assistant-only label mask

### Q6: labels = -100 是什么意思？

答：`-100` 是 PyTorch / Transformers 交叉熵 loss 的 ignore label。训练时：

- `labels[i] == -100`：这个位置跳过，不计算 loss。
- `labels[i] != -100`：这个位置参与 loss，模型要学习预测该 token。

在本项目里，assistant-only label mask 的目标是：

```text
system              labels = -100
user question       labels = -100
assistant reply     labels = token_id，包括该 assistant turn 的 <|im_end|>
tool response       labels = -100
assistant answer    labels = token_id，包括该 assistant turn 的 <|im_end|>
padding             labels = -100
```

这样模型仍然能通过 `input_ids` 看到完整上下文，但只学习 assistant 应该如何输出短 `<think>` + JSON `<tool_call>` 或 `<answer>...</answer>`。

### Q7: 为什么不能让整段 chat 都参与 loss？

答：如果整段 chat 都参与 loss，模型会被训练去复现：

- system prompt
- user 问题
- `<tool_response>` 检索证据
- assistant 工具调用和答案

这会让模型更容易复读用户问题或复读检索结果。对 Agent SFT 来说，正确目标应该是：

```text
给定 system/user/tool_response 上下文，学习 assistant 下一步应该输出什么。
```

因此本项目在 `training\sft_label_mask.py` 中只保留 assistant 内容作为 labels，其它位置全部置为 `-100`。
