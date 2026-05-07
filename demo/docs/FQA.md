# FQA

本文记录本项目 SFT 训练中常见的数据结构问题，重点解释 `input_ids`、`attention_mask`、`labels` 和 assistant-only label mask。

## messages、chat 文本、input_ids 是什么关系？

原始 SFT 样本首先是 `messages` 结构：

```python
[
    {"role": "system", "content": "你是一个中文小说阅读问答 Agent..."},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>检索证据...</tool_response>"},
    {"role": "assistant", "content": "<answer>答案</answer>"}
]
```

tokenizer 会先通过 chat template 把它渲染成模型实际看到的 chat 文本：

```text
<|im_start|>system
你是一个中文小说阅读问答 Agent...<|im_end|>
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
<tool_call>...</tool_call><|im_end|>
<|im_start|>user
<tool_response>检索证据...</tool_response><|im_end|>
<|im_start|>assistant
<answer>答案</answer><|im_end|>
```

然后 tokenizer 再把这段文本编码成整数数组：

```python
input_ids = [151644, ..., 151645, ...]
```

所以严格关系是：

```text
messages --chat template 渲染--> chat 文本 --tokenizer 编码--> input_ids
```

训练时模型真正接收的是 `input_ids`，但 `input_ids` 承载的信息来自完整 chat 文本。

## 151644 对应什么？

`151644` 是 tokenizer 词表里的一个 token id。对 Qwen/Qwen3 tokenizer 来说，它通常对应特殊 token：

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
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('unsloth/Qwen3-4B-Instruct-2507', trust_remote_code=True); print(t.convert_ids_to_tokens([151644,151645,151643]))"
```

## attention_mask 有什么用？

`attention_mask` 用来告诉模型哪些位置是真实 token，哪些位置是 padding。

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

## labels 和 input_ids 是怎么对应的？

`labels` 和 `input_ids` 等长、同下标一一对应：

```text
位置 i: input_ids[i] -> labels[i]
```

核心规则是：

```python
labels[i] = input_ids[i]   # 如果这个 token 属于 assistant 内容
labels[i] = -100           # 如果这个 token 不属于 assistant 内容
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
    -100,          # <|im_end|> 不算 loss
    -100, ...,      # tool_response 是 user 内容，不算 loss
    -100,
    -100, ...,      # assistant 起始模板 token，不算 loss
    234, 567, 890, # assistant 答案内容，算 loss
    -100
]
```

所以 `labels` 不是另一段独立文本，而是同一条 token 序列上的监督开关。

## labels = -100 是什么意思？

`-100` 是 PyTorch / Transformers 交叉熵 loss 的 ignore label。训练时：

- `labels[i] == -100`：这个位置跳过，不计算 loss。
- `labels[i] != -100`：这个位置参与 loss，模型要学习预测该 token。

在本项目里，assistant-only label mask 的目标是：

```text
system              labels = -100
user question       labels = -100
assistant reply     labels = token_id
user tool_response  labels = -100
assistant answer    labels = token_id
padding             labels = -100
```

这样模型仍然能通过 `input_ids` 看到完整上下文，但只学习 assistant 应该如何输出 `<tool_call>...</tool_call>` 或 `<answer>...</answer>`。

## 为什么不能让整段 chat 都参与 loss？

如果整段 chat 都参与 loss，模型会被训练去复现：

- system prompt
- user 问题
- `<tool_response>` 检索证据
- assistant 工具调用和答案

这会让模型更容易复读用户问题或复读检索结果。对 Agent SFT 来说，正确目标应该是：

```text
给定 system/user/tool_response 上下文，学习 assistant 下一步应该输出什么。
```

因此本项目在 `training\sft_label_mask.py` 中只保留 assistant 内容作为 labels，其它位置全部置为 `-100`。

## 三个数组各自负责什么？

```text
input_ids        模型看什么
attention_mask   哪些输入是真的，哪些是 padding
labels           哪些位置要计算训练 loss
```

它们共同组成最终给训练器的一条样本：

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
    ]
}
```
