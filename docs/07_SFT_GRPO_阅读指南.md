# SFT / GRPO 阅读指南

这个阶段先读，不急着训。

## 先看什么

### 1. SFT 样本从哪来

核心问题不是“格式长什么样”，而是“监督轨迹是谁提供的”。

原文选择 `Oracle trajectory`，原因很直接：

- 大模型 rollout 质量不稳定
- 多跳轨迹要求高精度
- 训练前先保证格式和路径质量更重要

### 2. prompt / template 对齐为什么重要

原文明确提到：SFT 和 GRPO 如果 prompt 模板不一致，模型在 RL 阶段会发生明显行为漂移。

你之后读训练代码时，要重点盯这件事：

- system message 里是否包含 tools 描述
- SFT / rollout / eval 三处模板是否一致

### 3. reward 为什么会带偏

如果 reward 只关心答案 token 重合，模型可能直接猜答案，不检索。

这是最经典的 reward hacking。

### 4. 为什么两阶段课程学习有效

因为：

- `faithfulness` 是基础行为
- `correctness` 是更高层目标

先学前者，能把策略空间限制在“基于证据回答”的区域。

## 你现阶段应达到的训练认知

你应该能解释下面四件事：

1. 什么是 `Oracle trajectory`
2. 什么是 reward hacking
3. 为什么要先 grounding 后 correctness
4. 为什么模板不对齐会让 RL 阶段退化

## 等资源升级后再做什么

等你后续真要上多卡 GPU，再把阅读重点扩展到：

- Qwen3 SFT 数据格式
- checkpoint 选择
- `verl` rollout 机制
- GRPO 的 group 内相对优势估计
- reward 组合与信号密度

