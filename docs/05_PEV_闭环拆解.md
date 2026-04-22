# PEV 闭环拆解

这一部分是原项目和普通 RAG 工程真正拉开差距的地方。

## 状态结构

最关键的状态不是 `answer`，而是：

- `plan`
- `current_step`
- `evidence`
- `tool_calls`
- `verification_result`
- `verification_feedback`
- `iteration_count`
- `trace`

这也是为什么原项目用 LangGraph 合适：它天然适合描述状态流与条件边。

## 节点职责

## Planner

负责把最终问题拆成可执行的子查询，并决定：

- 用什么工具
- 先搜什么
- 哪些步骤有依赖关系

## Executor

负责执行工具调用，把结果转成 `evidence`。

## Verifier

负责判断当前证据是否足够。如果不够，不只是给“不够”，还要说明“不够在哪里”。

## Synthesizer

负责基于已有证据输出最终答案。

## 最值得学的工程结论

不要把 Verifier 理解成“多一个判断节点”。它真正有价值的前提是：

- Verifier 的反馈足够具体
- Planner 能根据反馈改搜索策略
- 下一轮 evidence 真能带来新信息

如果做不到这三点，多轮 Agent 只会变成重复调用。

## 本目录里的最小实现

`src/agentic_rag_study/pev_graph.py` 里放的是规则化版本：

- 不依赖外部大模型
- 用 LangGraph 组织状态和条件跳转
- 用手工规则模拟 `replan feedback`

它不是为了追求效果，而是为了让你先把“状态是怎么流”的问题看明白。

