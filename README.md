# 垂直领域多跳 Agentic RAG & RL 学习工作台

这个目录现在不是单一 PDF 存放点，而是一套可以直接开始学习和最小复现的本地工作台。目标不是完整复刻原项目，而是按 `单卡/CPU 友好` 的方式，把你后续需要反复回看的骨架、样例、脚手架和指标都落到本地。

## 你会得到什么

- 分阶段学习文档：先吃透项目结构，再跑最小检索，再理解 PEV 闭环、评测和训练。
- 最小数据集：手工构造的金融半年报风格 `2-hop` 样本，方便你先理解多跳问题是什么。
- 最小代码骨架：覆盖 `PDF -> chunk -> BM25/dense/RRF -> PEV -> hop-aware 评测`。
- Windows 11 + PowerShell 使用路径：所有说明默认按当前环境写，不走 `cmd`。

## 推荐学习顺序

1. 先读 [docs/01_项目骨架图与数据流.md](docs/01_项目骨架图与数据流.md)
2. 再读 [docs/02_术语表.md](docs/02_术语表.md)
3. 然后读 [docs/03_实验因果链.md](docs/03_实验因果链.md)
4. 开始实操前看 [docs/04_最小复现指南.md](docs/04_最小复现指南.md)
5. Agent 层看 [docs/05_PEV_闭环拆解.md](docs/05_PEV_闭环拆解.md)
6. 评测层看 [docs/06_评测与手工_2hop_样本.md](docs/06_评测与手工_2hop_样本.md)
7. 训练层最后看 [docs/07_SFT_GRPO_阅读指南.md](docs/07_SFT_GRPO_阅读指南.md)

## 目录

```text
.
├── README.md
├── requirements.txt
├── docs/
├── data/demo_financial/
├── scripts/
├── src/agentic_rag_study/
└── tests/
```

## 快速开始

### 1. 准备环境

PowerShell 中执行：

```powershell
# 如果你的 python 命令跳到 Windows Store，请先安装 Python 3.11+，
# 或改用你自己的 Conda 环境。
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 跑最小检索示例

```powershell
python scripts/run_retrieval_demo.py
```

你会看到：

- `keyword_search` 命中人名/公司名的结果
- `dense_search` 的补充召回
- `RRF + rerank` 之后的统一排序

### 3. 跑最小 PEV 闭环

```powershell
python scripts/run_pev_demo.py
```

这个示例不会依赖外部大模型 API，而是用规则化的 `Planner / Verifier / Synthesizer` 来演示：

- `plan`
- `execute`
- `verify`
- `replan`
- `evidence` 累积

### 4. 看 hop-aware 评测

```powershell
python scripts/eval_demo.py
```

## 与原文项目的关系

这个工作台保留的是“学习主链”，不是“完整工程复刻”。已经覆盖：

- 为什么 `AgenticRAGTracer` 适合诊断多跳系统
- 为什么金融语料里 `BM25 + dense + reranker` 比单路检索稳
- 为什么 PEV 的瓶颈不在单纯多迭代，而在 `replan feedback`
- 为什么 RL 里要先学 `grounding / faithfulness`

暂时没有完整落地的内容：

- 大规模 PDF 语料清洗流水线
- 知识图谱三元组抽取与图遍历全链路
- Qwen3 SFT 与 `verl` GRPO 训练
- 外部 Judge 模型接入

## 参考资料

- AgenticRAGTracer 论文：<https://arxiv.org/abs/2602.19127>
- AgenticRAGTracer 数据集：<https://huggingface.co/datasets/YqjMartin/AgenticRAGTracer>
- LangGraph 总览：<https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph Agentic RAG：<https://docs.langchain.com/oss/python/langgraph/agentic-rag>
- BGE-M3 论文：<https://arxiv.org/abs/2402.03216>
- BGE-M3 模型卡：<https://huggingface.co/BAAI/bge-m3>
- bge-reranker-v2-m3：<https://huggingface.co/BAAI/bge-reranker-v2-m3>
- GraphRAG 说明：<https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/>
- Qwen3 Technical Report：<https://arxiv.org/abs/2505.09388>
- Qwen3-4B 模型卡：<https://huggingface.co/Qwen/Qwen3-4B>
- DeepSeekMath / GRPO：<https://arxiv.org/abs/2402.03300>
- verl 文档：<https://verl.readthedocs.io/en/latest/index.html>

