# 垂直领域多跳 Agentic RAG & RL 复现工程

这个目录是 `9.2.1 垂直领域多跳 Agentic RAG & RL 简历项目.pdf` 的工程化复现版本，目标不是只保留一个最小学习样例，而是把完整链路拆成可运行、可扩展、可迁移到外部 GPU 的项目结构。

## 复现范围

- 金融 PDF 解析与 chunk 切分
- `BM25 + dense + rerank` 轻量检索骨架
- 多跳 seed QA 与合成样本生成
- Oracle traces、SFT 数据转换、LLaMA-Factory 兼容导出
- Agentic GRPO 数据准备、reward 计算、retrieval server
- Pipeline/Agentic 评测、Judge 打分、hop-aware 诊断
- Windows 11 本机 smoke test
- 外部 Linux / WSL / A100 环境的 Stage1 / Stage2 / Stage3 启动脚本

## 目录

```text
demo/
├── bootstrap/
├── data/
├── docs/
├── requirements.txt
├── requirements-verl.txt
├── scripts/
├── src/
├── tests/
└── training/
```

## 本机快速开始

### 1. 创建虚拟环境

```powershell
Set-Location C:\Workspace\AI\Learning\AgenticRAG-RL\demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

### 2. 运行测试

```powershell
python -m pytest
```

### 3. 运行 smoke 检索服务

```powershell
python .\training\tools\retrieval_server.py --port 8790 --corpus .\data\smoke_financial\corpus.jsonl
```

### 4. 运行 smoke agentic 评测

```powershell
python .\scripts\eval_agentic.py `
  --data .\data\smoke_financial\qa_pairs.jsonl `
  --corpus .\data\smoke_financial\corpus.jsonl `
  --max-samples 2
```

## 推荐执行顺序

1. `scripts\parse_pdf_corpus.py`
2. `scripts\build_index.py`
3. `scripts\gen_seed_qa.py`
4. `scripts\domain_multihop_synthesis.py`
5. `scripts\clean_synthesis.py`
6. `scripts\judge_synthesis.py`
7. `scripts\split_train_test.py`
8. `scripts\gen_enhanced_aliases.py`
9. `scripts\build_oracle_traces.py`
10. `scripts\trace_to_sft.py`
11. `scripts\convert_sft_to_llamafactory.py`
12. `scripts\prepare_agentic_grpo_data.py`
13. `training\run_stage1_v11e.ps1`
14. `training\run_stage2_v14e.ps1`

## 环境说明

- Windows 11 本机仅承担数据处理、SFT 数据转换、CPU retrieval server 与 smoke 验证。
- `verl + vLLM + flash-attn` 的标准训练路径默认面向 Linux / WSL2 / 远端 GPU。
- PowerShell 启动脚本会尽量保持入口统一，但真正的 GRPO 训练建议在 A100 级 Linux 环境执行。
