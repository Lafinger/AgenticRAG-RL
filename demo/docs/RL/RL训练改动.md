# GRPO 训练改动

本文记录 demo 工程 GRPO 主线从旧 TRL/Unsloth 单轮入口，切换到 example 风格 `verl + vLLM + Ray + tool_agent + multi-turn tools` 的改动。这里的 GRPO 指 SFT v4 之后的强化学习阶段，不改变 SFT v4 数据、checkpoint 和 merged model 的有效性。

相关操作文档：

| 文档 | 内容 |
| --- | --- |
| `RL数据流.md` | GRPO parquet 生成、schema 和 tool-agent 消费方式 |
| `RL训练&观测.md` | retrieval server、verl 启动入口、核心参数和日志观测 |
| `RL训练测评.md` | GRPO 后 Agent loop smoke / 50 条主测评 |
| `RL训练样本长度计算.md` | rollout、tool response、batch 和上下文长度预算 |
| `RL训练追溯.md` | 当前可用的日志、raw turns、reward 和检索追溯口径 |
| `RL_资源推算.md` | GRPO 显存、rollout token 和多卡资源估算 |

## 当前结论

当前严格复刻 example 的 GRPO 主线是：

```text
SFT v4 merged model
-> grpo_agentic_train.parquet / grpo_agentic_val.parquet
-> retrieval_server.py
-> verl main_ppo
-> actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
-> actor_rollout_ref.rollout.multi_turn.enable=True
-> novel BaseTool 调用 /search
-> reward_agentic_rag.compute_score
```

旧入口 `scripts/train_grpo_unsloth.py` 只保留为历史 TRL/Unsloth 实验入口，不再代表当前严格复刻 example 的 multi-turn tool-agent GRPO 主线。

正式 GRPO 推荐在 Linux/WSL/远程 GPU 环境运行；Windows 侧负责数据准备、检索服务、单测和小样本 smoke。

## 版本演进

| 阶段 | 训练入口 | Rollout 形态 | 主要问题 | 当前结论 |
| --- | --- | --- | --- | --- |
| 旧 GRPO | `scripts/train_grpo_unsloth.py` | TRL/Unsloth `GRPOTrainer`，不在 rollout 中真实执行工具 | 与 example 的 tool-agent 多轮检索不等价 | 保留为历史实验入口 |
| 当前 GRPO | `training/start_grpo_tool_agent.sh` | verl `tool_agent` + `multi_turn.enable=True` + `format=hermes` | 需要 Linux/WSL/远程 GPU 和 verl/vLLM/Ray 环境 | 当前严格复刻主线 |

## 起点模型

GRPO 初始模型固定使用已通过主测评的 SFT v4 merged model：

```text
models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged
```

该模型来自：

```text
training/outputs/unsloth_sft_qwen3_4b_lora_react_v4/checkpoint-3633
```

SFT v4 的 50 条无锚点主测评指标为：

| 指标 | 当前值 |
| --- | ---: |
| `avg_em` | 0.84 |
| `avg_f1` | 0.8433 |
| `avg_hop_recall` | 0.75 |
| `answer_tag_rate` | 1.0 |
| `valid_tool_call_rate` | 1.0 |
| `think_tag_rate` | 1.0 |
| `starts_with_closing_tool_rate` | 0.0 |
| `malformed_tool_fragment_rate` | 0.0 |

GRPO 后的 checkpoint 不应让这些协议指标退化。

## 数据改动

GRPO parquet 不再使用旧的 `prompt/tools/agent_name/reward_model` 简化结构，而是对齐 example 的 tool-agent schema：

```text
data_source / prompt / ability / reward_model / extra_info / metadata
```

生成命令：

```powershell
uv run python `
  ./scripts/prepare_agentic_grpo_data.py `
  --input ./data/novel_eval/qa_pairs.jsonl `
  --train-output ./data/novel_eval/grpo_agentic_train.parquet `
  --val-output ./data/novel_eval/grpo_agentic_val.parquet `
  --val-ratio 0.1 `
  --seed 42
```

当前数据规模：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data/novel_eval/grpo_agentic_train.parquet` | 2700 | verl GRPO train rollout |
| `data/novel_eval/grpo_agentic_val.parquet` | 300 | verl GRPO validation rollout |

字段约束：

| 字段 | 约束 |
| --- | --- |
| `data_source` | 固定 `novel_agentic_rag` |
| `prompt` | 只包含 system 和 user question，不泄漏 oracle evidence 或 gold chunks |
| `ability` | 固定 `multi_hop_qa` |
| `reward_model.ground_truth` | 包含 `target/answer/question/answer_aliases/gold_chunks/hop_count` |
| `extra_info.need_tools_kwargs` | 固定 `true` |
| `extra_info.tools_kwargs` | 给 `keyword_search/dense_search/hybrid_search` 注入 `question/ground_truth/data_source` |
| `metadata.tool_names` | 固定 `keyword_search/dense_search/hybrid_search` |

工具 schema 不写入 parquet，由 verl 的 tool config 加载，避免 parquet 和 tool loader 各维护一份 schema。

## 工具改动

新增 verl `BaseTool` 实现：

```text
training/tools/novel_search_tool.py
```

当前主线工具：

| 工具 | class | 后端请求 |
| --- | --- | --- |
| `keyword_search` | `KeywordSearchTool` | `POST /search` with `tool=keyword_search` |
| `dense_search` | `DenseSearchTool` | `POST /search` with `tool=dense_search` |
| `hybrid_search` | `HybridSearchTool` | `POST /search` with `tool=hybrid_search` |

`graph_search` 不进入当前 canonical GRPO tool schema。索引层可以保留 graph 能力，但主训练协议只暴露 SFT v4 已学习的三个工具。

每个工具只接收：

```json
{"query": "..."}
```

工具内部请求 retrieval server：

```json
{"query": "...", "tool": "keyword_search", "top_k": 3}
```

返回给模型的 tool response 文本固定为：

```text
[chunk_id] chunk text...
```

`execute()` 的 `extra_info` 返回 `query/tool/retrieved_chunk_ids/num_results`，用于 reward 和调试。

## Tool Config

当前配置文件：

```text
training/config/novel_tool_config.yaml
```

它使用 verl `type: native` 工具类：

```text
training.tools.novel_search_tool.KeywordSearchTool
training.tools.novel_search_tool.DenseSearchTool
training.tools.novel_search_tool.HybridSearchTool
```

该配置是 GRPO rollout 的唯一工具 schema 来源。若未来改工具参数，必须同步检查：

1. SFT canonical tool schema。
2. `novel_tool_config.yaml`。
3. retrieval server `/search` 请求字段。
4. Agent loop eval parser 和 reward。

## Retrieval Server

正式 GRPO 前必须启动检索服务，并带 `--index-dir` 使用正式索引：

```powershell
uv run python `
  ./training/tools/retrieval_server.py `
  --corpus ./data/novel/corpus.jsonl `
  --index-dir ./data/novel/indexes `
  --embedding-model ./models/bge-m3 `
  --reranker-model ./models/bge-reranker-v2-m3 `
  --host 127.0.0.1 `
  --port 8790
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8790/health -UseBasicParsing
```

搜索 smoke：

```powershell
Invoke-WebRequest http://127.0.0.1:8790/search `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"query":"卖饼老者自称是哪个集市的人？","tool":"keyword_search","top_k":3}' `
  -UseBasicParsing
```

如果不传 `--index-dir`，server 会退化为 corpus-only `HybridRetriever`，只能用于本地功能 smoke，不代表正式检索质量。

## 训练入口

正式入口：

```text
training/start_grpo_tool_agent.sh
```

核心 verl 参数：

```text
algorithm.adv_estimator=grpo
data.return_raw_chat=True
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.multi_turn.tool_config_path=training/config/novel_tool_config.yaml
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
actor_rollout_ref.rollout.n=4
reward.custom_reward_function.path=training/reward_agentic_rag.py
reward.custom_reward_function.name=compute_score
```

Linux/WSL/远程 GPU 运行：

```bash
export VERL_DIR=/path/to/verl
export PROJECT_DIR=/path/to/AgenticRAG-RL/demo
bash ./training/start_grpo_tool_agent.sh
```

小样本 smoke 可覆盖参数：

```bash
ROLLOUT_N=2 TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
OUTPUT_DIR=./training/outputs/grpo_tool_agent_react_v4_smoke \
bash ./training/start_grpo_tool_agent.sh
```

## Reward 改动

当前 reward 入口仍是：

```text
training/reward_agentic_rag.py
```

关键约束：

- 只从 `<answer>...</answer>` 抽取最终答案；无标签文本不当作预测答案。
- 只把合法 JSON `<tool_call>` 计为有效工具调用。
- `<tool_response>` 和 `[chunk_id]` 用于提取证据与 `retrieved_chunk_ids`。
- 默认 reward version 为 `v9a`。
- 裸查询 `<tool_call>查询文本</tool_call>` 不给格式奖励，不做 repair。

底层 `score_response()` 的格式分也已改为只奖励合法 JSON tool call，避免 GRPO 对坏格式输出给正反馈。

## 测评与验收

GRPO 后仍使用主线 Agent loop eval，主测评不启用 anchor、strict 或 repair：

```powershell
uv run python `
  ./scripts/eval_hf_agentic.py `
  --model ./training/outputs/grpo_tool_agent_react_v4/<candidate-checkpoint-or-merged> `
  --data ./data/novel_eval/test.jsonl `
  --output ./results/grpo_tool_agent/react_v4_grpo_50.jsonl `
  --max-samples 50
```

验收门槛：

| 指标 | 要求 |
| --- | ---: |
| `think_tag_rate` | >= 0.95 |
| `valid_tool_call_rate` | >= 0.95 |
| `answer_tag_rate` | >= 0.95 |
| `starts_with_closing_tool_rate` | 0.0 |
| `malformed_tool_fragment_rate` | <= 0.05 |
| `avg_f1` | 不应明显低于 SFT v4 的 0.8433 |
| `avg_hop_recall` | 不应明显低于 SFT v4 的 0.75 |

如果 GRPO 后协议退化，优先检查 reward 是否给坏格式正反馈、tool response 是否过长、rollout stop 是否过早、以及 tool config 是否和 SFT schema 漂移。不能用 `--assistant-start-anchor think` 或 strict constraints 冒充主结果。

## 测试覆盖

当前新增或更新的测试覆盖：

```text
tests/test_grpo_data.py
tests/test_grpo_tool_agent.py
tests/test_reward_agentic_rag.py
tests/test_rewards.py
tests/test_traces.py
```

已通过：

```text
uv run pytest
146 passed
```

## 当前状态

已经完成：

1. GRPO parquet schema 对齐 example。
2. novel verl `BaseTool` 接入 retrieval server `/search`。
3. tool config 切为 native BaseTool class。
4. verl `main_ppo` 启动脚本补齐。
5. reward 对齐 multi-turn output，不奖励裸查询 tool call。
6. README 和 docs 同步当前主线。

未在 Windows 本机执行完整 verl GRPO 训练。原因是正式训练依赖 Linux/WSL/远程 GPU 的 verl/vLLM/Ray 环境；Windows 当前只作为数据、检索服务、单测和 smoke 准备环境。
