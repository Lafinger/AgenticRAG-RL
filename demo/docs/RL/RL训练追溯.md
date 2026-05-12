# RL 训练追溯

当前 RL 主线没有 SFT 那种 optimizer step 到样本的精确追溯文件。SFT 的 `step_sample_trace.jsonl` 来自自定义训练入口；verl tool-agent GRPO 的训练、采样、工具调用、reward 聚合跨 Ray、vLLM 和 worker 进程，当前仓库还没有实现等价的 per-step 样本映射。

因此 RL 追溯以日志、评测 raw turns、工具返回、reward breakdown 和检索 smoke 为主。

## 可追溯对象

| 对象 | 来源 | 用途 |
| --- | --- | --- |
| 训练日志 | `logs/grpo_tool_agent_react_v4.log` | 定位启动、Ray/vLLM、保存、验证和异常 |
| checkpoint | `training/outputs/grpo_tool_agent_react_v4` | 对比不同训练阶段效果 |
| eval raw turns | `results/grpo_tool_agent/*.jsonl` | 还原模型每轮 `<think>/<tool_call>/<answer>` |
| `retrieved_chunk_ids` | eval 输出和 tool extra info | 判断检索是否覆盖 gold chunks |
| reward 组件 | `training/reward_agentic_rag.py` 与 `agentic_rag_rl.rewards` | 判断 correctness、faithfulness、hop F1、format 的贡献 |
| retrieval smoke | `/search` 响应 | 判断问题来自检索服务还是模型策略 |

## 训练失败追溯

先查日志：

```powershell
Select-String -Path .\logs\grpo_tool_agent_react_v4.log `
  -Pattern "ERROR|Traceback|OOM|CUDA|Ray|vllm|tool|reward|nan|inf" `
  -CaseSensitive:$false
```

常见定位：

| 日志现象 | 可能原因 |
| --- | --- |
| retrieval server not ready | `/health` 不通或端口不一致 |
| ModuleNotFoundError | `PYTHONPATH`、`VERL_DIR` 或 native tool class path 不正确 |
| CUDA OOM | batch、rollout.n、response length 或 KV cache 过大 |
| reward 长期为 0 | 模型没有 `<answer>`、工具格式错误或 judge/evidence 为空 |
| max turns 频繁触发 | 模型不会停止、证据不足或 answer 触发条件太弱 |

## 评测样本追溯

GRPO 后先跑 `RL训练测评.md` 中的 5 条或 50 条 Agent loop 测评。对低分样本，查看 JSONL 中这些字段：

```text
question
gold
pred
raw_turns
tool_calls
retrieved_chunk_ids
gold_chunks
hop_recall
status
```

判断顺序：

1. `raw_turns` 是否出现首轮 `</tool_call>`、裸查询 `<tool_call>` 或 malformed JSON。
2. `tool_calls[].name` 是否只使用 `keyword_search/dense_search/hybrid_search`。
3. `retrieved_chunk_ids` 是否覆盖 `gold_chunks`。
4. 如果 hop recall 低，检查同一 query 的 retrieval server smoke 是否能召回 gold chunk。
5. 如果 hop recall 高但答案错，检查最终 `<answer>` 是否抽取错、别名缺失或证据被模型误读。
6. 如果达到 max turns，检查是否反复搜索相同 query 或没有在证据足够后停止。

## Reward 追溯

当前 reward 入口：

```text
training/reward_agentic_rag.py::compute_score
```

底层 reward version 默认：

```text
AGENTIC_RAG_REWARD_VERSION=v9a
```

`v9a` 主要由以下部分组成：

| 组件 | 权重 |
| --- | ---: |
| `hop_precision_recall` | 0.30 |
| `judge_faithfulness` | 0.25 |
| `judge_correctness` | 0.25 |
| `grounded_answer` | 0.10 |
| `format` | 0.10 |

其中 `format` 只奖励 `<answer>` 和合法 JSON `<tool_call>`；裸查询 `<tool_call>查询文本</tool_call>` 不给正反馈。

如果怀疑 reward 给错信号，用已有测试样式构造一个 `solution_str`，调用：

```powershell
$code = @'
from training.reward_agentic_rag import compute_score_breakdown
solution = '<answer>侯赢</answer>'
truth = {"target": "侯赢", "answer_aliases": ["侯赢"], "gold_chunks": ["chunk-a"], "hop_count": 1}
print(compute_score_breakdown(solution, truth))
'@
uv run python -c $code
```

## 检索追溯

对低 hop recall 样本，直接复现工具请求：

```powershell
Invoke-WebRequest http://127.0.0.1:8790/search `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"query":"这里替换为 raw_turns 中的 query","tool":"hybrid_search","top_k":3}' `
  -UseBasicParsing
```

如果 retrieval server 能召回 gold chunk，但模型训练后没问到正确 query，问题在策略；如果 server 本身召回不到，优先处理索引、query 改写、`top_k` 或语料 chunk。

## 当前限制

当前仓库还不能回答“某个 GRPO optimizer step 的梯度突刺来自哪几条样本”。如果后续需要 SFT 等级的 RL 追溯，应新增 verl callback 或 worker 侧日志，把 step、prompt index、rollout response、tool extra info、reward breakdown 和 checkpoint step 写成结构化 JSONL。未实现前，不要在文档或实验报告中声称具备 step 级 RL 样本追溯。
