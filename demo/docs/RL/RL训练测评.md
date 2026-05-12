# RL 训练测评

GRPO 后仍使用主线 Agent loop 测评：同一套 Qwen3 tools schema、项目 canonical Qwen3 ReAct renderer、无 assistant 起始 anchor、无 strict protocol constraints。测评目标不是证明“能跑”，而是确认 RL 没有破坏 SFT v4 已学稳的协议，并观察答案质量和检索覆盖是否改善。

## 测评口径

主线入口：

```text
scripts/eval_hf_agentic.py
```

主报告必须保持：

```text
--assistant-start-anchor none
--protocol-constraints none
```

`--assistant-start-anchor think` 和 `--protocol-constraints strict` 只能作为诊断或生产保护对照，不能替代主结果。

## SFT v4 对照基线

当前 RL 起点是 SFT v4 merged model；SFT v4 checkpoint-3633 在 50 条 held-out 主测评中的基线为：

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

GRPO 后 checkpoint 不应明显低于这组协议指标；否则优先回查 reward、tool response 长度、stop 条件和工具 schema 漂移。

## Smoke 测评

先对候选 GRPO checkpoint 或 merged 目录跑 5 条 smoke：

```powershell
uv run python `
  ./scripts/eval_hf_agentic.py `
  --model ./training/outputs/grpo_tool_agent_react_v4/<candidate-checkpoint-or-merged> `
  --data ./data/novel_eval/test.jsonl `
  --output ./results/grpo_tool_agent/react_v4_grpo_smoke_5.jsonl `
  --max-samples 5
```

继续正式测评前，smoke 至少满足：

```text
starts_with_closing_tool_rate = 0.0
think_tag_rate >= 0.90
valid_tool_call_rate >= 0.80
answer_tag_rate >= 0.80
```

如果 5 条 smoke 已经出现首轮 `</tool_call>`、裸查询 `<tool_call>` 或缺 `<answer>`，不要扩大测评；先排查训练输出和 reward。

## 50 条正式测评

```powershell
uv run python `
  ./scripts/eval_hf_agentic.py `
  --model ./training/outputs/grpo_tool_agent_react_v4/<candidate-checkpoint-or-merged> `
  --data ./data/novel_eval/test.jsonl `
  --output ./results/grpo_tool_agent/react_v4_grpo_50.jsonl `
  --max-samples 50
```

输出约定：

```text
results/grpo_tool_agent/react_v4_grpo_50.jsonl
results/grpo_tool_agent/react_v4_grpo_50_summary.json
```

## 验收门槛

| 指标 | 要求 |
| --- | ---: |
| `think_tag_rate` | >= 0.95 |
| `valid_tool_call_rate` | >= 0.95 |
| `answer_tag_rate` | >= 0.95 |
| `starts_with_closing_tool_rate` | 0.0 |
| `malformed_tool_fragment_rate` | <= 0.05 |
| `avg_f1` | 不应明显低于 SFT v4 的 0.8433 |
| `avg_hop_recall` | 不应明显低于 SFT v4 的 0.75 |

如果质量指标下降但协议指标稳定，优先分析低分样本的检索召回、答案别名和证据充分性；不要先改 ReAct 协议。

## 结果审查

先查看 summary：

```powershell
uv run python `
  -m json.tool `
  ./results/grpo_tool_agent/react_v4_grpo_50_summary.json
```

再抽查原始 turn：

```powershell
$code = @'
from pathlib import Path
lines = Path("./results/grpo_tool_agent/react_v4_grpo_50.jsonl").read_text(encoding="utf-8").splitlines()
print("\n".join(lines[:3]))
'@
uv run python -c $code
```

重点检查：

1. `raw_turns` 中每个 assistant turn 是否只有一个 action。
2. 工具轮是否先输出短 `<think>`，再输出 JSON `<tool_call>`。
3. `<tool_call>` 的 `name` 是否只属于 `keyword_search/dense_search/hybrid_search`。
4. tool response 是否以 `[chunk_id] text` 进入历史。
5. `retrieved_chunk_ids` 是否覆盖 `gold_chunks`。
6. 最终答案是否只通过 `<answer>...</answer>` 输出。

## LLM Judge

格式和基本指标稳定后，可以用 LLM-as-Judge 辅助判断 correctness、faithfulness 和 context precision：

```powershell
uv run python `
  ./scripts/run_llm_judge.py `
  ./results/grpo_tool_agent/react_v4_grpo_50.jsonl `
  --output ./results/grpo_tool_agent/react_v4_grpo_50_judged.json `
  --llm-provider common `
  --judge-model gpt-5.5 `
  --max-concurrency 5
```

Judge 结果是诊断辅助；主线验收仍以 `eval_hf_agentic.py` 的 summary 为准。
