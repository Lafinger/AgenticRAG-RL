# sft_predictions.jsonl 审核报告

生成日期：2026-05-07

## 1. 审核依据

本报告依据 `demo/README.md` 与 `demo/docs/训练&观测.md` 中对 SFT 训练后测评的约定进行审核。当前文件来自 `eval_hf_model.py --prompt-mode direct_answer` 的直接答案评测链路，该模式的系统提示要求：只输出 `<answer>最终答案</answer>`，不要输出 `<tool_call>`、分析过程或额外文字。

README 中 Step 14 的诊断指标强调同时观察答案指标、协议遵循和生成行为，其中 `answer_tag_rate` 用于判断是否稳定包含 `<answer>...</answer>`，`tool_call_rate` 与 `valid_tool_call_rate` 用于观察工具调用行为。对于 direct-answer 评测，合理预期是：答案标签稳定出现，工具调用标签不出现，最终答案能被抽取并参与 EM/F1 计算。

## 2. 输入与字段

| 项目 | 值 |
| --- | --- |
| 审核文件 | `demo/results/sft_compare/sft_predictions.jsonl` |
| 样本数 | 50 |
| 模型路径 | `.\models\Qwen3-4B-Instruct-2507-Unsloth-SFT-merged` |
| adapter | `null` |
| template | `qwen3_nothink` |
| prompt_mode | `direct_answer` |
| max_new_tokens | 128 |
| temperature | 0.0 |
| 输出报告 | `demo/results/sft_compare/审核sft_predictions.md` |

核心字段说明：

| 字段 | 含义 | 审核关注点 |
| --- | --- | --- |
| `prediction` | 从 `<answer>...</answer>` 抽取的答案；若无答案标签则退回原始生成文本 | 是否是真正的最终答案 |
| `raw_prediction` | 模型原始生成文本 | 是否违反 direct-answer 协议 |
| `em` / `f1` | 与 gold/aliases 的答案匹配指标 | correctness 趋势 |
| `answer_tag_present` | 原始输出是否包含 `<answer>` 标签 | 协议遵循 |
| `tool_call_count` | 成对 `<tool_call>...</tool_call>` 的数量 | 是否生成工具调用 |
| `valid_tool_call_count` | 可解析 JSON 的有效工具调用数量 | 工具协议有效性 |

## 3. 指标汇总

| 指标 | 数值 | 结论 |
| --- | ---: | --- |
| 样本数 | 50 | 覆盖本次 SFT direct-answer 小样本评测 |
| `avg_em` | 0.000000 | 没有任何样本精确命中 |
| `avg_f1` | 0.004848 | 几乎无有效答案重叠 |
| `answer_tag_rate` | 0.000000 | 0/50 输出包含 `<answer>...</answer>` |
| 记录的 `tool_call_rate` | 0.000000 | 评测脚本没有识别到成对工具调用 |
| `valid_tool_call_rate` | 0.000000 | 没有有效 JSON 工具调用 |
| 实际含 `<tool_call>`/`</tool_call>` 片段比例 | 1.000000 | 50/50 原始输出都含残缺工具标签片段 |
| `avg_generation_chars` | 51.50 | 平均生成很短，多数不是答案文本 |
| `max_generation_chars` | 831 | 个别样本出现长串重复 `<tool_call>` |

需要特别注意：`tool_call_rate=0.0` 不代表模型没有输出工具相关内容，而是因为脚本只统计成对的 `<tool_call>...</tool_call>`。本文件的主要问题是大量生成孤立、残缺或重复的 `<tool_call>` / `</tool_call>` 片段，因此没有被计入有效工具调用，但仍然严重违反 direct-answer 评测协议。

## 4. 典型样例

| index | gold | prediction / raw 现象 | EM | F1 | 判断 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 段誉 | `</tool_call>\n\n<tool_call>` | 0.0 | 0.0 | 没有 `<answer>`，只输出残缺工具标签 |
| 2 | 冰火岛 | `</tool_call>\n\n<tool_call>` | 0.0 | 0.0 | 与问题无关，无法抽取答案 |
| 3 | 段誉 | `</tool_call>\n\n<tool_call>` | 0.0 | 0.0 | 同类协议崩坏 |
| 9 | 七天 | 长串重复 `</tool_call>` / `<tool_call>` | 0.0 | 0.0 | 存在标签循环或停止条件异常迹象 |
| 23 | 灵蛇岛 | 长串重复 `</tool_call>` / `<tool_call>` | 0.0 | 0.0 | 同类重复输出问题 |
| 28 | 父子 | 包含 `<quote>殷天正与殷野王是父子关系。</quote>`，但无 `<answer>` 且末尾仍有 `<tool_call>` | 0.0 | 0.2424 | 有局部语义重叠，但没有按协议给最终答案 |
| 50 | 常金鹏 | `岑夫人的丈夫是谁？\n</tool_call>\n\n<tool_call>` | 0.0 | 0.0 | 复述/变形问题并混入残缺工具标签 |

## 5. 审核结论

该 `sft_predictions.jsonl` 不满足 SFT direct-answer 评测要求，结论为不合格。

主要失败不是普通的知识答案偏差，而是协议遵循崩坏：50 条样本全部没有 `<answer>...</answer>`，同时 50 条样本全部含有 `<tool_call>` 或 `</tool_call>` 片段。模型在 direct-answer 提示下没有稳定输出最终答案，反而倾向输出残缺工具调用标签，导致评测脚本无法抽取答案，EM 为 0，F1 也几乎为 0。

从现象看，当前 SFT merged model 不适合作为 direct-answer 能力结果汇报。若直接把该文件纳入 SFT 效果结论，会把格式崩坏和答案能力混在一起，无法说明模型是否真正学会了小说多跳问答。

## 6. 关于 base_predictions 的对照限制

同目录存在 `base_predictions.jsonl` 和 `base_predictions_as_eval.json`，但不建议把它们作为本次 SFT 文件的正式对照依据。原因是 base 文件中的问题示例来自《平凡的世界》相关语料，而当前 `sft_predictions.jsonl` 的问题主要来自金庸小说语料，gold chunk ID 体系也不同。直接比较 base 与 SFT 的 EM/F1、answer tag 或生成长度，会引入数据域不一致，容易得出误导性结论。

因此，本报告只把 `sft_predictions.jsonl` 自身的协议遵循和答案指标作为正式审核对象。

## 7. 建议

1. 优先检查 SFT 导出数据和训练配置，确认训练样本是否仍以 agent/tool-call trace 为主，而当前 direct-answer 评测却要求禁用工具调用。如果训练目标主要是工具轨迹学习，direct-answer 评测需要补充对应格式样本或单独做格式约束。
2. 检查 tokenizer chat template、`qwen3_nothink`、停止词/生成截断、LoRA 合并路径，重点排查为什么生成经常从 `</tool_call>` 起始，以及为什么没有生成 `<answer>`。
3. 重新跑完整评测前，先用 3-5 条手工 prompt 做最小验证：要求模型只输出 `<answer>...</answer>`，确认不会再输出 `<tool_call>`、分析过程或裸标签片段。
4. 如果目标是评估工具调用能力，应改用 Agent loop 评测，例如 `eval_hf_agentic.py`，并观察真实工具调用、检索闭环和 `hop_recall`。如果目标是评估最终答案能力，应继续使用 direct-answer，但需要确保训练数据和提示模板支持该输出形态。
5. 重新生成评测结果后，应补齐同一数据集上的 base/SFT 对照 `summary.json`，避免跨语料、跨 chunk ID 体系比较。

## 8. 复核清单

- [x] 已读取 `demo/README.md` 与 `demo/docs/训练&观测.md` 的训练、测评和协议说明。
- [x] 已审核 `demo/results/sft_compare/sft_predictions.jsonl` 的 50 条记录。
- [x] 已确认 50 条记录均无 `<answer>...</answer>`。
- [x] 已确认 50 条记录均含 `<tool_call>` 或 `</tool_call>` 片段。
- [x] 已在报告中区分“评测脚本记录的成对工具调用为 0”和“原始输出实际含残缺工具标签为 100%”。
- [x] 已说明 `base_predictions*` 不作为正式对照依据。