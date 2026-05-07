# sft_agentic_eval.jsonl 审核报告

生成日期：2026-05-07

## 1. 审核依据

本报告依据 `demo/docs/训练测评.md` 中 SFT Agent loop 评测约定进行审核。Agent loop 评测要求模型在每轮生成一个合法 `<tool_call>` 或最终 `<answer>`；脚本解析合法 JSON 工具调用，调用本地检索器，并把检索结果作为 `<tool_response>` 追加回消息历史。

因此，本报告不按 direct-answer 口径审查“禁止工具调用”，而是重点检查工具调用是否合法、检索是否命中证据、是否最终收敛到 `<answer>...</answer>`，以及 `avg_em` / `avg_f1` / `avg_hop_recall` / `answer_tag_rate` / `valid_tool_call_rate` 等指标。

## 2. 转换结果与输入字段

| 项目 | 值 |
| --- | --- |
| 源文件 | `demo/results/sft_compare/sft_agentic_eval.json` |
| JSONL 输出 | `demo/results/sft_compare/sft_agentic_eval.jsonl` |
| summary 输出 | `demo/results/sft_compare/sft_agentic_eval_summary.json` |
| 审核报告 | `demo/results/sft_compare/审核sft_agentic_eval.md` |
| 样本数 | 50 |
| 模型路径 | `.\models\Qwen3-4B-Instruct-2507-Unsloth-SFT-merged` |
| 评测模式 | HF Agent loop / SFT Agent loop |
| template | `qwen3_nothink` |
| max_turns | 5 |
| per_turn_max_new_tokens | 256 |
| temperature | 0 |
| top_k | 3 |

核心字段说明：

| 字段 | 含义 | 审核关注点 |
| --- | --- | --- |
| `prediction` | 从最终 `<answer>` 抽取的答案 | 是否完成最终回答 |
| `raw_turns` | 每轮模型原始输出、解析状态和检索结果 | 是否每轮只生成合法工具调用或最终答案 |
| `tool_calls` / `valid_tool_call_count` | 已解析出的合法工具调用及数量 | 工具协议有效性与查询质量 |
| `retrieved_chunk_ids` | Agent loop 检索到的 chunk | 是否实际调用检索并获得证据 |
| `gold_chunks` / `evidence` | 标注证据 | `hop_recall` 的判断基础 |
| `em` / `f1` / `hop_recall` | 最终答案和证据召回指标 | 闭环问答质量 |
| `status` | 样本最终状态 | 失败原因定位 |

## 3. 指标汇总

| 指标 | 数值 | 结论 |
| --- | ---: | --- |
| 样本数 | 50 | 覆盖本次 SFT Agent loop 小样本评测 |
| `avg_em` | 0.000000 | 没有任何样本精确命中最终答案 |
| `avg_f1` | 0.000000 | 最终答案为空，Token F1 为 0 |
| `avg_hop_recall` | 0.516667 | 有一定证据召回能力，但未转化为最终答案 |
| `answer_tag_rate` | 0.000000 | 0/50 输出最终 `<answer>` |
| `valid_tool_call_rate` | 0.980000 | 49/50 至少产生过一次合法工具调用 |
| 非空 `prediction` 数 | 0 | 没有样本形成可抽取最终答案 |
| raw turn 含 `<answer>` 样本数 | 0 | 原始多轮输出中也未见答案标签 |
| 有检索结果样本数 | 49 | 大多数样本完成过本地检索 |
| 合法工具调用总数 | 168 | 平均每条约 3.36 次合法工具调用 |
| 平均轮数 | 3.82 | 许多样本消耗多轮后仍未回答 |
| 最大轮数 | 5 | 与配置 `max_turns=5` 一致 |
| 平均原始生成字符数 | 3100.32 | 输出很长，但主要消耗在重复/残缺工具标签和查询上 |

## 4. 状态分布

| status | 数量 | 占比 | 平均 hop_recall | 平均合法工具调用数 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `max_turns_exceeded` | 27 | 54.00% | 0.5926 | 5 | 多轮检索后仍没有最终答案 |
| `invalid_tool_call_json: Expecting value` | 17 | 34.00% | 0.4902 | 1.65 | 后续轮次退化为非法或残缺工具调用 |
| `missing_tool_call` | 6 | 12.00% | 0.25 | 0.83 | 后续轮次既无合法工具调用也无最终答案 |

## 5. 典型样例

| index | gold | status | 现象 | 判断 |
| ---: | --- | --- | --- | --- |
| 1 | 段誉 | `missing_tool_call` | 首轮开头出现残缺 `</tool_call>`，随后虽然解析出一次 `keyword_search`，第二轮退化为大量裸 `<tool_call>` 标签。 | 工具调用链不能稳定延续，且没有最终 `<answer>`。 |
| 2 | 冰火岛 | `max_turns_exceeded` | 产生合法 `keyword_search`，但连续 5 轮仍只继续检索，最终 `max_turns_exceeded`。 | 会调用工具，但不会停止并整合答案。 |
| 3 | 段誉 | `max_turns_exceeded` | 达到 5 轮，查询围绕“保定帝/段誉/皇位”等线索反复展开。 | 多轮检索未形成最终回答。 |
| 4 | 小丐 | `max_turns_exceeded` | 达到 5 轮，输出以多个工具调用片段为主。 | 工具格式和停止决策都不稳定。 |
| 5 | 义兄弟 | `max_turns_exceeded` | 达到 5 轮，围绕石破天关系持续检索。 | 检索闭环未收敛。 |
| 28 | 父子 | `invalid_tool_call_json: Expecting value` | 两次合法查询“萧峰与萧远山是什么关系”，触及“父子”答案线索，但第 3 轮退化为裸 `<tool_call>` / `</tool_call>`。 | 有语义接近的检索行为，但没有最终答案标签，EM/F1 仍为 0。 |

## 6. 逐条审核

| index | gold | status | turns | prediction | valid_tool_calls | hop_recall | 主要问题 |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- |
| 1 | 段誉 | `missing_tool_call` | 2 | 无 | 1 | 0 | 已有检索后输出不含可解析工具调用，也无 `<answer>` |
| 2 | 冰火岛 | `max_turns_exceeded` | 5 | 无 | 5 | 0.6667 | 达到最大轮数仍未输出 `<answer>` |
| 3 | 段誉 | `max_turns_exceeded` | 5 | 无 | 5 | 0 | 达到最大轮数仍未输出 `<answer>` |
| 4 | 小丐 | `max_turns_exceeded` | 5 | 无 | 5 | 0.3333 | 达到最大轮数仍未输出 `<answer>` |
| 5 | 义兄弟 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 6 | 黄伯流 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 7 | 崆峒派 | `max_turns_exceeded` | 5 | 无 | 5 | 0.3333 | 达到最大轮数仍未输出 `<answer>` |
| 8 | 长剑倒挑 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 9 | 七天 | `max_turns_exceeded` | 5 | 无 | 5 | 0.6667 | 达到最大轮数仍未输出 `<answer>` |
| 10 | 珠花 | `max_turns_exceeded` | 5 | 无 | 5 | 0.3333 | 达到最大轮数仍未输出 `<answer>` |
| 11 | 光明顶 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 12 | 曾阿牛 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 13 | 段正明 | `max_turns_exceeded` | 5 | 无 | 5 | 0.6667 | 达到最大轮数仍未输出 `<answer>` |
| 14 | 屠龙宝刀 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 15 | 丁珰 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 16 | 金花婆婆 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 17 | 父母与儿子 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 18 | 少林寺 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 19 | 师叔 | `missing_tool_call` | 2 | 无 | 1 | 0.3333 | 已有检索后输出不含可解析工具调用，也无 `<answer>` |
| 20 | 大好人 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 21 | 金元宝 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.6667 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 22 | 灵蛇岛 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 1 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 23 | 灵蛇岛 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0.6667 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 24 | 白眉鹰王殷天正 | `max_turns_exceeded` | 5 | 无 | 5 | 0.6667 | 达到最大轮数仍未输出 `<answer>` |
| 25 | 侠客岛 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 26 | 纪晓芙 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 27 | 蝴蝶谷 | `invalid_tool_call_json: Expecting value` | 4 | 无 | 3 | 1 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 28 | 父子 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0 | 查询触及萧峰/萧远山关系，但第 3 轮退化为裸工具标签且无最终答案 |
| 29 | 徒儿 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 30 | 屠龙宝刀 | `missing_tool_call` | 2 | 无 | 1 | 0 | 已有检索后输出不含可解析工具调用，也无 `<answer>` |
| 31 | 蜜糖 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.6667 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 32 | 自毁容貌 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 33 | 到张三丰坐关的门外磕头 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 34 | 朱长龄自己房中 | `missing_tool_call` | 2 | 无 | 1 | 0.6667 | 已有检索后输出不含可解析工具调用，也无 `<answer>` |
| 35 | 钟姑娘 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 36 | 张无忌 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 37 | 白自在 | `missing_tool_call` | 1 | 无 | 0 | 0 | 未产出可解析工具调用，也无 `<answer>` |
| 38 | 探那大汉的深浅虚实 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 39 | 圆真 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 40 | 静虚 | `max_turns_exceeded` | 5 | 无 | 5 | 0.3333 | 达到最大轮数仍未输出 `<answer>` |
| 41 | 俞莲舟 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0.6667 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 42 | 第一座石室 | `max_turns_exceeded` | 5 | 无 | 5 | 0.6667 | 达到最大轮数仍未输出 `<answer>` |
| 43 | 张无忌 | `missing_tool_call` | 2 | 无 | 1 | 0.5 | 已有检索后输出不含可解析工具调用，也无 `<answer>` |
| 44 | 后艄 | `invalid_tool_call_json: Expecting value` | 2 | 无 | 1 | 0.3333 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 45 | 谢逊 | `max_turns_exceeded` | 5 | 无 | 5 | 0.3333 | 达到最大轮数仍未输出 `<answer>` |
| 46 | 赵敏 | `max_turns_exceeded` | 5 | 无 | 5 | 0.5 | 达到最大轮数仍未输出 `<answer>` |
| 47 | 屠龙刀 | `invalid_tool_call_json: Expecting value` | 4 | 无 | 3 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 48 | 大理 | `max_turns_exceeded` | 5 | 无 | 5 | 1 | 达到最大轮数仍未输出 `<answer>` |
| 49 | 赵敏 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0.5 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |
| 50 | 常金鹏 | `invalid_tool_call_json: Expecting value` | 3 | 无 | 2 | 0.3333 | 后续轮次生成非法/残缺工具调用 JSON，未输出 `<answer>` |

## 7. 审核结论

该 `sft_agentic_eval.jsonl` 显示当前 Agentic SFT 模型在 Agent loop 下不是完全失效：49/50 样本至少产生过一次合法工具调用，49/50 样本拿到了检索结果，`avg_hop_recall≈0.5167` 表明检索证据召回有一定基础。

但本次 Agent loop 闭环仍判定为不合格。核心原因是模型没有稳定完成“检索后给最终答案”的最后一步：50 条样本全部没有 `<answer>...</answer>`，`prediction` 全为空，导致 `avg_em=0`、`avg_f1=0`。失败模式主要集中在两类：一类是持续发起工具调用直到 `max_turns_exceeded`；另一类是在后续轮次输出裸 `<tool_call>`、残缺 JSON 或不可解析标签。

因此，这份结果不能作为“Agentic 闭环能力达标”的证据。它更准确地说明：模型已经学到了一部分工具调用触发和查询生成行为，但缺少可靠的停止条件、工具响应吸收能力和最终答案格式收敛。

## 8. 建议

1. 优先补强 Agent trace 中“检索后停止并输出 `<answer>`”的监督比例，尤其是最后一轮 assistant 只输出 `<answer>最终答案</answer>` 的样本。
2. 检查 `qwen3_nothink` chat template、停止词和 per-turn 截断逻辑，重点排查为什么输出经常从 `</tool_call>` 起始，以及为什么会生成多段工具调用。
3. 在 Agent loop 解码侧增加更严格的单轮输出约束或诊断：每轮只接受第一个完整 `<tool_call>...</tool_call>` 或 `<answer>...</answer>`，并记录被截断/丢弃的后续片段。
4. 训练或评测前做 3-5 条手工 smoke：要求模型先合法调用一次检索，再基于 `<tool_response>` 输出最终 `<answer>`，确认不再持续重复工具标签。
5. 后续报告中把 Agent loop 指标与 direct-answer 指标分开呈现：本文件可以说明工具调用和检索召回问题，不能替代 direct-answer 最终答案能力评估。

## 9. 复核清单

- [x] 已将旧 `sft_agentic_eval.json` 转换为逐样本 `sft_agentic_eval.jsonl`。
- [x] 已将旧顶层 `summary` 写入 `sft_agentic_eval_summary.json`。
- [x] 已确认 JSONL 样本数为 50。
- [x] 已逐条审核 50 条样本的 `prediction`、`status`、`raw_turns`、合法工具调用数和 `hop_recall`。
- [x] 已在报告中区分“工具调用触发率较高”和“最终答案闭环失败”。
- [x] 已保留源文件 `sft_agentic_eval.json` 不删除。
