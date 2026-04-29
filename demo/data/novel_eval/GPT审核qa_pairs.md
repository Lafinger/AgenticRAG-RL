# GPT 审核 qa_pairs.jsonl
- 审核对象：`demo/data/novel_eval/qa_pairs.jsonl`
- 支撑语料：`demo/data/novel/corpus.jsonl`
- 当前 seed 对照：`demo/data/novel_eval/seeds.jsonl`
- 审核时间：2026-04-30 00:08:20
- 样本总数：200

## 审核依据
本次按 `demo/README.md` 中 “Step 4: 合成多跳 QA” 的正式数据质量门槛，以及根项目《垂直领域多跳AgenticRAG&RL项目.md》的逐跳扩展和四重验证思想进行规则化审核。

- 字段完整：必须包含 `final_question/final_answer/hop_count/qa_type/subset/hops/answer_aliases`。
- hop 结构：`hop_count >= 2`，且等于 `len(hops)`；`hop_idx` 从 1 连续递增。
- 证据可用：每个 hop 的 `doc_chunk_id` 必须存在于 corpus，且 hop answer 应能在对应 chunk 中直接定位。
- 链路约束：同一条样本内不能重复使用同一个 `doc_chunk_id`。
- 类型约束：顶层 `qa_type` 固定为 `inference`，hop 级类型只能是小说域 5 类。
- 自然性：`final_question` 不应出现“第1步/hop/逐步检索/根据文档”等合成或查表痕迹。
- 无泄露：`final_question` 不应直接包含中间 hop answer 或最终答案。
- 多跳必要性：不能退化为最后一跳单跳问题，也不能只通过作品级弱连接拼接无关事实。
- 别名约束：`answer_aliases` 应为 1-3 个短别名，不能重复，不能引入冲突事实。

说明：本报告是规则审核，不等价于完整 LLM 四重验证。语义合理性、纯推理不可答、单文档不可答、全文档可答仍建议后续用 LLM Judge/Verifier 复查。

## 总体结论
- PASS：56 / 200
- WARN：26 / 200
- FAIL：118 / 200

## hop_count 分布
| hop_count | 数量 |
| ---: | ---: |
| 2 | 101 |
| 3 | 99 |

## qa_type 分布
| qa_type | 数量 |
| --- | ---: |
| `inference` | 200 |

## 问题类型统计
| 问题类型 | 数量 |
| --- | ---: |
| FAIL: final_question 退化为最后一跳单跳问题 | 86 |
| FAIL: final_question 有合成/查表痕迹 | 61 |
| FAIL: answer_aliases 重复 | 61 |
| WARN: 未发现明确的早期 hop 非答案线索，需复核多跳必要性 | 38 |
| WARN: answer_aliases 未包含 final_answer | 27 |
| FAIL: final_question 泄露中间 hop1 answer | 24 |
| FAIL: final_question 泄露中间 hop2 answer | 23 |
| WARN: final_answer 非最后一跳 answer，也不在 aliases 中 | 7 |
| FAIL: final_question 泄露 final_answer | 7 |
| WARN: final_answer 含并列符号，需确认是否复合答案 | 3 |
| FAIL: final_question 存在弱连接/作品级连接痕迹 | 1 |
| FAIL: final_answer 超过20字 | 1 |

## 修复建议
1. 优先处理 FAIL，尤其是 `final_question 泄露中间 hop answer` 和 `final_question 存在弱连接/作品级连接痕迹`。这类样本很容易变成机械拼接或单跳题。
2. 重新生成多跳问题时，不要把前一跳答案原文塞进最终问题，应改成需要检索前一跳才能获得下一跳查询条件的自然问法。
3. 对 “不在当前 seeds.jsonl 中” 的 WARN 样本，说明它们引用了已被清洗删除或当前 seed 文件不存在的 hop，建议重新用清洗后的 seed 集生成 qa_pairs。
4. 对 PASS/WARN 样本继续做 LLM 四重验证：语义合理、纯推理不可答、单文档不可答、全文档可答。规则检查不能替代这一步。

## 逐条审核明细
| 行号 | 结论 | hop_count | gold_chunks | final_answer | final_question | 审核意见 |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | FAIL | 2 | `corpus_chunkids_000003,corpus_chunkids_000111` | 金波 | 那位旧黄胶鞋用白线绳代替鞋带系着的瘦高个青年人，在放假前的最后一个星期六借了谁的自行车送自己的破烂铺盖回家？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 2 | FAIL | 2 | `corpus_chunkids_000001,corpus_chunkids_000269` | 二斗小米 | 1999年二三月间县立高中给学生分饭菜的地点所在的叙事作品中，贺耀宗托顺车给孙少安家捎来了什么物品？ | final_question 存在弱连接/作品级连接痕迹; final_question 退化为最后一跳单跳问题; answer_aliases 未包含 final_answer |
| 3 | FAIL | 3 | `corpus_chunkids_000001,corpus_chunkids_000269,corpus_chunkids_000268` | 黄原出的羊毛毯 | 在同时记载了1999年二三月间县立高中在校园内南墙根下给学生分饭菜、贺耀宗托顺车给孙少安家捎来二斗小米这两处情节的小说中，一九七六年临近结束前… | final_question 泄露中间 hop2 answer; answer_aliases 未包含 final_answer |
| 4 | FAIL | 3 | `corpus_chunkids_000003,corpus_chunkids_000111,corpus_chunkids_000110` | 县文化馆 | 旧黄胶鞋用白线绳代替鞋带系着的瘦高个青年人，在放假前最后一个星期六借了金波的自行车送自己的破烂铺盖回家，请问这个人上学期临放假前一个星期想起郝… | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 5 | FAIL | 3 | `corpus_chunkids_000003,corpus_chunkids_000004,corpus_chunkids_000010` | 顾养民 | 和在饭场馍筐前拾了两个高粱面馍的瘦高个青年人一样，开学后每次吃饭都最后来取黑高粱面馍的女生所在班级，开展文章学习的课堂上负责念报纸的班长叫什么… | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 6 | FAIL | 2 | `corpus_chunkids_000003,corpus_chunkids_000004` | 郝红梅 | 第1步回答“来到饭场的瘦高个青年人在馍筐前拾了多少个高粱面馍？”；第2步回答“和孙少平一样开学后每次吃饭最后来取黑高粱面馍的女生叫什么名字？”… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 7 | PASS | 2 | `corpus_chunkids_000004,corpus_chunkids_000010` | 顾养民 | 和孙少平一样开学后每次吃饭最后来取黑高粱面馍的女生所在的班级，在开展文章学习的课堂上负责念报纸的班长叫什么名字？ | 通过 |
| 8 | PASS | 2 | `corpus_chunkids_000005,corpus_chunkids_000108` | 黑高粱面馍 | 孙少平读高中阶段每顿饭只能啃的食物所属的品类下，他高中第二个学期刚开学时大部分日子里啃的具体主食是什么？ | 通过 |
| 9 | FAIL | 3 | `corpus_chunkids_000004,corpus_chunkids_000010,corpus_chunkids_000009` | 侯玉英 | 《平凡的世界》中，和孙少平开学后每次吃饭最后来取黑高粱面馍的女生所在的班级里，开展文章学习的课堂上负责念报纸的班长所在的班级中，向班主任揭发孙… | final_question 退化为最后一跳单跳问题 |
| 10 | FAIL | 2 | `corpus_chunkids_000004,corpus_chunkids_000014` | 城里买的吃食 | 第1步回答“开学以来孙少平每次吃饭最后来取的饭食是什么？”；第2步回答“田润叶每次回村到孙少平家串门时，会给孙少平的祖母带什么？”。请沿着这些… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 11 | FAIL | 3 | `corpus_chunkids_000004,corpus_chunkids_000014,corpus_chunkids_000076` | 队长 | 第1步回答“开学以来孙少平每次吃饭最后来取的饭食是什么？”；第2步回答“田润叶每次回村到孙少平家串门时，会给孙少平的祖母带什么？”；第3步回答… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 12 | PASS | 3 | `corpus_chunkids_000005,corpus_chunkids_000108,corpus_chunkids_000113` | 顾养民 | 《平凡的世界》中，孙少平读高中时每顿饭只能啃同一种主食，在他高中第二个学期刚开学、日常仍食用该种主食的那段时期，他在篮球场上向郝红梅要球时，郝… | 通过 |
| 13 | WARN | 3 | `corpus_chunkids_000006,corpus_chunkids_000115,corpus_chunkids_000109` | 劳动干事 | 和孙少平高中时所在班级的班长同班，还曾当众污蔑郝红梅是孙少平"婆姨"的人所在的班级，在上学期班干部选举中孙少平被选任的职务是什么？ | answer_aliases 未包含 final_answer |
| 14 | FAIL | 2 | `corpus_chunkids_000006,corpus_chunkids_000115` | 侯玉英 | 第1步回答“孙少平就读高中时所在班级的班长是谁？”；第2步回答“当众污蔑郝红梅是孙少平“婆姨”的人是谁？”。请沿着这些线索逐步检索，最终答案是… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 15 | PASS | 3 | `corpus_chunkids_000007,corpus_chunkids_000113,corpus_chunkids_000109` | 劳动干事 | 《平凡的世界》中，金波在县城上学期间曾多次给孙少平塞一类票证，拿到该票证的孙少平某次到篮球场上向郝红梅要球时，郝红梅把球传给了另一名男同学，请… | 通过 |
| 16 | PASS | 2 | `corpus_chunkids_000008,corpus_chunkids_000010` | 顾养民 | 班级开展初中最后一年孙少平在润生家发现的厚书的主人公相关文章学习的课堂上，负责念报纸的班长是谁？ | 通过 |
| 17 | FAIL | 2 | `corpus_chunkids_000007,corpus_chunkids_000113` | 顾养民 | 第1步回答“在县城上学期间，金波多次给孙少平塞过什么物品？”；第2步回答“孙少平在篮球场上向郝红梅要球时，郝红梅把球传给了谁？”。请沿着这些线… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 18 | FAIL | 2 | `corpus_chunkids_000007,corpus_chunkids_000021` | 罐子村 | 第1步回答“孙少平和金波在公社上初中的两年时间里，金波的自行车最终是什么状态？”；第2步回答“孙少平的姐姐兰花出嫁的村子是哪个？”。请沿着这些… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 19 | FAIL | 3 | `corpus_chunkids_000007,corpus_chunkids_000021,corpus_chunkids_000023` | 王满银 | 《平凡的世界》里，曾和孙少平在公社一同上了两年初中、所骑的自行车最终破烂不堪的金波所认识的孙少平的姐姐兰花，她出嫁到罐子村后的丈夫叫什么名字？ | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 20 | PASS | 2 | `corpus_chunkids_000009,corpus_chunkids_000010` | 顾养民 | 向班主任揭发孙少平在班上看“反动书”的人所在的班级里，开展文章学习的课堂上负责念报纸的班长叫什么名字？ | 通过 |
| 21 | FAIL | 3 | `corpus_chunkids_000009,corpus_chunkids_000010,corpus_chunkids_000005` | 高粱面馍 | 向班主任揭发孙少平在班上看“反动书”的告密者所在的高中班级中，负责在班级文章学习课堂上念报纸的班长的同班同学孙少平，读高中阶段每顿饭只能啃的食… | final_question 退化为最后一跳单跳问题 |
| 22 | PASS | 2 | `corpus_chunkids_000008,corpus_chunkids_000243` | 顾养民 | 在孙少平星期天躲起来看完那本讲保尔·柯察金的厚书的村子的学校里，他于教室抄诗的某天晚上，发现谁在偷看他放在桌上的诗歌笔记本？ | 通过 |
| 23 | PASS | 2 | `corpus_chunkids_000010,corpus_chunkids_000009` | 侯玉英 | 开展文章学习课上班长负责念报纸的那个班级里，向班主任揭发孙少平在班上看“反动书”的人是谁？ | 通过 |
| 24 | FAIL | 3 | `corpus_chunkids_000008,corpus_chunkids_000010,corpus_chunkids_000009` | 侯玉英 | 第1步回答“上初中最后一年，孙少平在润生家发现的厚书的主人公是谁？”；第2步回答“班级开展文章学习的课堂上，负责念报纸的班长是谁？”；第3步回… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 25 | FAIL | 3 | `corpus_chunkids_000008,corpus_chunkids_000243,corpus_chunkids_000244` | 田晓霞 | 《平凡的世界》中，曾在星期天躲在村子打麦场的麦秸垛后面看完讲保尔·柯察金的厚书，且在教室抄诗的某天晚上发现顾养民偷看自己放在桌上的诗歌笔记本的… | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 26 | FAIL | 3 | `corpus_chunkids_000010,corpus_chunkids_000009,corpus_chunkids_000110` | 县文化馆 | 在顾养民任班长负责文章学习课上念报纸、侯玉英曾揭发孙少平在班上看“反动书”的班级里，上学期临放假的前一个星期，孙少平想起郝红梅借走的他的书是从… | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 27 | PASS | 3 | `corpus_chunkids_000011,corpus_chunkids_000013,corpus_chunkids_000010` | 顾养民 | 孙少平曾被老师没收过一本借来的书，那段时期他还会把自己看完的书借给同班同学，请问当时他们班级开展文章学习的课堂上，负责念报纸的班长叫什么名字？ | 通过 |
| 28 | PASS | 3 | `corpus_chunkids_000010,corpus_chunkids_000116,corpus_chunkids_000110` | 县文化馆 | 郝红梅打算送给孙少平用来弥补未及时还书过失、会和要还的书一起送出的物品，与孙少平每天吃饭等众人散尽后才取的食物同属面制主食，那本被郝红梅借走的… | 通过 |
| 29 | FAIL | 2 | `corpus_chunkids_000010,corpus_chunkids_000116` | 白面饼 | 第1步回答“孙少平每天吃饭时等众人散尽后取的食物是什么？”；第2步回答“临近放假时，郝红梅打算和要还给孙少平的书一起送给对方以弥补未及时还书过… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 30 | FAIL | 2 | `corpus_chunkids_000011,corpus_chunkids_000013` | 郝红梅 | 第1步回答“孙少平被老师没收的书是从什么地方借来的？”；第2步回答“孙少平继续把自己看完的书借给谁看？”。请沿着这些线索逐步检索，最终答案是什… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 31 | PASS | 2 | `corpus_chunkids_000013,corpus_chunkids_000010` | 顾养民 | 孙少平继续将自己看完的书借给的那位同学所在的班级里，在开展文章学习的课堂上负责念报纸的班长是谁？ | 通过 |
| 32 | FAIL | 2 | `corpus_chunkids_000011,corpus_chunkids_000207` | 打枣 | 第1步回答“第一次当面和孙少平搭话的姑娘，走到孙少平跟前之前刚取完什么物品？”；第2步回答“金波从学校赶回来是为了做什么？”。请沿着这些线索逐… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 33 | WARN | 3 | `corpus_chunkids_000012,corpus_chunkids_000011,corpus_chunkids_000013` | 郝红梅 | 孙少平平时看完的书都会固定借给同一个人，哪怕在他有一本书被老师没收之后，也依然继续将自己看完的书借给这个人，请问这个人是谁？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 34 | FAIL | 3 | `corpus_chunkids_000011,corpus_chunkids_000207,corpus_chunkids_000007` | 白面票 | 第1步回答“第一次当面和孙少平搭话的姑娘，走到孙少平跟前之前刚取完什么物品？”；第2步回答“金波从学校赶回来是为了做什么？”；第3步回答“在县… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 35 | FAIL | 2 | `corpus_chunkids_000012,corpus_chunkids_000011` | 县文化馆 | 第1步回答“孙少平看过的书都会借给哪个人？”；第2步回答“孙少平被老师没收的书是从什么地方借来的？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 36 | FAIL | 2 | `corpus_chunkids_000014,corpus_chunkids_000076` | 队长 | 《平凡的世界》中，每次田润叶回村到他家串门时都会给他祖母带城里买的吃食的那位少年，他的哥哥十八岁时被一队社员一致推选担任什么职务？ | final_question 泄露中间 hop1 answer; answer_aliases 未包含 final_answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 37 | PASS | 3 | `corpus_chunkids_000013,corpus_chunkids_000010,corpus_chunkids_000009` | 侯玉英 | 孙少平看完书后会继续借给某个人，这个人所在的班级开展文章学习课时由班长负责念报纸，请问这个班里向班主任告发孙少平看"反动书"的人是谁？ | 通过 |
| 38 | FAIL | 3 | `corpus_chunkids_000014,corpus_chunkids_000076,corpus_chunkids_000075` | 城里买的吃食 | 孙少安开启农民生涯时立志要在某个村子当出众的庄稼人，他十八岁时还被该村一队社员一致推选为队长，请问田润叶每次回这个村子到孙少安的弟弟孙少平家串… | final_question 泄露中间 hop2 answer; answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 39 | FAIL | 2 | `corpus_chunkids_000014,corpus_chunkids_000095` | 马兰花 | 第1步回答“田润叶每次回村时会提着点心看望她户族里的哪位傻瓜叔叔？”；第2步回答“孙少安送给田润叶的花是什么？”。请沿着这些线索逐步检索，最终… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 40 | FAIL | 3 | `corpus_chunkids_000015,corpus_chunkids_000242,corpus_chunkids_000245` | 孙少平 | 《平凡的世界》中，曾用发绿的柳枝制作哨子，且被田晓霞到宿舍找他后在无人小山沟里接过对方递的绿皮笔记本，还在大暴雨引发沟道洪水时救下被困石崖下的… | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 41 | FAIL | 3 | `corpus_chunkids_000014,corpus_chunkids_000095,corpus_chunkids_000132` | 石圪节的公路上 | 第1步回答“田润叶每次回村时会提着点心看望她户族里的哪位傻瓜叔叔？”；第2步回答“孙少安送给田润叶的花是什么？”；第3步回答“孙少安是在什么地… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 42 | FAIL | 2 | `corpus_chunkids_000015,corpus_chunkids_000242` | 绿皮笔记本 | 第1步回答“孙少平用发绿的柳枝制作了什么物品？”；第2步回答“田晓霞到宿舍找孙少平后，在无人的小山沟里递给孙少平的物品是什么？”。请沿着这些线… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 43 | FAIL | 2 | `corpus_chunkids_000015,corpus_chunkids_000014` | 城里买的吃食 | 第1步回答“田润叶安排谁去叫孙少平到她二爸家？”；第2步回答“田润叶每次回村到孙少平家串门时，会给孙少平的祖母带什么？”。请沿着这些线索逐步检… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 44 | FAIL | 2 | `corpus_chunkids_000016,corpus_chunkids_000343` | 同班同学 | 《平凡的世界》中田福堂的儿子与孙少平是什么关系？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 45 | FAIL | 3 | `corpus_chunkids_000015,corpus_chunkids_000014,corpus_chunkids_000076` | 队长 | 在包含“田润叶安排人去叫孙少平到她二爸家”以及“田润叶每次回村到孙少平家串门时会给孙少平的祖母带礼物”这两个情节的文学作品中，孙少安十八岁时被… | final_question 退化为最后一跳单跳问题; answer_aliases 未包含 final_answer |
| 46 | FAIL | 3 | `corpus_chunkids_000016,corpus_chunkids_000343,corpus_chunkids_000344` | 姚淑芳 | 孙少平的同班同学田润生的父亲所在的双水村学校里，唯一的公派教师是谁？ | final_question 泄露中间 hop2 answer |
| 47 | PASS | 2 | `corpus_chunkids_000016,corpus_chunkids_000015` | 哨子 | 《平凡的世界》中，孙少平跟着田润叶进入单位大门的那次经历里，他用发绿的柳枝制作了什么物品？ | 通过 |
| 48 | FAIL | 3 | `corpus_chunkids_000017,corpus_chunkids_000349,corpus_chunkids_000055` | 王满银 | 田晓霞曾提到自己长到十七岁都没回过的老乡所在的村子，后来她在阳历年前的一天如期回到了这个村子，请问这个村子里兰花的女婿是谁？ | final_question 退化为最后一跳单跳问题 |
| 49 | WARN | 3 | `corpus_chunkids_000016,corpus_chunkids_000015,corpus_chunkids_000242` | 绿皮笔记本 | 在孙少平跟着田润叶进入某单位大门的那段时期，他曾用发绿的柳枝制作过小物品，请问这段时期田晓霞到宿舍找他之后，在无人的小山沟里递给孙少平的是什么… | answer_aliases 未包含 final_answer |
| 50 | PASS | 2 | `corpus_chunkids_000017,corpus_chunkids_000349` | 双水村 | 阳历年前的一天，田晓霞如期回到了她曾提到的自己长到十七岁都没回过的老乡所在的村子，这个村子叫什么名字？ | 通过 |
| 51 | FAIL | 2 | `corpus_chunkids_000017,corpus_chunkids_000023` | 王满银 | 第1步回答“孙少平在田润叶家一共吃了多少个馒头？”；第2步回答“孙少平的姐夫叫什么名字？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 52 | FAIL | 3 | `corpus_chunkids_000017,corpus_chunkids_000023,corpus_chunkids_000022` | 罐子村 | 第1步回答“孙少平在田润叶家一共吃了多少个馒头？”；第2步回答“孙少平的姐夫叫什么名字？”；第3步回答“孙少平骑行至哪个地点时看见了站在公路边… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 53 | WARN | 2 | `corpus_chunkids_000019,corpus_chunkids_000276` | 粮票、多兜黄挂包 | 《平凡的世界》中，田润叶塞给孙少平的物品和田晓霞送给孙少平的毕业礼物分别是什么？ | final_answer 含并列符号，需确认是否复合答案; answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 54 | WARN | 2 | `corpus_chunkids_000019,corpus_chunkids_000081` | 谈一次 | 田润叶叮嘱孙少平转告他哥孙少安到城里后去找她的地点，她决定在那里和孙少安做什么？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 55 | FAIL | 3 | `corpus_chunkids_000019,corpus_chunkids_000081,corpus_chunkids_000139` | 河滩里 | 田润叶原本打算在自己叮嘱孙少平转告孙少安到城里后去找她的地点和他谈一次，请问田福堂看见润叶和少安正晌午坐着的地点是哪里？ | final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 56 | PASS | 3 | `corpus_chunkids_000019,corpus_chunkids_000276,corpus_chunkids_000283` | 学校 | 曾收到田润叶塞的物品、田晓霞赠送的毕业礼物的孙少平，在目送郝红梅的身影消失后要前往什么地方？ | 通过 |
| 57 | FAIL | 3 | `corpus_chunkids_000020,corpus_chunkids_000197,corpus_chunkids_000196` | 春节 | 在星期五孙少平买粮的地点所属的区域，恰逢金俊海两口子到田家圪崂的公路边搬东西的时期，孙少安计划举办婚事的时间是什么？ | final_question 泄露中间 hop2 answer; answer_aliases 未包含 final_answer |
| 58 | PASS | 3 | `corpus_chunkids_000021,corpus_chunkids_000023,corpus_chunkids_000022` | 王满银 | 孙少平骑行至某个地点时看见了站在公路边的兰香，该地点恰好是他姐姐兰花出嫁的村子，请问孙少平的姐夫叫什么名字？ | 通过 |
| 59 | PASS | 2 | `corpus_chunkids_000021,corpus_chunkids_000023` | 王满银 | 《平凡的世界》中，孙少平的姐姐兰花所嫁入的村子里，她的丈夫叫什么名字？ | 通过 |
| 60 | FAIL | 2 | `corpus_chunkids_000021,corpus_chunkids_000152` | 白明川 | 街上唯一像样的建筑物为供销社门市部的公社里，孙少安参加的批判会结束时，对受批判人员说鼓励话的公社主任叫什么名字？ | final_question 退化为最后一跳单跳问题 |
| 61 | FAIL | 2 | `corpus_chunkids_000020,corpus_chunkids_000197` | 搬东西 | 第1步回答“星期五孙少平买粮的地点是哪里？”；第2步回答“金俊海两口子到田家圪崂的公路边做什么？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 62 | PASS | 3 | `corpus_chunkids_000021,corpus_chunkids_000152,corpus_chunkids_000056` | 田福顺 | 街上唯一像样的建筑物为供销社门市部的公社里，孙少安参加的某场批判会结束时，有公社主任对受批判人员说鼓励话，请问这场批判会中被确定为批判对象的田… | 通过 |
| 63 | FAIL | 3 | `corpus_chunkids_000022,corpus_chunkids_000021,corpus_chunkids_000023` | 王满银 | 孙少平骑行至某地时看见了站在公路边的兰香，这个地方正是他姐姐兰花出嫁的村子，请问孙少平的姐夫叫什么名字？ | final_question 退化为最后一跳单跳问题 |
| 64 | PASS | 2 | `corpus_chunkids_000022,corpus_chunkids_000021` | 罐子村 | 孙少平骑行至哪个地点时看见了站在公路边的兰香，且该地点恰好是他姐姐兰花出嫁的村子？ | 通过 |
| 65 | WARN | 2 | `corpus_chunkids_000023,corpus_chunkids_000014` | 城里买的吃食 | 田润叶每次回孙兰香的姐姐家所在的村子，到孙少平家串门时，会给孙少平的祖母带什么？ | answer_aliases 未包含 final_answer |
| 66 | PASS | 2 | `corpus_chunkids_000023,corpus_chunkids_000022` | 罐子村 | 孙少平在骑车去姐夫家的路上，行至哪个村子时看见了站在公路边的兰香？ | 通过 |
| 67 | FAIL | 2 | `corpus_chunkids_000022,corpus_chunkids_000119` | 孙少平 | 第1步回答“孙少平让金波先回去时，嘱托金波暂存在他家的物品是什么？”；第2步回答“金波筹划殴打顾养民时为了不牵连谁而对其保密相关行动？”。请沿… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 68 | FAIL | 3 | `corpus_chunkids_000022,corpus_chunkids_000119,corpus_chunkids_000007` | 白面票 | 第1步回答“孙少平让金波先回去时，嘱托金波暂存在他家的物品是什么？”；第2步回答“金波筹划殴打顾养民时为了不牵连谁而对其保密相关行动？”；第3… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题 |
| 69 | PASS | 3 | `corpus_chunkids_000023,corpus_chunkids_000022,corpus_chunkids_000021` | 罐子村 | 孙少平骑行途中看见站在公路边的兰香的地点，恰好是他姐夫所在的村子，请问这个村子叫什么名字？ | 通过 |
| 70 | PASS | 2 | `corpus_chunkids_000025,corpus_chunkids_000041` | 老鼠药 | 孙少平的姐夫在石圪节集市上倒卖的物品是什么？ | 通过 |
| 71 | FAIL | 3 | `corpus_chunkids_000023,corpus_chunkids_000014,corpus_chunkids_000076` | 队长 | 第1步回答“孙兰香的姐姐家位于哪个村子？”；第2步回答“田润叶每次回村到孙少平家串门时，会给孙少平的祖母带什么？”；第3步回答“孙少安十八岁时… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 72 | FAIL | 2 | `corpus_chunkids_000024,corpus_chunkids_000011` | 县文化馆 | 孙少平和妹妹兰香绕路回家时淌过的河流附近的县城中，孙少平被老师没收的书是从什么地方借来的？ | final_question 退化为最后一跳单跳问题 |
| 73 | FAIL | 3 | `corpus_chunkids_000025,corpus_chunkids_000041,corpus_chunkids_000022` | 罐子村 | 姐夫曾在石圪节集市上倒卖老鼠药的人，骑行至哪个地点时看见了站在公路边的兰香？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 74 | FAIL | 3 | `corpus_chunkids_000024,corpus_chunkids_000011,corpus_chunkids_000013` | 郝红梅 | 第1步回答“孙少平和兰香绕路回家时淌过的河流叫什么名字？”；第2步回答“孙少平被老师没收的书是从什么地方借来的？”；第3步回答“孙少平继续把自… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 75 | FAIL | 2 | `corpus_chunkids_000025,corpus_chunkids_000213` | 刘志祥 | 第1步回答“王满银是被公社民兵小分队从哪个村子带到农田基建会战工地的?”；第2步回答“柳岔公社的副主任是谁？”。请沿着这些线索逐步检索，最终答… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 76 | PASS | 2 | `corpus_chunkids_000026,corpus_chunkids_000027` | 罐子村 | 当初反对女儿兰花嫁给王满银的那位父亲，他的女儿执意要嫁的王满银属于哪个村子？ | 通过 |
| 77 | FAIL | 3 | `corpus_chunkids_000025,corpus_chunkids_000213,corpus_chunkids_000216` | 五十四人 | 第1步回答“王满银是被公社民兵小分队从哪个村子带到农田基建会战工地的?”；第2步回答“柳岔公社的副主任是谁？”；第3步回答“柳岔公社大会战工地… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 78 | FAIL | 3 | `corpus_chunkids_000026,corpus_chunkids_000029,corpus_chunkids_000028` | 罐子村 | 老丈人是孙玉厚，且曾在双水村后河湾将一身外地买来的时新衣裳塞到兰花手里的男子所属的村子叫什么名字？ | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 79 | FAIL | 2 | `corpus_chunkids_000026,corpus_chunkids_000029` | 孙玉厚 | 在双水村后河湾里把一身外地买来的时新衣裳塞到兰花手里的人的老丈人是谁？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 80 | PASS | 3 | `corpus_chunkids_000026,corpus_chunkids_000027,corpus_chunkids_000055` | 罐子村 | 反对女儿兰花嫁给王满银的兰花的父亲，其女婿所属的村子叫什么名字？ | 通过 |
| 81 | WARN | 2 | `corpus_chunkids_000027,corpus_chunkids_000055` | 罐子村 | 孙兰花的女婿所属的村子是哪个？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 82 | PASS | 2 | `corpus_chunkids_000028,corpus_chunkids_000101` | 大前门 | 田福军的信中提到的被押到双水村公社农田基建工地劳教的罐子村社员，在石圪节卖完老鼠药后买的香烟是什么牌子的？ | 通过 |
| 83 | WARN | 3 | `corpus_chunkids_000028,corpus_chunkids_000101,corpus_chunkids_000360` | 大前门 | 田福军信中提到的被押到双水村公社农田基建工地劳教的罐子村社员，在双水村秧歌最为知名的那个公社卖完老鼠药后购买的香烟是什么牌子的？ | answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 84 | WARN | 2 | `corpus_chunkids_000028,corpus_chunkids_000208` | 扯结婚衣裳 | 与王满银同属一个村子的孙少安，在打完枣又过了中秋节后，张罗着和贺秀莲去米家镇做什么？ | answer_aliases 未包含 final_answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 85 | WARN | 3 | `corpus_chunkids_000027,corpus_chunkids_000055,corpus_chunkids_000033` | 水果糖 | 孙兰花的女婿就是她当初想要嫁给的那个人，孙少平前往该女婿所属的村子看望姐姐一家时，塞给猫蛋和狗蛋的物品是什么？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 86 | FAIL | 3 | `corpus_chunkids_000028,corpus_chunkids_000208,corpus_chunkids_000191` | 贺耀宗 | 王满银所属村子里，在打完枣又过了中秋节后张罗着和贺秀莲去米家镇扯结婚衣裳的人的岳父叫什么名字？ | final_question 泄露中间 hop2 answer |
| 87 | FAIL | 2 | `corpus_chunkids_000029,corpus_chunkids_000028` | 罐子村 | 《平凡的世界》中，老丈人是孙玉厚的人所在的村子叫什么名字？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 88 | FAIL | 2 | `corpus_chunkids_000031,corpus_chunkids_000198` | 兰花的丈夫是王满银，公派教师姚淑芳的丈夫是金光明 | 兰花的丈夫和公派教师姚淑芳的丈夫分别叫什么名字？ | final_answer 超过20字; final_answer 含并列符号，需确认是否复合答案; answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 89 | PASS | 3 | `corpus_chunkids_000029,corpus_chunkids_000028,corpus_chunkids_000208` | 扯结婚衣裳 | 王满银的老丈人的大儿子孙少安，在王满银所属的村子打完枣又过了中秋节后，张罗着和贺秀莲去米家镇做什么？ | 通过 |
| 90 | PASS | 3 | `corpus_chunkids_000030,corpus_chunkids_000029,corpus_chunkids_000028` | 罐子村 | 王满银老丈人的大女儿兰花嫁的人所在的村子叫什么名字？ | 通过 |
| 91 | FAIL | 2 | `corpus_chunkids_000030,corpus_chunkids_000029` | 孙玉厚 | 第1步回答“孙玉厚的大女儿兰花所嫁之人来自哪个村子？”；第2步回答“王满银的老丈人是谁？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 92 | WARN | 2 | `corpus_chunkids_000031,corpus_chunkids_000063` | 金丝猴 | 孙少安去米家镇办事时，口袋里舍不得抽的香烟是什么牌子？ | answer_aliases 未包含 final_answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 93 | FAIL | 3 | `corpus_chunkids_000031,corpus_chunkids_000198,corpus_chunkids_000344` | 姚淑芳 | 第1步回答“兰花的丈夫叫什么名字？”；第2步回答“公派教师姚淑芳的丈夫是谁？”；第3步回答“双水村学校唯一的公派教师是谁？”。请沿着这些线索逐… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 94 | WARN | 3 | `corpus_chunkids_000031,corpus_chunkids_000063,corpus_chunkids_000065` | 双水村 | 孙少安某次前往米家镇的行程中随身带着舍不得抽的香烟，这次行程里他告知遇到的河南老师傅自己来自哪个村子？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 95 | FAIL | 3 | `corpus_chunkids_000032,corpus_chunkids_000332,corpus_chunkids_000326` | 银花 | 与兰香一同进入家门的人所在的村子和王家庄发生冲突时，王家庄的人淌过东拉河到田家圪崂要找的人的同村村民田海民的媳妇叫什么名字？ | final_question 退化为最后一跳单跳问题 |
| 96 | FAIL | 2 | `corpus_chunkids_000032,corpus_chunkids_000332` | 田福堂 | 第1步回答“和兰香一同进入家门的人是谁？”；第2步回答“与双水村发生冲突时，王家庄的人淌过东拉河到田家圪崂要找的人是谁？”。请沿着这些线索逐步… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 97 | PASS | 2 | `corpus_chunkids_000034,corpus_chunkids_000023` | 王满银 | 小时候每个夏天的早晨，会和孙少平一起去野地拔带露水珠的青草叶给奶奶淋眼睛的女孩，她的姐夫叫什么名字？ | 通过 |
| 98 | FAIL | 2 | `corpus_chunkids_000032,corpus_chunkids_000031` | 王满银 | 第1步回答“老祖母模模糊糊听到的和家里灾事相关的字是什么？”；第2步回答“兰花的丈夫叫什么名字？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 99 | WARN | 2 | `corpus_chunkids_000033,corpus_chunkids_000022` | 罐子村 | 孙少平有次回家后曾塞给猫蛋和狗蛋一些物品，请问他此次出行时骑行至哪个地点看见了站在公路边的兰香？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 100 | FAIL | 3 | `corpus_chunkids_000032,corpus_chunkids_000031,corpus_chunkids_000198` | 金光明 | 第1步回答“老祖母模模糊糊听到的和家里灾事相关的字是什么？”；第2步回答“兰花的丈夫叫什么名字？”；第3步回答“公派教师姚淑芳的丈夫是谁？”。… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 101 | PASS | 3 | `corpus_chunkids_000034,corpus_chunkids_000023,corpus_chunkids_000022` | 罐子村 | 孙少平某次骑行途中在公路边看到了小时候曾和自己一起在夏天清晨去野地拔带露水珠的青草叶给奶奶淋眼睛的人，而这个地点正好是他姐夫居住的村子，请问这… | 通过 |
| 102 | FAIL | 3 | `corpus_chunkids_000033,corpus_chunkids_000022,corpus_chunkids_000021` | 罐子村 | 第1步回答“孙少平回家后塞给猫蛋和狗蛋的物品是什么？”；第2步回答“孙少平骑行至哪个地点时看见了站在公路边的兰香？”；第3步回答“孙少平的姐姐… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 103 | FAIL | 3 | `corpus_chunkids_000035,corpus_chunkids_000034,corpus_chunkids_000023` | 王满银 | 曾给奶奶购买止痛物品、小时候每个夏天早晨都会和兰香一起去野地拔带露水珠的青草叶给奶奶淋眼睛的人，他的姐夫叫什么名字？ | final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 104 | PASS | 2 | `corpus_chunkids_000035,corpus_chunkids_000034` | 兰香 | 《平凡的世界》中，曾给奶奶购买止痛物品的孙少平，小时候每个夏天的早晨会和谁一起去野地拔带露水珠的青草叶给奶奶淋眼睛？ | 通过 |
| 105 | FAIL | 2 | `corpus_chunkids_000035,corpus_chunkids_000243` | 顾养民 | 那个曾安排万一自己的哥哥回来时就去队上的饲养室凑合一晚上的人，在教室抄诗的某天晚上，发现谁在偷看他放在桌上的诗歌笔记本？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 106 | PASS | 2 | `corpus_chunkids_000036,corpus_chunkids_000249` | 石圪节公社中学 | 孙兰香就读的初中所在地有一所她和金秀十三岁时共同升入的中学，这所中学的全称是什么？ | 通过 |
| 107 | PASS | 2 | `corpus_chunkids_000037,corpus_chunkids_000029` | 石圪节 | 王满银的老丈人家最小的孩子上初中的地点是哪里？ | 通过 |
| 108 | PASS | 3 | `corpus_chunkids_000036,corpus_chunkids_000249,corpus_chunkids_000251` | 专心学习 | 在孙兰香十三岁时与金秀一同升入的、位于其就读初中所在地的中学上学期间，她放弃回家劳动的打算后重新开始做什么？ | 通过 |
| 109 | FAIL | 2 | `corpus_chunkids_000036,corpus_chunkids_000269` | 二斗小米 | 孙少安的妹妹是孙少平母亲想起要喂猪时就已经把猪喂好的人，那么贺耀宗托顺车给孙少安家捎来了什么物品？ | final_question 退化为最后一跳单跳问题; answer_aliases 未包含 final_answer |
| 110 | FAIL | 3 | `corpus_chunkids_000035,corpus_chunkids_000243,corpus_chunkids_000244` | 田晓霞 | 第1步回答“孙少平安排万一哥哥回来时要去哪里凑合一晚上？”；第2步回答“孙少平在教室抄诗的某天晚上，发现谁在偷看他放在桌上的诗歌笔记本？”；第… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 111 | WARN | 3 | `corpus_chunkids_000036,corpus_chunkids_000269,corpus_chunkids_000268` | 黄原出的羊毛毯 | 在《平凡的世界》中，当少平母亲想起要喂猪时就已经把猪喂好的女孩的家中，在收到贺耀宗托顺车捎来物品的一九七六年临近结束前不久，这家的长子孙少安给… | answer_aliases 未包含 final_answer |
| 112 | PASS | 2 | `corpus_chunkids_000038,corpus_chunkids_000040` | 石圪节 | 双水村田姓人家大都居住的片区附近的小学，学生上完五年级后要到哪里上初中？ | 通过 |
| 113 | WARN | 3 | `corpus_chunkids_000037,corpus_chunkids_000045,corpus_chunkids_000046` | 拜把兄弟 | 《平凡的世界》中，站在自家院子里失神望着山的孙玉厚，冬天农闲时为石圪节商行驮瓷器要前往山西的某个地点，他和该地点的陶窑主是什么关系？ | answer_aliases 未包含 final_answer |
| 114 | FAIL | 3 | `corpus_chunkids_000037,corpus_chunkids_000029,corpus_chunkids_000028` | 罐子村 | 第1步回答“孙玉厚家最小的孩子上初中的地点是哪里？”；第2步回答“王满银的老丈人是谁？”；第3步回答“王满银所属的村子叫什么名字？”。请沿着这… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题 |
| 115 | FAIL | 2 | `corpus_chunkids_000037,corpus_chunkids_000045` | 柳林镇 | 第1步回答“孙玉厚站在自家院子里失神望着的山叫什么名字？”；第2步回答“冬天农闲时孙玉厚给石圪节商行驮瓷器要前往山西的哪个地点？”。请沿着这些… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 116 | FAIL | 3 | `corpus_chunkids_000038,corpus_chunkids_000040,corpus_chunkids_000075` | 双水村 | 第1步回答“双水村的田姓人家大都居住在什么地方？”；第2步回答“双水村小学的学生上完五年级后要到哪里上初中？”；第3步回答“孙少安开启农民生涯… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 117 | PASS | 2 | `corpus_chunkids_000039,corpus_chunkids_000346` | 落榜 | 双水村有庙的三角洲所在区域的孙少平，在十月份教育部发布当年大学招生消息后参加高考的结果是什么？ | 通过 |
| 118 | FAIL | 2 | `corpus_chunkids_000038,corpus_chunkids_000011` | 县文化馆 | 《平凡的世界》里，二爸名为孙玉亭的主人公被老师没收的书是从什么地方借来的？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 119 | PASS | 3 | `corpus_chunkids_000038,corpus_chunkids_000011,corpus_chunkids_000013` | 郝红梅 | 请说出路遥作品《平凡的世界》中，孙少平的二爸叫什么名字，他被老师没收的书是从什么地方借来的，他看完书后会继续借给谁阅读？ | 通过 |
| 120 | PASS | 2 | `corpus_chunkids_000040,corpus_chunkids_000075` | 石圪节 | 孙少安开启农民生涯时决心要在那里做出众庄稼人的村子，其小学的学生上完五年级后要到哪里上初中？ | 通过 |
| 121 | FAIL | 3 | `corpus_chunkids_000039,corpus_chunkids_000346,corpus_chunkids_000249` | 石圪节公社中学 | 第1步回答“双水村有庙的三角洲被称为什么？”；第2步回答“十月份教育部发布当年大学招生消息后，孙少平参加高考的结果是什么？”；第3步回答“孙兰… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 122 | FAIL | 3 | `corpus_chunkids_000039,corpus_chunkids_000038,corpus_chunkids_000040` | 石圪节 | 庙坪上种植的成片林木为枣树、且村内田姓人家大都居住在田家圪崂的村子，其村小学的学生上完五年级后要到哪里上初中？ | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 123 | FAIL | 2 | `corpus_chunkids_000039,corpus_chunkids_000038` | 田家圪崂 | 第1步回答“双水村庙坪上种植的成片林木是什么树？”；第2步回答“双水村的田姓人家大都居住在什么地方？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 124 | PASS | 2 | `corpus_chunkids_000041,corpus_chunkids_000101` | 铺盖卷 | 田福军的信中提到的被押到双水村公社农田基建工地劳教的罐子村社员，在双水村学校被劳教期间，孙少平带给对方的物品除了饭罐还有什么？ | 通过 |
| 125 | FAIL | 3 | `corpus_chunkids_000040,corpus_chunkids_000075,corpus_chunkids_000285` | 田福堂 | 哪座村子的小学学生上完五年级后要到石圪节上初中，且是孙少安开启农民生涯时决心要在其中做出众庄稼人的村子，这座村子的大队书记是谁？ | final_question 泄露中间 hop1 answer |
| 126 | PASS | 2 | `corpus_chunkids_000042,corpus_chunkids_000007` | 白面票 | 孙少平每星期六回村时通常过夜的人家的主人，在县城上学期间多次给孙少平塞过什么物品？ | 通过 |
| 127 | FAIL | 2 | `corpus_chunkids_000041,corpus_chunkids_000022` | 罐子村 | 第1步回答“孙少平的姐夫是谁？”；第2步回答“孙少平骑行至哪个地点时看见了站在公路边的兰香？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 128 | PASS | 3 | `corpus_chunkids_000041,corpus_chunkids_000101,corpus_chunkids_000360` | 铺盖卷 | 田福军信中提到的、被押到秧歌为全公社最有名的双水村所属公社的农田基建工地劳教的罐子村社员，在双水村学校被劳教期间，孙少平带给该社员的物品除了饭… | 通过 |
| 129 | PASS | 3 | `corpus_chunkids_000041,corpus_chunkids_000022,corpus_chunkids_000021` | 罐子村 | 孙少平的姐夫的妻子是兰花，孙少平骑行到某个地点时看见了站在公路边的兰香，该地点恰好是兰花出嫁的村子，请问这个地点的名称是什么？ | 通过 |
| 130 | FAIL | 2 | `corpus_chunkids_000043,corpus_chunkids_000052` | 纸烟 | 在公社会战指挥部计划召开批判会的地点，金俊山给前来谈事的孙玉亭递了什么物品？ | final_question 退化为最后一跳单跳问题; answer_aliases 未包含 final_answer |
| 131 | WARN | 2 | `corpus_chunkids_000043,corpus_chunkids_000244` | 田晓霞 | 孙少平告知二妈自己接下来要去的地方，他在该处抄完诗后将绿皮笔记本还给了谁？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 132 | FAIL | 3 | `corpus_chunkids_000042,corpus_chunkids_000007,corpus_chunkids_000113` | 顾养民 | 县城上学时期，孙少平每星期六回村通常在好友家过夜，该好友曾多次给他塞白面票，请问这一时期孙少平在篮球场上向郝红梅要球时，郝红梅把球传给了谁？ | final_question 泄露中间 hop2 answer; final_question 退化为最后一跳单跳问题; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 133 | FAIL | 2 | `corpus_chunkids_000042,corpus_chunkids_000011` | 县文化馆 | 《平凡的世界》中孙少平的二妈叫什么名字？同时孙少平被老师没收的书是从什么地方借来的？ | final_question 退化为最后一跳单跳问题 |
| 134 | FAIL | 3 | `corpus_chunkids_000042,corpus_chunkids_000011,corpus_chunkids_000013` | 郝红梅 | 第1步回答“孙少平的二妈叫什么名字？”；第2步回答“孙少平被老师没收的书是从什么地方借来的？”；第3步回答“孙少平继续把自己看完的书借给谁看？… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 135 | PASS | 3 | `corpus_chunkids_000043,corpus_chunkids_000052,corpus_chunkids_000319` | 纸烟 | 公社会战指挥部计划召开批判会的地点所在的村子中，田福堂准备召开社员大会动员搬迁前，主动提出去和金俊文、金俊武两兄弟协商搬迁事宜的人，给前来谈事… | 通过 |
| 136 | PASS | 2 | `corpus_chunkids_000045,corpus_chunkids_000046` | 拜把兄弟 | 冬天农闲时孙玉厚给石圪节商行驮瓷器需要前往的山西地点的陶窑主，和孙玉厚是什么关系？ | 通过 |
| 137 | WARN | 3 | `corpus_chunkids_000043,corpus_chunkids_000244,corpus_chunkids_000243` | 顾养民 | 在孙少平告知二妈自己接下来的去向，到他抄完诗将绿皮笔记本还给对方的这段时间里，他曾在教室抄诗的某天晚上发现谁在偷看他放在桌上的诗歌笔记本？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 138 | FAIL | 2 | `corpus_chunkids_000044,corpus_chunkids_000048` | 太原钢厂 | 公社会战指挥部中孙玉亭担任的职务是什么？担任该职务的孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？ | final_question 退化为最后一跳单跳问题; answer_aliases 未包含 final_answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 139 | WARN | 3 | `corpus_chunkids_000044,corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 曾在公社会战指挥部任职、早年间工作时能顿顿吃白蒸馍大肉菜的孙玉亭，少安妈提出可以拿给他的物品是什么？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 140 | WARN | 2 | `corpus_chunkids_000046,corpus_chunkids_000045` | 拜把兄弟 | 冬天农闲时，孙玉厚给石圪节商行驮瓷器需要前往的山西地点的陶窑主，和孙玉厚是什么关系？ | answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 141 | PASS | 2 | `corpus_chunkids_000045,corpus_chunkids_000355` | 石圪节公社 | 请问孙玉亭叫住即将走上土坡的孙玉厚索要的物品是什么，田福堂就孙少安的相关问题向上级汇报去的公社名称又是什么？ | 通过 |
| 142 | FAIL | 3 | `corpus_chunkids_000045,corpus_chunkids_000046,corpus_chunkids_000159` | 山西 | 第1步回答“冬天农闲时孙玉厚给石圪节商行驮瓷器要前往山西的哪个地点？”；第2步回答“孙玉厚和山西柳林镇的陶窑主是什么关系？”；第3步回答“孙少… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 泄露中间 hop1 answer; final_question 退化为最后一跳单跳问题 |
| 143 | PASS | 3 | `corpus_chunkids_000045,corpus_chunkids_000355,corpus_chunkids_000357` | 坚决制止 | 在孙玉亭叫住即将走上土坡的孙玉厚索要物品的相关事件中，田福堂就孙少安的相关问题向上级汇报后，地区革委会主任苗凯针对双水村的相关情况给出的指示是… | 通过 |
| 144 | FAIL | 3 | `corpus_chunkids_000046,corpus_chunkids_000045,corpus_chunkids_000157` | 南瓜 | 冬天农闲时会给石圪节商行驮瓷器前往山西某地、且与该地陶窑主是拜把兄弟的孙玉厚，在和孙玉亭讨论找媳妇的话题时，贺凤英正在锅台上切什么？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 145 | FAIL | 2 | `corpus_chunkids_000047,corpus_chunkids_000361` | 田五 | 与孙玉亭在柳林镇小学同过学的女子所在的双水村此次闹秧歌的秧歌队伞头是谁？ | final_question 退化为最后一跳单跳问题 |
| 146 | FAIL | 2 | `corpus_chunkids_000046,corpus_chunkids_000038` | 田家圪崂 | 第1步回答“1954年孙玉亭初中毕业后到什么地方当了工人？”；第2步回答“双水村的田姓人家大都居住在什么地方？”。请沿着这些线索逐步检索，最终… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 147 | FAIL | 3 | `corpus_chunkids_000047,corpus_chunkids_000361,corpus_chunkids_000360` | 石圪节公社 | 和孙玉亭在柳林镇小学同过学的女子所在的村庄，此次闹秧歌的伞头为田五，该村庄的秧歌是全哪个公社最有名的？ | final_question 泄露中间 hop2 answer |
| 148 | FAIL | 3 | `corpus_chunkids_000046,corpus_chunkids_000038,corpus_chunkids_000040` | 石圪节 | 第1步回答“1954年孙玉亭初中毕业后到什么地方当了工人？”；第2步回答“双水村的田姓人家大都居住在什么地方？”；第3步回答“双水村小学的学生… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 149 | FAIL | 3 | `corpus_chunkids_000048,corpus_chunkids_000289,corpus_chunkids_000317` | 姚淑芳 | 包含孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位、少安妈提出可以拿给孙玉亭的物品这两个情节的故事中，金光明是接到谁的信赶回村里知晓搬家通知的？ | final_question 退化为最后一跳单跳问题 |
| 150 | FAIL | 2 | `corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 第1步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？”；第2步回答“少安妈提出可以拿给孙玉亭的物品是什么？”。请沿着这些线索逐步检… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 151 | PASS | 2 | `corpus_chunkids_000049,corpus_chunkids_000288` | 金俊山 | 双水村学校的贫管会主任在书记田福堂去公社开会不在村里时，要找的双水村大队副书记是谁？ | 通过 |
| 152 | WARN | 2 | `corpus_chunkids_000048,corpus_chunkids_000055` | 徐治功、王满银 | 《平凡的世界》中，石圪节公社在双水村开展的农田基建大会战的总指挥与兰花的女婿分别是谁？ | final_answer 含并列符号，需确认是否复合答案; answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 153 | WARN | 3 | `corpus_chunkids_000048,corpus_chunkids_000055,corpus_chunkids_000033` | 水果糖 | 《平凡的世界》里，在石圪节公社于双水村开展农田基建大会战的总指挥任职期间，兰花的女婿家的孩子猫蛋和狗蛋，收到了孙少平回家后塞给他们的什么物品？ | answer_aliases 未包含 final_answer |
| 154 | FAIL | 2 | `corpus_chunkids_000049,corpus_chunkids_000055` | 王满银 | 总指挥徐治功提到的晚上的批判会没有批判对象的村子里，兰花的女婿是谁？ | final_question 退化为最后一跳单跳问题 |
| 155 | PASS | 2 | `corpus_chunkids_000050,corpus_chunkids_000060` | 世事要变了 | 孙玉亭过了哭咽河后想到的可以充当批判角色的人最爱说的一句话是什么？ | 通过 |
| 156 | PASS | 3 | `corpus_chunkids_000049,corpus_chunkids_000055,corpus_chunkids_000033` | 水果糖 | 总指挥徐治功提到晚上的批判会没有批判对象的村子里，兰花的女婿家的孩子猫蛋和狗蛋，收到孙少平回家后塞给他们的物品是什么？ | 通过 |
| 157 | FAIL | 3 | `corpus_chunkids_000049,corpus_chunkids_000288,corpus_chunkids_000289` | 旧棉花 | 书记田福堂去公社开会不在村里时，孙玉亭要找的双水村大队副书记所在村子的学校贫管会主任，少安妈提出可以拿给他的物品是什么？ | final_question 泄露中间 hop2 answer |
| 158 | PASS | 2 | `corpus_chunkids_000051,corpus_chunkids_000201` | 石圪节学校 | 金俊山的儿子的同校同学兰香的上学地点是哪里？ | 通过 |
| 159 | FAIL | 3 | `corpus_chunkids_000050,corpus_chunkids_000060,corpus_chunkids_000059` | 憨牛 | 第1步回答“孙玉亭过了哭咽河后想到的可以充当批判角色的人是谁？”；第2步回答“田二最爱说的一句话是什么？”；第3步回答“田二的儿子叫什么名字？… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 泄露中间 hop1 answer; final_question 退化为最后一跳单跳问题 |
| 160 | FAIL | 2 | `corpus_chunkids_000050,corpus_chunkids_000177` | 大前门 | 第1步回答“一九四八年解放军向国民党军队大反攻时，金俊山跟随部队最终打到了哪个城市？”；第2步回答“金俊武等三人完成挖掘任务回到双水村大队部院… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 161 | FAIL | 3 | `corpus_chunkids_000050,corpus_chunkids_000177,corpus_chunkids_000189` | 石圪节公社 | 第1步回答“一九四八年解放军向国民党军队大反攻时，金俊山跟随部队最终打到了哪个城市？”；第2步回答“金俊武等三人完成挖掘任务回到双水村大队部院… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 162 | FAIL | 2 | `corpus_chunkids_000051,corpus_chunkids_000063` | 金丝猴 | 《平凡的世界》中，在金俊山的女儿金芳出嫁前往的乡镇相关情节里，孙少安口袋里舍不得抽的香烟是什么牌子？ | final_question 退化为最后一跳单跳问题 |
| 163 | WARN | 2 | `corpus_chunkids_000052,corpus_chunkids_000319` | 纸烟 | 田福堂准备召开社员大会动员搬迁前，主动提出去和金俊文、金俊武两兄弟协商搬迁事宜的人，给前来谈事的孙玉亭递了什么物品？ | answer_aliases 未包含 final_answer; final_answer 非最后一跳 answer，也不在 aliases 中 |
| 164 | FAIL | 3 | `corpus_chunkids_000051,corpus_chunkids_000201,corpus_chunkids_000075` | 双水村 | 第1步回答“金俊山的儿子叫什么名字？”；第2步回答“兰香上学的地点是哪里？”；第3步回答“孙少安开启农民生涯时决心要在哪个村子做出众的庄稼人？… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 165 | FAIL | 3 | `corpus_chunkids_000051,corpus_chunkids_000063,corpus_chunkids_000065` | 双水村 | 《平凡的世界》中，在金俊山的女儿金芳出嫁的镇子里，口袋里揣着舍不得抽的金丝猴牌香烟的孙少安，当时告知河南老师傅自己来自哪个村子？ | final_question 泄露中间 hop2 answer |
| 166 | FAIL | 3 | `corpus_chunkids_000052,corpus_chunkids_000319,corpus_chunkids_000320` | 小学校的院子 | 曾给前来谈事的孙玉亭递过纸烟的金俊山，在田福堂筹备动员搬迁的社员大会时主动请缨去和金俊文、金俊武两兄弟协商搬迁事宜，请问双水村召开这次全体社员… | final_question 泄露中间 hop1 answer; final_question 泄露中间 hop2 answer; answer_aliases 未包含 final_answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 167 | FAIL | 2 | `corpus_chunkids_000053,corpus_chunkids_000052` | 纸烟 | 孙玉亭提议找人去给徐主任赔罪充数的那次事件中，金俊山给前来谈事的孙玉亭递了什么物品？ | final_question 退化为最后一跳单跳问题 |
| 168 | FAIL | 2 | `corpus_chunkids_000052,corpus_chunkids_000048` | 太原钢厂 | 第1步回答“孙玉亭提议在公社会战指挥部的批判会上批判的人是谁？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？”。请沿着这… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 169 | FAIL | 3 | `corpus_chunkids_000052,corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 第1步回答“孙玉亭提议在公社会战指挥部的批判会上批判的人是谁？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？”；第3步回… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 170 | FAIL | 3 | `corpus_chunkids_000053,corpus_chunkids_000052,corpus_chunkids_000319` | 金俊山 | 第1步回答“孙玉亭提议去给徐主任赔罪充数的人是谁？”；第2步回答“金俊山给前来谈事的孙玉亭递了什么物品？”；第3步回答“田福堂准备召开社员大会… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 171 | WARN | 3 | `corpus_chunkids_000053,corpus_chunkids_000183,corpus_chunkids_000184` | 田福堂的老婆 | 在孙玉亭离开金俊山家后要赶往的地点所在的村落中，为打棺材安排人伐倒队里树木的金俊山，和金俊武一起在半路上被谁碰见后带到了田福堂家？ | answer_aliases 未包含 final_answer |
| 172 | FAIL | 2 | `corpus_chunkids_000053,corpus_chunkids_000183` | 槐树 | 孙玉亭离开谁家之后会前往金家湾后面的小学，这个人为打棺材安排人伐倒了队里的哪种树？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 173 | PASS | 2 | `corpus_chunkids_000054,corpus_chunkids_000190` | 贺凤英 | 双水村的哪位妇女主任是孙少安带回的山西姑娘的远房本家人？ | 通过 |
| 174 | PASS | 3 | `corpus_chunkids_000054,corpus_chunkids_000190,corpus_chunkids_000235` | 春节 | 孙少安带回双水村的那位身为双水村妇女主任远房本家的山西姑娘，和孙少安计划的结婚时间是什么时候？ | 通过 |
| 175 | WARN | 2 | `corpus_chunkids_000055,corpus_chunkids_000033` | 水果糖 | 孙少平回到家后，塞给兰花女婿的两个孩子的物品是什么？ | 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 176 | WARN | 3 | `corpus_chunkids_000054,corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 在提及今晚双水村召开的批判大会主席台桌面铺有特定物品的相关剧情中，早年工作时能顿顿吃白蒸馍大肉菜的孙玉亭，少安妈提出可以拿给他的物品是什么？ | answer_aliases 未包含 final_answer |
| 177 | FAIL | 2 | `corpus_chunkids_000054,corpus_chunkids_000048` | 太原钢厂 | 第1步回答“今晚双水村召开的批判大会的主席台桌面铺的是什么物品？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？”。请沿着… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 178 | FAIL | 3 | `corpus_chunkids_000055,corpus_chunkids_000033,corpus_chunkids_000022` | 罐子村 | 第1步回答“兰花的女婿是谁？”；第2步回答“孙少平回家后塞给猫蛋和狗蛋的物品是什么？”；第3步回答“孙少平骑行至哪个地点时看见了站在公路边的兰… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 179 | FAIL | 2 | `corpus_chunkids_000055,corpus_chunkids_000188` | 金俊山 | 第1步回答“双水村的书记是谁？”；第2步回答“双水村大队的副书记是谁？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 180 | PASS | 3 | `corpus_chunkids_000055,corpus_chunkids_000188,corpus_chunkids_000049` | 金俊山 | 双水村的书记去公社开会不在村里时，孙玉亭要找的双水村大队副书记是谁？ | 通过 |
| 181 | FAIL | 2 | `corpus_chunkids_000057,corpus_chunkids_000056` | 田福顺 | 在大批判会上拿起话筒喊话要求民兵小分队严防阶级敌人破坏捣乱的人所参与的那场批判会中，双水村被确定为批判对象的田二的本名是什么？ | final_question 退化为最后一跳单跳问题 |
| 182 | PASS | 2 | `corpus_chunkids_000056,corpus_chunkids_000050` | 田福顺 | 孙玉亭过了哭咽河后想到的可以充当批判角色的双水村被确定为批判对象的人的本名是什么？ | 通过 |
| 183 | WARN | 2 | `corpus_chunkids_000057,corpus_chunkids_000048` | 太原钢厂 | 《平凡的世界》中，走在十几个被劳教的“阶级敌人”最前面的人的岳父之弟孙玉亭，早年间在哪家单位工作时能顿顿吃上白蒸馍大肉菜？ | answer_aliases 未包含 final_answer |
| 184 | PASS | 3 | `corpus_chunkids_000057,corpus_chunkids_000056,corpus_chunkids_000050` | 田福顺 | 在大批判会上拿起话筒喊话要求民兵小分队严防阶级敌人破坏捣乱的那场批判会中，孙玉亭过了哭咽河后想到的可以充当批判角色的人的本名是什么？ | 通过 |
| 185 | FAIL | 3 | `corpus_chunkids_000056,corpus_chunkids_000050,corpus_chunkids_000060` | 世事要变了 | 孙玉亭过了哭咽河后想到的可以充当批判角色的那位双水村批判对象田二，平日里最爱说的一句话是什么？ | final_question 泄露中间 hop2 answer |
| 186 | FAIL | 3 | `corpus_chunkids_000057,corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 第1步回答“走在十几个被劳教的“阶级敌人”最前面的人是谁？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？”；第3步回答“… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 187 | FAIL | 2 | `corpus_chunkids_000058,corpus_chunkids_000048` | 太原钢厂 | 第1步回答“双水村召开的批判大会上，被孙玉亭恭请讲话的公社徐主任全名是什么？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 188 | FAIL | 3 | `corpus_chunkids_000058,corpus_chunkids_000048,corpus_chunkids_000289` | 旧棉花 | 第1步回答“双水村召开的批判大会上，被孙玉亭恭请讲话的公社徐主任全名是什么？”；第2步回答“孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 189 | FAIL | 2 | `corpus_chunkids_000058,corpus_chunkids_000057` | 杨高虎 | 第1步回答“双水村的憨老汉田二在批判大会的阶级敌人队列中反复嘟囔的内容是什么？”；第2步回答“在大批判会上拿起话筒喊话要求民兵小分队严防阶级敌… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 190 | FAIL | 3 | `corpus_chunkids_000058,corpus_chunkids_000057,corpus_chunkids_000056` | 田福顺 | 第1步回答“双水村的憨老汉田二在批判大会的阶级敌人队列中反复嘟囔的内容是什么？”；第2步回答“在大批判会上拿起话筒喊话要求民兵小分队严防阶级敌… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 191 | FAIL | 2 | `corpus_chunkids_000059,corpus_chunkids_000056` | 田福顺 | 文革时期孙玉亭曾企图扯碎双水村被确定为批判对象的某位村民的大红烟布袋，请问这位村民的本名是什么？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 192 | FAIL | 2 | `corpus_chunkids_000059,corpus_chunkids_000056` | 田福顺 | 第1步回答“田二的儿子叫什么名字？”；第2步回答“双水村被确定为批判对象的田二的本名是什么？”。请沿着这些线索逐步检索，最终答案是什么？ | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 193 | FAIL | 3 | `corpus_chunkids_000059,corpus_chunkids_000056,corpus_chunkids_000050` | 憨牛 | 本名为田福顺的双水村被批判对象，是孙玉亭过了哭咽河后想到的可以充当批判角色的人，请问他的儿子叫什么名字？ | final_question 泄露中间 hop2 answer |
| 194 | FAIL | 2 | `corpus_chunkids_000060,corpus_chunkids_000052` | 纸烟 | 在双水村本次批判会的举办地点，金俊山给前来谈事的孙玉亭递了什么物品？ | final_question 退化为最后一跳单跳问题 |
| 195 | FAIL | 2 | `corpus_chunkids_000060,corpus_chunkids_000059` | 憨牛 | 《平凡的世界》中最爱说"世事要变了"的角色的儿子叫什么名字？ | final_question 泄露中间 hop1 answer; 未发现明确的早期 hop 非答案线索，需复核多跳必要性 |
| 196 | FAIL | 3 | `corpus_chunkids_000059,corpus_chunkids_000056,corpus_chunkids_000050` | 田二 | 第1步回答“文革时期孙玉亭企图扯碎田二的什么物品？”；第2步回答“双水村被确定为批判对象的田二的本名是什么？”；第3步回答“孙玉亭过了哭咽河后… | final_question 有合成/查表痕迹; final_question 泄露 final_answer; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 197 | FAIL | 2 | `corpus_chunkids_000061,corpus_chunkids_000060` | 世事要变了 | 孙玉亭收拾停当会场、最后一个离开学校院子走到土坡下面时，发现立在哭咽河畔的两个人中的田二最爱说的一句话是什么？ | final_question 退化为最后一跳单跳问题 |
| 198 | PASS | 3 | `corpus_chunkids_000060,corpus_chunkids_000052,corpus_chunkids_000319` | 纸烟 | 田福堂准备召开社员大会动员搬迁前，主动提出去和金俊文、金俊武两兄弟协商搬迁事宜的人，在双水村本次批判会的举办地点同孙玉亭谈事时，给对方递了什么… | 通过 |
| 199 | FAIL | 3 | `corpus_chunkids_000060,corpus_chunkids_000059,corpus_chunkids_000056` | 田福顺 | 第1步回答“田二最爱说的一句话是什么？”；第2步回答“田二的儿子叫什么名字？”；第3步回答“双水村被确定为批判对象的田二的本名是什么？”。请沿… | final_question 有合成/查表痕迹; answer_aliases 重复; final_question 退化为最后一跳单跳问题 |
| 200 | PASS | 2 | `corpus_chunkids_000061,corpus_chunkids_000330` | 刘玉升 | 孙玉亭与田二父子三人相跟着返回的地点发生了孙玉亭在王彩娥窑里出事的事件，请问是谁摸黑赶到王家庄向王彩娥娘家人报告了这一消息？ | 通过 |
