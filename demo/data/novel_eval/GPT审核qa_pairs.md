# GPT 审核 qa_pairs.jsonl

## 审核依据

- 主依据：`demo/README.md` 的 `Step 4: 合成多跳 QA`。
- 审核对象：`demo/data/novel_eval/qa_pairs.jsonl`。
- 证据来源：`demo/data/novel/corpus.jsonl`；同时检查 hop 是否仍存在于当前 `demo/data/novel_eval/seeds_clean.jsonl`。
- 判定口径：结构检查通过不等于语义通过；只要存在答案泄露、单跳退化、机械拼接、final answer 不短或不匹配，即判为不满足 Step 4。

## 汇总统计

- 输入记录数：50
- 结论分布：通过=7；需修订=3；不通过=40
- hop_count 分布：2=25；3=25
- 问题码分布：ALIAS_INVALID=2；ANSWER_LEAK=26；ANSWER_NOT_SHORT=2；EVIDENCE_WEAK=1；FINAL_ANSWER_MISMATCH=5；HOP_REDUNDANT=8；HOP_SEED_NOT_IN_CLEAN=2；INCOMPLETE_ANSWER=5；SINGLE_HOP_DEGENERATE=17；SYNTHETIC_TRACE=6；WEAK_LINKAGE=24
- 结构/元数据自动检查异常：ALIAS_INVALID=1；HOP_SEED_NOT_IN_CLEAN=2

## 问题码说明

| 问题码 | 含义 |
| --- | --- |
| STRUCTURE_OK | 结构字段、hop_count、hop 顺序、chunk 引用等规则项通过。 |
| ANSWER_LEAK | final_question 直接包含中间 hop answer 或 final_answer。 |
| SYNTHETIC_TRACE | final_question 有“先确定/再结合/最后/先弄清”等合成轨迹。 |
| SINGLE_HOP_DEGENERATE | 只看最后一跳或单个 chunk 即可回答，未体现多跳必要性。 |
| WEAK_LINKAGE | hop 间只是表面词、同名、类别或牵强联想，缺少真实剧情/实体/事件依赖。 |
| HOP_REDUNDANT | 链路中存在对 final_question 没有必要的 hop。 |
| FINAL_ANSWER_MISMATCH | final_answer 不符合最后一跳短答案默认规则，或与问题所问不匹配。 |
| INCOMPLETE_ANSWER | final_question 实际询问多个事实，但 final_answer 只回答其中一部分。 |
| ANSWER_NOT_SHORT | final_answer 过长或包含多个并列事实。 |
| ALIAS_INVALID | answer_aliases 重复、超过/少于要求或引入冲突事实。 |
| HOP_SEED_NOT_IN_CLEAN | hop 三元组已不在当前 seeds_clean.jsonl 中，链路来源需复核。 |
| EVIDENCE_WEAK | hop answer 或 final_answer 与对应 chunk 证据不够稳。 |
| FIELD_MISSING | 缺少顶层或 hop 必需字段。 |
| HOP_COUNT_INVALID | hop_count 与 hops 长度不一致或 hop_count < 2。 |
| HOP_ORDER_INVALID | hop_idx 不是从 1 连续递增。 |
| CHUNK_MISSING | hop doc_chunk_id 不存在于 corpus。 |
| DUP_CHUNK | 同一条样本内重复使用 doc_chunk_id。 |
| TYPE_INVALID | 顶层 qa_type 不是 inference 或 hop qa_type 非法。 |

## 逐条审核结果

| 序号 | hop_count | subset | 结论 | 满足 Step 4 | 问题码 | hop_doc_ids | final_question | final_answer | 审查说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, SINGLE_HOP_DEGENERATE | xkx_0001,tlbb_0001 | 《金庸作品集》新序中提到，主要人物具备“有武功”这一能力的那类小说，其情节主要侧重什么内容？ | 激烈的斗争 | final_question 直接写出 hop1 答案“有武功”，且核心答案只需 hop2。 |
| 2 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, WEAK_LINKAGE, SINGLE_HOP_DEGENERATE | xkx_0001,tlbb_0001 | 《鲁滨逊飘流记》中那个仆人的名字对应一周中的一天；如果说的是这一天，武侠小说的情节通常主要侧重什么内容？ | 激烈的斗争 | 用“星期五=一周中的一天”连接到武侠小说情节，属于表面联想，hop1 对回答无真实必要。 |
| 3 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, SINGLE_HOP_DEGENERATE, HOP_REDUNDANT | xkx_0001,tlbb_0001,yttlj_0001 | 《金庸作品集》新序中提到，主要人物有武功、情节侧重激烈斗争的是武侠小说；那么在并非这一类型的《鲁滨逊飘流记》中，出现的仆人叫什么名字？ | 星期五 | final_question 暴露“有武功/激烈斗争”等中间信息，最后只是在问《鲁滨逊飘流记》仆人名。 |
| 4 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE | xkx_0002,yttlj_0002 | 阅读小说时，读者会把哪两部分内容结合起来，并在这种基础上使自己的个性与感情和小说中所表现的个性与感情相接触，从而产生什么？ | 化学反应 | 两个 hop 来自高度重复的序文片段，单个相关 chunk 已能回答结合内容和化学反应。 |
| 5 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, FINAL_ANSWER_MISMATCH, HOP_REDUNDANT | xkx_0002,yttlj_0002,tlbb_0002 | 阅读小说时，读者会把小说的内容与自己的心理状态结合起来；在这种情况下，当读者的个性与感情和小说中所表现的个性与感情相接触时，会产生什么？ | 化学反应 | final_question 直接写出 hop1 答案，final_answer 对应 hop2 而非最后一跳，hop3 冗余。 |
| 6 | 2 | 2hop_novel_stepwise | 不通过 | 否 | INCOMPLETE_ANSWER, SINGLE_HOP_DEGENERATE | xkx_0002,yttlj_0002 | 在艺术领域，判断作品好坏被归入什么范畴；而在这一范畴下，当读者的个性与感情和小说中所表现的个性与感情相接触时，会产生什么？ | 化学反应 | 问题同时问“范畴”和“产生什么”，final_answer 只答后半部分，且后半部分可单跳回答。 |
| 7 | 2 | 2hop_novel_stepwise | 通过 | 是 | - | xkx_0003,yttlj_0003 | 两段文字都提到中国人长期以来秉持的文艺观，这一共同观点是什么？ | 文以载道 | 两段文字共同指向同一文艺观，需比较两个 hop 后确认共同答案。 |
| 8 | 3 | 3hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, ANSWER_LEAK, SINGLE_HOP_DEGENERATE | xkx_0001,tlbb_0001,yttlj_0001 | 如果先确定《鲁滨逊飘流记》里的仆人名叫“星期五”，再结合对武侠小说情节通常侧重内容的概括，最后回到《鲁滨逊飘流记》确认，这位仆人叫什么？ | 星期五 | final_question 有“先确定/再结合/最后回到”合成痕迹，并直接泄露最终答案“星期五”。 |
| 9 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, SINGLE_HOP_DEGENERATE | xkx_0002,yttlj_0002,tlbb_0002 | 既然在艺术领域判断作品好坏属于美的范畴，那么在会让读者的个性与感情同小说中表现的个性与感情发生“化学反应”的小说阅读过程中，读者会将什么与自己的心理状态结合起来？ | 小说的内容 | final_question 泄露“美的范畴/化学反应”，最终只问 hop3 的单跳事实。 |
| 10 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0004,tlbb_2885 | 那位表示自己创作武侠小说并没有“载道”意图的作者，前年在北京和谁会谈？ | 何祚麻先生 | final_question 写出 hop1 答案“没有”，该 hop 对定位会谈对象帮助很弱。 |
| 11 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE, WEAK_LINKAGE | xkx_0005,tlbb_0006 | 哪个门派的武功被视作武学中的泰山北斗，这种武功具体叫什么？ | 少林武功 | 两个 hop 答案同义，问题实质可由“少林武功”单跳回答。 |
| 12 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0004,tlbb_2885,yttlj_1294 | 那位表示自己创作武侠小说并无载道意图、且前年在北京会谈对象为何祚麻先生的作者，其作品中空相为表示不敢携带兵刃进入武当观时，把什么物品交给了另一名道人？ | 戒刀 | final_question 泄露“何祚麻先生”，前两跳与空相交出戒刀的剧情关联牵强。 |
| 13 | 3 | 3hop_novel_stepwise | 通过 | 是 | - | xkx_0005,tlbb_0006,xkx_0006 | 在这些片段里，哪一个门派的武功一再被称为武学中的“泰山北斗”？ | 少林武功 | 多段证据共同确认同一门派武功一再被称为泰山北斗。 |
| 14 | 3 | 3hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE, HOP_REDUNDANT | xkx_0003,yttlj_0003,tlbb_0003 | 文中提到，“中国人长期以来的文艺观”和“中国人长期以来秉持的文艺观”其实是同一种说法。明确这一共同观念后，再回答：与之相对照，中世纪欧洲的所有绘画都以什么为题材？ | 圣经故事 | 最终题目可在包含中世纪欧洲绘画题材的单个 chunk 中回答，前两跳重复且冗余。 |
| 15 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE, SINGLE_HOP_DEGENERATE | xkx_0006,tlbb_0006 | 在举例说爱好某类菜的人不必主张禁止其他各类菜系、其中所指的是广东菜的那段内容之外，另一段内容中被视作武学中“泰山北斗”的武功是什么？ | 少林武功 | final_question 泄露“广东菜”，且该 hop 与少林武功问题缺少真实依赖。 |
| 16 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_NOT_SHORT, FINAL_ANSWER_MISMATCH, INCOMPLETE_ANSWER | xkx_0006,tlbb_0006,yttlj_0006 | 文中先举例说，爱好哪类菜的人不必主张禁止其他各类菜系；而在另外两段文字中，又有哪种武功都被称为武学中的“泰山北斗”？ | 广东菜，少林武功 | final_answer 合并“广东菜，少林武功”两个事实，违反短答案和默认最后一跳答案规则。 |
| 17 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0006,xajh_1191 | 金庸小说里，被视作武学领域“泰山北斗”的是少林武功；而在另一部作品中，令狐冲摸索铁板上的字迹时，唯独没有出现哪个字？ | 剑 | final_question 直接写出“少林武功”，再转到令狐冲刻字，链路只是牵强排除。 |
| 18 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE, WEAK_LINKAGE | xkx_0007,xajh_0007 | 金庸曾把天津百花文艺出版社支付的《书剑恩仇录》版税缴税后的余数捐给几家文化机构并支助围棋活动，那么这种翻版本向作者支付版税吗？ | 不付 | 版税捐赠与翻版本不付版税只是对照，最终答案由 hop2 单独给出。 |
| 19 | 3 | 3hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, WEAK_LINKAGE, SINGLE_HOP_DEGENERATE | xkx_0006,xajh_1191,xajh_1190 | 令狐冲摸到一处刻字，其中没有出现的那个字并不在被视作武学领域泰山北斗的武功名称里；留下这处刻字的人是谁？ | 任我行 | “并不在武功名称里”的连接是人为拼接，留下刻字者可由最后一跳回答。 |
| 20 | 3 | 3hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE, HOP_REDUNDANT | xkx_0008,tlbb_0007,xkx_0007 | 除冯其庸、严家炎外，其自行点评也得到金庸认可的那位先生所提到的版本情况中，在中国大陆“三联版”出版之前，经金庸授权出版《书剑恩仇录》的那家出版社所支付的版税，金庸缴付所得税后的余数后来被他用来做什么？ | 捐给几家文化机构及支助围棋活动 | 最终问题已经描述出版与版税背景，答案主要由 xkx_0007 单个 chunk 支持，前两跳必要性不足。 |
| 21 | 2 | 2hop_novel_stepwise | 通过 | 是 | - | xkx_0010,yttlj_0010 | 唐人传奇中，那个既能缩小身体潜入别人肚肠、其武侠故事又千余年来一直为人所喜爱的人物叫什么名字？ | 聂隐娘 | 两个 hop 分别给出同一人物的不同特征，组合后才能确认目标人物。 |
| 22 | 3 | 3hop_novel_stepwise | 需修订 | 部分 | ALIAS_INVALID, HOP_SEED_NOT_IN_CLEAN | xkx_0010,yttlj_0010,xajh_0010 | 唐人传奇中，那个既能缩小身体潜入别人肚肠、其武侠故事又千余年来一直为人所喜爱的人物，出自哪部作品？ | 《聂隐娘》 | 主链路可成立，但 answer_aliases 重复且包含人物名“聂隐娘”作为作品别名，hop3 来源也不在当前 seeds_clean 中。 |
| 23 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, ANSWER_LEAK, WEAK_LINKAGE | xkx_0010,yttlj_0010 | 以作者后期创作中“中华民族各族一视同仁”这一核心基调为线索，文中提到哪位人物的武侠故事千余年来一直为人所喜爱？ | 聂隐娘 | final_question 泄露“中华民族各族一视同仁”，该基调与聂隐娘故事缺少真实依赖。 |
| 24 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, SINGLE_HOP_DEGENERATE | xkx_0008,tlbb_0007 | 除冯其庸、严家炎外，另一位自行点评得到其认可的先生是陈墨；那么，这位认可者的《书剑恩仇录》在中国大陆“三联版”出版之前，经其授权是由哪家出版社出版的？ | 天津百花文艺出版社 | final_question 直接写出“陈墨”，最后只问出版者，可由 hop2 回答。 |
| 25 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0007,xajh_0007,yttlj_0007 | 金庸曾把天津百花文艺出版社支付的《书剑恩仇录》版税缴税后的余款捐给几家文化机构并支助围棋活动；而与这种正规出版相对，翻版本连版税都不付。说到这类未经授权的情形，在自行点评金庸作品的人士中，除冯其庸、严家炎外，还有哪位先生？ | 陈墨 | final_question 泄露“不付”，把授权版税与翻版本不付版税硬接到点评者陈墨。 |
| 26 | 3 | 3hop_novel_stepwise | 不通过 | 否 | HOP_SEED_NOT_IN_CLEAN, ANSWER_LEAK, WEAK_LINKAGE | xkx_0010,yttlj_0010,xajh_0010 | 以后期创作小说的核心基调“中华民族各族一视同仁”著称的这位作者，曾提到一个千余年来一直为人喜爱的武侠人物；这位人物有缩小身体潜入别人肚肠的故事，这个故事出自哪部作品？ | 《聂隐娘》 | final_question 直接写出“中华民族各族一视同仁”，与聂隐娘出处的链路牵强。 |
| 27 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK | xkx_0011,xajh_0012 | 刘再复和与他是父女关系的刘剑梅合写的作品叫什么？ | 父女两地书 | final_question 直接写出关系答案“父女”，违反无中间答案泄露。 |
| 28 | 3 | 3hop_novel_stepwise | 需修订 | 部分 | HOP_REDUNDANT, WEAK_LINKAGE | xkx_0011,tlbb_0011,xkx_0012 | 《书剑恩仇录》的主角并不是那位既和刘再复合写过“父女两地书”又与他合写《共悟人间》的人；那么，后者是谁？ | 刘剑梅 | 最终答案由“父女两地书/共悟人间”两跳即可定位，陈家洛排除条件弱且可删。 |
| 29 | 2 | 2hop_novel_stepwise | 通过 | 是 | - | xkx_0012,xajh_0012 | 与刘再复合写《共悟人间》的那位人物，后来又与刘再复合写的作品叫什么？ | 父女两地书 | 先由《共悟人间》合作者定位刘剑梅，再查询其与刘再复合写作品，链路自然。 |
| 30 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, FINAL_ANSWER_MISMATCH, INCOMPLETE_ANSWER | xkx_0011,xajh_0012,yttlj_0012 | 与刘再复是父女关系的那位女儿叫什么名字？她和刘再复合写的作品叫什么？ | 父女两地书 | 问题同时问姓名和作品，却只答作品；还直接泄露“父女”关系。 |
| 31 | 3 | 3hop_novel_stepwise | 不通过 | 否 | FINAL_ANSWER_MISMATCH, HOP_REDUNDANT | xkx_0012,xajh_0012,yttlj_0012 | 刘再复的女儿中，曾与他合写《共悟人间》的那一位，后来又与他合写了哪部作品？ | 父女两地书 | final_answer 对应 hop2，不是最后一跳；hop3 反向重复定位刘剑梅，链路顺序不合格。 |
| 32 | 2 | 2hop_novel_stepwise | 需修订 | 部分 | WEAK_LINKAGE | xkx_0011,tlbb_0011 | 与《书剑恩仇录》主角一样名字也是三个字、并且和刘再复合写过《父女两地书》的人是谁？ | 刘剑梅 | 通过“三个字”连接陈家洛和刘剑梅较机械，建议改成更真实的人物/作品依赖。 |
| 33 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0012,tlbb_0012 | 那位在二○○二年四月于香港写下这篇文字的作者，后来与谁合写了《共悟人间》？ | 刘剑梅 | final_question 泄露“香港”，写作地点对合作者刘剑梅的推理依赖很弱。 |
| 34 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE, HOP_REDUNDANT | xkx_0012,tlbb_0012,xajh_0012 | 二○○二年四月在香港撰写这篇文字的作者，与那位曾和他合写《共悟人间》的人一起写的作品叫什么？ | 父女两地书 | final_question 泄露“香港”，地点 hop 对合写作品问题贡献不足。 |
| 35 | 2 | 2hop_novel_stepwise | 通过 | 是 | - | xkx_0014,tlbb_1147 | 当乔峰来到战国时魏国大梁城所在、也就是如今的哪个地方时，为避免受守关官兵盘查，他选择从何处绕道而行？ | 关西的高岭 | 先确定大梁城今地，再以该地理参照回答乔峰绕行地点，符合两跳依赖。 |
| 36 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SINGLE_HOP_DEGENERATE, HOP_REDUNDANT | xkx_0015,xkx_0014 | 在杂货铺里，伙计喊出“老哥们来啦”后，王掌柜当即作势要教训他；而他们随后谈到的战国时魏国大梁城，放到今天是哪里？ | 河南开封 | 杂货铺伙计动作只是剧情背景，最终“大梁城今地”由 hop2 单跳回答。 |
| 37 | 2 | 2hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, WEAK_LINKAGE | xkx_0014,yttlj_1050 | 先弄清侯监集是因哪位历史人物得名，再回答书中提到的明教是从哪里传入中土的？ | 波斯 | “先弄清侯监集得名人物”与明教来源无真实依赖。 |
| 38 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_NOT_SHORT, FINAL_ANSWER_MISMATCH, INCOMPLETE_ANSWER | xkx_0014,tlbb_1147,tlbb_1146 | 战国时魏国的大梁城放到今天是哪里？以这一带为参照继续往西时，乔峰为避开守关官兵是从哪里绕道而行的，又在后来到了京西路汝州梁县、银两用尽后做了什么？ | 大梁城在今河南开封；乔峰从关西的高岭绕道而行，后来潜入县衙公库盗了几百两银子。 | final_answer 是长句组合三件事，违反短答案要求，也偏离默认最后一跳答案。 |
| 39 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0014,yttlj_1050,yttlj_1655 | 侯监集得名于历史人物侯赢；而在《倚天屠龙记》中，明教是从波斯传入中土的。根据这一来源地，铸造十二枚明教圣火令的人是谁？ | 霍山 | final_question 泄露“侯赢/波斯”，侯监集得名与圣火令铸造者无必要关系。 |
| 40 | 2 | 2hop_novel_stepwise | 不通过 | 否 | INCOMPLETE_ANSWER, SINGLE_HOP_DEGENERATE | xkx_0015,tlbb_2393 | 杂货铺里，那个在贫嘴伙计话未说完时被马上的大汉用来勾住脖子的物品是什么；而在另一场景中，慕容博见金算盘崔百泉和过彦之急扑过来后，二人分别中了什么？ | 袖中指 | 问题问马鞭和袖中指两部分，final_answer 只答后半部分；可由 hop2 单独回答。 |
| 41 | 2 | 2hop_novel_stepwise | 通过 | 是 | - | xkx_0016,xkx_0015 | 后边赶上来的马前蹄踩中大腿的那个人，在杂货铺里说出“老哥们来啦”后，王掌柜作势要对他做什么动作？ | 往那伙计头顶拍落 | 先由被踩中者定位“那伙计”，再问其在杂货铺中受到的动作，链路清晰。 |
| 42 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, SINGLE_HOP_DEGENERATE, WEAK_LINKAGE | xkx_0016,tlbb_1432 | 当皮靴声最终停在烧饼铺外时，阿紫用什么东西擦拭左脚的皮靴？ | 红烧牛肉 | final_question 泄露“烧饼铺外”，阿紫擦靴事实可由 hop2 单跳回答。 |
| 43 | 3 | 3hop_novel_stepwise | 通过 | 是 | - | xkx_0016,xkx_0015,xkx_0014 | 那个被后边赶上来的马前蹄踩中大腿、又在喊出“老哥们来啦”后让杂货铺王掌柜作势往他头顶拍落的伙计所在情节中，提到战国时魏国的大梁城如今是哪里？ | 河南开封 | 通过连续剧情定位伙计及杂货铺情节，再追问同一情节提到的大梁城今地，链路可成立。 |
| 44 | 3 | 3hop_novel_stepwise | 不通过 | 否 | SYNTHETIC_TRACE, ANSWER_LEAK, WEAK_LINKAGE | xkx_0015,xkx_0014,tlbb_1147 | 杂货铺里王掌柜听到伙计说“老哥们来啦”后，作势要往伙计头顶拍落，让人先想到“高处”；再想到战国魏国的大梁城今在河南开封，属关东一带，那么乔峰为避开守关官兵，最终是从哪处地方绕道而行的？ | 关西的高岭 | “让人先想到高处”是人为联想，并直接泄露“河南开封”。 |
| 45 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, EVIDENCE_WEAK, WEAK_LINKAGE | xkx_0016,tlbb_1432,xkx_0034 | 皮靴声最终停在烧饼铺外时，阿紫曾用红烧牛肉擦拭过的那只皮靴，后来石清是从周牧的什么位置搜出了得自吴道通的物件？ | 左脚皮靴的靴筒 | final_question 泄露“烧饼铺外/红烧牛肉”，且最终位置答案在证据中是语义改写，链路牵强。 |
| 46 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0017,xkx_0562 | 文中先写到高个儿从腰间抽出一种兵器，那兵器是双钩；那么，长白山畔快刀门的掌门人是谁？ | 吕正平 | final_question 泄露“双钩”，该兵器与快刀门掌门问题缺少必要依赖。 |
| 47 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0017,xkx_0562,xkx_0563 | 那个从腰间抽出双钩的高个儿出现后，文中又提到长白山畔快刀门的掌门是吕正平，那么与他并提的鹤笔门掌门是谁？ | 范一飞 | final_question 泄露“双钩/吕正平”，只是把相邻门派掌门并列串接。 |
| 48 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE, ALIAS_INVALID | xkx_0015,tlbb_2393,tlbb_0601 | 那个曾与过彦之一同扑向慕容博、结果中了“袖中指”的“金算盘”崔百泉，外号里的器物名和另一段里马上的大汉勾住杂货铺伙计脖子的“马鞭”一样都是三字器物名；那么他随后从怀里掏出的那件金光灿烂之物是什么？ | 算盘 | final_question 同时泄露“马鞭/袖中指/算盘”，且“金算盘”作为别名会混淆人物外号和器物。 |
| 49 | 3 | 3hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK, WEAK_LINKAGE | xkx_0017,xajh_0108,xajh_0036 | 那个自称是侯监集人的卖饼老者，与卖酒少女是师兄妹；在这两人出现的相关情节里，林平之杀死那个汉子时用的是什么武器？ | 匕首 | final_question 泄露“侯监集/师兄妹”，再转到林平之杀人武器，链路依赖弱。 |
| 50 | 2 | 2hop_novel_stepwise | 不通过 | 否 | ANSWER_LEAK | xkx_0017,xajh_0108 | 那个自称是侯监集人的卖饼老者，与卖酒少女是什么关系？ | 师兄妹 | final_question 直接写出 hop1 答案“侯监集”，违反中间答案不泄露。 |
