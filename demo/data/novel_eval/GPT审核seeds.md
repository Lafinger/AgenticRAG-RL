# GPT 审计 seeds.jsonl
- 审计对象：`demo/data/novel_eval/seeds.jsonl`
- 支撑语料：`demo/data/novel/corpus.jsonl`
- 更新时间：2026-04-29 22:32:08
- 删除前样本数：688
- 本次删除 FAIL：103
- 删除后样本数：585

## 删除结果
- 删除前 PASS：105
- 删除前 WARN：480
- 删除前 FAIL：103
- 删除后 FAIL：0

说明：本次按上一版审计报告中的 `FAIL` 结论删除对应 `seeds.jsonl` 行。保留下来的 WARN 仍建议复核，尤其是时间/阶段限定、答案唯一性、entities 准确性。

## 已删除 FAIL 明细
| 原行号 | 结论 | doc_chunk_id | qa_type | answer | 问题 | 审计意见 |
| ---: | --- | --- | --- | --- | --- | --- |
| 7 | DELETED | `corpus_chunkids_000005` | `action_result` | 回家务农 | 孙少平的大哥十三岁高小毕业后为了供弟弟妹妹上学选择了做什么？ | 答案未在对应 chunk 中直接出现 |
| 19 | DELETED | `corpus_chunkids_000012` | `action_result` | 把书交到郝红梅手里 | 郝红梅向孙少平提出借书请求的次日，孙少平做了什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 21 | DELETED | `corpus_chunkids_000013` | `place` | 学校后面的拐沟里 | 孙少平所在班级下午挖地劳动的地点是哪里？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 班级挖地劳动 |
| 40 | DELETED | `corpus_chunkids_000024` | `character` | 孙少平 | 王满银的小舅子是谁？ | 答案未在对应 chunk 中直接出现 |
| 46 | DELETED | `corpus_chunkids_000027` | `relation` | 姐弟 | 孙少安和孙兰花是什么亲属关系？ | 答案未在对应 chunk 中直接出现 |
| 56 | DELETED | `corpus_chunkids_000033` | `action_result` | 提猪食桶出去喂猪 | 孙兰香看见母亲和姐姐坐在炕上哭后做出的行为是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 75 | DELETED | `corpus_chunkids_000044` | `relation` | 兄弟 | 孙玉厚和孙玉亭是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 82 | DELETED | `corpus_chunkids_000047` | `action_result` | 接回了贺凤英 | 孙玉亭亲自前往柳林镇的行为结果是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 100 | DELETED | `corpus_chunkids_000056` | `action_result` | 因板凳失去平衡栽倒在地 | 杨高虎站起来准备示意人群安静时孙玉亭遭遇了什么状况？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 113 | DELETED | `corpus_chunkids_000063` | `character` | 孙少平 | 孙少安的弟弟叫什么名字？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 115 | DELETED | `corpus_chunkids_000064` | `action_result` | 借火点烟 | 孙少安前往铁匠铺的最初目的是什么？ | 答案未在对应 chunk 中直接出现 |
| 120 | DELETED | `corpus_chunkids_000067` | `character` | 金俊斌 | 金俊武的弟弟叫什么名字？ | 重复 question; 重复 question+answer; chunk 含时间/阶段信息但问题未显式限定 |
| 124 | DELETED | `corpus_chunkids_000069` | `character` | 孙少平 | 兰花告诉孙少安是谁把猫蛋和狗蛋引到外面去的？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 135 | DELETED | `corpus_chunkids_000075` | `action_result` | 第三名 | 孙少安参加全县升初中统一考试的排名是多少？ | 答案未在对应 chunk 中直接出现 |
| 138 | DELETED | `corpus_chunkids_000076` | `action_result` | 不去县城 | 孙少安最初得知润叶让他去县城时的第一决定是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 140 | DELETED | `corpus_chunkids_000077` | `character` | 孙少安 | 田润叶嘱咐谁这几天到城里去找她？ | 答案未在对应 chunk 中直接出现 |
| 146 | DELETED | `corpus_chunkids_000080` | `character` | 润叶的二妈 | 向前的母亲托谁转告润叶，说向前看上了她？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 147 | DELETED | `corpus_chunkids_000081` | `character` | 小学教师 | 田润叶的职业是什么？ | 重复 question; chunk 含时间/阶段信息但问题未显式限定 |
| 149 | DELETED | `corpus_chunkids_000082` | `character` | 田晓霞 | 田润叶刚端起饭碗准备吃饭时，敲门闯进来的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 152 | DELETED | `corpus_chunkids_000083` | `place` | 晓霞的二妈家 | 润叶锁门后和晓霞一同前往的地点是哪里？ | 答案未在对应 chunk 中直接出现 |
| 162 | DELETED | `corpus_chunkids_000089` | `character` | 孙少安 | 田润叶进入学校大门后，在自己宿舍门口看到的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 181 | DELETED | `corpus_chunkids_000100` | `action_result` | 捡起冒烟的手榴弹扔了出去 | 1970年冬天城关公社民兵冬训时，白明川采取了什么行动避免了一场大灾祸？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 1970年城关公社民兵冬训 |
| 189 | DELETED | `corpus_chunkids_000104` | `relation` | 叔侄 | 孙玉亭和孙少安是什么亲属关系？ | 答案未在对应 chunk 中直接出现 |
| 190 | DELETED | `corpus_chunkids_000104` | `character` | 田福军 | 田福堂的弟弟叫什么名字？ | 答案未在对应 chunk 中直接出现 |
| 219 | DELETED | `corpus_chunkids_000119` | `action_result` | 用拳头捅到顾养民脸上 | 金波对顾养民做出的第一个攻击动作是什么？ | 答案未在对应 chunk 中直接出现 |
| 220 | DELETED | `corpus_chunkids_000120` | `place` | 校园东南角的小树林 | 被金波带人殴打后，顾养民没有返回自己的宿舍，去往了哪个地点？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 221 | DELETED | `corpus_chunkids_000120` | `action_result` | 打人事件发生后的第二天 | 孙少平得知金波串联他人殴打顾养民的时间是什么时候？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 223 | DELETED | `corpus_chunkids_000121` | `action_result` | 偷偷到医院看望顾养民 | 顾养民被打后请假去医院的当天，郝红梅做了什么？ | 答案未在对应 chunk 中直接出现 |
| 229 | DELETED | `corpus_chunkids_000125` | `place` | 双水村 | 星期六的时候，文中的“他”带着从黄原城买的稀罕东西返回的村子是哪个？ | 问题含文档/片段指代; chunk 含时间/阶段信息但问题未显式限定 |
| 230 | DELETED | `corpus_chunkids_000125` | `object` | 一包蛋糕 | 文中的“他”给奶奶买的礼物是什么？ | 问题含文档/片段指代; chunk 含时间/阶段信息但问题未显式限定 |
| 235 | DELETED | `corpus_chunkids_000128` | `character` | 孙少安 | 田润叶叮嘱孙少平星期六回去时要叫到城里来的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 236 | DELETED | `corpus_chunkids_000128` | `action_result` | 到自己家里吃饭 | 向前妈到学校找田润叶是想邀请她做什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 237 | DELETED | `corpus_chunkids_000129` | `character` | 刘志英 | 李登云的爱人叫什么名字？ | 重复 question; 重复 question+answer; chunk 含时间/阶段信息但问题未显式限定 |
| 239 | DELETED | `corpus_chunkids_000130` | `character` | 刘志英 | 李登云的爱人叫什么名字？ | 重复 question; 重复 question+answer; chunk 含时间/阶段信息但问题未显式限定 |
| 245 | DELETED | `corpus_chunkids_000133` | `character` | 孙少平 | 孙少安计划等谁高中毕业之后再考虑自己的婚姻问题？ | 答案未在对应 chunk 中直接出现 |
| 247 | DELETED | `corpus_chunkids_000134` | `character` | 孙少平 | 润叶托谁给孙少安捎话让他再到城里去？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 258 | DELETED | `corpus_chunkids_000139` | `relation` | 父子 | 孙少安和孙玉厚的关系是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 267 | DELETED | `corpus_chunkids_000144` | `relation` | 同班同学 | 顾养民和田润生是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 283 | DELETED | `corpus_chunkids_000152` | `action_result` | 他要给队里办点事 | 孙少安参加完批判会走出公社院子后，让父亲和妹妹先回家的原因是什么？ | 答案未在对应 chunk 中直接出现 |
| 290 | DELETED | `corpus_chunkids_000156` | `place` | 孙玉亭家 | 孙玉厚整夜没合眼后，第二天早晨没忙着出山去了谁家？ | 答案未在对应 chunk 中直接出现 |
| 292 | DELETED | `corpus_chunkids_000157` | `character` | 孙少安 | 孙玉厚询问孙玉亭冬天公社在村里会战时的女娃娃里有没有合适人选，是为了给谁找媳妇？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 冬天公社会战 |
| 302 | DELETED | `corpus_chunkids_000162` | `action_result` | 每天到报栏前看报纸 | 孙少平结识田晓霞后养成的、在之后各种环境都一直坚持的良好习惯是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 312 | DELETED | `corpus_chunkids_000167` | `action_result` | 打一盆热水让他泡脚 | 孙少平到金波家准备睡觉时，金秀会做什么帮他解乏？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 322 | DELETED | `corpus_chunkids_000172` | `place` | 石圪节村 | 金俊武建议双水村首先要豁的是哪个村的坝？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 334 | DELETED | `corpus_chunkids_000180` | `character` | 金俊斌 | 金俊武的弟弟叫什么名字？ | 重复 question; 重复 question+answer; 答案在 chunk 中出现多次，唯一性需复核 |
| 343 | DELETED | `corpus_chunkids_000185` | `action_result` | 追认金俊斌为党员 | 在讨论金俊斌丧事的大队党支部会议上，孙玉亭提出的额外建议是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 353 | DELETED | `corpus_chunkids_000190` | `action_result` | 给一队联系小麦良种 | 孙少安此次外出临走时对外声称的出行目的是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 366 | DELETED | `corpus_chunkids_000197` | `action_result` | 为儿子孙少安的婚事借钱 | 孙玉厚前往金俊海家的目的是什么？ | 重复 question; 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 368 | DELETED | `corpus_chunkids_000198` | `action_result` | 杀掉自家养的猪 | 孙玉厚夫妇筹办少安婚事时，打算怎么获取所需的猪肉？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 371 | DELETED | `corpus_chunkids_000199` | `action_result` | 借钱 | 孙玉厚前往金俊海家的目的是什么？ | 重复 question; 答案在 chunk 中出现多次，唯一性需复核 |
| 382 | DELETED | `corpus_chunkids_000205` | `character` | 孙少平 | 孙少安一家人去庙坪参加打枣活动时，背着老祖母的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 396 | DELETED | `corpus_chunkids_000212` | `relation` | 高中同班同学 | 柳岔公社主任周文龙和石圪节公社主任白明川是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 398 | DELETED | `corpus_chunkids_000213` | `action_result` | 捉偷跑的两个民工 | 周文龙带领民兵小分队今早上外出的目的是什么？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 偷跑民工; chunk 含时间/阶段信息但问题未显式限定 |
| 403 | DELETED | `corpus_chunkids_000215` | `relation` | 父子 | 周文龙和常到柳岔公社灶上免费吃饭的胖老头是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 404 | DELETED | `corpus_chunkids_000216` | `place` | 柳岔公社大会战工地 | 田福军、张有智在柳岔公社灶房吃完饭后，和刘志祥一同前往的地点是哪里？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 411 | DELETED | `corpus_chunkids_000219` | `action_result` | 打电话 | 周文龙向冯世宽汇报田福军、张有智在柳岔公社的活动时采用的联络方式是什么？ | 答案未在对应 chunk 中直接出现 |
| 422 | DELETED | `corpus_chunkids_000225` | `character` | 孙少安 | 临近春节的前十几天，孙玉厚一家人正忙着为谁的婚事做准备？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 春节前十几天; chunk 含时间/阶段信息但问题未显式限定 |
| 428 | DELETED | `corpus_chunkids_000228` | `relation` | 王满银是孙少安的姐夫 | 孙少安和王满银是什么亲属关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 442 | DELETED | `corpus_chunkids_000235` | `character` | 教师 | 田润叶的职业是什么？ | 重复 question |
| 448 | DELETED | `corpus_chunkids_000238` | `character` | 武惠良 | 杜莉莉的男朋友叫什么名字？ | 重复 question; 重复 question+answer; chunk 含时间/阶段信息但问题未显式限定 |
| 450 | DELETED | `corpus_chunkids_000239` | `character` | 武惠良 | 杜莉莉的男朋友叫什么名字？ | 重复 question; 重复 question+answer; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 460 | DELETED | `corpus_chunkids_000244` | `action_result` | 锄高粱地的最后一遍草 | 立秋之前孙少平所在班级到原西城外的山沟里做什么劳动？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 467 | DELETED | `corpus_chunkids_000018` | `character` | 田润生 | 高中时期，孙少平和谁是同班同学？ | 答案未在对应 chunk 中直接出现 |
| 471 | DELETED | `corpus_chunkids_000250` | `relation` | 兄妹 | 孙兰香和孙少安的亲属关系是什么? | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 480 | DELETED | `corpus_chunkids_000255` | `character` | 田润叶 | 田福堂的女儿叫什么名字？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 487 | DELETED | `corpus_chunkids_000258` | `character` | 李登云 | 李向前的父亲叫什么名字？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 490 | DELETED | `corpus_chunkids_000259` | `action_result` | 亲自去李登云家传递喜讯 | 徐国强老汉获知田润叶同意和李向前结婚后采取了什么行动？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 491 | DELETED | `corpus_chunkids_000259` | `character` | 田润生 | 徐爱云打发谁回双水村请田福堂来县城筹办田润叶的婚事？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 503 | DELETED | `corpus_chunkids_000264` | `place` | 双水村的阳土坡 | 田润叶童年时和少安刨“蛮蛮草”的地点是哪里？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 504 | DELETED | `corpus_chunkids_000267` | `action_result` | 以后干脆自己给自己盛饭 | 为了不让贺秀莲给自己捞稠饭搞特殊化，孙少安采取了什么做法？ | 答案未在对应 chunk 中直接出现 |
| 507 | DELETED | `corpus_chunkids_000269` | `character` | 奶奶（老祖母） | 秀莲放到孙少安碗里的白面馍原本是留给谁吃的？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 509 | DELETED | `corpus_chunkids_000268` | `action_result` | 给孙少安缝一件大氅 | 一九七六年临近结束前不久，贺秀莲原本打算用父亲留下的五十块钱做什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 510 | DELETED | `corpus_chunkids_000270` | `relation` | 夫妻 | 孙少安和秀莲是什么关系？ | 答案未在对应 chunk 中直接出现 |
| 511 | DELETED | `corpus_chunkids_000270` | `action_result` | 放回馍篮里 | 孙少安拿到秀莲给他的奶奶的白面馍后，对该白面馍做了什么处理？ | 答案未在对应 chunk 中直接出现 |
| 513 | DELETED | `corpus_chunkids_000271` | `action_result` | 在秀莲肩膀上捣了一拳头 | 孙少安失去理智冲动时对秀莲做出了什么行为？ | 答案未在对应 chunk 中直接出现 |
| 530 | DELETED | `corpus_chunkids_000282` | `character` | 郝红梅 | 孙少平要求侯主任让谁拿走十几块手帕？ | 答案未在对应 chunk 中直接出现 |
| 532 | DELETED | `corpus_chunkids_000280` | `relation` | 同班同学 | 郝红梅和侯玉英是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 536 | DELETED | `corpus_chunkids_000283` | `object` | 外面束着两条红丝线的精致大笔记本 | 毕业分别时侯玉英送给孙少平的礼物是什么？ | 答案未在对应 chunk 中直接出现 |
| 541 | DELETED | `corpus_chunkids_000288` | `character` | 孙少平 | 田福堂提出办初中新增教师时首先考虑的人选是谁？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 新增教师人选 |
| 546 | DELETED | `corpus_chunkids_000289` | `character` | 孙少平 | 孙玉亭打算安排谁去学校当教师？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 学校教师; chunk 含时间/阶段信息但问题未显式限定 |
| 549 | DELETED | `corpus_chunkids_000287` | `action_result` | 让田润生教书 | 孙玉亭给田福堂提出的安排田润生的工作建议是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 551 | DELETED | `corpus_chunkids_000293` | `action_result` | 与田润叶结婚 | 经过长时间不屈不挠的追求，李向前最终达成了什么结果？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 562 | DELETED | `corpus_chunkids_000302` | `action_result` | 打开各队粮库将战备粮借发给缺粮户 | 田福军发现后子头公社存在严重缺粮问题后当即做出了什么决定？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 567 | DELETED | `corpus_chunkids_000297` | `action_result` | 脸上挨了一记耳光 | 李向前醒来发现自己被田润叶安顿在脚地后，试图拥抱田润叶时得到了什么结果？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 568 | DELETED | `corpus_chunkids_000305` | `relation` | 叔侄 | 润叶和田福军是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 589 | DELETED | `corpus_chunkids_000313` | `action_result` | 四人帮'的那一套做法还在作怪 | 高老认为当前最主要的问题是什么？ | 答案未在对应 chunk 中直接出现 |
| 598 | DELETED | `corpus_chunkids_000322` | `character` | 金俊斌 | 金俊武父亲的坟下方的新坟里埋葬的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 599 | DELETED | `corpus_chunkids_000322` | `relation` | 亲兄弟 | 金俊文和金俊武是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 604 | DELETED | `corpus_chunkids_000317` | `action_result` | 撇下金俊武先做其他几户人家的搬家动员工作 | 孙玉亭给田福堂献上的应对金俊武不肯搬家的妙计内容是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 612 | DELETED | `corpus_chunkids_000326` | `action_result` | 看守现场 | 金俊武安排金强重新返回三妈的门下的任务是什么？ | 答案未在对应 chunk 中直接出现; entities 未在问答或 chunk 中出现: 三妈门下 |
| 616 | DELETED | `corpus_chunkids_000328` | `action_result` | 把裤子前后穿反了 | 孙玉亭慌乱穿衣服时出现了什么失误？ | 答案未在对应 chunk 中直接出现 |
| 621 | DELETED | `corpus_chunkids_000325` | `place` | 金俊斌家 | 王彩娥所住的窑洞位于谁家的院子中？ | 答案未在对应 chunk 中直接出现 |
| 634 | DELETED | `corpus_chunkids_000337` | `action_result` | 金老太太放开声痛哭 | 导致金俊文一家人忍不住跟着哭的直接诱因是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 639 | DELETED | `corpus_chunkids_000338` | `relation` | 母子 | 金俊武和金老太太是什么关系？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 644 | DELETED | `corpus_chunkids_000339` | `action_result` | 双膝跪倒在炕上 | 田福堂凑到金老太太身边后做出的首个动作是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 659 | DELETED | `corpus_chunkids_000355` | `relation` | 叔侄 | 孙玉亭和孙少安的亲属关系是什么？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 662 | DELETED | `corpus_chunkids_000348` | `action_result` | 给金三锤辅导作文 | 孙少平临时决定晚上去金光亮家做什么？ | 答案未在对应 chunk 中直接出现 |
| 669 | DELETED | `corpus_chunkids_000351` | `action_result` | 看儿子 | 孙少安的儿子降生后的冬天，孙少安急着收工回家是为了做什么？ | 答案未在对应 chunk 中直接出现 |
| 670 | DELETED | `corpus_chunkids_000351` | `character` | 孙少安的母亲 | 孙少安的妻子秀莲生完孩子后，大部分时间到饲养院侍候她的人是谁？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 677 | DELETED | `corpus_chunkids_000361` | `place` | 金家湾的小学院子 | 双水村此次闹秧歌的大秧歌队排练地点是哪里？ | 答案未在对应 chunk 中直接出现; chunk 含时间/阶段信息但问题未显式限定 |
| 678 | DELETED | `corpus_chunkids_000365` | `object` | 五卷集 | 文中作者称自己四十岁之前文学活动的基本总结是什么？ | 问题含文档/片段指代; 答案在 chunk 中出现多次，唯一性需复核 |
| 679 | DELETED | `corpus_chunkids_000365` | `character` | 刑良俊 | 文中提到的五卷集的编选者除了陈泽顺还有谁？ | 问题含文档/片段指代 |
| 688 | DELETED | `corpus_chunkids_000366` | `place` | 西安 | 1992年春天写下的该段文字的撰写地点是哪里？ | 问题含文档/片段指代 |

## 保留样本明细
| 原行号 | 结论 | doc_chunk_id | qa_type | answer | 问题 | 审计意见 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | PASS | `corpus_chunkids_000001` | `place` | 校园内的南墙根下 | 1999年二三月间县立高中给学生分饭菜的地点在哪里？ | 通过 |
| 2 | WARN | `corpus_chunkids_000003` | `object` | 白线绳 | 瘦高个青年人的旧黄胶鞋用什么物品代替鞋带系着？ | chunk 含时间/阶段信息但问题未显式限定 |
| 3 | WARN | `corpus_chunkids_000003` | `action_result` | 两个 | 来到饭场的瘦高个青年人在馍筐前拾了多少个高粱面馍？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 4 | PASS | `corpus_chunkids_000004` | `character` | 郝红梅 | 和孙少平一样开学后每次吃饭最后来取黑高粱面馍的女生叫什么名字？ | 通过 |
| 5 | PASS | `corpus_chunkids_000004` | `object` | 黑高粱面馍 | 开学以来孙少平每次吃饭最后来取的饭食是什么？ | 通过 |
| 6 | WARN | `corpus_chunkids_000005` | `object` | 高粱面馍 | 孙少平读高中阶段每顿饭只能啃的食物是什么？ | entities 未在问答或 chunk 中出现: 饭食 |
| 8 | WARN | `corpus_chunkids_000006` | `character` | 顾养民 | 孙少平就读高中时所在班级的班长是谁？ | entities 未在问答或 chunk 中出现: 高中班级 |
| 9 | WARN | `corpus_chunkids_000007` | `object` | 白面票 | 在县城上学期间，金波多次给孙少平塞过什么物品？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 10 | WARN | `corpus_chunkids_000007` | `action_result` | 破烂不堪 | 孙少平和金波在公社上初中的两年时间里，金波的自行车最终是什么状态？ | entities 未在问答或 chunk 中出现: 公社初中 |
| 11 | WARN | `corpus_chunkids_000008` | `character` | 保尔·柯察金 | 上初中最后一年，孙少平在润生家发现的厚书的主人公是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 12 | WARN | `corpus_chunkids_000008` | `place` | 村子打麦场的麦秸垛后面 | 孙少平星期天躲在哪里看完了那本讲保尔·柯察金的厚书？ | chunk 含时间/阶段信息但问题未显式限定 |
| 13 | WARN | `corpus_chunkids_000009` | `character` | 侯玉英 | 向班主任揭发孙少平在班上看“反动书”行为的告密者是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 14 | WARN | `corpus_chunkids_000010` | `character` | 顾养民 | 班级开展文章学习的课堂上，负责念报纸的班长是谁？ | entities 未在问答或 chunk 中出现: 班级文章学习课堂; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 15 | WARN | `corpus_chunkids_000010` | `object` | 黑高粱面馍 | 孙少平每天吃饭时等众人散尽后取的食物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 16 | WARN | `corpus_chunkids_000011` | `place` | 县文化馆 | 孙少平被老师没收的书是从什么地方借来的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 17 | WARN | `corpus_chunkids_000011` | `object` | 两个黑面馍 | 第一次当面和孙少平搭话的姑娘，走到孙少平跟前之前刚取完什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 18 | WARN | `corpus_chunkids_000012` | `character` | 郝红梅 | 孙少平看过的书都会借给哪个人？ | chunk 含时间/阶段信息但问题未显式限定 |
| 20 | WARN | `corpus_chunkids_000013` | `character` | 郝红梅 | 孙少平继续把自己看完的书借给谁看？ | chunk 含时间/阶段信息但问题未显式限定 |
| 22 | WARN | `corpus_chunkids_000014` | `object` | 城里买的吃食 | 田润叶每次回村到孙少平家串门时，会给孙少平的祖母带什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 23 | WARN | `corpus_chunkids_000014` | `character` | 田二 | 田润叶每次回村时会提着点心看望她户族里的哪位傻瓜叔叔？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 24 | WARN | `corpus_chunkids_000015` | `object` | 哨子 | 孙少平用发绿的柳枝制作了什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 25 | WARN | `corpus_chunkids_000015` | `character` | 润生 | 田润叶安排谁去叫孙少平到她二爸家？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 26 | WARN | `corpus_chunkids_000016` | `relation` | 田福堂 | 田润生的父亲是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 27 | WARN | `corpus_chunkids_000016` | `place` | 县革委会 | 孙少平跟着田润叶进入的单位大门属于哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 28 | WARN | `corpus_chunkids_000017` | `action_result` | 五个 | 孙少平在田润叶家一共吃了多少个馒头？ | chunk 含时间/阶段信息但问题未显式限定 |
| 29 | WARN | `corpus_chunkids_000017` | `place` | 双水村 | 田晓霞提到的自己长到十七岁都没回过的老乡所在的村子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 30 | WARN | `corpus_chunkids_000019` | `object` | 粮票 | 田润叶塞给孙少平的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 31 | WARN | `corpus_chunkids_000019` | `place` | 小学 | 田润叶叮嘱孙少平转告他哥到城里后要去哪里找她？ | entities 未在问答或 chunk 中出现: 孙少安; chunk 含时间/阶段信息但问题未显式限定 |
| 32 | WARN | `corpus_chunkids_000020` | `place` | 城关粮站 | 星期五孙少平买粮的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 33 | WARN | `corpus_chunkids_000021` | `place` | 罐子村 | 孙少平的姐姐兰花出嫁的村子是哪个？ | chunk 含时间/阶段信息但问题未显式限定 |
| 34 | WARN | `corpus_chunkids_000021` | `object` | 供销社的门市部 | 石圪节公社街上唯一像样的建筑物是什么？ | entities 未在问答或 chunk 中出现: 供销社门市部; chunk 含时间/阶段信息但问题未显式限定 |
| 35 | WARN | `corpus_chunkids_000022` | `place` | 罐子村 | 孙少平骑行至哪个地点时看见了站在公路边的兰香？ | chunk 含时间/阶段信息但问题未显式限定 |
| 36 | WARN | `corpus_chunkids_000022` | `object` | 面 | 孙少平让金波先回去时，嘱托金波暂存在他家的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 37 | WARN | `corpus_chunkids_000023` | `character` | 王满银 | 孙少平的姐夫叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 38 | WARN | `corpus_chunkids_000023` | `place` | 罐子村 | 孙兰香的姐姐家位于哪个村子？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 39 | PASS | `corpus_chunkids_000024` | `place` | 东拉河 | 孙少平和兰香绕路回家时淌过的河流叫什么名字？ | 通过 |
| 41 | WARN | `corpus_chunkids_000025` | `object` | 老鼠药 | 王满银在石圪节集市上倒卖的物品是什么? | chunk 含时间/阶段信息但问题未显式限定 |
| 42 | WARN | `corpus_chunkids_000025` | `place` | 罐子村 | 王满银是被公社民兵小分队从哪个村子带到农田基建会战工地的? | chunk 含时间/阶段信息但问题未显式限定 |
| 43 | WARN | `corpus_chunkids_000026` | `object` | 一身外地买来的时新衣裳 | 王满银在双水村后河湾里塞到兰花手里的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 44 | WARN | `corpus_chunkids_000026` | `character` | 孙玉厚 | 反对女儿兰花嫁给王满银的兰花的父亲是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 45 | WARN | `corpus_chunkids_000027` | `place` | 罐子村 | 孙兰花想要嫁给的王满银所属的村子是哪个？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 47 | WARN | `corpus_chunkids_000028` | `place` | 罐子村 | 王满银所属的村子叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 48 | PASS | `corpus_chunkids_000028` | `object` | 大前门 | 王满银在石圪节卖完老鼠药后买的香烟是什么牌子的？ | 通过 |
| 49 | WARN | `corpus_chunkids_000029` | `character` | 孙玉厚 | 王满银的老丈人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 50 | WARN | `corpus_chunkids_000030` | `place` | 罐子村 | 孙玉厚的大女儿兰花所嫁之人来自哪个村子？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 51 | WARN | `corpus_chunkids_000031` | `character` | 王满银 | 兰花的丈夫叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 52 | WARN | `corpus_chunkids_000031` | `action_result` | 给队里的牲口看病 | 孙少安去米家镇做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 53 | WARN | `corpus_chunkids_000032` | `character` | 少平 | 和兰香一同进入家门的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 54 | WARN | `corpus_chunkids_000032` | `object` | 枪 | 老祖母模模糊糊听到的和家里灾事相关的字是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 55 | WARN | `corpus_chunkids_000033` | `object` | 水果糖 | 孙少平回家后塞给猫蛋和狗蛋的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 57 | WARN | `corpus_chunkids_000034` | `character` | 兰香 | 小时候每个夏天的早晨，孙少平会和谁一起去野地拔带露水珠的青草叶给奶奶淋眼睛？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 58 | WARN | `corpus_chunkids_000035` | `object` | 止痛片 | 孙少平给奶奶买的用来止痛的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 59 | WARN | `corpus_chunkids_000035` | `place` | 队上的饲养室 | 孙少平安排万一哥哥回来时要去哪里凑合一晚上？ | entities 未在问答或 chunk 中出现: 孙少平的哥哥 |
| 60 | PASS | `corpus_chunkids_000036` | `place` | 石圪节 | 兰香就读的初中位于什么地方？ | 通过 |
| 61 | WARN | `corpus_chunkids_000036` | `character` | 兰香 | 当少平母亲想起要喂猪时，谁已经把猪喂好了？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 62 | PASS | `corpus_chunkids_000037` | `place` | 石圪节 | 孙玉厚家最小的孩子上初中的地点是哪里？ | 通过 |
| 63 | WARN | `corpus_chunkids_000037` | `place` | 庙坪山 | 孙玉厚站在自家院子里失神望着的山叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 64 | WARN | `corpus_chunkids_000038` | `place` | 田家圪崂 | 双水村的田姓人家大都居住在什么地方？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 65 | WARN | `corpus_chunkids_000038` | `character` | 孙玉亭 | 孙少平的二爸叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 66 | WARN | `corpus_chunkids_000039` | `place` | 庙坪 | 双水村有庙的三角洲被称为什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 67 | WARN | `corpus_chunkids_000039` | `object` | 枣树 | 双水村庙坪上种植的成片林木是什么树？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 68 | WARN | `corpus_chunkids_000040` | `place` | 石圪节 | 双水村小学的学生上完五年级后要到哪里上初中？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 69 | WARN | `corpus_chunkids_000041` | `character` | 王满银 | 孙少平的姐夫是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 70 | WARN | `corpus_chunkids_000041` | `object` | 铺盖卷 | 王满银在双水村学校被劳教期间，孙少平带给王满银的物品除了饭罐还有什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 71 | WARN | `corpus_chunkids_000042` | `place` | 金波家 | 孙少平每星期六回村时通常在谁家过夜？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 72 | WARN | `corpus_chunkids_000042` | `character` | 贺凤英 | 孙少平的二妈叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 73 | WARN | `corpus_chunkids_000043` | `place` | 学校院子里 | 公社会战指挥部计划召开批判会的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 74 | WARN | `corpus_chunkids_000043` | `place` | 金波家 | 孙少平告诉二妈自己接下来要去的地方是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 76 | WARN | `corpus_chunkids_000044` | `character` | 副总指挥 | 当时孙玉亭在公社会战指挥部担任什么职务？ | chunk 含时间/阶段信息但问题未显式限定 |
| 77 | PASS | `corpus_chunkids_000045` | `place` | 柳林镇 | 冬天农闲时孙玉厚给石圪节商行驮瓷器要前往山西的哪个地点？ | 通过 |
| 78 | WARN | `corpus_chunkids_000045` | `object` | 旱烟 | 孙玉亭叫住即将走上土坡的孙玉厚索要的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 79 | WARN | `corpus_chunkids_000046` | `relation` | 拜把兄弟 | 孙玉厚和山西柳林镇的陶窑主是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 80 | PASS | `corpus_chunkids_000046` | `place` | 太原钢厂 | 1954年孙玉亭初中毕业后到什么地方当了工人？ | 通过 |
| 81 | WARN | `corpus_chunkids_000047` | `character` | 贺凤英 | 在柳林镇小学与孙玉亭同过学的女子的官名是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 83 | WARN | `corpus_chunkids_000048` | `place` | 太原钢厂 | 孙玉亭早年间工作时能顿顿吃白蒸馍大肉菜的单位是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 84 | WARN | `corpus_chunkids_000048` | `character` | 徐治功 | 石圪节公社在双水村开展的农田基建大会战的总指挥是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 85 | WARN | `corpus_chunkids_000049` | `character` | 金俊山 | 书记田福堂去公社开会不在村里时，孙玉亭要找的双水村大队副书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 86 | WARN | `corpus_chunkids_000049` | `place` | 双水村 | 总指挥徐治功提到，晚上的批判会没有批判对象的是哪个村？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 87 | WARN | `corpus_chunkids_000050` | `character` | 田二 | 孙玉亭过了哭咽河后想到的可以充当批判角色的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 88 | WARN | `corpus_chunkids_000050` | `place` | 兰州 | 一九四八年解放军向国民党军队大反攻时，金俊山跟随部队最终打到了哪个城市？ | entities 未在问答或 chunk 中出现: 解放军大反攻; chunk 含时间/阶段信息但问题未显式限定 |
| 89 | WARN | `corpus_chunkids_000051` | `character` | 金成 | 金俊山的儿子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 90 | WARN | `corpus_chunkids_000051` | `place` | 米家镇 | 金俊山的女儿金芳出嫁到了哪个地方？ | chunk 含时间/阶段信息但问题未显式限定 |
| 91 | WARN | `corpus_chunkids_000052` | `object` | 纸烟 | 金俊山给前来谈事的孙玉亭递了什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 92 | WARN | `corpus_chunkids_000052` | `character` | 田二 | 孙玉亭提议在公社会战指挥部的批判会上批判的人是谁？ | entities 未在问答或 chunk 中出现: 公社会战指挥部批判会; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 93 | WARN | `corpus_chunkids_000053` | `character` | 田二 | 孙玉亭提议去给徐主任赔罪充数的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 94 | WARN | `corpus_chunkids_000053` | `place` | 金家湾后面的小学 | 孙玉亭离开金俊山家后要赶往的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 95 | WARN | `corpus_chunkids_000054` | `character` | 贺凤英 | 双水村的妇女主任是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 96 | WARN | `corpus_chunkids_000054` | `object` | 条格布门帘 | 今晚双水村召开的批判大会的主席台桌面铺的是什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 97 | WARN | `corpus_chunkids_000055` | `character` | 王满银 | 兰花的女婿是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 98 | WARN | `corpus_chunkids_000055` | `character` | 田福堂 | 双水村的书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 99 | WARN | `corpus_chunkids_000056` | `character` | 田福顺 | 双水村被确定为批判对象的田二的本名是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 101 | WARN | `corpus_chunkids_000057` | `character` | 杨高虎 | 在大批判会上拿起话筒喊话要求民兵小分队严防阶级敌人破坏捣乱的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 102 | WARN | `corpus_chunkids_000057` | `character` | 王满银 | 走在十几个被劳教的“阶级敌人”最前面的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 103 | WARN | `corpus_chunkids_000058` | `character` | 徐治功 | 双水村召开的批判大会上，被孙玉亭恭请讲话的公社徐主任全名是什么？ | entities 未在问答或 chunk 中出现: 双水村批判大会; chunk 含时间/阶段信息但问题未显式限定 |
| 104 | WARN | `corpus_chunkids_000058` | `action_result` | 世事要变了 | 双水村的憨老汉田二在批判大会的阶级敌人队列中反复嘟囔的内容是什么？ | entities 未在问答或 chunk 中出现: 双水村批判大会; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 105 | WARN | `corpus_chunkids_000059` | `character` | 憨牛 | 田二的儿子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 106 | WARN | `corpus_chunkids_000059` | `object` | 大红烟布袋 | 文革时期孙玉亭企图扯碎田二的什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 107 | WARN | `corpus_chunkids_000060` | `action_result` | 世事要变了 | 田二最爱说的一句话是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 108 | WARN | `corpus_chunkids_000060` | `place` | 双水村小学院子 | 双水村本次批判会的举办地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 109 | WARN | `corpus_chunkids_000061` | `character` | 田二父子 | 孙玉亭收拾停当会场、最后一个离开学校院子走到土坡下面时，发现立在哭咽河畔的两个人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 110 | WARN | `corpus_chunkids_000061` | `place` | 田家圪崂 | 孙玉亭与田二父子三人相跟着返回的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 111 | WARN | `corpus_chunkids_000062` | `place` | 米家镇 | 孙少安带生产队生病的牛去看病的兽医站位于哪个镇？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 112 | WARN | `corpus_chunkids_000062` | `object` | 好毛驴 | 前年二队队长金俊武提出换孙少安所在生产队的好牛时，除两头牛之外额外搭的牲畜是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 114 | WARN | `corpus_chunkids_000063` | `object` | 金丝猴 | 孙少安口袋里舍不得抽的香烟是什么牌子？ | chunk 含时间/阶段信息但问题未显式限定 |
| 116 | WARN | `corpus_chunkids_000065` | `place` | 双水村 | 孙少安告知河南老师傅自己来自哪个村子？ | chunk 含时间/阶段信息但问题未显式限定 |
| 117 | WARN | `corpus_chunkids_000065` | `object` | 金丝猴 | 孙少安掏出来的纸烟是什么品牌的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 118 | WARN | `corpus_chunkids_000066` | `place` | 双水村 | 孙少安第二天早晨起身返回的村庄是哪个？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 119 | WARN | `corpus_chunkids_000066` | `action_result` | 世事要变了 | 田二见到孙少安时嘴里嘟囔的内容是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 121 | WARN | `corpus_chunkids_000067` | `character` | 田万江 | 孙少安把牛送到饲养室后，给哪位饲养员交待了药？ | chunk 含时间/阶段信息但问题未显式限定 |
| 122 | WARN | `corpus_chunkids_000068` | `relation` | 小学同学 | 孙少安和石圪节公社文书刘根民的关系是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 123 | WARN | `corpus_chunkids_000068` | `place` | 石圪节公社 | 孙少安认为解决姐夫的相关事件必须通过哪个公社？ | chunk 含时间/阶段信息但问题未显式限定 |
| 125 | WARN | `corpus_chunkids_000069` | `object` | 蛋糕 | 孙少安从米家镇买回来的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 126 | WARN | `corpus_chunkids_000070` | `object` | 十几斤白面 | 孙少平手中提的口袋里装的是什么物品？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 127 | WARN | `corpus_chunkids_000070` | `character` | 润叶 | 给孙少平十几斤白面的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 128 | PASS | `corpus_chunkids_000071` | `object` | 破瓷器片 | 小时候的孙少安和润叶冬天去金家湾搜寻的玩具是什么？ | 通过 |
| 129 | WARN | `corpus_chunkids_000072` | `action_result` | 砍柴 | 孙少安六岁时，父亲提议带他去做什么活？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 130 | WARN | `corpus_chunkids_000072` | `object` | 玉米面馍 | 田润叶偷拿自己家的什么食物给砍柴回家的孙少安？ | chunk 含时间/阶段信息但问题未显式限定 |
| 131 | PASS | `corpus_chunkids_000073` | `character` | 金俊海家 | 1960年最困难的时期，孙少安一家从田家圪崂搬出来后借住了谁家的窑洞？ | 通过 |
| 132 | PASS | `corpus_chunkids_000073` | `action_result` | 第一名 | 孙少安在双水村小学就读的四年里，每次班级考试的名次是多少？ | 通过 |
| 133 | WARN | `corpus_chunkids_000074` | `place` | 哭咽河 | 传说中由痛哭而死的男人的眼泪流成的河叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 134 | WARN | `corpus_chunkids_000074` | `place` | 石圪节高小 | 一九六四年，和润叶双双考上的高小名称是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 136 | WARN | `corpus_chunkids_000075` | `place` | 双水村 | 孙少安开启农民生涯时决心要在哪个村子做出众的庄稼人？ | chunk 含时间/阶段信息但问题未显式限定 |
| 137 | WARN | `corpus_chunkids_000076` | `character` | 队长 | 孙少安十八岁时被一队社员一致推选担任什么职务？ | chunk 含时间/阶段信息但问题未显式限定 |
| 139 | WARN | `corpus_chunkids_000077` | `place` | 县城 | 孙少安打算前往哪个地点找田润叶帮忙？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 141 | WARN | `corpus_chunkids_000078` | `object` | 红缨枪 | 田润叶任职的学校安排红小兵学军时，规定女学生要拿什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 142 | WARN | `corpus_chunkids_000078` | `character` | 县贸易经理部的汽车司机 | 李向前的职业是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 143 | WARN | `corpus_chunkids_000079` | `place` | 省城 | 李向前给她捎买的线衣是在哪里购买的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 144 | WARN | `corpus_chunkids_000079` | `character` | 她二妈 | 李向前的母亲托谁转告说李向前看上了她？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 145 | WARN | `corpus_chunkids_000080` | `character` | 孙少安 | 在二妈告知润叶李向前家提亲的事后，润叶首次考虑婚姻大事时第一个想到的共度一生的男性是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 148 | WARN | `corpus_chunkids_000081` | `action_result` | 谈一次 | 田润叶决定和孙少安做什么事？ | chunk 含时间/阶段信息但问题未显式限定 |
| 150 | WARN | `corpus_chunkids_000082` | `object` | 五十斤粮票 | 田润叶去中学找孙少平时，把自己省下的什么物品给了孙少平？ | chunk 含时间/阶段信息但问题未显式限定 |
| 151 | PASS | `corpus_chunkids_000083` | `action_result` | 学校事多 | 润叶称自己这几天不回家吃饭的原因是什么？ | 通过 |
| 153 | WARN | `corpus_chunkids_000084` | `relation` | 老下级 | 李登云过去和徐国强是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 154 | WARN | `corpus_chunkids_000084` | `place` | 花坛 | 徐国强退休后没事时常在院子的哪个区域修修整整？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 155 | WARN | `corpus_chunkids_000085` | `character` | 徐国强 | 李登云过去一直是谁的老下级？ | chunk 含时间/阶段信息但问题未显式限定 |
| 156 | WARN | `corpus_chunkids_000085` | `place` | 西北大学 | 田福军的大儿子田晓晨就读的大学是哪所？ | chunk 含时间/阶段信息但问题未显式限定 |
| 157 | WARN | `corpus_chunkids_000086` | `object` | 盐 | 田福军第一次炒好的肉丝缺少什么调料？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 158 | WARN | `corpus_chunkids_000086` | `character` | 徐爱云 | 田福军的爱人叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 159 | WARN | `corpus_chunkids_000087` | `object` | 蛋糕 | 李登云看望徐国强时携带的礼物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 160 | WARN | `corpus_chunkids_000088` | `character` | 向前 | 李登云给徐国强带的生日蛋糕是吩咐谁从省城里购买的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 161 | WARN | `corpus_chunkids_000088` | `object` | 肉食 | 徐国强六十五大寿的宴桌上大部分菜品是什么类型？ | chunk 含时间/阶段信息但问题未显式限定 |
| 163 | WARN | `corpus_chunkids_000089` | `action_result` | 到医院看牙 | 李登云向徐国强告辞时声称自己要去做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 164 | PASS | `corpus_chunkids_000090` | `character` | 田福高 | 孙少安把队里冬小麦返青后的除草施肥等相关事务安顿给了哪位副队长？ | 通过 |
| 165 | PASS | `corpus_chunkids_000090` | `place` | 街上食堂 | 孙少安洗完脸后，田润叶提出要带他去什么地方吃饭？ | 通过 |
| 166 | WARN | `corpus_chunkids_000091` | `place` | 国营食堂 | 孙少安跟着田润叶进入了县城最大的什么场所？ | chunk 含时间/阶段信息但问题未显式限定 |
| 167 | WARN | `corpus_chunkids_000091` | `relation` | 亲妹妹 | 孙少安看待田润叶的情感类似什么亲属关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 168 | WARN | `corpus_chunkids_000092` | `place` | 原西河 | 孙少安和田润叶吃完饭后散步所沿的河畔名称是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 169 | PASS | `corpus_chunkids_000093` | `place` | 原西河 | 少安坐在草坡旁时望见的草坡下的河叫什么名字？ | 通过 |
| 170 | WARN | `corpus_chunkids_000093` | `object` | 马兰花 | 润叶坐在草坡旁摘来摆弄的花是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 171 | WARN | `corpus_chunkids_000094` | `action_result` | 把那小子打得鼻子口里直淌血 | 在石圪节上高小时，有男同学故意往润叶身上扔篮球，孙少安做了什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 172 | WARN | `corpus_chunkids_000094` | `object` | 手绢 | 润叶用什么物品抹去了脸上的泪痕？ | chunk 含时间/阶段信息但问题未显式限定 |
| 173 | WARN | `corpus_chunkids_000095` | `object` | 马兰花 | 孙少安送给田润叶的花是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 174 | WARN | `corpus_chunkids_000095` | `place` | 石圪节 | 田润叶提出要和孙少安一起去什么地方找白叔叔和徐叔叔？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 175 | WARN | `corpus_chunkids_000096` | `character` | 少平 | 少安的弟弟叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 176 | PASS | `corpus_chunkids_000096` | `character` | 白明川 | 田福军写好信后要求少安将信交给哪个人？ | 通过 |
| 177 | WARN | `corpus_chunkids_000097` | `object` | 公共汽车 | 孙少安和田润叶回石圪节乘坐的交通工具是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 178 | PASS | `corpus_chunkids_000098` | `place` | 公路上 | 看完田润叶交给自己的信时，孙少安站在什么地方？ | 通过 |
| 179 | WARN | `corpus_chunkids_000099` | `object` | 大前门 | 田福堂递给白明川、徐治功的纸烟品牌是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 180 | WARN | `corpus_chunkids_000099` | `place` | 双水村 | 公社召开的大队书记会议结束后，田福堂要返回的村子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 182 | WARN | `corpus_chunkids_000100` | `character` | 冯世宽 | 徐治功称劳教农民的政策是县上哪位主任提出的？ | entities 未在问答或 chunk 中出现: 劳教农民政策; chunk 含时间/阶段信息但问题未显式限定 |
| 183 | WARN | `corpus_chunkids_000101` | `character` | 王满银 | 田福军的信中提到的被押到双水村公社农田基建工地劳教的罐子村社员是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 184 | WARN | `corpus_chunkids_000101` | `object` | 老鼠药 | 王满银是因为贩卖什么物品被劳教的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 185 | WARN | `corpus_chunkids_000102` | `relation` | 父女 | 田福堂和润叶的人物关系是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 186 | WARN | `corpus_chunkids_000102` | `object` | 黑回绒 | 田福堂的自行车大梁上缠绕的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 187 | WARN | `corpus_chunkids_000103` | `character` | 王满银 | 徐治功托田福堂捎话要求释放的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 188 | WARN | `corpus_chunkids_000103` | `object` | 永久牌 | 田福堂骑的自行车是什么品牌的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 191 | PASS | `corpus_chunkids_000105` | `character` | 杨高虎 | 田福堂到达双水村村头后最初呼喊要找的人是谁？ | 通过 |
| 192 | PASS | `corpus_chunkids_000105` | `object` | 旧鞋 | 田福堂打算送给孙玉亭的物品是什么？ | 通过 |
| 193 | WARN | `corpus_chunkids_000106` | `object` | 鸡蛋 | 王满银被释放后返程途中，兰花提到要煮给家人吃的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 194 | WARN | `corpus_chunkids_000106` | `place` | 罐子村 | 王满银在老丈人家吃完早饭后要返回的村子是哪个？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 195 | PASS | `corpus_chunkids_000107` | `object` | 瓜果菜蔬 | 夏天时石圪节的集市场上出现了什么往日没有的商品？ | 通过 |
| 196 | WARN | `corpus_chunkids_000107` | `action_result` | 修梯田 | 农业学大寨期间，县城各机关干部每天上午到城外山上做什么？ | entities 未在问答或 chunk 中出现: 县城机关干部 |
| 197 | PASS | `corpus_chunkids_000108` | `object` | 黑高粱面馍 | 孙少平高中第二个学期刚开学时大部分日子里啃的主食是什么？ | 通过 |
| 198 | WARN | `corpus_chunkids_000108` | `character` | 郝红梅 | 孙少平不忍心中途退学、不情愿离开县城高中的不可告人的原因涉及的人物是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 199 | WARN | `corpus_chunkids_000109` | `character` | 劳动干事 | 上学期班级选举班干部时，孙少平被选任的职务是什么？ | entities 未在问答或 chunk 中出现: 上学期班干部选举; 答案在 chunk 中出现多次，唯一性需复核 |
| 200 | PASS | `corpus_chunkids_000109` | `action_result` | 冠军 | 上学期全校乒乓球比赛中，孙少平获得的名次是什么？ | 通过 |
| 201 | PASS | `corpus_chunkids_000110` | `place` | 县文化馆 | 上学期临放假的前一个星期，孙少平想起郝红梅借走的他的书是从哪里借来的？ | 通过 |
| 202 | WARN | `corpus_chunkids_000110` | `character` | 金波 | 上学期临放假前的最后一个星期六，孙少平借了谁的自行车送自己的破烂铺盖回家？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 203 | WARN | `corpus_chunkids_000111` | `character` | 金波 | 放假前的最后一个星期六，孙少平借了谁的自行车送自己的破烂铺盖回家？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 204 | WARN | `corpus_chunkids_000111` | `object` | 白面饼 | 郝红梅夹在还给孙少平的书里掉落的食物是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 205 | WARN | `corpus_chunkids_000112` | `character` | 金波妈 | 给孙少平的家织老粗布裁剪成制服式样的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 206 | WARN | `corpus_chunkids_000112` | `place` | 篮球场 | 开学两个多星期时，每天下午吃完饭郝红梅通常和同学们在什么场地玩耍？ | entities 未在问答或 chunk 中出现: 下午饭后玩耍 |
| 207 | WARN | `corpus_chunkids_000113` | `character` | 顾养民 | 孙少平在篮球场上向郝红梅要球时，郝红梅把球传给了谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 208 | WARN | `corpus_chunkids_000113` | `place` | 县城外边的河滩里 | 孙少平被郝红梅无视要球请求后，最终糊里糊涂转到了什么地方？ | chunk 含时间/阶段信息但问题未显式限定 |
| 209 | PASS | `corpus_chunkids_000114` | `object` | 黑高粱面馍 | 郝红梅在县高中上学时顿顿吃的主食是什么？ | 通过 |
| 210 | WARN | `corpus_chunkids_000114` | `character` | 孙少平 | 郝红梅对和自己家庭情况类似的哪个男生产生了同病相怜的感情？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 211 | WARN | `corpus_chunkids_000115` | `character` | 侯玉英 | 当众污蔑郝红梅是孙少平“婆姨”的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 212 | WARN | `corpus_chunkids_000115` | `character` | 孙少平 | 临近放假时，郝红梅发现自己箱底的书是借自谁的？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 213 | WARN | `corpus_chunkids_000116` | `object` | 白面饼 | 临近放假时，郝红梅打算和要还给孙少平的书一起送给对方以弥补未及时还书过失的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 214 | WARN | `corpus_chunkids_000116` | `character` | 顾养民 | 站在篮球场边看球时，邀请郝红梅一起打篮球的她的同班班长是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 215 | WARN | `corpus_chunkids_000117` | `character` | 顾养民 | 这学期开学后某天下午打篮球时，郝红梅收到孙少平的传球请求后，把球传给了谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 216 | WARN | `corpus_chunkids_000118` | `character` | 金波 | 找到在河岸边情绪低落的孙少平的好朋友是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 217 | WARN | `corpus_chunkids_000118` | `object` | 玉米面烧饼 | 金波带给孙少平的食物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 218 | WARN | `corpus_chunkids_000119` | `character` | 孙少平 | 金波筹划殴打顾养民时为了不牵连谁而对其保密相关行动？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 222 | WARN | `corpus_chunkids_000121` | `character` | 孙少平 | 金波殴打顾养民是为了谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 224 | WARN | `corpus_chunkids_000122` | `relation` | 兄妹 | 在学校文艺宣传队排演的小戏里，田晓霞扮演的角色与孙少平扮演的角色是什么关系？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 225 | WARN | `corpus_chunkids_000123` | `character` | 田晓霞 | 县上决定让孙少平和哪个人一同参加全区革命故事调讲？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 226 | WARN | `corpus_chunkids_000123` | `place` | 石圪节公社 | 孙少平参加文艺宣传队时期盼宣传队能巡回到哪个公社演出？ | chunk 含时间/阶段信息但问题未显式限定 |
| 227 | WARN | `corpus_chunkids_000124` | `character` | 贾冰 | 孙少平在黄原地区故事会上认识的原西县籍诗人叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 228 | WARN | `corpus_chunkids_000124` | `action_result` | 二等奖 | 孙少平和晓霞在黄原地区故事调讲中获得了什么奖项？ | chunk 含时间/阶段信息但问题未显式限定 |
| 231 | WARN | `corpus_chunkids_000126` | `object` | 一包点心 | 田润叶到孙少安家拜访时，给少安奶带的礼物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 232 | PASS | `corpus_chunkids_000126` | `action_result` | 回家吃饭 | 田润叶临走前叮嘱少安妈转告孙少安第二天中午要做什么？ | 通过 |
| 233 | WARN | `corpus_chunkids_000127` | `character` | 队长 | 田润叶回村找孙少安时，孙少安的身份是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 234 | WARN | `corpus_chunkids_000127` | `object` | 公共车 | 田润叶返回县城乘坐的交通工具是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 238 | WARN | `corpus_chunkids_000129` | `action_result` | 把茶水倒了一桌子 | 李向前给客人倒茶时发生了什么失误？ | chunk 含时间/阶段信息但问题未显式限定 |
| 240 | WARN | `corpus_chunkids_000130` | `action_result` | 找借口躲出去 | 李向前到田润叶的学校宿舍找她时，田润叶采取了什么应对方式？ | chunk 含时间/阶段信息但问题未显式限定 |
| 241 | WARN | `corpus_chunkids_000131` | `place` | 双水村 | 田润叶突然决定很快要返回的村子是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 242 | WARN | `corpus_chunkids_000131` | `character` | 李向前 | 全校老师口中田润叶的女婿是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 243 | PASS | `corpus_chunkids_000132` | `place` | 石圪节的公路上 | 孙少安是在什么地方看完田润叶的纸条的？ | 通过 |
| 244 | PASS | `corpus_chunkids_000132` | `action_result` | 站在公路上幸福地哭起来 | 孙少安反应过来田润叶纸条的内容后做出了什么行为？ | 通过 |
| 246 | WARN | `corpus_chunkids_000133` | `place` | 石圪节 | 兰香要赶去上第一节课的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 248 | WARN | `corpus_chunkids_000134` | `place` | 自留地 | 如果中午不在山里吃饭，孙少安回家吃完饭碗一撂会去什么地方？ | entities 未在问答或 chunk 中出现: 中午吃饭; 答案在 chunk 中出现多次，唯一性需复核 |
| 249 | WARN | `corpus_chunkids_000135` | `action_result` | 坐汽车回县城去了 | 年轻的队长跑回家后从母亲口中得知润叶做了什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 250 | WARN | `corpus_chunkids_000135` | `relation` | 友情 | 年轻的队长认为自己拒绝润叶的爱情后，两人之间的什么关系也被割断了？ | chunk 含时间/阶段信息但问题未显式限定 |
| 251 | WARN | `corpus_chunkids_000136` | `place` | 米家镇 | 孙少安家的自留地位于通往哪个地方方向的公路上面？ | chunk 含时间/阶段信息但问题未显式限定 |
| 252 | WARN | `corpus_chunkids_000136` | `character` | 润叶 | 孙少安在东拉河边的自留地附近碰到的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 253 | WARN | `corpus_chunkids_000137` | `place` | 罐子村 | 田福堂购买几斤羊肉的地点是哪个村庄？ | chunk 含时间/阶段信息但问题未显式限定 |
| 254 | WARN | `corpus_chunkids_000137` | `character` | 田万有 | 少安和润叶听到的在对面远山梁上唱信天游的人是谁？ | entities 未在问答或 chunk 中出现: 孙少安,田润叶; chunk 含时间/阶段信息但问题未显式限定 |
| 255 | PASS | `corpus_chunkids_000138` | `character` | 少安 | 父亲在前面公路上碰见的担水的人是谁？ | 通过 |
| 256 | PASS | `corpus_chunkids_000138` | `place` | 罐子村 | 田福堂买的几斤羊肉是在哪个村子购买的？ | 通过 |
| 257 | WARN | `corpus_chunkids_000139` | `place` | 河滩里 | 田福堂看见润叶和少安正晌午坐着的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 259 | WARN | `corpus_chunkids_000140` | `character` | 孙玉亭 | 田福堂不在村子里的时候，通常主要委托谁管理队里的工作？ | entities 未在问答或 chunk 中出现: 队里工作; 答案在 chunk 中出现多次，唯一性需复核 |
| 260 | WARN | `corpus_chunkids_000140` | `object` | 几双旧鞋 | 田福堂让老婆拿报纸包起来准备带给孙玉亭的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 261 | WARN | `corpus_chunkids_000141` | `object` | 几双旧鞋 | 田福堂去孙玉亭家时送给孙玉亭的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 262 | WARN | `corpus_chunkids_000141` | `object` | 金黄的旱烟叶 | 田福堂到县城后带给徐国强的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 263 | WARN | `corpus_chunkids_000142` | `object` | 金黄的旱烟叶 | 田福堂此次带给徐国强老汉的礼物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 264 | WARN | `corpus_chunkids_000142` | `place` | 县医院 | 田福堂吃完饭后去哪个地点找徐爱云？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 265 | PASS | `corpus_chunkids_000143` | `place` | 县医院 | 田福堂吃完饭后独自前往的地点是哪里？ | 通过 |
| 266 | PASS | `corpus_chunkids_000143` | `character` | 向前 | 徐爱云向田福堂提到的喜欢田润叶的李登云的儿子叫什么名字？ | 通过 |
| 268 | WARN | `corpus_chunkids_000144` | `character` | 顾老先生 | 徐爱云打算带田福堂找谁开治疗气管炎的药物？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 269 | WARN | `corpus_chunkids_000145` | `character` | 田福高 | 临近中午时分田福堂骑着车子回到石圪节时，在石圪节的小桥上看到的同村人是谁？ | entities 未在问答或 chunk 中出现: 石圪节小桥; 答案在 chunk 中出现多次，唯一性需复核 |
| 270 | WARN | `corpus_chunkids_000145` | `place` | 大庄河 | 田福高的姨夫所属的村子是哪个？ | chunk 含时间/阶段信息但问题未显式限定 |
| 271 | WARN | `corpus_chunkids_000146` | `character` | 孙玉亭 | 提出用抓纸蛋的方式解决双水村养猪任务落实难题的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 272 | WARN | `corpus_chunkids_000146` | `place` | 石圪节 | 双水村村民年底交售生猪需要前往的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 273 | WARN | `corpus_chunkids_000147` | `place` | 石圪节 | 庄稼人喂好猪后要去哪个地点交售生猪？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 274 | PASS | `corpus_chunkids_000148` | `action_result` | 用眼睛估量 | 孙少安给社员扩大自留地时采用的土地丈量方式是什么？ | 通过 |
| 275 | PASS | `corpus_chunkids_000148` | `character` | 徐治功 | 田福堂去公社找哪位主任帮忙调查猪饲料地的问题？ | 通过 |
| 276 | WARN | `corpus_chunkids_000149` | `character` | 徐治功 | 石圪节公社针对扩大猪饲料地事件召开的批判会的主持人是谁？ | entities 未在问答或 chunk 中出现: 扩大猪饲料地批判会; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 277 | WARN | `corpus_chunkids_000149` | `action_result` | 作批判发言 | 在石圪节公社针对扩大猪饲料地事件的批判会上，孙玉亭被公社安排了什么任务？ | entities 未在问答或 chunk 中出现: 扩大猪饲料地批判会; chunk 含时间/阶段信息但问题未显式限定 |
| 278 | WARN | `corpus_chunkids_000150` | `action_result` | 作批判发言 | 在孙少安接受批判的本次公社大会上，孙玉亭被公社安排做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 279 | WARN | `corpus_chunkids_000150` | `character` | 刘根民 | 和孙少安高小时是同班同学的公社文书叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 280 | WARN | `corpus_chunkids_000151` | `character` | 田福堂 | 孙少安明确意识到是谁将他推到接受批判的台子上的？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 281 | WARN | `corpus_chunkids_000151` | `character` | 田福高 | 给孙少安讲述在石圪节碰上田福堂的经过的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 282 | PASS | `corpus_chunkids_000152` | `character` | 白明川 | 孙少安参加的批判会结束时，对受批判人员说鼓励话的公社主任叫什么名字？ | 通过 |
| 284 | WARN | `corpus_chunkids_000153` | `place` | 双水村 | 孙少安太阳落山后通过石圪节小桥，踏上的是通往哪个村子的公路？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 285 | PASS | `corpus_chunkids_000153` | `object` | 自卷的旱烟卷 | 孙少安独自行走在通往双水村的公路上时吸的是什么？ | 通过 |
| 286 | PASS | `corpus_chunkids_000154` | `place` | 东拉河对面的山洼 | 孙少安将拾起的路边石头甩向了什么地方？ | 通过 |
| 287 | PASS | `corpus_chunkids_000154` | `character` | 兰香 | 孙玉厚让谁先回了家？ | 通过 |
| 288 | WARN | `corpus_chunkids_000155` | `place` | 高粱地 | 孙少安父子俩回家前是从什么地方走出来的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 289 | WARN | `corpus_chunkids_000155` | `object` | 旱烟卷 | 孙少安重新点着的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 291 | PASS | `corpus_chunkids_000156` | `action_result` | 给孙少安娶媳妇 | 孙玉厚夜里睡不着一直惦记着关于孙少安的什么事？ | 通过 |
| 293 | WARN | `corpus_chunkids_000157` | `object` | 南瓜 | 贺凤英听到孙玉厚和孙玉亭讨论找媳妇的话题时，正在锅台上切什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 294 | WARN | `corpus_chunkids_000158` | `place` | 柳林 | 孙玉厚打算让孙少安去哪个地方相看贺凤英提到的远门侄女？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 295 | WARN | `corpus_chunkids_000158` | `action_result` | 出山劳动 | 孙玉厚从孙玉亭家赶回家里时，孙少安去做什么了？ | chunk 含时间/阶段信息但问题未显式限定 |
| 296 | WARN | `corpus_chunkids_000159` | `place` | 山西 | 孙少安的父母同意让孙少安去什么地方相亲？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 297 | WARN | `corpus_chunkids_000159` | `character` | 润叶 | 孙少安得知自己要去相亲的消息后立刻想到的女性是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 298 | WARN | `corpus_chunkids_000160` | `action_result` | 相亲 | 孙少安答应父母亲去山西做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 299 | WARN | `corpus_chunkids_000160` | `place` | 庙坪山 | 孙少安安排完队里的事务后爬上了哪座山？ | chunk 含时间/阶段信息但问题未显式限定 |
| 300 | WARN | `corpus_chunkids_000161` | `character` | 顾养民 | 孙少平进入县高中后的这段时间里，郝红梅使用的大红皮笔记本是谁送给她的？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 301 | PASS | `corpus_chunkids_000161` | `object` | 两个黑馍 | 孙少平刚进入县高中时，等其他同学打完饭才去取的食物是什么？ | 通过 |
| 303 | WARN | `corpus_chunkids_000162` | `relation` | 同村人 | 孙少平和田晓霞除了一同演过戏、讲过故事，还有什么明确的同乡关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 304 | WARN | `corpus_chunkids_000163` | `character` | 初澜 | 孙少平与田晓霞一同看报纸时，田晓霞手指的那篇文章的署名是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 305 | WARN | `corpus_chunkids_000163` | `action_result` | 一星期 | 田晓霞计划每隔多久给孙少平拿一次她父亲订阅的内部报纸？ | chunk 含时间/阶段信息但问题未显式限定 |
| 306 | WARN | `corpus_chunkids_000164` | `character` | 田晓霞 | 给孙少平带来各类读物的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 307 | WARN | `corpus_chunkids_000164` | `character` | 金波 | 在学校忆苦思甜报告会上，用胳膊肘戳孙少平告知他父亲来了的人是谁？ | entities 未在问答或 chunk 中出现: 孙少平父亲; chunk 含时间/阶段信息但问题未显式限定 |
| 308 | WARN | `corpus_chunkids_000165` | `character` | 贺凤英 | 给孙少安提去山西看媳妇的亲事的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 309 | WARN | `corpus_chunkids_000165` | `object` | 烟锅 | 孙少平在学校会场后方见到孙玉厚时，孙玉厚手中拿着的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 310 | PASS | `corpus_chunkids_000166` | `character` | 金成 | 孙少平去金家圪崂睡觉前，会顺路到谁的办公室里看完当天的报纸？ | 通过 |
| 311 | WARN | `corpus_chunkids_000166` | `object` | 猪头肉 | 快吃饭时金波从街上买回来的食物除了烧饼还有什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 313 | WARN | `corpus_chunkids_000167` | `character` | 田万有 | 孙少平赤着脚走到东拉河边时，认出跪在水井前的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 314 | WARN | `corpus_chunkids_000168` | `action_result` | 祈雨 | 孙少平撞见跪在水井边的田万有，当时田万有正在进行什么行为？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 315 | PASS | `corpus_chunkids_000168` | `character` | 一队的老饲养员 | 田万有排行第四的哥哥田万江的身份是什么？ | 通过 |
| 316 | PASS | `corpus_chunkids_000169` | `place` | 米家镇 | 双水村发生严重旱情时，村民在通往哪个镇方向的村前东拉河上拦河水抗旱？ | 通过 |
| 317 | WARN | `corpus_chunkids_000170` | `object` | 纸烟 | 田福堂在自家窑里烦乱来回走时手里拿着的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 318 | WARN | `corpus_chunkids_000170` | `character` | 玉亭 | 向田福堂汇报村里谁在骂他的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 319 | WARN | `corpus_chunkids_000171` | `place` | 大队部 | 田福堂召集大小队干部开会的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 320 | WARN | `corpus_chunkids_000171` | `character` | 孙少安 | 未出席田福堂召集的本次干部会议的一队长是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 321 | WARN | `corpus_chunkids_000172` | `character` | 孙玉亭 | 谁建议大队批判田五的封建迷信活动？ | chunk 含时间/阶段信息但问题未显式限定 |
| 323 | WARN | `corpus_chunkids_000173` | `place` | 罐子村 | 田福堂安排金俊武前往哪个村的坝执行任务？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 324 | WARN | `corpus_chunkids_000173` | `object` | 拖拉机 | 孙玉亭提议去石圪节豁坝时乘坐什么交通工具加快行动速度？ | chunk 含时间/阶段信息但问题未显式限定 |
| 325 | PASS | `corpus_chunkids_000174` | `character` | 孙玉亭 | 双水村准备向石圪节进军的十几个年轻后生的任务是由谁交待的？ | 通过 |
| 326 | PASS | `corpus_chunkids_000174` | `place` | 村前米家镇方向的东拉河里 | 双水村村民加高坝梁的地点在哪里？ | 通过 |
| 327 | WARN | `corpus_chunkids_000175` | `character` | 田五大叔 | 孙少平在坝梁工地往架子车上装土时，负责推车的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 328 | WARN | `corpus_chunkids_000175` | `place` | 石圪节 | 孙玉亭原本安排孙少平去哪个地点放水？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 329 | WARN | `corpus_chunkids_000176` | `place` | 坝梁中间 | 金富和金强最初挖掘石圪节坝梁的位置是哪里？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 330 | WARN | `corpus_chunkids_000177` | `object` | 大前门 | 金俊武等三人完成挖掘任务回到双水村大队部院子时，田福堂给每人递的纸烟是什么牌子？ | chunk 含时间/阶段信息但问题未显式限定 |
| 331 | WARN | `corpus_chunkids_000177` | `place` | 罐子村 | 在孙玉亭等人还没动手挖坝之前，二队长金俊武带人完成挖掘任务的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 332 | WARN | `corpus_chunkids_000178` | `place` | 大队部的窑洞 | 田福堂和金俊武等待金成、田海民报告水来消息的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 333 | PASS | `corpus_chunkids_000179` | `action_result` | 启动两台抽水机 | 水到达双水村土坝时，田福堂立刻下达了什么命令？ | 通过 |
| 335 | PASS | `corpus_chunkids_000180` | `object` | 铁锨 | 金俊斌去前河道大便时扛的工具是什么？ | 通过 |
| 336 | WARN | `corpus_chunkids_000181` | `character` | 王彩娥 | 金俊斌的妻子是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 337 | WARN | `corpus_chunkids_000181` | `place` | 东拉河进入米家川大河的入口处 | 金俊斌的尸首是在哪里被找到的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 338 | WARN | `corpus_chunkids_000182` | `place` | 庙坪的破庙 | 金俊武的母亲想去什么地方看死去的金俊斌？ | chunk 含时间/阶段信息但问题未显式限定 |
| 339 | PASS | `corpus_chunkids_000183` | `object` | 槐树 | 金俊山为打棺材安排人伐倒了队里的哪种树？ | 通过 |
| 340 | WARN | `corpus_chunkids_000183` | `character` | 孙玉亭 | 田福堂身体不适时，安排老婆去叫谁来商量事情？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 341 | PASS | `corpus_chunkids_000184` | `character` | 田福堂的老婆 | 是谁半路上碰见金俊山和金俊武后将二人带到了田福堂家？ | 通过 |
| 342 | PASS | `corpus_chunkids_000184` | `object` | 温开水 | 田福堂吃完药之后喝的是什么？ | 通过 |
| 344 | WARN | `corpus_chunkids_000185` | `character` | 孙玉亭 | 讨论金俊斌丧事的大队党支部会议上，安排谁负责筹备追悼会相关事宜？ | entities 未在问答或 chunk 中出现: 金俊斌追悼会; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 345 | WARN | `corpus_chunkids_000186` | `place` | 学校 | 孙玉亭最初打算将金俊斌的追悼会设置在什么地方？ | chunk 含时间/阶段信息但问题未显式限定 |
| 346 | WARN | `corpus_chunkids_000186` | `character` | 金波他妈 | 为金俊斌缝制入殓服装的工作是由谁领料的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 347 | WARN | `corpus_chunkids_000187` | `character` | 金俊山 | 双水村为金俊斌举办的追悼会由谁主持？ | entities 未在问答或 chunk 中出现: 金俊斌追悼会; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 348 | WARN | `corpus_chunkids_000187` | `place` | 金家祖坟 | 双水村护送金俊斌灵柩的送葬队伍最终前往的安葬地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 349 | WARN | `corpus_chunkids_000188` | `character` | 金俊山 | 双水村大队的副书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 350 | WARN | `corpus_chunkids_000188` | `object` | 大型拖拉机 | 罐子村半夜起床上厕所的村民看到双水村从村中开过的交通工具是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 351 | PASS | `corpus_chunkids_000189` | `place` | 石圪节公社 | 双水村是哪个公社的农业学大寨先进典型？ | 通过 |
| 352 | WARN | `corpus_chunkids_000189` | `character` | 金俊山 | 田福堂因病不能到公社做检查时，代替他代表双水村大队党支部向全公社人民做检查的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 354 | WARN | `corpus_chunkids_000190` | `character` | 贺凤英 | 孙少安带回的山西姑娘是哪位双水村妇女主任的远房本家人？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 355 | WARN | `corpus_chunkids_000191` | `character` | 贺耀宗 | 贺秀莲的父亲叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 356 | WARN | `corpus_chunkids_000191` | `place` | 双水村 | 孙少安与贺秀莲的亲事确定后原本计划返回的村子是哪个？ | entities 未在问答或 chunk 中出现: 返回村子; chunk 含时间/阶段信息但问题未显式限定 |
| 357 | PASS | `corpus_chunkids_000192` | `place` | 柳林 | 孙少安带贺秀莲回双水村时，是从哪里坐汽车出发的？ | 通过 |
| 358 | PASS | `corpus_chunkids_000192` | `object` | 一床从未沾身的新铺盖 | 贺秀莲到孙少安家后，金大婶拿出了什么给她盖？ | 通过 |
| 359 | PASS | `corpus_chunkids_000193` | `character` | 田福高 | 孙少安回家的第二天上午找的生产队副队长是谁？ | 通过 |
| 360 | PASS | `corpus_chunkids_000194` | `object` | 冬小麦 | 白露刚过时，山野阳坡上正在播种的农作物是什么？ | 通过 |
| 361 | PASS | `corpus_chunkids_000194` | `character` | 王彩娥 | 俊斌的媳妇叫什么名字？ | 通过 |
| 362 | WARN | `corpus_chunkids_000195` | `object` | 粮票 | 孙少安的二妈把夏天分得的麦子在石圪节粮站换成了什么？ | entities 未在问答或 chunk 中出现: 孙少安二妈 |
| 363 | WARN | `corpus_chunkids_000195` | `place` | 金俊武家里 | 孙少安离开二爸后径直去了哪里？ | entities 未在问答或 chunk 中出现: 孙少安二爸; chunk 含时间/阶段信息但问题未显式限定 |
| 364 | WARN | `corpus_chunkids_000196` | `action_result` | 春节 | 孙少安计划什么时候举办婚事？ | chunk 含时间/阶段信息但问题未显式限定 |
| 365 | WARN | `corpus_chunkids_000196` | `object` | 大队储备粮 | 田福堂承诺孙少安借粮食时可以从哪里取用？ | chunk 含时间/阶段信息但问题未显式限定 |
| 367 | WARN | `corpus_chunkids_000197` | `action_result` | 搬东西 | 金俊海两口子到田家圪崂的公路边做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 369 | WARN | `corpus_chunkids_000198` | `character` | 金光明 | 公派教师姚淑芳的丈夫是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 370 | WARN | `corpus_chunkids_000199` | `character` | 刘根民 | 孙少安在公社当文书的同学叫什么名字？ | entities 未在问答或 chunk 中出现: 公社文书 |
| 372 | PASS | `corpus_chunkids_000200` | `relation` | 叔伯兄弟 | 金俊海和金俊山是什么关系？ | 通过 |
| 373 | PASS | `corpus_chunkids_000200` | `place` | 包头 | 金俊海顺路回家后第二天要去什么地方拉货？ | 通过 |
| 374 | PASS | `corpus_chunkids_000201` | `place` | 石圪节学校 | 兰香上学的地点是哪里？ | 通过 |
| 375 | PASS | `corpus_chunkids_000201` | `action_result` | 扯几身时新衣裳 | 孙玉厚打算过两天让孙少安带秀莲去县城做什么？ | 通过 |
| 376 | WARN | `corpus_chunkids_000202` | `character` | 金俊山 | 孙玉厚原本打算向谁借粮食给孙少安操办婚事？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 377 | WARN | `corpus_chunkids_000202` | `place` | 米家镇 | 孙少安决定带秀莲去什么地方扯结婚用的新衣裳？ | entities 未在问答或 chunk 中出现: 扯新衣裳; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 378 | PASS | `corpus_chunkids_000203` | `place` | 柳林镇 | 贺秀莲上完本村小学后没有前往哪个地方上初中？ | 通过 |
| 379 | WARN | `corpus_chunkids_000203` | `character` | 常有林 | 贺秀莲的姐姐贺秀英招的上门女婿是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 380 | PASS | `corpus_chunkids_000204` | `place` | 双水村 | 孙少安的家位于哪个村子？ | 通过 |
| 381 | WARN | `corpus_chunkids_000204` | `action_result` | 春节 | 贺秀莲提出的与孙少安最迟结婚办事的时间是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 383 | WARN | `corpus_chunkids_000205` | `action_result` | 八月十四 | 双水村每年打红枣的固定日期是农历哪一天？ | entities 未在问答或 chunk 中出现: 农历日期; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 384 | WARN | `corpus_chunkids_000206` | `character` | 孙玉亭 | 在庙坪负责捡枣相关组织工作的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 385 | WARN | `corpus_chunkids_000206` | `place` | 河对面一队的禾场上 | 大队会计田海民和队干部们称量统计枣子的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 386 | WARN | `corpus_chunkids_000207` | `action_result` | 打枣 | 金波从学校赶回来是为了做什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 387 | PASS | `corpus_chunkids_000207` | `action_result` | 过完中秋节 | 孙少平准备什么时候回学校？ | 通过 |
| 388 | WARN | `corpus_chunkids_000208` | `action_result` | 扯结婚衣裳 | 打完枣又过了中秋节后，孙少安张罗着和贺秀莲去米家镇做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 389 | WARN | `corpus_chunkids_000208` | `character` | 田海民 | 双水村一队禾场上负责打算盘、报人名和斤称数码的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 390 | WARN | `corpus_chunkids_000209` | `place` | 米家镇 | 少安和秀莲扯布料的商店位于哪个镇？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 391 | PASS | `corpus_chunkids_000209` | `character` | 贺耀宗 | 寒露过了十来天时，从山西写信询问贺秀莲未归原因的人是谁？ | 通过 |
| 392 | WARN | `corpus_chunkids_000210` | `character` | 金俊武 | 孙少安带秀莲去石圪节公社时借的自行车属于谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 393 | WARN | `corpus_chunkids_000210` | `relation` | 同学 | 在石圪节公社当文书的刘根民和孙少安是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 394 | WARN | `corpus_chunkids_000211` | `place` | 黄原地区 | 原西县隶属于哪个地区？ | chunk 含时间/阶段信息但问题未显式限定 |
| 395 | PASS | `corpus_chunkids_000211` | `action_result` | 全部被坚冰封盖 | 1976年严寒时节黄土高原的大小河流处于什么状态？ | 通过 |
| 397 | WARN | `corpus_chunkids_000212` | `place` | 柳岔公社 | 田福军和张有智元月二日动身坐吉普车首先去了原西县的哪个公社？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 399 | WARN | `corpus_chunkids_000213` | `character` | 刘志祥 | 柳岔公社的副主任是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 400 | WARN | `corpus_chunkids_000214` | `place` | 柳岔街 | 田福军提出释放被劳教人员后，安排民兵小分队去哪个地点执行“堵资本主义”的任务？ | chunk 含时间/阶段信息但问题未显式限定 |
| 401 | WARN | `corpus_chunkids_000214` | `action_result` | 一起跑了 | 贾家沟胳膊被打坏的人和羊湾村被吊打的民工最后采取了什么行动？ | entities 未在问答或 chunk 中出现: 羊湾村被吊打民工; chunk 含时间/阶段信息但问题未显式限定 |
| 402 | PASS | `corpus_chunkids_000215` | `place` | 石圪节 | 田福军提到和周文龙谈过话之后，晚上争取赶到的地点是哪里？ | 通过 |
| 405 | WARN | `corpus_chunkids_000216` | `action_result` | 五十四人 | 柳岔公社大会战工地的被劳教民工昨晚偷跑2人后还剩多少人？ | chunk 含时间/阶段信息但问题未显式限定 |
| 406 | WARN | `corpus_chunkids_000217` | `character` | 刘志祥 | 负责详细记录田福军指示的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 407 | WARN | `corpus_chunkids_000217` | `place` | 石圪节公社 | 田福军和张有智在柳岔公社未等到周文龙返回，没吃晚饭便前往的公社是哪个？ | chunk 含时间/阶段信息但问题未显式限定 |
| 408 | WARN | `corpus_chunkids_000218` | `character` | 刘志祥 | 周文龙回到柳岔公社后，向他汇报县上两位领导来柳岔情况的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 409 | WARN | `corpus_chunkids_000218` | `action_result` | 放了 | 田主任离开柳岔公社时吩咐刘志祥让周文龙如何处置捉回来的那两个人？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 410 | WARN | `corpus_chunkids_000219` | `character` | 周文龙 | 冯世宽最看重的公社书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 412 | WARN | `corpus_chunkids_000220` | `character` | 徐治功 | 白明川回公社机关时，委托谁全面负责牛家沟公社会战工地的事？ | chunk 含时间/阶段信息但问题未显式限定 |
| 413 | WARN | `corpus_chunkids_000220` | `object` | 西风酒 | 白明川拿出来招待田福军和张有智的酒是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 414 | WARN | `corpus_chunkids_000221` | `action_result` | 七号 | 县革委会办事组通知田福军和张有智最迟需在几号前返回县城参加紧急会议？ | chunk 含时间/阶段信息但问题未显式限定 |
| 415 | WARN | `corpus_chunkids_000221` | `object` | 西凤酒 | 田福军、白明川、张有智三人深夜交谈时喝光的是什么酒？ | chunk 含时间/阶段信息但问题未显式限定 |
| 416 | WARN | `corpus_chunkids_000222` | `place` | 双水村 | 田福军原本打算回的村子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 417 | WARN | `corpus_chunkids_000222` | `action_result` | 拉灭电灯 | 田福军听到爱云的提醒后做了什么动作？ | chunk 含时间/阶段信息但问题未显式限定 |
| 418 | WARN | `corpus_chunkids_000223` | `relation` | 侄女 | 李登云的儿子正在追求的女性是田福军的什么亲属？ | chunk 含时间/阶段信息但问题未显式限定 |
| 419 | PASS | `corpus_chunkids_000223` | `action_result` | 牙疼 | 原西县常委会争论期间，李登云声称自己无法发言的原因是什么？ | 通过 |
| 420 | PASS | `corpus_chunkids_000224` | `place` | 北京 | 周恩来同志逝世的地点是哪里？ | 通过 |
| 421 | WARN | `corpus_chunkids_000224` | `character` | 周恩来 | 一九七六年一月八日逝世的中华人民共和国国务院总理是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 423 | WARN | `corpus_chunkids_000225` | `character` | 田福高 | 孙少安首先向谁诉说了自己结婚没有住处的难处？ | entities 未在问答或 chunk 中出现: 结婚住处难处; chunk 含时间/阶段信息但问题未显式限定 |
| 424 | WARN | `corpus_chunkids_000226` | `object` | 籽种 | 田福高提议借给孙少安结婚暂住的窑洞原本是用来存放什么的？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 425 | WARN | `corpus_chunkids_000226` | `character` | 金波 | 给正在裱糊结婚窑洞窗户的孙少安递浆糊和麻纸的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 426 | WARN | `corpus_chunkids_000227` | `character` | 王满银 | 孙少安走到罐子村的小石桥上时看到了谁？ | entities 未在问答或 chunk 中出现: 罐子村小石桥; chunk 含时间/阶段信息但问题未显式限定 |
| 427 | WARN | `corpus_chunkids_000227` | `object` | 待客的烟酒 | 孙少安吃完午饭后去石圪节街上买什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 429 | WARN | `corpus_chunkids_000228` | `character` | 刘根民 | 孙少安置办完供销社的物品后打算去石圪节公社找哪位同学参加自己的婚礼？ | chunk 含时间/阶段信息但问题未显式限定 |
| 430 | WARN | `corpus_chunkids_000229` | `character` | 胡得禄 | 石圪节公社唯一的专业理发师是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 431 | WARN | `corpus_chunkids_000229` | `object` | 二毛五分钱 | 孙少安在石圪节理发店理发花费了多少钱？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 432 | WARN | `corpus_chunkids_000230` | `character` | 贺秀莲 | 在快要临近春节的一天，与孙少安在自家举行简朴婚礼的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 433 | WARN | `corpus_chunkids_000230` | `place` | 罐子村 | 王满银在孙少安婚礼前一天晚饭后要返回哪个村睡觉？ | entities 未在问答或 chunk 中出现: 睡觉地点; chunk 含时间/阶段信息但问题未显式限定 |
| 434 | WARN | `corpus_chunkids_000231` | `character` | 金俊文 | 少安的婚礼上，负责制作席面菜的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 435 | WARN | `corpus_chunkids_000231` | `character` | 刘根民 | 少安的婚礼上，到场的唯一国家干部是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 436 | WARN | `corpus_chunkids_000232` | `object` | 两块杭州出的锦花缎被面 | 田润叶托田福堂转送给新婚的孙少安夫妇的礼物是什么？ | entities 未在问答或 chunk 中出现: 贺秀莲; chunk 含时间/阶段信息但问题未显式限定 |
| 437 | WARN | `corpus_chunkids_000232` | `character` | 干部 | 孙少安提到田润叶的工作身份是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 438 | WARN | `corpus_chunkids_000233` | `character` | 田润叶 | 1976年春天，坐在原西河边草坡上的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 439 | PASS | `corpus_chunkids_000233` | `object` | 马兰花 | 1976年春天，坐在原西河边的田润叶手里捏着的花是什么？ | 通过 |
| 440 | WARN | `corpus_chunkids_000234` | `character` | 徐爱云 | 田润叶的二妈叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 441 | WARN | `corpus_chunkids_000234` | `character` | 少安 | 去年的现在和田润叶一起坐在河岸上的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 443 | PASS | `corpus_chunkids_000235` | `action_result` | 春节 | 孙少安和他带回双水村的山西姑娘计划的结婚时间是什么时候？ | 通过 |
| 444 | WARN | `corpus_chunkids_000236` | `object` | 马兰花 | 润叶在原西河畔坐着时手里拿着的鲜艳的花是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 445 | WARN | `corpus_chunkids_000236` | `action_result` | 一上午 | 润叶在原西河畔坐了多久才站起来往回走？ | chunk 含时间/阶段信息但问题未显式限定 |
| 446 | PASS | `corpus_chunkids_000237` | `relation` | 同班同学 | 从初中到高中时期，田润叶和杜莉莉是什么关系？ | 通过 |
| 447 | WARN | `corpus_chunkids_000237` | `place` | 黄原 | 田润叶为躲避清明节被向前家缠磨，决定前往的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 449 | WARN | `corpus_chunkids_000238` | `relation` | 同学 | 田润叶和杜莉莉是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 451 | WARN | `corpus_chunkids_000239` | `character` | 贾冰 | 杜莉莉所在文化馆的贾老师全名叫什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 452 | PASS | `corpus_chunkids_000240` | `place` | 柳岔公社 | 贾冰的籍贯所属公社是什么？ | 通过 |
| 453 | WARN | `corpus_chunkids_000241` | `object` | 课外书 | 高中最后一个学期开始时，孙少平在空闲时会阅读什么类型的书籍？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 454 | WARN | `corpus_chunkids_000241` | `action_result` | 一年半 | 截止到高中最后一个学期开始，孙少平已经在原西中学就读了多长时间？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 455 | WARN | `corpus_chunkids_000242` | `object` | 绿皮笔记本 | 田晓霞到宿舍找孙少平后，在无人的小山沟里递给孙少平的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 456 | WARN | `corpus_chunkids_000242` | `place` | 她家里 | 田晓霞平时从什么地方拿书给孙少平阅读？ | chunk 含时间/阶段信息但问题未显式限定 |
| 457 | WARN | `corpus_chunkids_000243` | `character` | 顾养民 | 孙少平在教室抄诗的某天晚上，发现谁在偷看他放在桌上的诗歌笔记本？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 458 | WARN | `corpus_chunkids_000243` | `place` | 教室 | 孙少平会在夜深人静时到什么地方抄写田晓霞给的诗歌？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 459 | WARN | `corpus_chunkids_000244` | `character` | 田晓霞 | 孙少平抄完诗后将绿皮笔记本还给了谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 461 | WARN | `corpus_chunkids_000245` | `character` | 孙少平 | 大暴雨引发沟道洪水时，将被困在石崖下的侯玉英救上来的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 462 | WARN | `corpus_chunkids_000245` | `object` | 锄 | 侯玉英不听其他女同学劝阻独自走到石崖下时携带的农具是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 463 | WARN | `corpus_chunkids_000248` | `place` | 石圪节公社 | 孙玉厚家的兰香正在哪里上初中？ | entities 未在问答或 chunk 中出现: 孙兰香 |
| 464 | WARN | `corpus_chunkids_000248` | `object` | 带露水的草叶 | 孙兰香和二哥摘来给患眼病的奶奶淋眼睛的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 465 | WARN | `corpus_chunkids_000246` | `object` | 丛草 | 侯玉英遭遇洪水遇险时，死死揪着什么才没有被水卷走？ | chunk 含时间/阶段信息但问题未显式限定 |
| 466 | WARN | `corpus_chunkids_000246` | `character` | 孙少平 | 在侯玉英遭遇洪水生命遇险时，冒着生命危险抢救她的人是谁？ | entities 未在问答或 chunk 中出现: 洪水遇险; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 468 | PASS | `corpus_chunkids_000247` | `character` | 孙玉亭 | 文化革命初期带领双水村贫下中农造反队刨过金光明弟兄三家窑洞和院子的人是谁？ | 通过 |
| 469 | PASS | `corpus_chunkids_000247` | `object` | 烧酒 | 侯主任两口子为招待女儿的救命恩人孙少平摆的宴席上准备的酒是什么？ | 通过 |
| 470 | WARN | `corpus_chunkids_000250` | `character` | 金秀 | 孙兰香将自己打算退学的想法最先告诉的好朋友是谁? | entities 未在问答或 chunk 中出现: 退学打算; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 472 | WARN | `corpus_chunkids_000249` | `place` | 石圪节公社中学 | 孙兰香和金秀十三岁时升入的中学是哪所？ | chunk 含时间/阶段信息但问题未显式限定 |
| 473 | WARN | `corpus_chunkids_000249` | `character` | 汽车司机 | 金秀的父亲的职业是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 474 | WARN | `corpus_chunkids_000251` | `action_result` | 专心学习 | 孙兰香放弃回家劳动的打算后重新开始做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 475 | WARN | `corpus_chunkids_000251` | `action_result` | 金波要去参军了 | 九月初从县城传回的和金波相关的消息是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 476 | WARN | `corpus_chunkids_000253` | `place` | 中学的操场上 | 石圪节公社追悼会的中心会场设在哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 477 | WARN | `corpus_chunkids_000253` | `action_result` | 四人帮”被抓起来了 | 十月二十一日从北京传来的爆炸性消息是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 478 | WARN | `corpus_chunkids_000252` | `object` | 绿皮笔记本 | 金波参军送行时，兰香在县第二百货门市部购买的送给金波的纪念品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 479 | WARN | `corpus_chunkids_000252` | `object` | 钢笔 | 金波参军出发前，给兰香和金秀每人送的礼物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 481 | WARN | `corpus_chunkids_000255` | `action_result` | 开汽车 | 徐国强提到的李向前从事的职业是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 482 | PASS | `corpus_chunkids_000257` | `object` | 棉大衣 | 这天下午田润叶回家取的物品是什么？ | 通过 |
| 483 | WARN | `corpus_chunkids_000257` | `character` | 向前 | 徐国强想要促成田润叶和谁的亲事？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 484 | WARN | `corpus_chunkids_000254` | `character` | 田福堂 | 徐国强老汉抽的旱烟是谁带来的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 485 | WARN | `corpus_chunkids_000254` | `object` | 各种庄稼 | 日常没事的时候徐国强老汉会在院子花坛的小块土地上营务什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 486 | WARN | `corpus_chunkids_000258` | `character` | 孙少安 | 田润叶内心真正爱慕的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 488 | WARN | `corpus_chunkids_000256` | `character` | 李登云 | 志英的丈夫是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 489 | WARN | `corpus_chunkids_000256` | `character` | 冯世宽 | 李登云站在哪个一把手的一边？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 492 | PASS | `corpus_chunkids_000262` | `relation` | 同班同学 | 白明川与周文龙高中时的关系是什么？ | 通过 |
| 493 | WARN | `corpus_chunkids_000262` | `object` | 毛毯 | 孙少安托孙少平捎给润叶的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 494 | WARN | `corpus_chunkids_000261` | `object` | 红烧肘子 | 石圪节食堂的胖炉头胡得福名扬全县的拿手菜尤其是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 495 | PASS | `corpus_chunkids_000261` | `character` | 周文龙 | 与石圪节公社主任白明川是高中同班同学的柳岔公社主任叫什么名字？ | 通过 |
| 496 | WARN | `corpus_chunkids_000260` | `place` | 县招待所的大餐厅 | 田润叶与李向前的婚礼举办地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 497 | WARN | `corpus_chunkids_000260` | `character` | 徐爱云 | 给田润叶脖颈系米色纱巾的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 498 | WARN | `corpus_chunkids_000263` | `object` | 毛毯 | 少安夫妇送给田润叶的结婚礼物是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 499 | PASS | `corpus_chunkids_000263` | `character` | 冯世宽主任 | 田润叶婚礼上，走在最前面进入餐厅的县上领导是谁？ | 通过 |
| 500 | WARN | `corpus_chunkids_000265` | `character` | 田五叔 | 给孙少安和贺秀莲编排调侃小曲的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 501 | WARN | `corpus_chunkids_000265` | `place` | 田家圪崂饲养院 | 孙少安与贺秀莲结婚近十个月时居住的小窑位于哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 502 | WARN | `corpus_chunkids_000264` | `character` | 马国雄 | 田润叶婚礼上的特邀司仪是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 505 | WARN | `corpus_chunkids_000266` | `place` | 石圪节医院 | 孙少安和秀莲为检查秀莲未怀孕的原因去了哪家医院？ | entities 未在问答或 chunk 中出现: 生育检查; chunk 含时间/阶段信息但问题未显式限定 |
| 506 | WARN | `corpus_chunkids_000269` | `object` | 二斗小米 | 贺耀宗托顺车给孙少安家捎来了什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 508 | WARN | `corpus_chunkids_000268` | `object` | 黄原出的羊毛毯 | 一九七六年临近结束前不久，孙少安给即将结婚的田润叶准备的结婚礼物是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 512 | WARN | `corpus_chunkids_000271` | `object` | 两个白面馍 | 婆婆让秀莲给孙少安带的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 514 | WARN | `corpus_chunkids_000273` | `character` | 兰香 | 孙少平为攒够毕业的花费，暑假时和谁一起挖了二十多天药材？ | entities 未在问答或 chunk 中出现: 毕业花费,挖药材 |
| 515 | WARN | `corpus_chunkids_000273` | `object` | 手帕 | 孙少平毕业时送给班里女同学的礼物是什么？ | entities 未在问答或 chunk 中出现: 毕业礼物; 答案在 chunk 中出现多次，唯一性需复核 |
| 516 | WARN | `corpus_chunkids_000274` | `place` | 青海 | 金波当兵去往的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 517 | WARN | `corpus_chunkids_000274` | `object` | 长笛 | 金波在师部文工团演奏的乐器是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 518 | WARN | `corpus_chunkids_000277` | `character` | 金光明 | 在二门市抓住偷手帕的郝红梅的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 519 | WARN | `corpus_chunkids_000277` | `object` | 手帕 | 郝红梅在二门市偷的物品是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 520 | WARN | `corpus_chunkids_000279` | `character` | 金光明 | 同学们即将离校时，抓获郝红梅偷手帕的售货员叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 521 | WARN | `corpus_chunkids_000279` | `action_result` | 五块 | 郝红梅在二门市部付钱购买的手帕有多少块？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 522 | PASS | `corpus_chunkids_000276` | `object` | 多兜黄挂包 | 田晓霞送给孙少平的毕业礼物是什么？ | 通过 |
| 523 | WARN | `corpus_chunkids_000278` | `character` | 顾养民 | 临近高中毕业时，给郝红梅的生活带来无限美好希望的人物是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 524 | WARN | `corpus_chunkids_000278` | `place` | 黄原地区 | 顾养民的父母亲属于哪个地区的人物？ | chunk 含时间/阶段信息但问题未显式限定 |
| 525 | PASS | `corpus_chunkids_000272` | `place` | 原西县 | 一九七七年元月中旬，孙少平即将在哪个县的高中毕业？ | 通过 |
| 526 | PASS | `corpus_chunkids_000272` | `place` | 双水村 | 一九七七年元月中旬，即将从原西县高中毕业的孙少平打算毕业后回到哪个村子？ | 通过 |
| 527 | PASS | `corpus_chunkids_000275` | `object` | 白馍 | 快毕业时期的某个下午，孙少平报的饭是什么？ | 通过 |
| 528 | PASS | `corpus_chunkids_000275` | `place` | 街上的国营食堂 | 快毕业时，田晓霞带孙少平去吃饭的地点是哪里？ | 通过 |
| 529 | PASS | `corpus_chunkids_000282` | `place` | 双水村 | 金光明夸赞孙少平是哪个村的人才？ | 通过 |
| 531 | WARN | `corpus_chunkids_000280` | `character` | 孙少平 | 侯玉英的救命恩人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 533 | WARN | `corpus_chunkids_000281` | `character` | 郝红梅 | 冬夜中的孙少平回忆两年前的寒冷日子时，想起自己常和谁在中学饭场不期而遇？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 534 | PASS | `corpus_chunkids_000284` | `action_result` | 烧掉了 | 孙少平看完侯玉英写给自己的恋爱信后，最终是如何处理这封信的？ | 通过 |
| 535 | PASS | `corpus_chunkids_000284` | `place` | 厕所 | 孙少平烧掉侯玉英所写的恋爱信的地点是哪里？ | 通过 |
| 537 | WARN | `corpus_chunkids_000283` | `place` | 学校 | 孙少平目送郝红梅的身影消失后要前往什么地方？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 538 | WARN | `corpus_chunkids_000286` | `character` | 孙玉亭 | 田福堂的主要助手是谁? | chunk 含时间/阶段信息但问题未显式限定 |
| 539 | PASS | `corpus_chunkids_000286` | `place` | 村里的农田基建工地 | 孙少平回到双水村后的第三天去了哪里劳动? | 通过 |
| 540 | WARN | `corpus_chunkids_000288` | `character` | 孙玉亭 | 双水村学校的贫管会主任是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 542 | WARN | `corpus_chunkids_000285` | `character` | 田福堂 | 双水村的大队书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 543 | WARN | `corpus_chunkids_000285` | `action_result` | 秋后庄稼收割毕 | 田福堂计划在什么时间之后干有震动性的工作？ | chunk 含时间/阶段信息但问题未显式限定 |
| 544 | PASS | `corpus_chunkids_000290` | `character` | 孙少平 | 孙玉亭准备让谁担任村里新办初中班的教师？ | 通过 |
| 545 | PASS | `corpus_chunkids_000290` | `object` | 一卷旧棉絮 | 孙玉亭在大哥家吃完中午饭后，大嫂给了他什么物品让他带走？ | 通过 |
| 547 | WARN | `corpus_chunkids_000289` | `object` | 旧棉花 | 少安妈提出可以拿给孙玉亭的物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 548 | WARN | `corpus_chunkids_000287` | `character` | 孙玉亭 | 田福堂的主要助手是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 550 | WARN | `corpus_chunkids_000293` | `object` | 电动刮胡子刀 | 李向前拥有的、原西县很多人还没听说过的日用物品是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 552 | WARN | `corpus_chunkids_000292` | `object` | 驾驶执照 | 李向前跟县供销社老司机当助手不到一年，考取了什么证件？ | chunk 含时间/阶段信息但问题未显式限定 |
| 553 | WARN | `corpus_chunkids_000292` | `character` | 润叶 | 李向前结婚后，一直不肯与他同床的妻子叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 554 | PASS | `corpus_chunkids_000295` | `action_result` | 拍发了一封电报 | 李向前在省城下火车后给润叶做了什么事？ | 通过 |
| 555 | WARN | `corpus_chunkids_000295` | `place` | 运输公司的家属院 | 李向前结婚后的居住地点是哪里？ | entities 未在问答或 chunk 中出现: 运输公司家属院 |
| 556 | WARN | `corpus_chunkids_000291` | `place` | 石圪节 | 双水村未办初中班时，村里的娃娃上初中需要去哪个地方上学？ | entities 未在问答或 chunk 中出现: 上学地点; 答案在 chunk 中出现多次，唯一性需复核 |
| 557 | WARN | `corpus_chunkids_000294` | `place` | 北京 | 李向前找借口独自前往的城市是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 558 | WARN | `corpus_chunkids_000299` | `place` | 后子头公社 | 一个星期前田福军前往检查工作的全县最偏远的公社名称是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 559 | WARN | `corpus_chunkids_000299` | `object` | 柴棍 | 快过端阳节时田福军前往最后一个“死角”村的山路上手里拉着什么物品？ | chunk 含时间/阶段信息但问题未显式限定 |
| 560 | WARN | `corpus_chunkids_000296` | `action_result` | 强奸未遂 | 李向前强奸妻子田润叶的行为最终结果是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 561 | WARN | `corpus_chunkids_000296` | `action_result` | 踩瘪了 | 李向前发现家里空无一人后对地上的大皮箱做了什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 563 | WARN | `corpus_chunkids_000302` | `character` | 冯世宽 | 派吉普车到后子头公社接田福军回城的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 564 | WARN | `corpus_chunkids_000301` | `object` | 糠团子 | 田福军在队长家揭开锅盖后，看到锅里除两个玉米面馍外的其余食物是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 565 | PASS | `corpus_chunkids_000300` | `place` | 土崖凹 | 临近中午时田福军抵达的小村子名称是什么？ | 通过 |
| 566 | WARN | `corpus_chunkids_000300` | `object` | 玉米面馍 | 田福军在土崖凹村生产队长家吃午饭时拿起的食物是什么？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 569 | WARN | `corpus_chunkids_000305` | `action_result` | 如何接待中央老首长的问题 | 冯世宽走进田福军的办公室要和他商量什么事？ | chunk 含时间/阶段信息但问题未显式限定 |
| 570 | PASS | `corpus_chunkids_000298` | `action_result` | 三个 | 一个星期以来，田福军走过了多少个偏僻的死角村子？ | 通过 |
| 571 | PASS | `corpus_chunkids_000298` | `action_result` | 不在 | 田福军此次造访偏僻死角村子的安排是否在他的原工作计划内？ | 通过 |
| 572 | PASS | `corpus_chunkids_000306` | `action_result` | 如何接待中央的高老 | 1977年端阳节原西县常委们正在讨论的议题是什么？ | 通过 |
| 573 | WARN | `corpus_chunkids_000306` | `place` | 高家园公社高店则村 | 中央高老的故乡是原西县的哪个村庄？ | chunk 含时间/阶段信息但问题未显式限定 |
| 574 | WARN | `corpus_chunkids_000304` | `place` | 原西 | 润叶明确表示自己不想继续待的地点是哪里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 575 | WARN | `corpus_chunkids_000304` | `action_result` | 离婚 | 润叶针对自己和李向前的婚姻明确表示当前不会做出什么行为？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 576 | WARN | `corpus_chunkids_000303` | `relation` | 二爸 | 田福军是田润叶的什么亲属？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 577 | WARN | `corpus_chunkids_000303` | `place` | 学校 | 田润叶称自己婚后住在什么地方？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 578 | WARN | `corpus_chunkids_000308` | `character` | 吴克俭 | 原西县民政方面的负责人是谁？ | entities 未在问答或 chunk 中出现: 民政方面负责人 |
| 579 | WARN | `corpus_chunkids_000308` | `object` | 红蓝铅笔 | 冯世宽在常委会上发言时手里拿的是什么笔？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 580 | WARN | `corpus_chunkids_000310` | `place` | 黄原地区 | 吃过午饭后苗凯坐车返回的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 581 | WARN | `corpus_chunkids_000310` | `action_result` | 按秩序停放汽车 | 原西县招待所院子里用白灰划出的方格子用途是什么？ | entities 未在问答或 chunk 中出现: 白灰方格; chunk 含时间/阶段信息但问题未显式限定 |
| 582 | WARN | `corpus_chunkids_000309` | `place` | 县革委会的客房 | 散会后冯世宽陪同苗凯前往何处休息？ | entities 未在问答或 chunk 中出现: 县革委会客房; chunk 含时间/阶段信息但问题未显式限定 |
| 583 | WARN | `corpus_chunkids_000309` | `character` | 张有智 | 田福军在会上发表关于接待高老的看法后，第一个表态完全同意他观点的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 584 | WARN | `corpus_chunkids_000307` | `character` | 苗凯 | 黄原地区革委会主任是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 585 | WARN | `corpus_chunkids_000307` | `character` | 马国雄 | 原西县革委会成立的接高老办公室由谁挂帅？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 586 | WARN | `corpus_chunkids_000311` | `place` | 原西县招待所 | 高老回到原西县时，冯世宽等人等候迎接他的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 587 | WARN | `corpus_chunkids_000312` | `place` | 高店则 | 高老的出生地是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 588 | PASS | `corpus_chunkids_000312` | `character` | 顾健翎 | 高老当年在本县打仗负伤时为他治伤的人是谁？ | 通过 |
| 590 | WARN | `corpus_chunkids_000315` | `character` | 孙玉亭 | 田福堂在为炸山拦坝的事焦虑时想到的高参是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 591 | WARN | `corpus_chunkids_000315` | `place` | 金家湾北头 | 孙玉亭提出给炸山拦坝需搬家的人家箍新窑洞的地点是哪里？ | entities 未在问答或 chunk 中出现: 搬家人家; chunk 含时间/阶段信息但问题未显式限定 |
| 592 | PASS | `corpus_chunkids_000316` | `place` | 大队部的办公窑里 | 田福堂和孙玉亭拉谈完相关事宜的第二天晚上，双水村有职务的干部被集中到了什么地方？ | 通过 |
| 593 | WARN | `corpus_chunkids_000316` | `action_result` | 先开个干部会 | 田福堂询问孙玉亭开展相关工作先从哪里下手时，孙玉亭提出的首个举措是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 594 | WARN | `corpus_chunkids_000314` | `place` | 哭咽河 | 田福堂计划炸掉半个神仙山和庙坪山拦成大坝，想要改造为米粮川的河流叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 595 | WARN | `corpus_chunkids_000314` | `place` | 石圪节公社 | 田福堂设想炸掉半个神仙山和庙坪山拦成的大坝，起码是哪个公社最大的坝？ | chunk 含时间/阶段信息但问题未显式限定 |
| 596 | WARN | `corpus_chunkids_000320` | `place` | 小学校的院子 | 双水村大队召开全体社员大会的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 597 | WARN | `corpus_chunkids_000320` | `character` | 孙玉亭 | 双水村大队负责卖掉几万斤储备粮的人是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 600 | WARN | `corpus_chunkids_000318` | `place` | 原西城 | 金光明工作的城市是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 601 | WARN | `corpus_chunkids_000318` | `character` | 金光亮 | 金光明的哥哥叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 602 | WARN | `corpus_chunkids_000321` | `object` | 柴草 | 金俊武在庙坪后山犁完麦地后独自扛着的是什么物品？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 603 | WARN | `corpus_chunkids_000321` | `action_result` | 搬家 | 接近白露时节，金俊武等人秋庄稼收割完之后要做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 605 | WARN | `corpus_chunkids_000317` | `character` | 妻子姚淑芳 | 金光明是接到谁的信赶回村里知晓搬家通知的？ | chunk 含时间/阶段信息但问题未显式限定 |
| 606 | WARN | `corpus_chunkids_000319` | `character` | 金俊山 | 田福堂准备召开社员大会动员搬迁前，主动提出去和金俊文、金俊武两兄弟协商搬迁事宜的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 607 | WARN | `corpus_chunkids_000323` | `character` | 孙玉亭 | 金俊文提议要扣在窑里捶一顿的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 608 | WARN | `corpus_chunkids_000323` | `character` | 桂兰嫂 | 金俊武建议派谁去探问王彩娥的正经打算？ | chunk 含时间/阶段信息但问题未显式限定 |
| 609 | WARN | `corpus_chunkids_000324` | `character` | 孙玉亭 | 金富和金强把谁扣在了金俊斌家里？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 610 | WARN | `corpus_chunkids_000324` | `object` | 镢头 | 金俊武没吃午饭就去自留地干活时扛的农具是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 611 | PASS | `corpus_chunkids_000326` | `character` | 银花 | 田海民的媳妇叫什么名字？ | 通过 |
| 613 | WARN | `corpus_chunkids_000327` | `place` | 院子里 | 田海民离开后，田福堂没有回家而是待在什么地方？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 614 | WARN | `corpus_chunkids_000327` | `character` | 田海民 | 金俊文和金俊武正在焦急等待谁的到来？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 615 | WARN | `corpus_chunkids_000328` | `character` | 孙玉亭 | 和王彩娥一同被关在窑里的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 617 | WARN | `corpus_chunkids_000334` | `object` | 拖拉机 | 田海民用什么交通工具将徐治功等人送回石圪节？ | chunk 含时间/阶段信息但问题未显式限定 |
| 618 | WARN | `corpus_chunkids_000334` | `action_result` | 不予追究 | 徐治功主持召开的双水村和王家庄大队领导人紧急联席会议对孙玉亭与王彩娥的男女关系问题作出了什么处理决定？ | chunk 含时间/阶段信息但问题未显式限定 |
| 619 | WARN | `corpus_chunkids_000330` | `character` | 刘玉升 | 是谁摸黑赶到王家庄向王彩娥娘家人报告了孙玉亭在王彩娥窑里出事的消息？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 620 | WARN | `corpus_chunkids_000325` | `character` | 田海民 | 金俊武安排金富去叫来处理孙玉亭相关事件的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 622 | WARN | `corpus_chunkids_000332` | `character` | 田福堂 | 与双水村发生冲突时，王家庄的人淌过东拉河到田家圪崂要找的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 623 | PASS | `corpus_chunkids_000331` | `place` | 哭咽河后沟道 | 王彩娥家的门被打开后的混乱中，孙玉亭逃往的地点是哪里？ | 通过 |
| 624 | WARN | `corpus_chunkids_000331` | `character` | 田牛 | 金家户族与王姓村民混战期间，毫无缘由参战的两旁世人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 625 | WARN | `corpus_chunkids_000335` | `object` | 几千斤炸药 | 孙玉亭用卖掉大队几万斤储备高粱的钱买回了什么？ | entities 未在问答或 chunk 中出现: 大队储备高粱; chunk 含时间/阶段信息但问题未显式限定 |
| 626 | WARN | `corpus_chunkids_000335` | `character` | 田福堂 | 谁从县上请来的工程专家早在初秋就选好了炸山和拦坝的具体地址？ | chunk 含时间/阶段信息但问题未显式限定 |
| 627 | WARN | `corpus_chunkids_000329` | `character` | 田福高 | 田福堂派谁去金家湾那面查看情况？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 628 | PASS | `corpus_chunkids_000329` | `object` | 纸烟 | 田福堂递给田福高的物品是什么？ | 通过 |
| 629 | WARN | `corpus_chunkids_000333` | `place` | 石圪节 | 王家庄的人进入双水村引发群架后，金俊山安排田海民开拖拉机去哪个地方找公社领导？ | chunk 含时间/阶段信息但问题未显式限定 |
| 630 | WARN | `corpus_chunkids_000333` | `character` | 杨高虎 | 徐治功决定前往双水村处理群架事件时，找来的公社武装专干是谁？ | entities 未在问答或 chunk 中出现: 双水村群架; 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 631 | WARN | `corpus_chunkids_000336` | `character` | 张桂兰 | 金俊文的妻子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 632 | WARN | `corpus_chunkids_000336` | `place` | 石圪节公社 | 王彩娥与孙玉亭的相关关系引发的武斗事件震惊了哪个公社？ | chunk 含时间/阶段信息但问题未显式限定 |
| 633 | WARN | `corpus_chunkids_000337` | `character` | 金俊山 | 金俊武做不通母亲搬迁工作时，打发金强去报告的大队副书记是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 635 | WARN | `corpus_chunkids_000340` | `character` | 孙玉亭 | 双水村哭咽河准备炸山期间，带领爆破组的人是谁? | entities 未在问答或 chunk 中出现: 哭咽河炸山 |
| 636 | WARN | `corpus_chunkids_000340` | `place` | 田家圪崂 | 田福堂告辞金家人后返回的地点是哪里? | chunk 含时间/阶段信息但问题未显式限定 |
| 637 | WARN | `corpus_chunkids_000341` | `action_result` | 拉一辆队里的架子车回来 | 孙少安得知秀莲可能要临产时，安排孙少平去做什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 638 | WARN | `corpus_chunkids_000341` | `place` | 石圪节医院 | 孙少安坚持要带临产的秀莲去的医院叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 640 | WARN | `corpus_chunkids_000338` | `action_result` | 以失败而告终 | 金俊山劝说金老太太挪窝的工作最终结果是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 641 | WARN | `corpus_chunkids_000342` | `character` | 田二 | 被众人送到石圪节公社医院抢救最终身亡的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 642 | WARN | `corpus_chunkids_000342` | `object` | 自己卷的旱烟卷 | 孙少安在石圪节公社医院院子等待贺秀莲生产时抽的烟是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 643 | WARN | `corpus_chunkids_000344` | `character` | 姚淑芳 | 双水村学校唯一的公派教师是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 645 | WARN | `corpus_chunkids_000339` | `object` | 大坝 | 双水村需要金老太太搬家是因为要在她居住的地方修建什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 646 | PASS | `corpus_chunkids_000346` | `action_result` | 落榜 | 十月份教育部发布当年大学招生消息后，孙少平参加高考的结果是什么？ | 通过 |
| 647 | WARN | `corpus_chunkids_000346` | `character` | 金波 | 十二月上旬，突然复员回到双水村的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 648 | WARN | `corpus_chunkids_000345` | `place` | 原西城郊 | 田晓霞插队的地点是哪里？ | chunk 含时间/阶段信息但问题未显式限定 |
| 649 | WARN | `corpus_chunkids_000345` | `character` | 孙玉厚家的二小子 | 在双水村的日常生活中，孙少平给自己的身份定位是什么？ | chunk 含时间/阶段信息但问题未显式限定 |
| 650 | WARN | `corpus_chunkids_000343` | `relation` | 同班同学 | 田润生和孙少平是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 651 | WARN | `corpus_chunkids_000347` | `action_result` | 立刻从学校赶到金波家 | 孙少平得知金波复员回到双水村后采取了什么行动？ | chunk 含时间/阶段信息但问题未显式限定 |
| 652 | WARN | `corpus_chunkids_000347` | `place` | 青海 | 金波和孙少平久别重逢时给孙少平叙说的是哪个地方的民情风俗？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 653 | WARN | `corpus_chunkids_000349` | `place` | 双水村 | 阳历年前的一天，田晓霞如期回到了哪个村子？ | chunk 含时间/阶段信息但问题未显式限定 |
| 654 | WARN | `corpus_chunkids_000349` | `character` | 金三锤 | 孙少平到金光亮家是给谁辅导作文？ | entities 未在问答或 chunk 中出现: 作文辅导; chunk 含时间/阶段信息但问题未显式限定 |
| 655 | WARN | `corpus_chunkids_000353` | `character` | 田福高 | 孙少安打算就冒险的事和社员商量后，首先找的一队副队长是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 656 | WARN | `corpus_chunkids_000353` | `place` | 饲养员田万江的窑洞 | 双水村一队的“会议室”是什么地方？ | chunk 含时间/阶段信息但问题未显式限定 |
| 657 | PASS | `corpus_chunkids_000352` | `place` | 石圪节 | 孙少安前不久到哪个地方赴集时听到了安徽铁匠说的承包制相关消息？ | 通过 |
| 658 | PASS | `corpus_chunkids_000352` | `action_result` | 给社员多划了点猪饲料地 | 孙少安前几年因为什么行为被公社批判？ | 通过 |
| 660 | WARN | `corpus_chunkids_000355` | `place` | 石圪节公社 | 田福堂就孙少安的相关问题向上级汇报去的是哪个公社？ | chunk 含时间/阶段信息但问题未显式限定 |
| 661 | WARN | `corpus_chunkids_000348` | `character` | 金三锤 | 孙少平班上金光亮的儿子叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
| 663 | WARN | `corpus_chunkids_000350` | `place` | 石圪节 | 孙少平告知家人要请田晓霞来家中吃饭时，提出要去哪个地方割羊肉？ | chunk 含时间/阶段信息但问题未显式限定 |
| 664 | WARN | `corpus_chunkids_000350` | `character` | 田晓霞 | 孙少平第一次带回家吃饭的客人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 665 | WARN | `corpus_chunkids_000356` | `character` | 孙玉亭 | 建议孙少安主动到公社投案以争取党和政府从宽处理的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 666 | PASS | `corpus_chunkids_000356` | `action_result` | 用长途电话向县革委会的领导作了汇报 | 公社主任白明川和徐治功得知双水村的“严重政治事件”后做出了什么行动？ | 通过 |
| 667 | WARN | `corpus_chunkids_000357` | `action_result` | 坚决制止 | 地区革委会主任苗凯针对双水村相关情况给出的指示是什么？ | 答案在 chunk 中出现多次，唯一性需复核 |
| 668 | PASS | `corpus_chunkids_000357` | `place` | 石圪节公社 | 县常委会决定要求哪个公社坚决制止双水村的资本主义复辟倾向？ | 通过 |
| 671 | WARN | `corpus_chunkids_000354` | `character` | 孙少安 | 双水村大队第一生产队一九七八年农业作业组生产合同的队长签名是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 672 | WARN | `corpus_chunkids_000354` | `character` | 田福林 | 双水村大队第一生产队一九七八年农业作业组生产合同的第三农业组长签名是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 673 | PASS | `corpus_chunkids_000358` | `character` | 孙少安 | 针对双水村的资本主义复辟倾向事件，县革委会要求公社严肃批评教育的生产队长是谁？ | 通过 |
| 674 | WARN | `corpus_chunkids_000359` | `character` | 李登云 | 一九七八年初临近春节时，接替冯世宽出任原西县革委会主任的人是谁？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 675 | WARN | `corpus_chunkids_000359` | `relation` | 同学 | 一九七八年春节前后被提拔为石圪节公社副主任的刘根民和孙少安是什么关系？ | chunk 含时间/阶段信息但问题未显式限定 |
| 676 | WARN | `corpus_chunkids_000361` | `character` | 田五 | 双水村此次闹秧歌的秧歌队伞头是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 680 | WARN | `corpus_chunkids_000360` | `place` | 石圪节公社 | 双水村的秧歌是全哪个公社最有名的？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 681 | WARN | `corpus_chunkids_000360` | `character` | 石钟 | 田福军的老上级是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 682 | WARN | `corpus_chunkids_000363` | `character` | 王明清 | 率先向田五发难的罐子村伞头叫什么名字？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 683 | WARN | `corpus_chunkids_000363` | `place` | 双水村 | 王明清的唱词中提到他们转九曲来到的是哪个村子？ | 答案在 chunk 中出现多次，唯一性需复核; chunk 含时间/阶段信息但问题未显式限定 |
| 684 | WARN | `corpus_chunkids_000364` | `character` | 孙少安 | 田福军说完关于农村未来局面的看法后，和谁紧紧握了握手？ | chunk 含时间/阶段信息但问题未显式限定 |
| 685 | WARN | `corpus_chunkids_000364` | `object` | 旱烟卷 | 孙少安站在小土坡上飞快卷起的是什么烟卷？ | chunk 含时间/阶段信息但问题未显式限定 |
| 686 | WARN | `corpus_chunkids_000362` | `character` | 孙少平 | 双水村大秧歌和小戏的总导演是谁？ | chunk 含时间/阶段信息但问题未显式限定 |
| 687 | WARN | `corpus_chunkids_000362` | `character` | 胡得禄 | 改嫁到石圪节的王彩娥的丈夫叫什么名字？ | chunk 含时间/阶段信息但问题未显式限定 |
