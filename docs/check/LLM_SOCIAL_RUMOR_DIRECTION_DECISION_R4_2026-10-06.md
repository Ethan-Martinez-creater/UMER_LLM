# UMER 第二部分工作：近期文献调研与方向选择记录（R4）

- 整理日期：2026-10-06。
- 性质：本次会话的调研结论与方向排序记录，不是方法立项、完整实验协议或执行授权。
- 研究目标：第一部分为传统社交网络谣言检测 UMER；第二部分保留“大模型＋社交网络谣言／虚假新闻”，与网络空间安全专业关联，能够形成独立研究问题，而非仅融合 LLM 提升分类准确率。
- 文献重点：近两年（约 2024-10 至本轮检索时点），必要纳入更早直接前驱；正式主会、Findings、工作坊、期刊、已接收待召开会议论文与预印本分别标注。
- 证据边界：本文整理此前已完成的远端仓库核查和文献检索；保存本报告时没有重新检索全部文献、更新远端 HEAD 或运行实验。不是穷尽性查重证明。

## 1. 最终方向排序

**关闭 BCR-Utility 当前方法线，不进入原 M2；保留大方向，选择一个优先候选和两个备选候选。**

| 排序 | 方向 | 研究对象 | 当前定位 |
|---|---|---|---|
| 优先 | 虚假社会共识下的依赖感知社会证据安全 | 单个事件、给定时间截点内的证据使用 | 优先进行数据与机制可行性筛查；不直接开始完整系统开发 |
| 备选一 | 社会信息污染下的事实核查知识写入安全 | 跨事件持续存在的知识库／检索记忆 | 主要转向出口；需补查记忆污染近邻和连续事件数据 |
| 备选二 | 时序证据变化下的可撤回事实核查 | 同一事件的判决与证据到达轨迹 | 较低优先级；近邻密集、逐截点标注成本高 |

这里的“最终”指本次讨论形成的候选排序，不代表三条路线的创新已经核实、数据已经就绪或效果已经成立。用户要求记录该排序，不等于授权启动新推理、训练、数据标注或服务器实验。

三条不同时开展实验。首选失败时，依据失败原因评估备选，而不是用相同机制换名称继续。

## 2. 与已有文档和历史结论的关系

保留以下历史报告，不覆盖、不回写其原有结论：

- [R1：查重、可行性与研究方向决策](LLM_RUMOR_RESEARCH_DECISION_2026-10-03.md)。
- [R2：第二轮深度选题检索](LLM_SOCIAL_MISINFORMATION_DIRECTION_SEARCH_R2_2026-10-04.md)。
- [R3：完整研究主线的第三轮评估](LLM_SOCIAL_RUMOR_THESIS_DIRECTION_SEARCH_R3_2026-10-04.md)。
- [此前研究准入审计计划](RESEARCH_ADMISSION_AUDIT_PLAN_2026-10-03.md)。

本文记录本次会话的新排序，但不取消旧文档识别出的近邻、数据和理论门槛：

1. 去重、provenance、一般 belief revision 和普通鲁棒性不能因重新命名而变成创新。
2. 本轮优先方向不是 R3 的“结构依赖安全认证”方案；不主张已取得认证界、决策稳定性证书或事实正确性保证。
3. 知识写入与时序修正备选仍可能退化为通用记忆管理或 belief revision，必须补充方法级查重。
4. 本次没有重新启用早期候选、准入审计计划或旧实验。

若后续采用隔离、聚合或认证机制，R3 已记录的 RobustRAG、图认证等近邻仍须纳入，不能因本轮表格未列出便视为已排除。

## 3. 为什么停止当前 BCR 方法线

### 3.1 研究状态的证据基线

此前核验的远端 `main` 与本地 HEAD 为：

`d71991dd9745a910e94ca91398d0b871a7ffda7f`

其正式结论为 `M1F_CLOSE_BCR_UTILITY_METHOD`。交接文档首页停留在 `b62010b`／M1-E，不能以其中早期 promising 表述覆盖后续 M1-F。

- [固定版本 M1-F 报告](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/d71991dd9745a910e94ca91398d0b871a7ffda7f/results/bcr_utility_v1/m1f/M1F_REPORT.md)。
- [固定版本 M1-F 判决](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/d71991dd9745a910e94ca91398d0b871a7ffda7f/results/bcr_utility_v1/m1f/M1F_VERDICT.json)。
- [研究交接文档](../../UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md)。

### 3.2 已有结果对选择的约束

| 路线 | 正式证据／结论 | 对后续的约束 |
|---|---|---|
| UMER | PHEME Accuracy／Macro-F1 为 0.8988／0.8925；Ma-Weibo 为 0.9689／0.9689 | 第一部分成果保留；不能夸大检索或传播图的事实推理能力 |
| TC-DSCR／MS-TSR | 最终证据支持压缩价值；reader transfer 不稳定，主张边界为 `MS_TSR_COMPRESSION_ONLY` | 不把压缩、teacher 保真或 evidence-ID grounding 当作可靠事实核验 |
| CR-TSER V2R1 | P2 的 edge 增量 0.012150，未达到 0.02；95% CI 为 [-0.002342, +0.027196]；正式 NO_GO | 不依赖已失败的强结构交互假设；不等于证明所有社会结构无效 |
| BCR M1／M1-E | LIGHT-TOUCH 曾获历史条件通过；fingerprint 增量未获稳定支持 | 保留历史判决，不以其替代最新审计 |
| BCR M1-F | D5 未超过 dev 选出的强简单基线 S2 source-state | 关闭当前方法线，不扩 reader 或调阈值事后挽救 |

Ma-Weibo 的 M1-F 关键结果：

- D5／S4 mean Macro-F1 = 0.3072；S2 = 0.3148。
- D5−S2 = -0.007549；95% event-bootstrap CI = [-0.095612, +0.079948]。
- 三个 reader 中两个点估计为正，但最差 reader 差值为 -0.144261。
- active HELPFUL／HARMFUL Macro-F1 差值为 -0.189262。
- 同快照 centered Spearman 增量约 +0.002364，CI 跨零，不能支撑稳定证据内排序能力。

这不是“所有 gate 都失败”：个别点估计条件通过，但核心有效性门槛未通过。PHEME 只作诊断，不能救回 primary。既有 SERVER verifier 的 66 checks、0 issues 说明实现审计通过，不等于科研假设成功；本文没有重跑 verifier。

### 3.3 不继续的实质理由

宽泛的 reader-specific utility、删除归因、跨 reader 检索监督与行为画像已有直接前作；当前狭义实现又未胜强基线。未检索到完全同设定论文不等于值得继续。需要新机制和独立证据，而不是仅调模型、增加 reader 或扩大实验规模。

## 4. 近期文献怎样约束选题

### 4.1 普通融合与常规任务升级已拥挤

| 文献与身份 | 已有研究内容 | 本项目不能重复包装的部分 |
|---|---|---|
| Exploring Large Language Models for Effective Rumor Detection on Social Media，NAACL 2025 主会 [链接](https://aclanthology.org/2025.naacl-long.128/) | 结合语义与传播信息筛选社会上下文供 LLM 判断 | “UMER／GAT 选评论＋LLM 判别” |
| LLM-enhanced Multiple Instance Learning for Joint Rumor and Stance Detection with Social Context Information，ACM TIST 2025 [DOI](https://doi.org/10.1145/3716856)／[原文](https://arxiv.org/html/2502.08888v1) | LLM 解释、传播结构与谣言／立场联合建模 | 普通图文融合、解释融合与弱监督 stance |
| LLM-based Rumor Detection via Influence Guided Sample Selection and Game-based Perspective Analysis，ACL 2025 主会 [链接](https://aclanthology.org/2025.acl-long.1378/) | 影响引导样本选择与多视角分析 | 筛 SFT 样本、多视角解释与贡献归因 |
| Cross-domain Rumor Detection via Test-Time Adaptation and Large Language Models，EMNLP 2025 主会 [链接](https://aclanthology.org/2025.emnlp-main.407/) | LLM 伪标签与 test-time adaptation | 普通跨域伪标注、适配加精度提升 |
| The Truth Becomes Clearer Through Debate!，SIGIR 2025 [DOI](https://doi.org/10.1145/3726302.3730092)／[原文](https://arxiv.org/html/2505.08532v1) | 双侧多智能体辩论与判决 | 辩论、解释汇总或生成辩论图；生成图不是真实传播图 |
| LLM-based Few-Shot Early Rumor Detection with Imitation Agent，KDD 2026 [DOI](https://doi.org/10.1145/3770854.3780315)／[预印本入口](https://arxiv.org/abs/2512.18352) | imitation agent 决定停止调用冻结 LLM | 普通早停代理；其 early rate 不能直接解释为真实时延 |
| Rethink Rumor Detection in the Era of LLMs: A Review，Findings of EMNLP 2025 [链接](https://aclanthology.org/2025.findings-emnlp.464/) | Cognition–Interaction–Behavior 综述框架 | 综述建议不能当成已验证空白，也不能误标主会 |

### 4.2 社会信息安全与证据安全的直接近邻

| 文献与身份 | 已核实的相关内容 | 对优先方向的限制 |
|---|---|---|
| MALCOM，ICDM 2020，必要前驱 [原文入口](https://arxiv.org/abs/2009.01048) | 固定内容、追加恶意评论攻击检测器 | 评论注入并非新问题 |
| Message Injection Attack on Rumor Detection under the Black-Box Evasion Setting Using Large Language Model，WWW 2024 Web4Good [DOI](https://doi.org/10.1145/3589334.3648139) | LLM 消息注入攻击 GNN | 不能以换成 LLM 生成评论作为贡献 |
| SINCon，ACL 2025 主会 [官方页](https://aclanthology.org/2025.acl-long.617/)／[原文](https://arxiv.org/html/2504.07135) | 节点影响遮蔽与对比学习防御 LLM 恶意消息 | 普通鲁棒损失或传播树防御已做 |
| AdComment，预印本 [原文](https://arxiv.org/html/2510.09712v3) | 多类恶意评论与 group-adaptive adversarial learning | 合理评论攻击、LLM victim、分组训练不构成单独创新 |
| LLM-TKT，SIGIR 2026 正式论文 [DOI](https://doi.org/10.1145/3805712.3809585) | LLM 叙事推理迁移、局部／全局对齐与抗攻击 | “LLM 蒸馏给 GNN 抗注入”已直接覆盖 |
| RAGuard，NeurIPS 2025 Datasets and Benchmarks [官方页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed25c00ff6900989116d3ba5d607d33d-Abstract-Datasets_and_Benchmarks_Track.html)／[原文](https://arxiv.org/html/2502.16101v4) | 误导性 Reddit 检索材料损害 LLM 事实核查 | “不可信社会材料会误导 LLM”不是新发现 |
| Whose Facts Win? LLM Source Preferences under Knowledge Conflicts，ACL 2026 主会 [官方页](https://aclanthology.org/2026.acl-long.1357/)／[原文](https://arxiv.org/html/2601.03746v3) | 来源框架、重复效应、同／不同作者控制及重复不变训练 | 去重、信誉提示和重复一致性已做；来源偏好不等于事实正确性 |
| Most LLM Conformity Needs No Speaker，2026 预印本 [原文](https://arxiv.org/html/2607.05545v1) | 区分内容重复与 speaker framing 的影响 | 简单重复机制诊断或迁移 PHEME 不足 |
| IMRRF，NAACL 2025 主会 [官方页](https://aclanthology.org/2025.naacl-long.461/) | 多源检索、知识转换与 redundancy filtering | 普通冗余过滤不是新方法 |
| GAVEL，Findings of ACL 2026 [官方页](https://aclanthology.org/2026.findings-acl.1789/) | Evidence Contract、出处绑定和机械检查 | 添加引用 ID／引用审计不足 |
| ProvenanceGuard，2026 预印本 [原文](https://arxiv.org/html/2606.18037v3) | 稳定来源 ID、主张路由、来源归属核验与修复 | 来源归属不等于来源独立性，但来源检查本身已有 |
| REAL／FAE，2026-09 预印本、已接收 CIKM 2026，会议尚未召开 [原文](https://arxiv.org/html/2609.08943v1)／[DOI](https://doi.org/10.1145/3799682.3841076) | 多轮证据消融及证据移除后的 NEI 训练 | 消融、证据依赖诊断和拒判不能单独作为创新 |

### 4.3 时间与知识更新近邻

- [DYNAMO，IJCAI 2025 Main](https://www.ijcai.org/proceedings/2025/334)：可信源知识图谱、分解检索与知识更新。其模型判定后写回机制可能产生自确认风险，是依据机制提出的假设，不是作者已经实证的失败。
- [LiveFact，ACL 2026 主会](https://aclanthology.org/2026.acl-long.546/)／[原文](https://arxiv.org/html/2604.04815v1)：时间变化、最终真假与当前证据充分性分离；不是社交回复树数据。
- [When Silence Is Golden，ICLR 2026](https://openreview.net/forum?id=PhUCxfS0yf)：时间问题与 abstention。普通时间推理加拒判已做。
- [AbstentionBench，NeurIPS 2025](https://neurips.cc/virtual/2025/poster/121675)：不可回答问题与拒判评测。不能把不可回答类别本身当新任务。
- [Conformal Arbitrage，NeurIPS 2025](https://openreview.net/forum?id=dX2BTCD02T)：风险控制仲裁。其边际期望保证不能改写为每个事件、每次停止或 accepted conditional risk 保证。

### 4.4 阅读深度与未排除风险

上述证据来自已完成的官方页面、公开原文和作者资源核查，不是全部论文逐字全文审计。NAACL 部分工作、LLM-TKT、IMRRF、GAVEL 等尚未完整取得或读完全文；部分 ACM／ScienceDirect 页面存在访问限制。

不能由摘要未提到某机制，推出论文没有研究该机制。正式立项前须补齐最直接近邻，特别是 LLM-TKT、IMRRF、GAVEL 以及两条备选对应的记忆安全、belief revision 文献。备选一的跨领域查重尚不充分，不能称其成熟度与优先方向相同。

纠错记录：KDD 2026 imitation agent 不能误标 WWW；LLM-MIL 为 TIST 正式期刊；GAVEL 不引用误命中的法律摘要论文 `2601.04424`；已接收 CIKM 2026 不写成会议已举行。

## 5. 优先方向：依赖感知社会证据安全

### 5.1 研究问题

在大量评论只是同一未经核验说法的复制、转述或附和时，能否避免 LLM 将其当成多份独立事实证据，同时保留根据少量真实、可核验反证纠正判断的能力？

研究的是信息污染下的判决依据完整性，不只是扰动前后分类精度。社交谣言必须是核心对象，而不是通用安全方法的附带示例。

### 5.2 暂定输入、输出与机制位置

- 输入：固定时间截点的源主张、社会讨论、可获得的出处及引用信息；只使用当时可用材料。
- 输出：与数据语义一致的判决／暂不判决，以及实际支撑判决的主张、材料和出处记录。
- 机制假设：建模“事实主张—支撑材料—来源依赖”，限制共同来源内容重复累计；出处不明时保留不确定性。
- 不强制将 UMER 接到最终判决前，也不让其预测作为事实 oracle。

两项性质必须同时检验：

1. 同源复述不应被反复累计为新增事实支持。
2. 独立、有效、可核验的新增反证应能促成正确更新。

这是待检验假设，不是已实现的新架构，更不是形式化事实正确性保证。

### 5.3 必须避免的假设

相似文本不一定同源，改写文本也可能同源；不同账号不等于独立来源；parent-child 不等于复制／引用关系；独立来源不等于内容真实；UMER 两视图不等于统计独立；LLM 生成解释不等于外部事实证据。

来源归属、来源依赖、内容可靠性与证据充分性是四个不同维度。不能仅用 stance 多数投票推断事实。

### 5.4 创新成立与停止条件

必须超过精确／语义去重、重复不变训练、信誉提示、来源归属检查、置信拒判及忽略所有评论，并保留有效少数纠错。

若最终只是聚类、降权、stance、prompt 和 abstention 的组合，判创新不足。若无法核验依赖与事实，或简单去重／source-only 已取得相同效果，关闭该候选，不扩模型救结果。

## 6. 备选一：事实核查知识写入安全

### 6.1 与优先方向的区别

优先方向保护当前判决；本方向保护跨事件持续存在的知识状态。研究一次误判或未经核验说法被写入记忆后，如何避免后续检索反复引用并自我证实。

### 6.2 待验证机制

区分待核验说法和已核验事实，保留原始出处及派生关系；依据被否定时撤回相关记录和派生结论。不能只通过禁止一切写入获胜，必须保留真实新知识的吸收能力。

可能评价：污染写入率、后续错误传播、撤回完整性、真实知识吸收、纠错能力和成本。具体定义和数值门槛尚未注册。

必要基线：不写入、仅允许核验来源写入、置信阈值写入、来源白名单、普通来源记录与已有动态更新方法。

### 6.3 资源与风险

可先采用冻结 reader 和小型检索库，不需大模型全参数训练。但需要连续事件、真实新知识与独立核验；不能以模型生成内容自行证明知识正确，也不能把旧 UMER 检索模块的微弱分类增益当成知识安全依据。

需补查 memory poisoning、持久记忆管理与安全知识更新，确保不是通用方法迁移。DYNAMO 只提供相关更新机制，未证明这里存在新空白。

停止条件：简单“仅核验来源写入”已解决问题；剩余效果来自检索过滤；无法建立可信连续评测；社交谣言属性只剩背景包装。

## 7. 备选二：可撤回的时序事实核查

### 7.1 研究问题

随着新证据到达，什么时候应保持、修正或撤回判断？如何区分依靠有效新证据纠错与被后续社会压力带偏？

不做普通早停代理，也不把最终真值直接套给每个早期截点。

### 7.2 待验证机制与评价

维护判决和支撑证据的对应关系；依据失效、有效反证出现或材料不足时作出有依据的修正。

联合评价无依据翻转、证据失效后的错误坚持、有效纠错、同覆盖率错误风险及延迟。只评价“翻转更少”会奖励永不更新，不能作为主指标。

必要基线：每个截点独立重判、保持旧判决、置信阈值更新、普通拒判及证据契约方法。

### 7.3 风险与停止条件

LiveFact、REAL／FAE、GAVEL、时序 abstention 与 belief revision 构成密集近邻。还需逐截点证据充分性及判决理由标注，最终 veracity 标签不足。

若普通重判／阈值更新同样有效，或收益只是延迟输出、增加拒判，则停止。不声称 fixed-time coverage 自动支持自适应停止；没有满足理论条件时不作 conformal 安全承诺。

## 8. UMER 承接与数据准入

### 8.1 两部分工作的合理联系

第一部分回答如何融合源帖与社会传播信息进行事件级谣言检测；第二部分研究 LLM 使用这些社会材料时的证据安全、知识积累或时序更新边界。

可继承事件组织、时间截点、回复树处理和事件隔离经验，不必复用全部模型。不能仅将 UMER 放在流水线开头就声称实质方法联系；图或检索若承担核心作用，需要另行证明。

### 8.2 数据状况

| 数据 | 已核实的可用价值 | 不能据此假设的内容 |
|---|---|---|
| PHEME 9-event | 6425 events；2402 rumour 中 1067 true、638 false、697 unverified；有源帖与回复树资源 | rumour/non-rumour 不是 false/true；无已核实来源依赖 gold；没有全量逐截点证据充分性标签 |
| RumourEval 2019 | 社交对话、stance 与 veracity 资源 | SDQC 不等于事实／观点类型；不能默认覆盖全 PHEME |
| Ma-Weibo | 既有 UMER 事件／传播资产 | non-rumour 不是逐条 verified true；不直接作真假核查主基准 |
| RAGuard | 2648 claims、16331 社交文档，含文档链接和事实核查 verdict | 展平文档，不是回复树；无已核实 parent／author／created_at 或依赖 gold；文档 usefulness 标签是模型生成 |
| FakeNewsNet | 新闻与 tweet ID 资源 | 完整社交数据可能需真实 API 权限及恢复；不按零成本可用数据立项 |

数据入口：

- [PHEME veracity](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)。
- [PHEME rumour/non-rumour](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619)。
- [PHEME rumour scheme](https://figshare.com/articles/dataset/PHEME_rumour_scheme_dataset_journalism_use_case/2068650)。
- [RumourEval 2019](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580)。
- [RAGuard](https://huggingface.co/datasets/UCSC-IRKM/RAGuard)。
- [Ma-Weibo](https://github.com/Jie-Ran/Ma-Weibo)。

未核实到同时具有真实嵌套回复、事实真值与来源依赖 gold 的现成公开数据。主要瓶颈是可核验材料和人工标注，不是模型规模。数据平台许可不能自动覆盖原社交内容全部权利。

最终真假、当时证据是否充分、系统是否选择拒判是不同概念。PHEME 的最终 unverified 不能直接成为所有早期截点的 unknown gold。

## 9. 优先方向的首轮筛查建议（未执行）

### 9.1 数据门槛

建议先检查约 60–100 个独立事件能否提供可核验依据和可判断依赖关系。数量是探索规模建议，不是已核实可用数量、统计充分性保证或批准的标注任务。

先制定人工核验规则，明确“可核验／不可核验／依赖未知”等状态；无法确定的情况保留，不让 LLM 自动生成依赖真值。问题发现集与最终确认集按事件隔离，发现集不参与确认性结论。

### 9.2 配对机制筛查

在固定源主张及有效事实材料下，比较同源复述增加与独立有效反证增加的响应。控制内容长度、token 数、位置、截断和身份呈现；不能通过给某一方法更少污染、更长上下文制造优势。

受控构造只能支持受控条件结论，不能冒充真实平台操纵或自然社会共识。是否需要构造、如何标注与具体预算须另行制定协议。

### 9.3 强比较与统计

- 对照 source-only、UMER-only、精确／语义去重、信誉提示、来源归属检查、重复不变方法和置信拒判；允许 oracle 诊断上界，但不能称可部署方法。
- 在同判决覆盖率下比较错误风险，同时检查有效独立反证的纠错保留。
- 报告错误方向、置信校准和拒判；LLM 自述置信度不当作校准概率。
- 完整事件作为划分和配对统计单位，多个截点／评论不作独立样本。
- 小规模结果仅作准入筛查。高置信错误的定义、模型、预算、开发集与正式门槛须在确认评测前冻结；本文不临时指定未经验证数值。

### 9.4 转向规则

- 若优先方向失败是来源依赖不可标注：评估基于可核验文档的知识写入安全，不强行代理独立性。
- 若核心机制已被前作覆盖：重新核查备选差异，不直接迁移同一机制。
- 若收益只靠减少社会信息：优先方向关闭，不能以扩 reader、扩大模型或换阈值救回。
- 若两条备选也不能满足数据和新增贡献：继续选题，不承诺必须在三条中选出已成立方法。

## 10. 执行边界与总结

本次仅新增本 Markdown 报告：不覆盖历史报告，不修改代码、协议或实验结果，不下载数据／模型，不申请账号，不运行本地或服务器推理／训练，不提交、不推送 Git。

后续若另行布置执行任务，提示词须独立加入：

> 具体执行规范和要求需要从 UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md 中检索。

实际执行还应核实最新版本，遵守 LOCAL／SERVER、正式 reader、历史结果只读与人工确认硬门。不能自动量化、改变 scoring／dtype／预算，不能因为交接首页过时而恢复 BCR。安全评测仅限获授权的自有离线基准，不操作真实平台或第三方系统。

**本轮结论：停止旧 BCR 方法线；优先筛查“依赖感知社会证据安全”，以“事实核查知识写入安全”为主要备选，以“可撤回时序事实核查”为第二备选。三者都具有论文选题潜力，但创新、数据与方法效果尚需证实，不保证论文发表。**
