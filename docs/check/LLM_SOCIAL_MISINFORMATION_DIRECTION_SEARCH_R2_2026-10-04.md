# LLM + 社交网络谣言 / 虚假新闻：第二轮深度选题检索

日期：2026-10-04。性质：文献与公开资源评审，不授权模型部署、推理、训练、付费服务或历史资产修改。

## 1. 本轮结论

不是没有其他小方向，而是必须把“值得研究的问题”与“已成立的新方法贡献”分开。本轮保留三个条件探索入口，不把任何一个称为已确认研究空白。

| 优先级 | 小方向 | 问题证据 | 数据 / 当前门槛 | 决策 |
|---|---|---|---|---|
| 1，条件推荐 | 数值与时间主张的证据适用范围约束核验 | TSVer 真实主张实验；VerifierFC 人工校正推理漂移实验 | TSVer 主张、CSV、单位元数据、时间区间和理由可直接访问；缺细粒度运算/范围 gold | 最值得进一步核查；不能只是 LLM + pandas/SQL |
| 2，社会对话相关性最强 | 局部回复立场何时不能组合为根主张立场 | LREC2024 target-dependence 实验；TruthStance 实际沿树组合方案 | TruthStance 公开全文 / parent / root / 人工局部 stance；缺人工双目标 gold | 只值得小规模可组合性试验；不做普通 stance 分类 |
| 3，额外资源成本较高 | 同图匹配但关键语境属性证据缺失时的选择性核验 | WACV2025 相似性捷径；COVE 预测语境与 gold 语境差距 | 5Pils-OOC / VERITE / RW-Post；图像可得性和属性充分性 gold 待验 | 较弱候选，强近邻密集，需要 VLM 授权 |

用户选择已更新：上一轮的任务语义错位审计暂不考虑；真实片段导致无依据组合推断及跨 agent 放大防御保留为备选。不自动执行上一轮 A0。此文替代前轮方向排序，但不覆盖或修改前轮证据记录。

严格要求：所有候选的主要贡献不得是传统 rumor/fake-news 分类精度提升。没有通过问题复现、数据准入和近邻对照，不开始方法训练。

## 2. 检索范围与证据标准

- 主窗口：2023-10-04 至 2026-10-04；分支初始按前日窗口检索，关键新近来源已在主线程核查。ProgramFC ACL2023、NewsCLIPpings、VisualNews、MOCHEG 等较早来源只作前驱 / 数据背景，不计为窗口内新增工作。
- 检索入口：ACL Anthology、CVF、ICML/PMLR 相关论文、PVLDB、ACM/DOI、IJCAI、arXiv/ar5iv、作者 GitHub、Zenodo、Hugging Face、CheckThat 官方。
- 检索不是穷尽证明。主会、Findings、industry、workshop 与 preprint 明确区分；arXiv 的论文许可不替代数据许可。
- 重点来源抓取官方页面、HTML 原文或作者资源。仅摘要、无法获取全文或下载未验证的情况单独说明。
- GitHub API 出现限流，ScienceDirect 某些全文 403，HF 部分端点不可达；没有凭据、付费或绕过访问控制。公开 raw 文本与 CSV 仅远程只读检查，未下载完整数据集。
- 访问使用 floating main / 当前网页，没有存每个响应的内容哈希；本文是检索记录，不是 commit-pinned 数据复现。正式执行须固定版本、保存哈希和许可。

## 3. 优先入口：数值与时间主张的证据适用范围约束

### 3.1 研究对象

研究“程序算对了，却用了错误证据范围”的事实核验：国家 / 人群选错、时间窗错位、百分比与百分点混淆、分母变化、币种或价格基年不一致、区间平均值被当端点值、缺失值被当零等。

例如“某国目前超过 60% 的就业人口从事农业”，不能用另一个国家、较早年度、农业增加值占 GDP 比例，或跨年平均值来直接核验。这个示例是任务说明，不是已复现实验。

拟研究的问题是：**在数值运算正确的情况下，模型能否识别证据对当前主张的适用范围，并避免输出无依据的确定性结论？** 不把“有数字所以难”当贡献。

### 3.2 直接问题证据

1. **TSVer，EMNLP2025 main**：304 个真实主张，来自 41 个核查机构，400 条时间序列。人工 verdict 一致性 κ=0.77；官方论文摘要报告 Gemini-2.5-Pro verdict accuracy 63.57、理由 Ev2R 48.63。原文明确刻意避免简单查表 / 简单算术，纳入跨国家、多序列及时间 / 数字歧义。
2. TSVer v2 的基线是元数据选择序列、时间区间和国家，再将切片作为文本表交 LLM 生成判词与理由。直接完整检索后只有约 31% 的样本少于 100 万 tokens，因此数据压缩 / 范围选择本身是障碍。这个问题不意味着必须训练时间序列大模型。
3. **Think Right, Not More，Findings EMNLP2025**：对分层抽样 100 个 QuanTemp 主张，先 LLM judge 再双人核验和纠正，约 34% 出现 reasoning drift。原始 judge 只有 69% 正确；不能把自动 judge 当新实验 gold。原文有币种不同却被当冲突等案例。这是特定样本与协议的结果，不是所有社交帖子 34%。

### 3.3 已做近邻与可能剩余空间

- **RePanda，ACL2025 main**已经 claim → 可执行 pandas、错误纠正、OOD 评估；仅加工具执行不是新方法。
- **CEDAR，PVLDB2025**已经 claim → SQL、多策略重试、成本优化；仅“低成本数值核验”也不新。
- **ProgramFC，ACL2023**已程序引导复杂核查，是窗口外必须考虑的前驱。
- **VerifierFC，Findings EMNLP2025 / CheckThat2026 Task2**已多推理路径、训练 verifier、adaptive best-of-N；不能把思维链排名当新贡献。
- **QuanTemp++，2025 preprint**已拆解、检索证据集与过滤；不是换数据即可新。
- **Verifying ambiguous claims with a reasoner-translator framework，Neurocomputing2025，DOI 10.1016/j.neucom.2025.130536**是危险近邻；[作者团队官方介绍](https://hlt.hitsz.edu.cn/info/1001/1604.htm)已明确用蕴含树分叉定位歧义、不同前提分支保留解释，在 AmbiFC 评估。因此“歧义保留 / 多解释 / 结构化解释”已经做过。全文仍未获取（403），不能确认更窄单位/分母/时间窗约束是否也覆盖。
- TSVer 原文也已讨论不同时间范围改变 verdict、指标差别与 cherry-picking；范围错配本身不是本项目首次发现。标注者可读原核查文章帮助消歧，测试模型不可读文章；必须把缺上下文与可判定的范围错误区分，不假定缺的信息可由算法恢复。

可能的狭义贡献：在固定真实证据下，将“执行正确”与“主张语义和证据范围匹配”分离，检测后者引起的错误支持，并建立可检验的约束 / 适用边界。**这只是尚待验证的差异，不是查重确认的创新。** 如果只是让模型输出单位、时间再调用 pandas，属于现有方法组合，应停止。

### 3.4 方法设想与评价

LLM 只承担语义选择：指标、国家 / 人群、时间范围、单位、分母、比较关系及聚合算子。确定性程序绑定 CSV 单元、执行白名单运算、保留数值来源、检查单位和缺失值。不要任意执行生成 Python，也不要让 LLM 同时捏造证据与证书。

有歧义时应列出可支持的解释和必要假设，不把一个方便算的解释当唯一事实；但歧义方法近邻未核清前不将其注册为贡献。

主指标：范围错配引起的 false-support、证据单元 / 时间区间对齐、可执行有效率、数值 / 单位一致率、在匹配覆盖率下的错误支持风险、预算。Verdict accuracy 只能辅助。TSVer 的 reference justification / verdict 不是完整程序 gold，Ev2R 也不是形式化保证。

### 3.5 数据与最小停止试验（建议，未执行）

优先 TSVer：作者 repo 直接提供 dev/test JSONL、CSV、units metadata，CC-BY-SA4.0。已远程读到真实 dev 记录与 doctors-per-1000-people CSV，并非只看到下载链接。**不等于全部 400 文件已下载、无误或适用每个主张。** 上游 OWID 等源许可亦须记录。

QuanTemp（SIGIR2024，CC-BY-NC4.0）和 QuanTemp++ 可作外部扩展，但 text snippet / 检索相关性 gold 不等于数值运算 gold。CheckThat2026 Task2 官方声明发布完整 corpus、claim-associated evidence 和每条 20 traces；官方入口已核，但全资产可读性 / 许可 / 标注语义尚待执行审计。

官方划分本轮逐行解析核验：dev=24、test=280，总304。不能从24条dev虚构40–60条pilot。准入先用全部24条dev，双人判断是否能从公开数据建立范围和操作 gold；κ≥0.7为项目筛选门槛，小样本仍须报告不确定性。若24条不足，不擅自把test转dev：另寻独立开发资源，或由用户批准将预先哈希选择的一部分test仅作问题发现集，永久退出最终确认评测，并冻结其余样本。任何方案均不得看完test再选样本/调机制。

试验必须区分检索不到、范围选错、执行错、真实歧义、原文上下文缺失和官方 gold 争议。先完成数据准入，再单独冻结问题复现与强近邻对照；不承诺24条能支持论文级统计。

必须比较 direct LLM、TSVer 原式两步、RePanda / CEDAR 式工具执行和受约束候选；同模型、同证据、同预算。oracle 范围仅作为诊断上界，不能当可部署方法。

停止条件：可靠范围 gold 覆盖不足；只剩简单算术；强工具基线已消除问题；剩余问题主要来自坏 gold；只出现 overall accuracy 增益；必须调用新付费检索；近邻已经覆盖全部所谓机制。小样本门槛仅筛选，不是显著性 / 论文保证。

单4090条件下适合未来串行现有文本 reader + CPU CSV 操作，但 TSVer 长表不能原样塞入 8B；预算受限时必须比较公平、披露截断和覆盖，不宣称完整复现论文长上下文设置。这个方向偏真实世界虚假信息核查，不是传播网络建模；若用户要求必须以社交图为核心，则不应作为第一候选。

## 4. 社交对话入口：局部立场的可组合边界

### 4.1 研究问题

“B 反对 A，C 反对 B，所以 C 支持根谣言”并非必然成立：C 可能反对 B 的措辞、人物评价或另一条主张，而不是原始事实命题。只有响应对象 / 命题关系保持时，简单正负立场组合才可能有效。

目标：区分 **局部 stance 预测错误** 与 **局部 stance 正确，但沿树组合错误**。贡献应是可组合条件和错误传播边界，而不是给 UMER 蒸馏 stance、增加 GNN 或提高总 F1。

### 4.2 实证与强近邻

- Li & Scarton，LREC-COLING2024：现有 target-aware 模型在必须看 target 的子集仍可不如 target-oblivious。已经做 target mask / shuffle、cross-attention、样本重权。因此普通“看目标”不是新增。
- MT-CSD，LREC-COLING2024：15,876 Reddit 人工目标立场，75.99% 深度 >3，明确指出隐式目标 / 指代困难；GLAN 已 local/global/structure 建模。五个 entity target，非谣言事实 gold。
- TruthStance，2026 preprint：已有 parent-child stance 链式传播，NEUTRAL 后的整棵子树被丢弃。约750个人工 stance 是 to-parent；约10万为 Gemini silver。**这提供一个可检查的近邻假设，不直接证明 gold 局部标签也会组合失败。** 后者必须独立验证。

### 4.3 数据门槛

TruthStance v3（750人工 stance / 750 argument）、v4（114MB，comment/parent/root 正文、时间、silver）开放下载，CC BY4.0，无需 tweet rehydration。但没有人工 parent/root 双目标 stance 和命题身份 gold，也没有 veracity gold。

MT-CSD repo 有全文 CSV / 索引，根目录未发现清楚许可证；不能把 GitHub 可见当使用许可。RumourEval / PHEME 可作最后迁移，但 root stance 也不等于 parent stance，不能擅自把旧标签重新解释。

最小建议：100条深度≥2分支，双人标 root 命题身份及 root stance，与已有可靠局部标签组合比较。项目 gold κ 门槛沿用≥0.7，不放宽为0.6。若可靠样本中少于15例局部正确而组合错，或主要是一般讽刺 / 立场分类误差，停止。数量门槛是探索建议，不是统计充分性。

通过后仍需足够人工双重 gold 和事件 / thread 分组外部验证。若用户连小规模新标注也不接受，此方向不能优先。它不是上一轮 R/V 标签定义审计，但确实仍有语义研究成分，不能伪装成完全不需要任务审计的路线。

## 5. 多模态入口：图像匹配不等于语境属性充分性

### 5.1 问题与证据

固定同一张真实图片、来源匹配与相似背景，判断当前 caption 的时间 / 地点 / 人物 / 处境是否有直接证据。

Similarity over Factuality（WACV2025）显示六个 CLIP 相似度 + MLP 在 NewsCLIPpings 可约90%，VERITE80.6%，不能据高分认定事实核验。COVE（NAACL2025）真实 5Pils-OOC 上 predicted-context 58.2% vs gold-context95.9%，表明语境获取 / grounding 瓶颈。**这些是宽问题实证，不是本候选特定“遮蔽属性仍误支持”的已完成实证。**

### 5.2 不能重复的内容

COVE 已七属性 context→veracity、同图语境复用；CMIE 已关系打分；DEFAME 已 NEI / 工具继续检索；RW-Post / AgentFact 已 evidence-bounded 和 keypoint→evidence ID；ReMMD 已原子 entity/time/place/event 与充分性；EFR 已 strong conflict vs weak mismatch、bbox/token binding；GroundMM 已 unsupported vs contradictory 定位。

因此普通“多模态证据充分性”“可解释定位”“原子核验”不新。只有同图匹配、相似性和预算受控时，遮蔽关键属性证据仍导致强近邻系统确定性误支持，且有新的有效机制，才继续。

### 5.3 数据与成本

- 5Pils-OOC：624图×正确 / OOC captions，CC-BY-SA4.0，context / 缓存证据可访问；图像由URL下载，正确 caption GPT4生成、部分属性 Llama分解，不是全人工 gold。
- VERITE：1000 pairs来自338文章，Zenodo仅110.7KB URL/caption/label，不含原图。需验证URL、图像 family 分组及 disputed gold。
- RW-Post：CVPR Workshops2026，不是CVPR主会；GitHub提供非图部分与5例demo，全图Drive / DataPort约426.88MB。DataPort有登录，Drive本轮未成功获取全包。非商业研究许可、原版权保留。claim_time 官方字段是核查网站上的发布时间，不能自动当原帖首次出现或严格oracle证据截止。
- DGM4：官方HF原图资产入口，操作位置gold，不是属性事实充分性gold；不作为本题现成gold。EFR已报告噪声。

需额外授权一个 VLM，现有三个文本 reader 不可假定能看图。未部署、未测显存，不能保证单4090完整训练。优先只用缓存证据，不调用付费反向图像搜索。

最小建议：50–80个独立图像family，双人标决定性属性 / 直接支持片段；完整、遮蔽、冲突三条件，预算相等，比较相似性、direct VLM、COVE / ReMMD / EFR式强近邻。资产有效率<90%、gold κ<0.7、强近邻已消除残余、无视觉相对文本价值或只有accuracy改善则停止。不改官方 gold 来“救”方法。

## 6. 社区注释与辟谣为什么不列为优先

Beyond the Crowd（ACL2026 main）实证健康注释延迟中位17.6小时，并指出流畅性与准确性混淆；但 CrowdNotes+ 已证据增强、自动注释、relevance/correctness/helpfulness分层与 HealthJudge。其 README 声明1,268 post-note 数据，但文中所指 HealthNotes 路径本轮返回404，未找到完整包并验证许可；不能声称可直接开跑。

Truth with a Twist（WWW2026）已大规模比较众包 / 专业辟谣说服手段，并报告整体相关性很小（ρ≈0.039）；其中“不存在普遍说服性偏差”的结果也应保留，不能只取支持预想的部分。

MURSE（EACL2026 industry）已个性化中文辟谣与模拟反馈；用户偏好 / 模拟接受不等于实际信念改变。若主贡献是实际辟谣有效性，则需要真实受试者、伦理 / 经费 / 新实验，现有静态 post-note 数据不足。当前资源偏好下不优先。

CheckThat2026 Task3 已给 claim、veracity、evidence生成完整核查文章，并评价 entailment/citation/evidence coverage；因此普通生成有引用辟谣也已是现成任务，且撞本项目生成 grounding 风险。

## 7. 仓库旧候选与新增排除表

| 路线 | 本轮决策与理由 |
|---|---|
| BCR reader source-state / familiarity | 旧交接为未注册候选，但 M1-F 强基线审计已关闭原方法路线，不能把旧积极描述当最新立项依据 |
| CR P3/P4 未运行 | 因先决结构效应门槛失败而停，不是独立未验证的新课题；不自动跳过P2运行 |
| 动态 social context | 未被所有旧实验普遍否定，但 SePro / EARD /风险受控早停/动态传播近邻密集；没有新机制不能重启 |
| TOWRV 时间截止开放核查 | 旧终止含资源 NOT_READY，不等于科学否定；TSVer离线CSV能避开网络瓶颈，但不能称修复了真实首次可得时间档案 |
| 认知 / stance 蒸馏，错误路由，生成证据卡 | 已失败；本轮stance候选必须以命题可组合边界为对象，不能改名回到这些路线 |
| claim correction / 跨事件继承 | 旧人工一致性失败；Agent-based Claim Matching、COLING2025、Claim2Vec已有近邻。匹配gold不等于安全verdict移植gold |
| 跨语言 / 新topic adaptation | T2ARD、Claim2Vec、EACL2026语言/检索偏差已有研究；只改泛化accuracy不符合用户要求 |
| 人类 vs LLM 虚假新闻传播 | Pandora / MegaFake 有文本资源，但未找到可直接下载且同时有作者来源、独立真假、自然cascade三重gold的匹配资源；不是断言不存在 |
| 冲突证据加权 | CONFACT已做；原文明确排除social-media claims，不能拿作社会原生证据。没有细粒度冲突类型gold |
| 来源重复 / 证据污染 | SINCon、On the Risk of Evidence Pollution、GraphEcho等直接近邻，不单列普通去重 /可信度方法 |

## 8. 公开资源可用性汇总

| 资源 | 实际可用状态 | gold / 许可边界 |
|---|---|---|
| TSVer | repo直接JSONL、CSV、metadata；已读代表性文件 | CC-BY-SA4；304真实claims，小样本、理由含LLM辅助，需复核，不是完整程序gold |
| QuanTemp | 官方repo发布claims / evidence snippets | CC-BY-NC4；text evidence不是structured arithmetic gold |
| QuanTemp++ | README提供Drive包、corpus/qrels说明 | 本轮未实下全包，独立数据许可待核；不是原生社会对话 |
| CheckThat2026 Task2 | 官方声称corpus + traces完整发布 | asset/license待核；trace utility是否只按最终label定义须读runner |
| TruthStance | Zenodo v3/v4开放，全文无需rehydration | CC-BY4；750局部stance，其余silver；无人工root或真假gold |
| MT-CSD | CSV / JSON直接repo | 许可证不清；实体立场非事实真假 |
| 5Pils-OOC / VERITE | captions/metadata/context缓存，图像需URL获取 | synthetic captions/URL失效/gold争议；不能假定全真人 |
| RW-Post | repo非图资产、demo；全图外部下载入口 | 原图下载本轮未完成，学术非商用，非细粒度充分性gold |
| NewsCLIPpings / VisualNews | IDs/embedding和大型原图包 | 91GB原图，合成错配、许可待核，不优先 |
| DGM4 | 官方HF原图入口 | 本轮HF不可达，许可/完整下载待核；操作gold≠事实gold |
| MOCHEG | Google Form申请 | CC-BY4声明，未申请获取；过时gold风险 |
| MultiClaim / MultiClaimNet / Claim2Vec | Restricted Zenodo申请 | 机构邮箱、研究用途、不转发；不能称免门公开 |
| MegaFake | 表单和邮件request | CC-BY4声明，策略预设标签不等于人工真假 |
| HealthNotes | README宣称发布 | 指向路径404；下载与许可未实核，不优先 |
| CommunityFact | arXiv论文及HF入口发现 | 本轮HF页面失败；claims/labels源自community pipeline，不宣称独立人工全部gold |
| LiveFact | ACL2026 main / 原文已核 | 动态benchmark已经提出，完整可用资产未实核；不重提普通时间感知 |

## 9. 下一步决策（等待用户选择）

建议先讨论第1候选的研究对象是否符合用户“大方向”定位：允许以真实虚假信息核查而非传播图为核心，则第1最适合现有文本reader与离线资产；必须以社会互动为核心，则第2更贴题，但须接受少量新增人工gold。第3成本与撞题风险更高，不优先。

**没有任何候选目前达到“创新已成立 / 可以开始训练”的标准。** 不因旧方案失败而把相邻的新名字当成功机会。任何试验必须先登记核心假设、强近邻对照、数据gold及停止条件。

将来给执行智能体的独立规范句必须保留：具体执行规范和要求须先从 [UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md](../../UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md) 检索；远端最新阶段判决优先于旧交接阶段状态。本轮仅文献研究，不构成执行授权。

## 10. 关键来源索引

### 数值 / 时间

- TSVer官网：https://aclanthology.org/2025.emnlp-main.1519/ ；v2正文：https://arxiv.org/html/2511.01101v2
- TSVer数据/基线：https://github.com/marekstrong/TSVer ；官方说明：https://raw.githubusercontent.com/marekstrong/TSVer/main/README.md
- VerifierFC，Findings2025：https://aclanthology.org/2025.findings-emnlp.1322/ ；正文：https://arxiv.org/html/2509.22101v1
- VerifierFC代码：https://github.com/VenkteshV/VerifierFC
- RePanda ACL2025：https://aclanthology.org/2025.acl-long.1549/ ；正文：https://arxiv.org/html/2503.11921
- CEDAR PVLDB2025：https://mail.vldb.org/pvldb/volumes/18/paper/CEDAR%3A%20A%20System%20for%20Cost-Efficient%20Data-Driven%20Claim%20Verification （官方摘要已核；PDF工具不支持，未冒称全文核验）
- ProgramFC ACL2023，窗口外：https://aclanthology.org/2023.acl-long.386/
- QuanTemp SIGIR2024：https://github.com/factiverse/QuanTemp ；DOI https://doi.org/10.1145/3626772.3657874
- QuanTemp++ preprint：https://arxiv.org/html/2510.22055 ；数据：https://github.com/VenkteshV/QuanTemp_Plus
- CheckThat2026 Task2：https://checkthat.gitlab.io/clef2026/task2/ ；参赛方法：https://arxiv.org/html/2607.25069
- Ambiguous claims Neurocomputing2025：https://doi.org/10.1016/j.neucom.2025.130536 ；作者官方方法介绍：https://hlt.hitsz.edu.cn/info/1001/1604.htm （介绍已核，全文访问受限）

### 社会对话 / 主张匹配

- Target dependence LREC-COLING2024：https://aclanthology.org/2024.lrec-main.253/ ；正文：https://ar5iv.labs.arxiv.org/html/2303.12665
- MT-CSD LREC-COLING2024：https://arxiv.org/html/2403.11145v2 ；数据：https://github.com/nfq729/MT-CSD/tree/main/data
- TruthStance preprint：https://arxiv.org/html/2602.14406v1 ；人工gold：https://zenodo.org/records/18363711 ；全文v4：https://zenodo.org/records/22664795
- COLING2025 Claim Matching：https://aclanthology.org/2025.coling-main.650/
- Agent-based Claim Matching preprint：https://arxiv.org/html/2510.23924v1
- MultiClaimNet Findings2025：https://aclanthology.org/2025.findings-emnlp.599/ ；正文：https://arxiv.org/html/2503.22280v1
- Claim2Vec preprint：https://arxiv.org/html/2604.09812v1
- EACL2026语言/检索偏差：https://aclanthology.org/2026.eacl-long.240/
- SemEval2025 Task7（workshop）：https://aclanthology.org/2025.semeval-1.323/
- T2ARD EMNLP2025：https://aclanthology.org/2025.emnlp-main.407/
- CONFACT IJCAI2025 AI & Social Good：https://www.ijcai.org/proceedings/2025/1073 ；排除social claims正文：https://arxiv.org/html/2505.17762v1

### 多模态

- Similarity over Factuality WACV2025：https://ar5iv.labs.arxiv.org/html/2407.13488
- COVE NAACL2025：https://aclanthology.org/2025.naacl-long.102/ ；正文：https://arxiv.org/html/2502.01194v1 ；数据：https://github.com/UKPLab/naacl2025-cove
- CMIE ACL2025：https://ar5iv.labs.arxiv.org/html/2505.23449
- DEFAME ICML2025：https://arxiv.org/html/2412.10510v2
- ASAP CVPR2025：https://arxiv.org/html/2412.12718v1
- GroundMM MM2025：https://ar5iv.labs.arxiv.org/html/2509.08008
- RW-Post CVPRW2026：https://openaccess.thecvf.com/content/CVPR2026W/PPMisDet/html/Xu_RW-Post_Auditable_Evidence-Grounded_Multimodal_Fact-Checking_in_the_Wild_CVPRW_2026_paper.html ；作者repo：https://github.com/xudanni0927/AgentFact
- ReMMD preprint：https://arxiv.org/html/2606.24112v2
- EFR preprint：https://arxiv.org/html/2608.08009v1
- VERITE数据：https://zenodo.org/records/10474450 ；5Pils：https://github.com/UKPLab/5pils
- DGM4官方：https://github.com/rshaojimmy/MultiModal-DeepFake ；MOCHEG：https://github.com/PLUM-Lab/Mocheg

### 社区治理 / 其他排除

- CrowdNotes+ ACL2026：https://aclanthology.org/2026.acl-long.233/ ；正文：https://arxiv.org/html/2510.11423 ；README：https://github.com/jiayingwu19/CrowdNotesPlus
- Truth with a Twist WWW2026：https://arxiv.org/html/2601.14105v3 ；DOI https://doi.org/10.1145/3774904.3792938
- MURSE EACL2026 industry：https://aclanthology.org/2026.eacl-industry.45/
- CheckThat2026文章生成：https://checkthat.gitlab.io/clef2026/task3/
- CommunityFact preprint：https://arxiv.org/html/2605.30241v1
- LiveFact ACL2026：https://aclanthology.org/2026.acl-long.546/ ；正文：https://arxiv.org/html/2604.04815v1
- Evidence Pollution ACL2025：https://arxiv.org/html/2410.12600v2
- Evidence-Ledger preprint：https://arxiv.org/html/2607.26512v1
- Trust but Don't Verify preprint：https://arxiv.org/html/2606.05403v1 （合成金融评估，不当社会谣言直接证据）
- Unraveling Misinformation Propagation preprint：https://arxiv.org/html/2505.18555v1 （数学题与合成错误输入，不当自然社会cascade）
- Pandora NAACL2025：https://aclanthology.org/2025.naacl-long.142/
- MegaFake作者资源：https://github.com/zhe-wang0018/MegaFake

### 项目约束来源

- 已尝试路线：https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/docs/LLM_INTEGRATION_ATTEMPTS.md
- 旧动态候选：https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/docs/动态社会上下文与LLM谣言检测_方法级查重调研.md
- 最新BCR强基线判决：https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/docs/BCR_UTILITY_M1F_STRONG_BASELINE_VALIDITY_AUDIT.md

本轮无模型实验，无完整数据集下载，无许可申请，无API账号授权，无历史文件修改。
