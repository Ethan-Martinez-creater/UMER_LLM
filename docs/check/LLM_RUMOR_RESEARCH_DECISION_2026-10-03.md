# LLM × 谣言检测：查重、可行性与研究方向决策

日期：2026-10-03。性质：研究评审，不授权新 GPU 实验、M2、模型部署或付费 API。

## 1. 直接回答两个问题

### 1.1 当前 BCR 方向还要不要继续？

**建议关闭当前成功方法主线，不执行原 M2，不把 M1-E 的 reader-state / familiarity 候选直接扶正。** 理由独立且相互加强：

1. **宽泛创新已被覆盖。** reader-specific utility、多 reader 检索监督、未知 reader 泛化、正确答案概率差、删除扰动归因和 behavioral fingerprint 都有明确论文。尤其 uRAG（SIGIR 2024）、SePer（ICLR 2025）、SCARLet（EMNLP 2025）及 LLM-Specific Utility / SpecUBench 构成直接近邻。
2. **狭义组合尚未查到完全相同论文，但未被验证有效。** 可能剩余的差别是“低成本行为画像/状态 → 未见 reader 在同一社交快照中的 signed full-context leave-one-evidence-out utility”。这不是检索确认的独占空白，也不能仅靠换数据集和 LORO 划分成为方法贡献。
3. **现有最严格实验失败。** M1-F 中 D5 比 dev 选出的 S2 source-state only 基线平均 Macro-F1 低 0.0075，95% CI 跨零；active HELPFUL/HARMFUL Macro-F1 低 0.1893，同快照证据排序接近随机。继续调模型或扩大 reader 面板，没有新的机制依据。

**不能宣称“其他人已经完整做出了同一个 BCR”。** 合理结论是：大部分宽泛主张已有成果，剩余狭义新增贡献尚未成立且当前实现的验证失败，因此不值得继续按原方案投入。

失败结果可保留为研究边界资产，但当前三个 reader、有限事件及特定协议不足以支持“普遍不可预测”论文。若要写负结果论文，需要提出比现有 SpecUBench/uRAG 更强的新规律，并独立复现；不能把失败日志拼接当作论文级发现。

### 1.2 大方向内还有什么值得做？

**目前没有足够证据推荐立刻投入另一轮完整方法开发。** 推荐顺序：

- **优先做小规模准入审计：任务语义与证据真值混淆。** 研究 LLM 谣言检测是否把“未经核实的传闻状态”误当“事实为假”，以及这种混淆怎样改变模型排名、解释与证据效用。它最符合当前资产与低成本约束，但仅发现定义错误或改提示词不够发高水平论文。
- **条件性方法候选：阻断真实片段被拼接成无依据结论及其跨 agent 放大。** 已有 Generative Montage 攻击，仅重复其问题与机制不足以形成新增贡献；可探索支持链约束下的防御，且必须保持合法多跳推断能力。现有 CoPHEME 并不能直接提供完备真值/支持链 gold，先过数据门槛才值得试验。
- **暂不推荐：重启动态 context、换域风险受控早停、普通冲突加权、去重/provenance、通用 belief revision、生成回复/蒸馏/精度融合。** 前两类并非理论上不可能，而是当前没有清晰新增机制与可行性证据；其余已有直接近邻或已触发本项目失败条件。

不能保证任何候选能发表。当前最优行动是花小成本判定“是否真的存在一个可复现的新失效模式”，而不是先写一个新模型名字。

## 2. 检索范围与可信度边界

- 当前运行日期 2026-10-03；滚动三年为 2023-10-03—2026-10-03，主要覆盖 2024—2026，纳入 2023 年 12 月直接前驱。
- 入口：ACL Anthology、PMLR、ICLR proceedings、ACM/DOI/Crossref、IJCAI、arXiv/ar5iv、大学机构库、Semantic Scholar、NSTL，以及相关作者公开代码。
- 方法：宽检索 → 方法级关键词拆分 → 关键近邻原文或官方出版页面 → 反向排除候选。不是拥有完整数据库访问权限的 PRISMA 系统综述；未声称穷尽所有论文。
- 检索主题：reader-conditioned utility / model-aware retrieval / behavioral fingerprint / harmful context；LLM rumor detection / social context / early detection；risk-controlled stopping / selective prediction；evidence conflict / provenance / repetition；belief revision；rumour-vs-veracity 标签语义与数据污染。
- 关键 RAG 近邻查阅 HTML 正文；一些谣言论文只能核查官方摘要/出版页。正文、摘要、元数据三种证据在第 8 节分开标注。无法读取的全文不据以作细粒度方法结论。
- ScienceDirect、部分 ACM、Figshare 返回 403；通过官方元数据或公开原文补充，不绕过权限、不付费、不使用凭据。
- CIKM 2026 的 LLM-Specific Utility 稿面含 DOI，但会议在 11 月；保守标“公开预印本，稿面标 CIKM 待发表”，不写已召开发表。SePer 为 ICLR 2025，不是 2026。workshop 不冒充主会，Findings 不冒充主会。
- 本轮没有本地数据、模型与服务器验证；除交接文档及本轮新建材料外，没有读取当前目录其他文件。项目事实以远端正式报告为主。

## 3. 与现有研究结果对齐

### 3.1 已成立的基础成果

UMER 是图+多视图文本+训练分区检索的单检查点事件分类器。PHEME 五折 Accuracy/Macro-F1 为 0.8988/0.8925；Ma-Weibo 为 0.9689/0.9689。其已有成果不受 BCR 失败影响。文本贡献最大，检索整体增益很小，不夸大。

依据：[UMER 正式实验](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/docs/UMER_EXPERIMENTS.md)。

### 3.2 历史失败对新方向的硬约束

| 历史路线 | 已知结论 | 新候选须规避什么 |
|---|---|---|
| 认知/stance/role 蒸馏、解释卡、选择性路由 | 下游或保真门槛失败 | 不能用新提示词包装同一监督信号；须有新任务/机制证据 |
| 改写、生成虚拟回复、合成干预 | 语义保持/grounding/真实校准不足 | 不把合成用户行为当真实传播；不让 LLM 同时构造与认证 gold |
| 跨事件纠正关系 | 人工一致性复现不足 | 新支持链 gold 也先做一致性 pilot，不能假设人一定能标 |
| TOWRV 时间核查 | 可靠外部档案不可得 | 不以外部资源“之后再解决”为前提立项 |
| CR-TSER | reader 差异成立；结构交互门槛未过 | 不以 propagation interaction 为已成立增益机制 |
| BCR M1/M1-E/M1-F | 当前 fingerprint 实现/协议未获支持，D5 不胜强基线 | 不再扩面板/调阈值事后挽救；事件级状态收益≠证据内排序 |

依据：[历史 LLM 尝试](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/docs/LLM_INTEGRATION_ATTEMPTS.md)、[交接文档](../../UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md)、[M1-F 报告](https://github.com/Ethan-Martinez-creater/UMER_LLM/blob/main/results/bcr_utility_v1/m1f/M1F_REPORT.md)。

**纠正一个过强表述：** CR-TSER 的 P2 失败，不等于所有动态 social context 选择都无效。其 Delta_edge=0.012150，要求 0.02，CI [-0.002342,0.027196]；只否定这套预注册强结构交互假设。动态方法若有全新的、独立于该交互机制的假设，仍可研究；本轮未找到足够强的理由授权重启。

### 3.3 BCR 最新实证判决

远端 HEAD：d71991dd9745a910e94ca91398d0b871a7ffda7f。交接文档停在 b62010b / M1-E，现应以 M1-F 为准。

| Ma-Weibo 主比较 | 数值 |
|---|---:|
| D5/S4 平均 Macro-F1 | 0.3072 |
| dev 选中 S2 平均 Macro-F1 | 0.3148 |
| 平均差 | -0.007549 |
| 95% event-bootstrap CI | [-0.095612,+0.079948] |
| positive reader | 2/3 |
| 最差 reader 差 | -0.144261 |
| active Macro-F1 差 | -0.189262 |
| HARMFUL AUPRC 差 | -0.041907 |
| 快照内 centered Spearman 差 | +0.002364，CI 跨零 |
| 正式判决 | M1F_CLOSE_BCR_UTILITY_METHOD |

PHEME 仅诊断，不可拯救主 gate。M2 未进入、未授权。M1 的历史 CONDITIONAL_GO 不被改写，但不能替代最新强基线判决。

## 4. 当前方向的方法级查重

| BCR 欲主张内容 | 已有强近邻 | 剩余区别与判定 |
|---|---|---|
| 效用依赖具体 reader，跨模型不能直接共享 | LLM-Specific Utility / SpecUBench；SePer | 大动机已覆盖，不能称首次发现 |
| 多 reader 监督，泛化到未知 reader | uRAG SIGIR 2024；IUM ICTIR 2025 | uRAG 已测未知模型；IUM需目标模型反馈。不是第一次跨 reader |
| 概率差定义 helpful/harmful utility | SePer ICLR 2025 | BCR 的 full-set LOO 和三类阈值不同，但“signed 概率差”不新 |
| context 删除扰动估计证据效用 | SCARLet EMNLP 2025 | 随机 mask+LIME 与固定 full-context LOO 不同；核心归因已有 |
| relevance≠utility，有害证据可以降低表现 | SIGIR 2024 Utility Judgments；Detrimental Contexts 2023；FER 2023 | 不能用这类现象作主要贡献 |
| behavioral fingerprint 作为 reader 表征 | Behavioral Fingerprinting 2025 arXiv | 指纹用于 utility transfer 的组合可能不同，但当前指纹方案在冻结协议下未获支持 |
| source-state/输出分布帮助跨模型行为分析 | LOS 2025 workshop；SePer | 目标不同；D5 的证据级能力未过强比较 |

**创新是否已被完全覆盖和当前是否值得做是两个不同问题。** 未发现完全同设定论文时，仍须通过强基线证明狭义新增机制有效；当前恰恰没有通过。

## 5. 候选一：LLM 谣言检测的任务语义混淆与可迁移失效

### 5.1 具体研究问题

在原生标签可核验的 PHEME 上，区分：

- R：历史传播中是否为未经核实的传闻（rumour/non-rumour）；
- V：传闻后来被核实为 true/false/unverified。

问：LLM 系统面对“rumor/谣言”提示，是否实际用真假代理 R？这种混淆是否在 true-rumour 子集、不同语言提示、不同 reader 和 source-only/context 输入下系统性改变排名或证据归因？能否构造不依赖昂贵新 gold 的任务对齐评估协议？

**不是**再提高 UMER 分类精度、翻译提示词刷分、宣布 PHEME 标签有错、或将 V=unverified 直接当“应该拒答”的 gold。

### 5.2 为什么值得先排查

- PHEME 原始论文明确 rumour 是 unverified circulation，并非 false；概念区分本身很早已存在，不是我们的创新。
- 本项目 M1-F 已承认 utility 只针对 rumour 标签；这一修正还没有回答跨方法评测是否被隐含真值任务污染。
- 本轮公开 CoPHEME 审计提供一个具体可核查对象：库内目标池有 false 和 unverified，证据池有 true 和 non-rumour；其 loader 默认路径把 false/unverified 都纳入目标。这要求进一步核查下游任务与标签映射：若目标是rumour状态，两类同时入选合理；若被当成全部false结论则存在错位。当前不证明作者正式结果错误。
- 不需要先增加模型、构建外部档案或生成新传播。

### 5.3 与最强近邻的区别

TripleFact 已研究训练污染、实时采集和实体偏差；本候选研究的是**任务目标错位**，不是知道某事件答案的 memorization。Generative Montage 已研究真片段诱发错误结论；本候选首先核验“真片段/假结论”是否有对应标注，不重复其攻击创新。M1-F 已做项目内部 task-validity，单纯复制这个检查不新。

尚未检索到完全相同的多系统双标签审计，但这只是候选，不是确认空白。若效果仅在一个含糊提示存在、定义明确后完全消失，或已有论文已系统解决，应停止作为独立论文路线。

### 5.4 论文级贡献需要什么

1. 可复现的输入、prompt、标签、任务目标映射；至少一个项目外系统的公开实现证据。
2. 原生双标签/已核实子集，不能靠 LLM 生成真假 gold。
3. 在未用于探索的话题/模型上复现，说明此失效会影响方法选择或证据解释，不仅整体 F1。
4. 与污染控制、提示歧义、事件重复、文本长度和语言翻译因素分离。
5. 有可使用/可发布的审计清单或派生 manifest，不擅自再分发原始用户数据。

发表价值：若发现稳定、跨系统的任务错位并导致结论逆转，可定位实证/评测论文；若只有定义澄清，科研价值有限，宜作为项目修正而非主线论文。**这是相对最可行的准入审计，不是已经具备高水平论文保障。**

## 6. 候选二：真实片段的无依据组合推断及跨 agent 放大防御

### 6.1 可考虑的窄问题

在同一证据集/相同推理预算下，阻止 LLM 把相关片段拼成缺少支持链的全局结论，同时保留合法多跳推断和正常回答覆盖率。进一步问：下游 agent 能否区分上游结论的原始证据与上游自行补出的桥接关系，避免把模型断言当新增事实？

潜在方法不是强制“只许单段直接蕴含”：可由 LLM提出有限的桥接前提，程序绑定原文/ID/出处，独立 verifier 检查支持路径；失败时输出缺口而非强行判假。是否有实质新增机制，必须与现有 provenance/NLI/GRIL比较后再确认。

### 6.2 已有覆盖与风险

- Generative Montage（ACL 2026 Outstanding Paper）已做此攻击，正文第 6 节还提出 belief tracking、entropy、cross-model divergence、provenance auditing、训练和符号检查等防御方向。**仅实现其建议，不自动成为新贡献。**
- GRIL（Findings ACL 2026）已做缺前提识别、暂停与澄清；普通“检查推理前提”不新。
- GraphEcho（2026 arXiv）已做路径重复、证据来源与预算策略；普通去重/来源标记不新。
- CONFACT / CAG 已做冲突与可信度；普通可信度加权不新。
- 本项目纠正关系 gold 一致性失败；开放域桥接推断的人工一致性可能同样不足。

### 6.3 数据门槛：当前还没通过

公开 CoPHEME 可访问且提供预生成攻击计划，减少重新调用商业攻击模型的必要，但：

- false 源帖的提取 conclusion 可能不是同一句命题，不能自动继承 false 标签；
- non-rumour 不是逐片段真值证明；
- unverified 不是 false；
- 公开池统计不等于论文正式实验选用集；
- 尚未核实使用/再分发许可，真假及推理支持链证书也未核实。

所以当前评级为**条件性备选**：资源与支持链 gold 能过关时，可在现有 7B/8B reader 上做有限防御 pilot；否则 NOT_READY。没有合法多跳正例对照的“降低攻击成功率”，可能只是模型拒答一切，不算有效防御。

## 7. 不建议立即立项的方向与理由

| 方向 | 近邻/本项目障碍 | 判定 |
|---|---|---|
| GNN/top-k评论 → LLM 的静态 context refinement | SePro NAACL 2025 | 已有直接方法，不能换 UMER 即新 |
| 动态树 context / reader utility选择 | 旧方向查重已识别，CR-TSER强交互失败，BCR证据排序失败 | 不靠旧效用机制重启；需另一个可证伪假设 |
| LLM stop/continue 早检测 | KDD 2026 Imitation Agent | 仅复用该stop/continue机制并换轻量停止器，不足以形成新增贡献 |
| 序列风险受控 stop/abstain | ICML 2024 accumulated gap control，Safe to Stop 2026 arXiv | 连反复查看/条件早停/风险覆盖控制也有近邻，换域不够 |
| 恶意回复注入攻击与普通防御 | SINCon ACL 2025 | 已有，不能再做同类攻击+鲁棒准确率 |
| 重复证据/来源依赖 | GraphEcho | 已有很近预印本，单独方向优先淘汰 |
| 错误证据可信度、冲突解决 | CAG EMNLP 2024、CONFACT IJCAI 2025 | 普通冲突权重已做；需新失效而不是堆模块 |
| 证据撤回/连续 belief revision | Belief-R EMNLP 2024；ReviseQA/AGM workshop近邻 | 公共评论否认≠真值撤回；原生逐步gold不足 |
| 反应生成/stance辅助/多视角SFT | DELL、CICAN、TIST MIL、ISGA；项目已有失败 | 拥挤且不满足用户主贡献要求 |
| 时间/污染评测本身 | DAWN、TripleFact、既有chronological研究 | 应作为协议要求，不把“改时间划分”当独占创新 |
| 外部核查agent、多模态路线 | 档案/额外数据与依赖未保证 | 暂不以缺失资产下注 |

## 8. 核心文献证据表

标记：**正文**=核查到可读 HTML 方法内容；**官方摘要**=核对出版页与摘要，不声称全文复现；**元数据**=核验题名/venue/date，只作线索。

### A. 当前 BCR 的直接近邻

1. **LLM-Specific Utility for Retrieval-Augmented Generation**，2025 初稿 / 2026 v3，arXiv，稿面标 CIKM 2026 待发表。**正文**：[v3](https://arxiv.org/html/2510.11358v3)。四 reader、三 QA 数据集；单 passage 相对空 context 的 accuracy效用及 SpecUBench。核心 reader-specific 现象已覆盖；非 fingerprint-LORO signed LOO。
2. **Towards a Search Engine for Machines: Unified Ranking for Multiple Retrieval-Augmented Large Language Models**，SIGIR 2024。**正文**：[uRAG](https://ar5iv.labs.arxiv.org/html/2405.00175)，[DOI](https://doi.org/10.1145/3626772.3657733)。18 训练+18未知系统，模型/task ID 个性化多数收益不显著；未知模型泛化已测。
3. **Learning to Rank for Multiple Retrieval-Augmented Models through Iterative Utility Maximization**，ICTIR 2025，非 SIGIR 主会。**正文，由专题检索核查**：[IUM](https://arxiv.org/html/2410.09942)，[DOI](https://doi.org/10.1145/3731120.3744584)。EM式多 agent反馈，在线还需目标 agent反馈，不等于 zero-touch。
4. **SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction**，ICLR 2025。**正文+官方venue**：[原文](https://arxiv.org/html/2503.01478)，[ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c44c4afd77d5ee760e7f4bed0c50f878-Abstract-Conference.html)。定义 P(correct|d)-P(correct)，语义采样/聚类，已有知识/冗余/负效用均讨论。
5. **Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models**，EMNLP 2025 主会。**正文+官方摘要**：[出版页](https://aclanthology.org/2025.emnlp-main.33/)，[原文](https://arxiv.org/html/2504.00573v2)。SCARLet random deletion+ridge归因，multi-task和inter-passage interaction，含fact checking。
6. **Bridging the Preference Gap between Retrievers and LLMs**，ACL 2024 主会。**正文，由专题检索核查**：[出版页](https://aclanthology.org/2024.acl-long.562/)，[原文](https://arxiv.org/html/2401.06954)。选择/排序/重复文档以匹配下游；personalized generation不是reader画像迁移。
7. **Are Large Language Models Good at Utility Judgments?**，SIGIR 2024。**正文，由专题检索核查**：[原文](https://arxiv.org/html/2403.19216v2)，[DOI](https://doi.org/10.1145/3626772.3657784)。utility判断与噪音/反事实；非reader-specific边际gold。
8. **Detrimental Contexts in Open-Domain Question Answering**，Findings EMNLP 2023（12月，在窗口内）。**官方摘要**：[出版页](https://aclanthology.org/2023.findings-emnlp.776/)。完整context可比子集差，正确context也可能带来错误。
9. **From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification**，Findings EMNLP 2023（12月）。**官方摘要**：[出版页](https://aclanthology.org/2023.findings-emnlp.422/)。verifier反馈训练证据检索，任务直接邻近。
10. **Behavioral Fingerprinting of Large Language Models**，2025 arXiv，未核实正式venue。**正文，由专题检索核查**：[原文](https://ar5iv.labs.arxiv.org/html/2509.04504)。诊断prompt聚合行为画像，不预测signed证据效用。
11. **Learning on LLM Output Signatures for Gray-Box Behavior Analysis**，2025 arXiv / ICML 2025 Reliable and Responsible Foundation Models **workshop** poster。**正文+官方workshop页**：[原文](https://ar5iv.labs.arxiv.org/html/2503.14043)，[官网](https://icml.cc/virtual/2025/50894)。输出概率签名用于跨模型幻觉/污染行为分析，不是ICML主会。

### B. LLM × 谣言/错误信息领域与候选排除

12. **Rumor Detection on Social Media with Crowd Intelligence and ChatGPT-Assisted Networks**，EMNLP 2023 主会（12月）。**官方摘要**：[CICAN](https://aclanthology.org/2023.emnlp-main.347/)。ChatGPT知识增强+实体句子异构图，传统精度融合已存在。
13. **DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection**，Findings ACL 2024。**官方摘要**：[出版页](https://aclanthology.org/2024.findings-acl.155/)。反应生成、代理任务解释、多专家融合，覆盖项目早期许多宽泛想法。
14. **Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation**，EMNLP 2024 主会。**官方摘要**：[CAG](https://aclanthology.org/2024.emnlp-main.1109/)。训练模型处理低可信context；普通可信度融合不新。
15. **Exploring Large Language Models for Effective Rumor Detection on Social Media**，NAACL 2025 主会。**官方摘要+项目已有方法级调研**：[SePro](https://aclanthology.org/2025.naacl-long.128/)。语义与传播GAT协作、context refinement，不能只换UMER做top-k。
16. **LLM-based Rumor Detection via Influence Guided Sample Selection and Game-based Perspective Analysis**，ACL 2025 主会。**官方摘要**：[ISGA](https://aclanthology.org/2025.acl-long.1378/)。influence挑选SFT样本，Shapley多视角分析，最终传统分类SOTA不是用户想要的主贡献。
17. **SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection**，ACL 2025 主会。**官方摘要**：[出版页](https://aclanthology.org/2025.acl-long.617/)。均匀节点预测影响的对比防御，普通LLM回复注入路线已有。
18. **TripleFact: Defending Data Contamination in the Evaluation of LLM-driven Fake News Detection**，ACL 2025 主会。**官方摘要**：[出版页](https://aclanthology.org/2025.acl-long.431/)。人类对抗、实时web、entity-controlled环境，污染评测已有。
19. **Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs**，IJCAI 2025 **AI and Social Good track**。**官方摘要**：[CONFACT](https://www.ijcai.org/proceedings/2025/1073)。冲突+媒体来源可信度进入retrieval/generation，不夸称主会普通track。
20. **LLM-enhanced Multiple Instance Learning for Joint Rumor and Stance Detection with Social Context Information**，ACM TIST 16(3),58，2025。**机构库摘要/出版信息**：[HKBU](https://scholars.hkbu.edu.hk/en/publications/llm-enhanced-multiple-instance-learning-for-joint-rumor-and-stanc/)，[DOI](https://doi.org/10.1145/3716856)。仅claim级监督的joint stance/rumor MIL。
21. **LLM-based Few-Shot Early Rumor Detection with Imitation Agent**，KDD 2026。**正文+Crossref正式信息**：[原文](https://arxiv.org/html/2512.18352v2)，[DOI](https://doi.org/10.1145/3770854.3780315)，[元数据](https://api.crossref.org/works/10.1145/3770854.3780315)。stop/continue模仿agent，冻结LLM；正式online日期2026-04-20，print日期2026-08-09，避免模板旧日期误读。
22. **K-FLARE: A knowledge-fused language and relational-learning framework for rumor detection**，Knowledge-Based Systems 334:115089，2026-02（DOI登记2025）。**元数据，全文403**：[Crossref](https://api.crossref.org/works/10.1016/j.knosys.2025.115089)，[出版入口](https://doi.org/10.1016/j.knosys.2025.115089)。本轮不把无法读到的详细方法当决定性证据；项目旧调研记录其稀疏回复双流设计。
23. **A New DAWN for Fake News Detection: Exploiting Engagement Earliness for Temporality-aware Evaluation**，ACM TIST，2026-05-07 online。**官方元数据/摘要+公开前稿**：[Crossref](https://api.crossref.org/works/10.1145/3815193)，[前稿](https://arxiv.org/html/2411.12775v1)。未来context泄漏与temporality-aware评测；不把该期刊题名误标WSDM 2025版本。
24. **Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage**，ACL 2026 主会 **Outstanding Paper**。**正文+出版页+公开代码**：[出版页](https://aclanthology.org/2026.acl-long.270/)，[原文](https://arxiv.org/html/2601.01685v2)，[仓库](https://github.com/CharlesJW222/Lying_with_Truth)。真片段组合误导、CoPHEME与下游放大；第6节已提出多种潜在防御。
25. **GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents**，2026-09 arXiv，未核实正式venue。**正文**：[原文](https://arxiv.org/html/2609.17695)。path重复与evidential origins分离，冲突、SciFact、预算策略；已有紧邻，不推荐普通echo去重。
26. **Belief Revision: The Adaptability of Large Language Models Reasoning**，EMNLP 2024 主会。**官方摘要**：[Belief-R](https://aclanthology.org/2024.emnlp-main.586/)。更新必要/不必要时的adaptivity trade-off；通用belief revision已做。
27. **Pause or Fabricate? Training Language Models for Grounded Reasoning**，Findings ACL 2026。**官方摘要**：[GRIL](https://aclanthology.org/2026.findings-acl.1155/)。识别缺前提、暂停/澄清、再推理，候选防御的重要强比较。
28. **Early Time Classification with Accumulated Accuracy Gap Control**，ICML 2024 主会。**官方摘要+专题全文核查**：[PMLR](https://proceedings.mlr.press/v235/ringel24a.html)，[原文](https://ar5iv.labs.arxiv.org/html/2402.00857)。累计停止时刻条件下finite-sample accuracy-gap控制，已有LLM阅读实验。
29. **Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents**，2026-09 arXiv。**正文**：[原文](https://arxiv.org/html/2609.09678v1)。整episode风险/coverage联合校准、分离selection/calibration、成本控制。移植谣言不自动新。
30. **Rethink Rumor Detection in the Era of LLMs: A Review**，Findings EMNLP 2025。**官方摘要**：[综述](https://aclanthology.org/2025.findings-emnlp.464/)。Cognition-Interaction-Behavior框架；作检索索引，不把综述未来建议当空白证明。

补充线索（不承担核心创新判决）：CLEAR / IP&M 2026 的 flow purification+Dirichlet evidential learning 出版入口存在，但全文受限且精确 DOI 未核实，故不凭标题声称覆盖候选；ReviseQA（ICML 2025 workshop）与 AGM-Bench（ICLR 2026 workshop）的增删信息任务由专题检索核查，不当主会。其出现足以要求belief-revision进一步查重，但不替代原生谣言gold审计。

### C. 窗口外的标签定义来源（不是近三年创新论文）

31. **Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads**，PLOS ONE 2016。**正文**：[原论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150989)。明确rumour定义与后来true/false；330-thread journalism标注集，不与后续6425-event all-rnr扩展混为一谈。Figshare扩展集本轮403，本地原始映射尚未核验。

## 9. 本轮可复查的数据观察与证据边界

公开 CoPHEME 仓库观察 HEAD：21183d3ce578d7d4bc05c51e29857e9e97237bfa。六个事件 JSON 通过HTTPS只读在内存解析，未保存/再发布原始数据。读取使用浮动main，未保存逐文件哈希或证明各响应绑定该HEAD；因此这些数值是读取时的观察，不构成严格commit级复现证据，后续A0须使用固定commit URL及SHA256。

| 事件 | false目标 | unverified目标 | non-rumour证据 | true证据 |
|---|---:|---:|---:|---:|
| charliehebdo |116|149|1621|193|
| ferguson |8|266|859|10|
| germanwings-crash |111|33|231|94|
| ottawashooting |72|69|420|329|
| putinmissing |9|117|112|0|
| sydneysiege |86|54|699|382|
| 合计 |402|688|3942|1008|

标签计数是**公开池的行数**，不是独立命题数、攻击成功样本数或论文实际eval集。公开 loader 的 `get_target_hypotheses` 默认 `only_rumour=True` 分支同时选 false/unverified，并未应用 `veracity_filter`；但这不证明所有正式实验都走此路径，也不证明 extracted conclusion 的真假。应审查具体 plans、runner、筛选与证书，不能对论文作未经验证的指控。

机器可读记录：[CoPHEME审计](COPHEME_PUBLIC_DATA_AUDIT_2026-10-03.json)。

## 10. 决策与下一步

1. 当前 BCR 方法：**CLOSE，历史产物不改写，M2仍不授权。**
2. 下一步最优投入：**一轮CPU/只读资源与任务语义准入审计**，同时判断候选一是否具有实证条件、候选二是否有可靠gold。
3. 暂不安排新reader inference、训练、下载模型、自动攻击生成、修改历史标签或阈值。
4. 若准入失败，结论是“当前条件下没有足够好的新主线”，应如实报告，不能无休止换名；若准入通过，再单独预注册pilot与计算预算。
5. 具体下一轮范围见[准入审计计划](RESEARCH_ADMISSION_AUDIT_PLAN_2026-10-03.md)。该计划是建议供用户选择，不代表本轮已执行或已授权后续GPU工作。
