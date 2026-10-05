# UMER / BCR 研究方向评审：文献证据索引

检索与判定截止：**2026-10-02**。本文件用于研究决策，不改变任何冻结实验结论。

## 阅读与发表状态说明

- **A：方法级阅读**：读到公开正文的研究问题、方法或实验设置，并核对出版信息；不表示逐页读完所有附录。
- **B：摘要级核查**：读到官方摘要与出版/版本信息；不据此推断未公开的实现细节。
- **C：元数据核查**：仅确定论文真实存在、题名、期刊/会议及日期；不作为方法相同的强证据。
- Findings、主会、专题轨、教程与 arXiv 预印本分别标明。作者声称 accepted 不自动等于截止日已正式发表。
- 本次使用 ACL Anthology、AAAI OJS、NeurIPS Proceedings、arXiv、出版商登记的 Crossref 元数据，并以普通学术搜索定位资料。部分 ACM/ScienceDirect/DBLP/OpenReview 页面遇到 403 或浏览器验证；不声称完整遍历 Web of Science、Scopus、Google Scholar 或所有付费全文。
- 下列 **33 条记录并非 33 篇已精读的顶会论文**，包括方法论文、评测论文、期刊扩展、教程、预印本及基础数据文献。没有检索到完全相同的方法，不等于证明不存在。

## 一、直接影响 BCR 创新性判断的工作

| ID | 论文与正式状态 | 深度 | 已核实内容与对本项目的影响 |
|---|---|---|---|
| R01 | Fangzheng Tian 等，**Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation**，ECIR 2026，正式论文集，2026-03-25 在线，pp.368–385 | A | 将有/无检索上下文的回答质量差定义为 RPP 目标，用 relevance/QPP、reader 的 query-conditioned context perplexity、文档质量等特征做线性回归。直接邻近 E3 + 浅层 utility predictor；实验为 NQ、Llama-3-8B，不等于本项目目标-reader-utility-label-free LORO。 |
| R02 | Hengran Zhang 等，**LLM-Specific Utility for Retrieval-Augmented Generation**，2025 首次公开，2026-08-27 v4；作者声明 CIKM 2026 accepted | A | SpecUBench 和 reader-specific utilitarian passages：不同 reader 需要不同证据，其他 reader 的最优证据并不完全通用。截止日未独立确认出版社正式上线，故按“公开预印本、作者声明录用”处理。不是无目标 utility 标签的 LORO predictor。 |
| R03 | Shi Zhou，**Preference Is Not Intervention: The Structure and Stability Boundaries of Reader-Specific Evidence Utility**，2026-08-18 arXiv 预印本 | A | 包含 LOO utility、九 reader、activity/ordinal/signed geometry，稳定偏好相似度不能可靠预测 help/harm transfer。连“静态 fingerprint 失败、query-local interaction 很重要”的负结果叙事也有直接先例。未确认同行评审发表。 |
| R04 | Dongwon Jung 等，**Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation**，Findings of EMNLP 2025，pp.16181–16196 | A | FaviComp 以压缩模型和目标 reader 的 token-level ensemble 生成更熟悉的证据，training-free 但需要 reader token probabilities。familiarity/perplexity 不是新机制；不同在于它改写/压缩，不预测 signed atomic deletion utility。 |
| R05 | Yilong Xu 等，**Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models**，EMNLP 2025 主会，pp.629–648 | A | SCARLet 在 shared context 上合成多任务数据，用删除扰动与 ridge surrogate 估计 passage utility，再训练 retriever；覆盖 passage interactions 与跨任务泛化。所读方法正文为公开 arXiv 后续版本，未逐字比对正式 PDF。 |
| R06 | Changle Qu 等，**Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation**，Findings of EMNLP 2025，pp.9632–9644 | B | 官方摘要明确：utility 是超过 LLM 内部知识的 marginal benefit；三个 alignment objectives 训练 reranker；动态选择弥补 knowledge gap 的证据。不能把“根据已有知识决定证据是否有益”当全新问题。具体损失细节未按全文核实。 |
| R07 | Yuhang Wu 等，**Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning**，ACL 2026 主会长文，pp.30474–30490 | A | RRPO 用回答质量反馈优化有序证据选择，reference-anchored baseline 稳定学习；官方摘要报告跨 reader 泛化。无人工 relevance 标注不等于无 gold answers；具体迁移划分不自动等价于 BCR 的 LORO。 |
| R08 | Zhanghao Hu 等，**Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score**，AAAI 2026 NLP 技术轨，2026-03-14 | B | SPS 用 reader 隐表示衡量摘要语义适配；xCompress 对摘要采样、排序、压缩。四 open-source readers、五 QA benchmarks。比纯 perplexity 更强的 reader-conditioned compatibility 已有正式工作。 |
| R09 | Chenxu Wang 等，**ICL-Router: In-Context Learned Model Representations for LLM Routing**，AAAI 2026，2026-03-14 | A | 用 anchor queries 的 query vectors 和 correctness 构成能力 profile，对新模型可不重训 router；但需要新模型 probe 的正确性标签。覆盖 behavioral profile → model representation → new-query/model prediction 的宽泛思想，不等同 signed evidence intervention。 |

### R01–R09 一手来源

- **R01**：[正文](https://arxiv.org/html/2601.14546v1)；[Springer DOI](https://doi.org/10.1007/978-3-032-21289-4_24)；[出版商登记元数据](https://api.crossref.org/works/10.1007/978-3-032-21289-4_24)。注意作者实现的 PerpC/PerpA 用 `exp(mean log probability)`，分数方向与通常 perplexity 相反。
- **R02**：[版本与摘要](https://arxiv.org/abs/2510.11358)；[v4 正文](https://arxiv.org/html/2510.11358v4)。摘要页写三 QA 数据集，所读 HTML 的 abstract/introduction 又写六个、部分表仍为三个；本报告不据此固定声称“全部六数据集均已完整验证”。正文所载 CIKM 日期为 2026-11-07—11，晚于检索截止日。
- **R03**：[摘要及版本](https://arxiv.org/abs/2608.17781)；[正文](https://arxiv.org/html/2608.17781v1)。其中 33% sign disagreement 等数值是作者报告，非本次复现实验。
- **R04**：[正式论文页](https://aclanthology.org/2025.findings-emnlp.878/)；[公开正文](https://arxiv.org/html/2409.12468)。
- **R05**：[正式论文页](https://aclanthology.org/2025.emnlp-main.33/)；[公开正文](https://arxiv.org/html/2504.00573)。
- **R06**：[正式论文页](https://aclanthology.org/2025.findings-emnlp.511/)。
- **R07**：[正式论文页](https://aclanthology.org/2026.acl-long.1406/)；[公开正文](https://arxiv.org/html/2604.02091)。
- **R08**：[AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/40371)。
- **R09**：[AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/40628)；[公开正文](https://arxiv.org/html/2510.09719)。公开版本作者列表与 proceedings 不完全相同，正式引文应以 proceedings 为准。

## 二、社会证据、时间与可信性：备选方向也必须面对的工作

| ID | 论文与正式状态 | 深度 | 与方向选择有关的证据 |
|---|---|---|---|
| R10 | Yirong Zeng 等，**Exploring Large Language Models for Effective Rumor Detection on Social Media**，NAACL 2025 主会长文 | B | SePro：长结构化社会上下文会妨碍 LLM，语义—传播图注意力及聚类用于上下文精炼。“GNN 选评论再喂 LLM”已不新。 |
| R11 | Fengzhu Zeng 等，**LLM-based Few-Shot Early Rumor Detection with Imitation Agent**，KDD 2026 正式论文，V.1，1856–1867 | A | 轻量代理决定 continue/stop，冻结 LLM 作检测；训练使用 gold 和 LLM 轨迹构造专家。已正面覆盖“何时停止观察/调用 LLM”；训练轨迹成本不能忽略。 |
| R12 | Junghoon Kim 等，**Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness**，WSDM 2025 | B | DAWN 的时间合法切分与 engagement-earliness 去噪；未来信息泄漏、随机切分过乐观已有直接领域先例。不是一个生成式 LLM 融合模型。 |
| R13 | 同一团队，**A New DAWN for Fake News Detection: Exploiting Engagement Earliness for Temporality-aware Evaluation**，ACM TIST 2026，2026-05-07 在线 | B | R12 的正式期刊扩展；出版商登记摘要仍强调 temporality-aware evaluation 与边权去噪。未按全文比较与前身的所有新增内容。 |
| R14 | Linda Zeng 等，**Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals**，NeurIPS 2025 Datasets and Benchmarks Track | B | RAGuard 用 Reddit 自然出现的误导信息，而非仅合成噪声；区分 supporting/misleading/unrelated。官方摘要报告测试 RAG 均低于无检索基线。“社会证据可能有害”已有正式强先例。 |
| R15 | Herun Wan 等，**On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs**，ACL 2025 主会长文 | A | 四任务十数据集；包括评论删除、重复、改写、Echo/生成污染，以及检测、专家融合、参数更新防御和校准退化分析。“回声攻击/合成评论污染”不能再次当首次发现。 |
| R16 | Lovisa Hagström 等，**A Reality Check on Context Utilisation for Retrieval-Augmented Generation**，ACL 2025 主会长文 | A | DRUID：真实检索证据及人工 relevance/stance，5,490 样本、1,329 claims；单一 perplexity 等属性与真实 context usage 的关系有限，synthetic 场景会夸大规律。可作为外部压力测试候选，不等于现成社交传播树。 |
| R17 | Cheng Xu 等，**LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection**，ACL 2026 主会长文 | B | 动态证据、时间不确定性、污染监测、classification/inference 双模式；22 LLM。动态 benchmark 和 early uncertainty 本身已有工作。本轮未完成数据下载/时间戳及许可逐项审核。 |
| R18 | Minghan Li 等，**S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA**，ACL 2026 主会长文 | B | 显式 sufficiency/gap judge 控制下一轮 retrieval；因此“轻量充分性判别器+继续检索”不是空白。 |
| R19 | Ekaterina Fadeeva 等，**Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval-Augmented Generation**，Findings of ACL 2026 | B | FRANQ 区分 factuality 与 faithfulness，再条件化不确定性估计。仅声称“证据一致不等于事实正确”也不是新贡献；其对象是 RAG 生成输出。 |
| R20 | Jiaqi Wu 等，**From Relevance to Utility: Faith-Rank for Utility-Driven Evidence Re-ranking and Reliable Answering in Retrieval-Augmented Generation**，Machine Learning 115(8):191，2026-07-27 在线 | C | Crossref 可确认正式期刊信息；Springer/ACM 页面受限，未读到可核验正文，不能据题名断言已实现我们的全部设置。 |
| R21 | Atharv Naphade，**Rational Synthesizers or Heuristic Followers? Analyzing LLMs in RAG-based Question-Answering**，Findings of ACL 2026 | B | GroupQA：1,635 questions、15,058 evidence documents；官方摘要报告重复改写有时比独立支持更有说服力、顺序偏差与解释不忠实。 |
| R22 | Sikun Wang 等，**GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents**，2026-09-15 arXiv 预印本 | B | 固定证据内容，操纵图路径和来源数量；provenance-aware post-training 降重复却可能降低真实科学 claims 准确率。去重/来源独立性不是全新切口，也不保证改善任务。 |
| R23 | Sebastien Kawada、Manolis Kellis，**Evidence Integration in Large Language Models**，2026-09-03 arXiv 预印本 | B | 作者提出 receiver prior 和 candidate tilt 理论，报告 familiarity、模型特征与证据整合关系。规模与结论均是作者声明，未同行评审；仍构成广义“即时模型状态决定证据效果”的竞争性公开先例。 |
| R24 | Yanyu Chen 等，**Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results**，2026-07-22 arXiv 预印本 | B | SelectBench/DAPO 同时追求保留有效证据、拒绝污染；作者诚实报告改进较小且未过 Holm 校正。故“选择性采纳”不是无人研究，但可靠性和外推仍未解决。 |

### R10–R24 一手来源

- **R10**：[NAACL 正式页](https://aclanthology.org/2025.naacl-long.128/)。
- **R11**：[arXiv](https://arxiv.org/abs/2512.18352)；[公开正文](https://arxiv.org/html/2512.18352v2)；[ACM 登记元数据](https://api.crossref.org/works/10.1145/3770854.3780315)；[DOI](https://doi.org/10.1145/3770854.3780315)。Crossref online 2026-04-20、print 2026-08-09；arXiv 正文会议信息中的一个 2025 日期不作为出版日期依据。
- **R12**：[公开页](https://arxiv.org/abs/2411.12775)；[DOI](https://doi.org/10.1145/3701551.3703524)。
- **R13**：[正式登记元数据与摘要](https://api.crossref.org/works/10.1145/3815193)；[DOI](https://doi.org/10.1145/3815193)。
- **R14**：[NeurIPS 正式页](https://papers.nips.cc/paper_files/paper/2025/hash/ed25c00ff6900989116d3ba5d607d33d-Abstract-Datasets_and_Benchmarks_Track.html)。不要与同名“RAGuard 防御框架”预印本混淆。
- **R15**：[ACL 正式页](https://aclanthology.org/2025.acl-long.480/)；[公开正文](https://arxiv.org/html/2410.12600v1)。
- **R16**：[ACL 正式页](https://aclanthology.org/2025.acl-long.968/)；[公开正文](https://arxiv.org/html/2412.17031v2)；[作者代码与数据入口](https://github.com/copenlu/context-utilisation-for-RAG)。
- **R17**：[ACL 正式页](https://aclanthology.org/2026.acl-long.546/)。
- **R18**：[ACL 正式页](https://aclanthology.org/2026.acl-long.1185/)。
- **R19**：[Findings 正式页](https://aclanthology.org/2026.findings-acl.338/)。
- **R20**：[出版商登记信息](https://api.crossref.org/works/10.1007/s10994-026-07109-8)；[DOI](https://doi.org/10.1007/s10994-026-07109-8)。
- **R21**：[Findings 正式页](https://aclanthology.org/2026.findings-acl.2003/)。
- **R22**：[版本及摘要](https://arxiv.org/abs/2609.17695)。
- **R23**：[版本及摘要](https://arxiv.org/abs/2609.04290)。未将 114 页预印本称为已全文精读。
- **R24**：[版本及摘要](https://arxiv.org/abs/2607.20090)。未将其受控污染/指令注入场景等同于自然社会评论。

## 三、既有失败路线的已发表近邻与基础定义

| ID | 论文与状态 | 深度 | 对项目的意义 |
|---|---|---|---|
| R25 | **Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**，AAAI 2024 AI for Social Impact 专题技术轨 | B | ARG/ARG-D：LLM 多视角理由、小模型自适应吸收、推理时不调用 LLM 的蒸馏。顾问/蒸馏大框架已有，不能仅换 teacher 救回旧路线。本索引以正式题名和官方页面引用。 |
| R26 | **DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection**，Findings of ACL 2024 | A | 合成用户反应/交互图、多代理任务解释、图模型专家与 LLM 融合；七项评测包含其他社会文本任务，并非七个 rumor datasets。 |
| R27 | **Reinforcement Tuning for Detecting Stances and Debunking Rumors Jointly with Large Language Models**，Findings of ACL 2024 | A | JSDRV：stance 与 rumor verification、两层选择器、伪标签与调优；使用人工 veracity 种子、P-stance 外部监督和 Llama-2 7B LoRA。不能写成完全零标注。 |
| R28 | **Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**，官方预印本标注 KDD 2024 Research Track 并提供 ACM DOI | B | SheepDog 的风格改写、一致性与内容导向监督，邻近旧 Round036/040。不能把再次改写增强当新方向。 |
| R29 | **K-FLARE: A knowledge-fused language and relational-learning framework for rumor detection**，Knowledge-Based Systems 334:115089，2026-02-15 | C | 仅核验正式元数据；不根据旧笔记或题名补写模型、数据集和训练细节。 |
| R30 | **DCAGTN: Dual-snapshot causal-attention graph transformer with feedback mechanism for real-time rumor detection**，Knowledge-Based Systems，2026-02-28 | C | 仅核验正式元数据。不能根据 causal-attention 名称声称做了因果识别，也未确认它是生成式 LLM 方法。 |
| R31 | Arkaitz Zubiaga 等，**Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads**，PLOS ONE 2016 | A | 基础定义明确：rumour 是传播中的未验证信息，可以最终为真、为假或未解决。用于解释 PHEME rumorhood 与 falsity 的区别。 |
| R32 | Hengran Zhang 等，**Beyond Relevance: Utility-Centric Retrieval in the LLM Era**，作者标注 SIGIR 2026，内容明确为 tutorial | B | 将 LLM-specific/context-dependent utility 归纳为领域方向；教程说明问题受重视，但不是一篇独立验证所有机制的研究论文。 |
| R33 | **Can Large Language Models Detect Rumors on Social Media?**，2024 arXiv，LeRuD；本轮未核实正式发表状态 | B | 线索提示与 Chain-of-Propagation，邻近传播上下文处理。不能算作已发表主会。 |

### R25–R33 一手来源

- **R25**：[AAAI 正式页](https://ojs.aaai.org/index.php/AAAI/article/view/30214)。
- **R26**：[Findings 正式页](https://aclanthology.org/2024.findings-acl.155/)；[公开正文](https://arxiv.org/html/2402.10426v1)。
- **R27**：[Findings 正式页](https://aclanthology.org/2024.findings-acl.796/)；[公开正文](https://arxiv.org/html/2406.02143v1)。
- **R28**：[公开记录](https://arxiv.org/abs/2310.10830)；[DOI](https://doi.org/10.1145/3637528.3671977)。
- **R29**：[出版商登记信息](https://api.crossref.org/works/10.1016/j.knosys.2025.115089)；[Elsevier coredata](https://api.elsevier.com/content/article/PII:S0950705125021276?httpAccept=text/xml)。
- **R30**：[Elsevier coredata](https://api.elsevier.com/content/article/PII:S0950705125022014?httpAccept=text/xml)；[DOI](https://doi.org/10.1016/j.knosys.2025.115167)。
- **R31**：[PLOS 正式全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150989)。
- **R32**：[公开摘要与版本](https://arxiv.org/abs/2604.08920)。
- **R33**：[公开摘要与版本](https://arxiv.org/abs/2402.03916)。

## 四、引用边界

1. 不能以“他人也报告负结果”推断本项目的一切机制均不可能；只能说明重复发现同一宽泛现象的新颖性有限。
2. 同理，别人正式发表了 utility 方法，不能据此推断我们的 reader-label-free social setting 已被完全解决。
3. 文献报告的性能不得与本项目 F1 横向拼表比较：任务、标签、划分、reader、监督和成本不同。
4. 预印本作为创新性风险线索，不能代替已同行评审的可靠性证据。
5. R20/R29/R30 的方法细节尚未验证，不用它们作“完全撞车”的决定性证据；仅 R01/R04/R05/R06/R07 的已核内容已足以否定泛化版 E3+MLP 的新颖性主张。
6. 本索引为可追溯的定向深检，而非穷尽性系统综述；没有对未访问的付费全文或未公布工作作保证。
