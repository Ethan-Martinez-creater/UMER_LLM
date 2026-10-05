# UMER / BCR 下一步研究方向评审

**日期：2026-10-02**  
**性质：文献与研究决策评审，不是新实验协议或阶段授权。**  
**依据：最新交接手册、冻结 M1/M1-E 产物、当前代码，以及本轮核查的公开文献。**

## 0. 先给决定

我的建议是：**不要原样继续 BCR 的静态 fingerprint 主线，不要直接执行旧 M2；也不要把 fingerprint 删除、接上 E3+MLP 就作为下一篇论文的主创新。**

但我也不建议立即放弃“LLM 如何使用社会证据”整个领域。更合理的取舍是：

1. **停止围绕原假设继续扩容，保留已有数据、缓存、reader 接口和负结果。**
2. **先完成任务语义与效用测量的短周期审查。** 本轮发现常数基线与 PHEME 标签构念两项重要问题，优先级高于增加 reader 或设计复杂模型。
3. 若审查后仍有实质信号，只对一个更窄的候选问题做有预算上限的可证伪验证：  
   **在新事件和未参与 utility 监督的 reader 上，能否减少社会证据把正确判断带偏的风险，同时保留它修正错误判断的收益？**
4. 若这个候选不能在实际证据采纳动作上超过简单/强基线，就应退出这一代 utility-selector 路线，转向有明确新数据条件的任务，而不是再换一个模块名。

这是**转研究问题、保留资产的受限转向**，不是给 BCR 换名字继续，也不是宣布新的方法已经可发表。

### 对用户两个问题的直接回答

- **方向有没有价值？有。** reader 对外部证据的脆弱性、相关性不等于实际效用、误导证据和模型先验冲突，都是已有高水平论文持续研究的真实问题。
- **现有版本能不能支撑一篇有竞争力的论文？目前证据不足。** 主要障碍不是代码不完整，而是主机制失败、近邻方法很近、基线强度不足、实用任务收益尚未建立，以及跨数据集解释存在混杂。
- **是不是别人已经做出来了？宽泛核心已经有人做并正式发表；完全相同的 BCR 设置，本轮未确认已有同构方法。** 但“尚未发现完全同构”不等于“有充分创新性”。把已有的 reader-centric utility 方法应用到两套 rumor 数据，并不足以自动构成新论文。
- **下一步是否还有可探索方向？有，但应围绕可验证的决策失败，而不是继续追求‘给 UMER 加一个 LLM 模块’。** 下文给出优先级和明确停止条件；不承诺论文录用，也不给没有依据的成功概率。

## 1. 本轮检索是什么、不是什么

本轮围绕四条线检索：

1. reader-specific / model-conditioned retrieval utility、行为表征和跨模型迁移；
2. source/query 状态、perplexity/familiarity、充分性与检索收益预测；
3. LLM 社会上下文谣言检测、蒸馏、早期判断和动态图；
4. 误导证据、证据污染、真实上下文利用、时间合法性及可靠性。

完整记录见 [文献证据索引](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/docs/research_review_2026-10-02/LITERATURE_EVIDENCE.md>)：33 条参考，区分方法级、摘要级、元数据级阅读，以及主会、Findings、期刊、教程、预印本。

核心依据来自 ACL Anthology、AAAI OJS、NeurIPS proceedings、arXiv 和出版商提交到 Crossref 的元数据。部分 ACM、ScienceDirect、DBLP 和 OpenReview 页面受到反爬或验证限制；未访问的全文没有被当成已读。**本轮不是对所有数据库的穷尽检索，也不能保证不存在未公开或尚未检索到的同类工作。**

## 2. 当前路线为什么值得研究，却不宜原样继续

### 2.1 已经成立的资产与结论

按 [最新交接手册](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/UMER_BCR_RESEARCH_HANDOFF_2026-10-02.md>)：

- CR-TSER 的 reader-specific utility 异质性成立；强结构交互门槛未通过，历史线已关闭。
- M1 的 Ma-Weibo ZERO-TOUCH 失败；LIGHT-TOUCH 相对 B0 的平均 utility-sign Macro-F1 增量为 +0.0616。
- M1-E 的 D5 去掉 fingerprint 后达到 +0.0710（相对 B0）；E3 存在时，fingerprint 的平均正增量未获支持，PHEME 上明显负向。
- M1-E 是 post-hoc diagnostic，不定义新 gate；M1_CONDITIONAL_GO 不能直接解释成下一阶段授权。

这些是有效的研究证据。问题在于：**有效证据不等于已经形成足够新颖、稳健且有用的方法。**

### 2.2 对当前机制最危险的近邻论文

| 我们可能想提出的贡献 | 已有工作的覆盖 | 仍需证明的真正差异 |
|---|---|---|
| 相关性不等于 utility | Uplift-RAG（Findings EMNLP 2025）、SCARLet（EMNLP 2025）、RRPO（ACL 2026） | 不是再次发现二者不同，而是解决一个已有方法解决不了的具体决策问题 |
| 不同 reader 需要不同证据 | LLM-Specific Utility / SpecUBench；作者声明 CIKM 2026 录用 | 无目标 reader utility 标签、真实新事件上的可靠迁移，而非测量已有 reader 差异 |
| 静态行为相似度不能迁移 help/harm | Preference Is Not Intervention（2026-08 预印本） | 超出已有负结果的机制或可操作解法；不能只复述 fingerprint 失败 |
| reader 的 source-state / familiarity 可预测收益 | Tian 等 ECIR 2026、FaviComp（Findings EMNLP 2025）、SPS（AAAI 2026） | 不能只是相同特征换成 MLP；需更强、可证伪的适配机制或稳定外推证据 |
| 社会证据可能把 LLM 带偏 | RAGuard（NeurIPS 2025 D&B）、Evidence Pollution（ACL 2025） | 不是再次展示有害性，而是解决降低伤害与保留帮助之间的实际取舍 |

最直接的一篇是 [Tian 等 ECIR 2026](https://arxiv.org/html/2601.14546v1)，其 [正式出版信息](https://api.crossref.org/works/10.1007/978-3-032-21289-4_24) 已核实：

> relevance/QPP + reader 对 query-conditioned context 的 perplexity + 文档质量 → 线性回归 → 检索收益预测。

它与“E3+通用特征+浅层 predictor”非常接近。区别确实存在：它研究完整 retrieved context 的 QA 增益，BCR 研究固定 SRC 下的原子删除 signed utility，并要求目标 reader 的 utility-label-unseen。但**这一差异需要带来新的科学困难、方法和结果，不能仅作为包装。**

[FaviComp](https://aclanthology.org/2025.findings-emnlp.878/) 已把目标 reader familiarity 直接用于证据压缩；[SPS](https://ojs.aaai.org/index.php/AAAI/article/view/40371) 进一步研究比 perplexity 更强的 reader 内部适配信号。我们不能再把“熟悉度影响证据使用”本身写成新机制。

另一方面，[Preference Is Not Intervention](https://arxiv.org/abs/2608.17781) 是预印本，不应当作已同行评审定论；但它已公开了很接近的 signed LOO、行为 profiles 与 transfer 失败分析，足以构成新颖性风险。

### 2.3 正式 gate 通过并不等于论文级基线已经充分

本轮额外做了一项**只读算术复核**：直接读取已经保存的 D5 预测，不训练、不推理、不更改 gate。结果保存在 [ARITHMETIC_AUDIT.json](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/docs/research_review_2026-10-02/ARITHMETIC_AUDIT.json>)。

对固定的“始终预测 NEUTRAL”规则，三类 Macro-F1 为：

```text
F1_constant = [2 × n_NEUTRAL / (n_total + n_NEUTRAL)] / 3
```

HELPFUL、HARMFUL 两类 F1 均为 0。规则没有使用 eval 标签选参；eval 标签仅用于计算该固定规则的指标。

| 数据集 | reader | 始终 NEUTRAL | D5 | D5 − 常数规则 |
|---|---|---:|---:|---:|
| Ma-Weibo | qwen | 0.29535 | 0.29427 | −0.00108 |
| Ma-Weibo | mistral | 0.30619 | 0.30351 | −0.00269 |
| Ma-Weibo | internlm | 0.29433 | 0.32388 | +0.02954 |
| **Ma-Weibo** | **三 reader 平均** | **0.29863** | **0.30722** | **+0.00859** |
| PHEME | qwen | 0.29806 | 0.18007 | −0.11799 |
| PHEME | mistral | 0.32822 | 0.37381 | +0.04559 |
| PHEME | internlm | 0.25736 | 0.27109 | +0.01373 |
| **PHEME** | **三 reader 平均** | **0.29455** | **0.27499** | **−0.01956** |

复核时还把 D5 Macro-F1 从逐条预测重新计算，和冻结 JSON 的数值在 1e-12 内一致；每个 reader 分别有 611 / 695 条唯一 evidence keys，但都只有 **25 个 eval events**。

正确解释：

- +0.0710 是相对 B0，并不等于相对一个最低限度的常数基线也有同等收益。
- D5 在 Ma 上相对常数规则只多约 **0.86 个 F1 百分点**；本轮没有为这个新对比计算 CI，不能声称显著。
- 这**不证明 D5 完全无用**：常数规则没有任何 HELPFUL/HARMFUL 检出能力；D5 的某些 harmful 排序信号仍可能有价值。
- 但这足以说明，当前 headline gain 不能单独支持“机制已经有效、可以扩容写论文”。至少还欠 constant/majority、线性/logistic、单独 source-state、单独 familiarity，以及强 utility 方法等对照。
- 本诊断不回写或推翻冻结的 M1_CONDITIONAL_GO；它改变的是我们对下一阶段投入价值的判断。

### 2.4 更深的风险：当前指标离实际应用还有一层

当前主结果是 **预测 evidence 的 HELPFUL/NEUTRAL/HARMFUL 类别的 Macro-F1**，并不是最终 rumor detector 的 F1。

另外，当前 utility 是：

```text
u_r(e | SRC) = p_r(gold | SRC) − p_r(gold | SRC without e)
```

这表示固定参考上下文里的局部删除效应，不是下面任一对象：

- evidence 本身是否事实正确；
- source-only 加入 evidence 的效应；
- 同时删除多条 evidence 的联合效应；
- 完整选择策略对最终预测正确率或风险的提升。

因此不能把逐条 utility 相加模拟任意子集，也不能因为 HARMFUL 分类变好，就直接声称检测器更安全。策略的证据必须来自**真实执行所选上下文后的最终 reader 输出**。

## 3. 本轮发现的任务语义问题：先处理，再谈 domain shift

### 3.1 PHEME 当前目标不是简单的真假二分类

本机 [PHEME adapter:34–44](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/project/tcdscr/data/pheme_adapter.py#L34-L44>) 直接把：

```text
rumours → 1
non-rumours → 0
```

作为标签。[CR-TSER 加载入口:66–70](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/scripts/cr_tser_common.py#L66-L70>) 沿用了这个 adapter。[冻结 reader prompt:24–45](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/project/cr_tser/readers/base_reader.py#L24-L45>) 使用 A=RUMOR、B=NON_RUMOR，没有展开解释 rumor 的语义。

[PHEME 原始研究](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150989) 明确说明：rumour 是当时未验证的传播信息，后来可能为真，也可能为假。

**所以当前 PHEME 的 RUMOR 不能直接译成 FALSE，NON_RUMOR 也不能直接当作 TRUE。**

这并不自动使原 BCR 实验无效：其仍可合法研究冻结 rumorhood 分类任务上的模型效用。但它意味着：

1. 不能无条件把本项目表述为跨数据集事实真假核验；
2. Ma/PHEME 的差异不只可能来自语言、domain 或 tokenizer，也需要考虑标签构念及模型对 label words 的理解；
3. “纠正”在当前标签下首先只能指**把模型的任务分类从错变对**，不必然是纠正了事实错误；
4. 若下一阶段选择事实核验，必须在新协议/新命名空间下明确 veracity 目标，不能静默改旧标签救结果。

本轮没有测量这一因素对 PHEME 失败的因果贡献，**不能声称已经找到了失败原因**。

### 3.2 还需分开的几件事

- 现有评测是两数据集各自训练的 LORO，不是 Ma-Weibo 训练后直接测试 PHEME。
- E3 的 raw source margin 可能混有任务标签先验、校准和语言效应；NLL/familiarity 不等于知道事实真相。
- `source-state` 在同事件多条 evidence 上被重复使用，统计独立单位仍应是 event，而不是把所有 atomic rows 当独立样本。
- 固定概率差存在 ceiling/floor 限制。一个极自信 reader 的小概率变化，未必代表 evidence 真没有影响；度量变更只能成为另行注册的诊断，不能修改历史缓存。
- 手册的 ±0.05 sign 描述是简写；冻结实现还有正确性翻转优先规则，见 [tri_class_label:153–163](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/project/cr_tser/models/utility_heads.py#L153-L163>)。

## 4. 这个领域现在在做什么：不是缺方向，而是宽泛套路已拥挤

### 4.1 顾问、蒸馏与大小模型协作

ARG/ARG-D（AAAI 2024）、DELL、JSDRV（均为 Findings ACL 2024）已经覆盖 rationale、stance、合成反应、专家融合和调优。我们过去的 Round033/034/039/046 与这些路线高度邻近。

**结论：换强 teacher、换 schema、加 attention/LoRA/agent，不能单独构成新的研究问题。**

### 4.2 更聪明地构造社会上下文

[SePro](https://aclanthology.org/2025.naacl-long.128/) 已研究如何用语义/传播图将长 social context 精炼给 LLM。utility 方向又有 Uplift-RAG、SCARLet、RRPO。

**结论：泛化版“为 LLM 选最有价值的评论”已经不是空白。** 传播结构在本项目中也不能只当标签：P2 失败后，不能为了保留 social-network 新颖性而重新假设结构一定有强增量。

### 4.3 早期判断与真实时间评测

[Few-Shot EARD](https://doi.org/10.1145/3770854.3780315) 已把轻量代理与冻结 LLM 结合做早期停止；[DAWN 的 TIST 扩展](https://doi.org/10.1145/3815193)研究时间合法评测；[LiveFact](https://aclanthology.org/2026.acl-long.546/) 将动态证据、不完整信息与污染监测纳入 LLM 核验评估。

**结论：只增加 time embedding、早停代理，或把静态数据切成几个时点，都不构成新的核心贡献。**

### 4.4 误导、污染、上下文信任与证据整合

[RAGuard](https://papers.nips.cc/paper_files/paper/2025/hash/ed25c00ff6900989116d3ba5d607d33d-Abstract-Datasets_and_Benchmarks_Track.html) 已研究 Reddit 自然误导证据；[Evidence Pollution](https://aclanthology.org/2025.acl-long.480/) 已覆盖重复、回声、改写和生成污染；[DRUID](https://aclanthology.org/2025.acl-long.968/) 已指出真实检索与人工构造情境存在落差。

新的公开工作还包括 GroupQA、GraphEcho 和 SelectBench。因此，“社会回声会骗人”“过滤坏证据”“同时拒绝污染、保留有用信息”也不是无人涉及。

**价值还在，但论文需要更精确的未解决问题，以及比现有简单控制/防御更好的结果。**

## 5. 三种下一步选择：优先级与否决理由

| 选择 | 科学价值 | 对本项目的可行性 | 创新性风险 | 建议 |
|---|---|---|---|---|
| 原 BCR fingerprint + 扩 reader | 问题重要，但主机制已有负证据 | 可执行不等于值得执行 | 极高：机制和负结果叙事均有近邻 | **不继续作为主线** |
| D5 / E3+MLP 直接升级为方法论文 | 有弱而局部的预测信号 | 资产可复用 | 极高：ECIR/FaviComp/SPS；常数基线差距小 | **不直接立项** |
| **A：任务语义审查后，研究 reader/source 条件下的伤害—修正双目标证据采纳** | 真实应用问题明确 | 当前最能复用资产，但需新动作级验证 | 中高：RAGuard/SelectBench/utility 方法相邻 | **有条件的第一候选** |
| **B：真实时间、证据不足与状态更新的核验任务** | 高，尤其适用于新事件 | 需要新数据可得性、时间依据与目标标签 | 高：LiveFact/DAWN/EARD 已覆盖宽泛框架 | **资源条件成立后的备选** |
| **C：证据来源依赖/伪多源支持的鲁棒使用** | 高，具有社会传播特性 | 需要可验证 provenance；旧数据未必具备 | 高：Evidence Pollution/GroupQA/GraphEcho 很近 | **储备，不作为近期第一选择** |

上述不是成功率估计。A 排第一，是因为**增量成本和已有资产匹配较好**，不是因为已证明它创新或会成功。

### 5.1 A：从“预测 utility 类别”转成“是否值得采纳证据”

研究问题应明确为：

> 目标 reader 的 utility 标签不用于训练时，能否根据当前 source 与可见证据，作出比始终使用/忽略证据、置信度规则和强 utility selector 更好的采纳决策，并同时减少伤害和保留修正机会？

#### 为什么不是旧 Round043/049 换名

旧路线主要预测 UMER 是否会错、固定比例转介，以及解释卡是否保真。新候选必须学习/评价的是**动作相对于不动作的成对结果**。如果只是把 E3 接入 error-AUROC/AURC 路由器，仍然是旧路线，不足以成立。

#### 最小可检验对象

先限制动作集合，而非一上来搜索所有证据子集：

- 使用 source-only；
- 使用冻结规则构造的 SRC；
- 使用一个事前指定的证据精炼策略。

每个动作的结果要通过真实 reader 执行得到。旧 I1 标签可作为诊断和初始化信息，**不能代替这些反事实动作结果**。

设 `correct_source` 是 source-only 的正确性，`correct_action` 是采纳动作后的正确性，至少同时报告：

```text
引入错误：P(correct_source = 1, correct_action = 0)
获得修正：P(correct_source = 0, correct_action = 1)
净变化：P(correct_action = 1) − P(correct_source = 1)
成本：总 forward / 输入 token / wall time / reader 特征提取成本
```

若允许拒答，再单列 coverage–risk 与等待/人工成本。**全部拒答、全部忽略评论不应因为降低伤害就被宣称成功。**

#### E3 在这里的合理角色

E3 是一个已有候选传感器，而不是预先认定的创新。要检验：

- familiarity 是在识别有效信息，还是只偏爱熟悉的叙述？
- 一个陌生但正确的纠错证据，会不会因 NLL 高被错误拒绝？
- source-state 的收益能否超过类别先验、单一 margin、校准后的简单规则？
- 只知道 relevance/NLI 的模型是否已经足够？
- reader-specific 条件化是否真的优于共享 utility 方法？

如果拟发表事实核验结论，应优先借助具有人工 evidence annotations 的公开资源做独立压力测试，例如 DRUID/RAGuard；**先核查数据可得性、许可与任务兼容性，本轮尚未完成下载与样本级 readiness。**

#### 基线不能缺失

1. always source-only / always SRC；constant/majority（用于 utility 预测诊断）；
2. semantic/token-matched/random 或固定规则（明确预算）；
3. source margin/entropy 单独、familiarity 单独、NLI 单独；
4. 同特征的 linear/ridge/logistic 与当前 D5；
5. Uplift-RAG、SCARLet、RRPO 类方法中至少选择任务和预算兼容的强对照；
6. SPS/perplexity compatibility 对照；
7. 目标 reader 有监督模型、动作 oracle 作为**另列上界**，不冒充部署基线。

B3 必须保留，但它复制的是**同 evidence key 的训练 reader 真实 utility 标签**，属于 same-evidence oracle-assisted transfer，不能与新事件无 oracle 的方法混为同一预算组。

#### 方法论文的最小新增贡献应是什么

不是一张新的网络结构图，而是至少满足以下一种实质贡献：

- 可解释、可复现的失败条件，并据此设计能改善伤害—修正前沿的轻量决策规则；
- 跨 reader 与新事件稳定的校准/条件适配机制，明确什么监督条件下有效；
- 对现有 utility/compatibility 方法都困难、且具真实社会证据意义的合法评测问题与可靠解法。

本轮尚未证明这些条件成立，不能先给方法取名再把它当作已立项创新。

### 5.2 B：真正转到时间约束的事实核验，而非“再早停一次”

如果 A 失败，且用户愿意改变数据条件，可以考虑：**证据不足时如何保持正确的不确定状态；新证据到来后何时应改变判断，以及如何避免过早自信。**

但这条线已经有 LiveFact、EARD、S2G-RAG 等。可行入口不是“第一个动态 verifier”，而是某个已有框架未解决、可量化的更新失效，例如晚到可靠证据无法纠正先入为主判断。

与历史 TOWRV 的区别必须来自**资源条件真的改变**：已有公开的时序证据集是否提供文本、可靠可得时间和相应标注？如果仍必须自己追溯失效的网页档案，就不能把网络可达或论文里有链接当成问题已解决。

在 readiness 前，不建议投入多代理、外部搜索工具链或大规模人工标注。

### 5.3 C：来源依赖与伪共识，但不是又一次“结构有用”

同一事实被转述十次，不等于十条独立证据。这个社会传播问题有价值，且不同于 CR-TSER P2 的强非加性交互假设。

然而 GraphEcho 已控制路径数与来源数，GroupQA 已研究重复论据，Evidence Pollution 已有 Repeat/Echo。因此**仅去重、计数或生成回声评论，不够新**。

只有在真实可验证的 provenance、自然传播场景、保留有效新信息的联合目标上形成明显差异，才值得再谈。这还要求不能把“同一回复树”自动解释为同一事实来源，也不能用未经验证的相似度群组假定因果依赖。

当前数据条件与历史 P2 结果使它不适合作为近期第一选择。

## 6. 为什么不再推荐那些热门组合

| 路线 | 文献与项目证据 | 当前决策 |
|---|---|---|
| LLM cognition/stance/salience → UMER 蒸馏 | ARG/DELL/JSDRV 已覆盖；旧系列多次未获净收益 | 不再以模块增益为主线 |
| LLM 合成回复/传播补全 | 已有方法；项目真实行为校准和 grounding 不足 | readiness 未变不重启 |
| 风格改写增强 | SheepDog 已有；项目语义保持审核未全过 | 不靠扩大改写数挽救 |
| 更复杂 GNN/Transformer/RL selector | SePro/utility reranker/动态方法拥挤；旧 structural gate 失败 | 先证明简单方法解决不了的问题 |
| 通用 sufficiency critic | S2G-RAG、context sufficiency 谱系已覆盖 | 不作为单独创新 |
| 更换大模型/扩六 reader | 不修复标签构念、弱基线或假设失败 | 只有明确新问题后才考虑 |
| 直接做更多 agent 的辩论融合 | 算力与成本高，且无法自动解决同源错误/评测真实性 | 不作为当前资源下的优先方向 |

过去的 FAIL 也不能一概解释为“LLM 无价值”。应区分**科学机制未支持、实现/输出接口失败、资源未就绪**：例如 TOWRV 的 NOT_READY 并不是科学反证；同样，修好 schema 也不等于有了论文贡献。

## 7. 建议的投入顺序与退出条件（仅供审批）

### 阶段 0：先明确任务和证据标准

建议作为短周期前置审查，优先利用已有资料：

- 书面区分 rumorhood、veracity、stance、evidence sufficiency 和 model utility；
- 保留冻结协议，列出哪些新问题必须建立新 protocol/namespace；
- 检查 label-word bias、source-only 状态、类别比例、概率饱和和 tokenization 的混杂；
- 把本轮常数基线纳入诊断，而非改历史 gate。

**退出条件**：如果原信号主要是任务语义或基础校准问题，就先修研究设计，不进入方法扩容。

### 阶段 1：给 A 一次有预算上限的验证机会

建议以 **1–2 周研发时间作为投入上限的讨论起点**，不是运行时间承诺，也不是已批准计划。预算应先按少量开发事件的实际 token/forward/wall time估计；不能省略 E3 成本。

- 固定很小的动作集合与强基线；
- 先在新的开发数据上估计方差和可检测的实际效应；
- 用未参与此次假设选择的新事件做确认；
- 条件允许且获得授权后，再增加一个明确 held-out family 用于证伪，而不是机械执行旧六-reader M2；
- 预注册需要达到的实际改善、非劣条件、区间估计和预算。不要事后看到结果再改阈值。

**退出条件**：不能同时减少引入错误、保留修正收益，或仅靠全拒/全不用证据、弱基线、某一 reader 或某一数据集支撑，则停止这条方法线。

### 阶段 2：只有通过才展开论文工作

- 引入新的时间/话题或独立数据；
- 区分各数据集内部 LORO、跨 dataset transfer 和真正新模型外推；
- 以 event 为统计单位，不能把几千 evidence rows 当作几千独立试验；
- 当前 utility_eval 已被反复查看，只能作为公开披露的诊断证据，不能作为新假设盲确认集；
- 记录 API/reader 调用成本、数据许可与模型访问条件。

如果使用 conformal/risk-control 术语，必须写清可交换性、校准标签和 shift 假设。**跨新 reader/domain 的无标签设置不能凭加一个 conformal 模块就宣称分布无关保证。**

## 8. 最后：关于“能否写论文”的诚实预期

### 现在能写什么

可以写可信的内部研究报告、负结果与证据账本。当前结果还不适合直接包装成“解决 unseen-reader evidence utility transfer”的方法论文。

负结果不必然没有论文价值，但要有新的、可推广的认识。现有 fingerprint 失败叙事与 R03 已相近，三 reader、两历史数据集和 25 eval events 也限制其外推能力。**把失败集合整理得工整，不自动等于一篇有竞争力的负结果论文。**

### 后续什么样的成果才值得投稿

- 若 A 在严格设置与强对照下形成清楚的决策收益和机制边界，才可能面向 ACL/EMNLP/SIGIR/WWW 等相关渠道讨论适配性；
- 若贡献主要是可靠的领域应用验证，需要更全面的实验与实质问题改善，不能把 IP&M/TKDE 等期刊当成“创新不足即可降投”的捷径；
- 若走 benchmark 路线，则需要数据质量、许可、代表性和充分基线，NeurIPS D&B 等不是绕开方法失败的低成本方案。

这些是贡献形态与研究社区的对应关系，不是录用预测。

## 9. 收尾发现的 M1-F 文件：单独注明，不冒充本轮实验

本报告收尾检查时，工作区出现了此前未列入交接的、**尚未跟踪的** [M1F_REPORT.md](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/results/bcr_utility_v1/m1f/M1F_REPORT.md>)、[M1F_VERDICT.json](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/results/bcr_utility_v1/m1f/M1F_VERDICT.json>) 和 [TASK_VALIDITY.md](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/results/bcr_utility_v1/m1f/TASK_VALIDITY.md>) 等产物。它们**不是本轮文献评审生成的**。

我已只读查看这三份报告，但本轮没有复跑 M1-F、审计其执行代码，或确认其来源与批准状态。因此，本评审的主要结论仍建立在权威交接及冻结 M1/M1-E、本轮独立算术复核上，**不把新文件自动提升为已授权的正式结论**。

这些新文件报告：

- `M1F_CLOSE_BCR_UTILITY_METHOD`，并明确不改变 M1/M1-E、不授权 M2；
- Ma-Weibo：D5 相对 source-state-only 的 S2，平均 Macro-F1 差为 **−0.00850**，95% CI **[−0.07730,+0.06948]**；
- D5 的 within-snapshot centered Spearman 为 **−0.0083**，区间跨 0，未显示有力的同一快照内证据区分能力；
- 文本报告的 PHEME 小节同时写了数值和“PHEME was not part of this run”。**不能把该小节的数字当作已完成的 PHEME M1-F 实验**，需由该轮负责人澄清。

如果这些是用户已经批准并认可的新审计，下一步应**先审阅其证据链，而不是重复安排同一强基线审查**；它们会进一步加强“停止把 D5 当作可扩容的方法主线”的建议。候选 A 若要继续，必须有独立的新问题和动作级证据，不能以重跑 M1-F 或修改旧门槛作为继续理由。

## 10. 一句话结论

**我建议停止原样推进 BCR；先做测量和任务语义审查，再只给“伤害—修正双目标的 reader 条件证据采纳”一次受限验证机会。保留现有资产，不执着于现有机制。若这一步仍不成立，就改变数据和任务条件，而不是继续在 fingerprint / E3 / selector 之间换模块。**

---

### 本轮生成的证据

- [完整文献证据索引](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/docs/research_review_2026-10-02/LITERATURE_EVIDENCE.md>)：发表状态、阅读深度、官方 URL 与限制。
- [只读算术诊断](<E:/Graduate_work_folder/Graduate_Project_Worksapace/UMER/docs/research_review_2026-10-02/ARITHMETIC_AUDIT.json>)：固定 NEUTRAL 规则与冻结 D5 的精确点估计、逐 reader 类别数量。

本轮没有调用研究服务器执行模型，没有训练或生成新 utility labels，没有改变冻结结果、协议、reader panel 或 M2 授权。
