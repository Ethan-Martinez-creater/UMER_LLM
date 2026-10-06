# SES-v1 P0-R1 近邻补充核查

- 执行：子代理网络核查（第二轮），执行者汇总；意见仍为 provisional。
- 相对 P0 的更新：GAVEL、IMRRF、RobustRAG 由摘要/受限升级为**正文**核查；WFW、Speaker-Free Floor 完成节号复核；LLM-TKT 仍 UNRESOLVED。
- 全部走合法公开入口（Anthology PDF / arXiv HTML）；无全文者记录已尝试入口。

## 核查级别统计（10 篇，程序化计数）

| 核查级别 | 篇数 | 篇目 |
|---|---|---|
| 正文 | 9 | WFW、RAGuard、SINCon、IMRRF、GAVEL、ProvenanceGuard、REAL/FAE、Speaker-Free Floor、RobustRAG |
| 官方摘要 | 0 | — |
| 元数据/受限 | 0 | — |
| FULLTEXT_UNRESOLVED | 1 | LLM-TKT（SIGIR 2026） |

（P0 报告"4 正文/2 摘要/1 元数据"为有限核查下的错误汇总，本表替代。）

## 各篇更新要点

1. **LLM-TKT — 仍 FULLTEXT_UNRESOLVED**。已尝试：ACM DL（403）、arXiv API（无作者版）、SIGIR 2026 官网程序页、Google Scholar、ResearchGate（登录墙）、GitHub 仓库（404）、Semantic Scholar（429）。摘要级确认：两阶段蒸馏 LLM narrative reasoning → GNN，对抗鲁棒方向；与本候选问题仅弱重叠。创新排除结论对该篇保持 UNRESOLVED。
2. **GAVEL（Findings of ACL 2026，正文）**：Evidence Contract 下辩论 + 机械化审查。**有重复处理但为引用层面**：§4.3 "Duplication and conflict checks for inconsistent citations"；附录 A.5 同一回复重复引用同一证据单元超阈值即标记；A.2 "after deduplicating identical strings"。**无按来源分组/上游依赖建模**；无少数纠错保留评估。FEVEROUS 41.80 vs 最佳基线 37.60。剩余差异：单元/格式级去重 ≠ 同源转述的依赖推断。
3. **IMRRF（NAACL 2025，正文）**：其"冗余信息过滤"（§3.2）是 **LLM 摘要式无关性过滤**（抽取 claim 相关信息压缩 ≤300 词，"filtering out irrelevant information"），**不是去重、不按来源归属/依赖建模**。P0 中"冗余≈不相关干扰、未区分四概念"的推断得到正文证实，原推断从"有限阅读"升级为正文证据。动机句："the retrieved evidence frequently contains substantial redundant information, which can interfere with the LLMs' judgment."
4. **RobustRAG（arXiv:2405.15556v2，正文，第 10 篇必查）**：isolate-then-aggregate 为**文档级隔离**+安全聚合器投票；**不建模"多篇 passage 是否同源"**；威胁模型是恶意注入，不是自然同源转述。ICML 2024 官网身份为 **workshop poster**，不标主会。剩余差异：依赖结构（谁转述谁）在其威胁模型之外。
5. **WFW 复核（arXiv:2601.03746，§5）**：三种条件（2-Table=两个不同来源、1-Table=无重复、Repetition=同一来源重复）；结论原句 "all models prefer repeated information (average SP gap of 30.04), even though no new source is provided and thus no true majority presented."。**度量是 source preference（选择题概率差），不得表述为"事实真假准确率受同源重复影响"**。§6 另含缓解：合并同源重复表可将部分设置 SP gap 从 45.7 降至约 1.0（摘要口径 99.8%、正文口径最高 95.9%，引用须注明口径差异）。威胁判定：高——现象与基本缓解均已被占位。
6. **Speaker-Free Floor 复核（arXiv:2607.05545v1，§3.4/§5）**：no-source 条件（去说话者）+ 剂量实验（重复条数 × 说话人数 N∈{1,2,3,6}）；no-source 66.5% 有害修改率 vs 重问 10.3%；结论 speaker-free floor 为主要成分、简单重校准不可恢复。未把"同一上游被多账号转述"的传播结构作为研究对象。威胁判定：高。

## 对机制假设的影响（F3/F7 收窄结论）

- **distinct_falsifiable_hypothesis = UNRESOLVED**：H1 的 URL/RT 分组计数与其最强规则去重基线未显示实质算法差异；H2 "只改选择与权重、不改 prompt" 不足以证明创新；且 WFW §6 已给出同源重复合并的缓解。原 H1/H2 降级为探索候选（SIDE 假设），不再作为已成立的剩余机制。
- 10 篇中仍无一篇同时覆盖"真实线程转述链依赖结构 + 同源复述不重复计数 + 独立纠错保留"三项；但"仅现成去重/规则方法"也未被排除——恰是 NOVELTY 风险所在。若不能在双人标注可标注性之外给出实质新机制，该方向应按计划转向备选或关闭。
