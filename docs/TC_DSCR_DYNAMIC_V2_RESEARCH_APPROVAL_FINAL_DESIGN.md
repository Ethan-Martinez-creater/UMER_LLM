# TC-DSCR Dynamic V2：深度调研、方案审批与最终方法设计

## 1. 结论

本轮调研与方法审批后，推荐不再延续原 E3 的：

\[
d_i^t=u_i^t+\lambda_n \cdot novelty_i^t+\lambda_p \cdot persistence_i^t
\]

这种“逐节点加分式 Dynamic Memory”。

最终推荐的 Dynamic V2 为：

# MF-TSR：Marginal-Fidelity Temporal Set Refinement
## 边际保真驱动的时序证据集合精炼

MF-TSR 继续嵌入 TC-DSCR，不推翻已经完成并验证有效的 Causal Social Encoder 与 Static Utility Selector。它只替换失败的 Dynamic Memory 层：

\[
G_t
\rightarrow
\text{Causal Social Encoder}
\rightarrow
\text{Static Utility}
\rightarrow
\boxed{\text{MF-TSR}}
\rightarrow
\text{Evidence Set }S_t
\rightarrow
\text{LLM}
\]

核心变化是：

> 不再问“某一个节点因为新颖或历史保留应该加多少分”，而是问“在当前 token budget 和当前已有 evidence set 下，加入、删除或替换某条社会证据，是否真的提高了整个 evidence set 对当前传播图判别信息的保真度”。

这一设计直接对应当前实验已经确认的三个失败根因：

1. Dynamic bonus 的尺度只有 Static utility 波动的约 3.9%（PHEME）和 1.9%（Ma-Weibo），因此很难真正改变重要排序；
2. 即使 ranking 变化，token-budget packing 后常常没有产生集合变化；
3. 即使集合变化，下游 Proxy prediction 改变率也只有约 0.71% / 0.43%。

因此 Dynamic V2 必须从 point-wise reranking 转为 set-wise marginal refinement。

---

# 2. 当前研究结果对下一版方法的约束

当前仓库已经给出了足够明确的经验事实。

## 2.1 已经成立的部分

Causal Social Encoder 已经能够在真实时序 snapshot 上形成稳定表示。

Corrected E2 进一步证明：

- PHEME：Static Selector 相比简单 baseline 有稳定但较小提升；
- Ma-Weibo：Static Selector 相比 Random/Semantic 具有明显优势；
- 因而“从传播图中精炼值得提供给下游模型的 social evidence”这一研究问题本身是成立的。

因此下一步不应重写 encoder，也不应放弃 Static Utility Selector。

## 2.2 已经失败的部分

原 Dynamic Memory 采用：

\[
u_i + \lambda_n novelty_i+\lambda_p persistence_i
\]

held-out test 上：

- PHEME Dynamic − Static Macro-F1 约 +0.0001；
- Ma-Weibo 约 +0.00001；
- PHEME FlipRate 反而显著变差；
- corrected paired bootstrap 的 Macro-F1 CI 均跨 0。

Failure Diagnosis 显示：

### Static / Dynamic 集合高度相似

- PHEME exact match ≈ 90.5%，mean Jaccard ≈ 0.969；
- Ma-Weibo exact match ≈ 88.5%，mean Jaccard ≈ 0.975。

### Dynamic bonus 尺度远小于 Static Utility

\[
\frac{\sigma(dynamic\ bonus)}{\sigma(u)}
\]

约为：

- PHEME：0.039；
- Ma-Weibo：0.019。

### ranking change 经常无法穿透 budget boundary

特别是 Ma-Weibo，大量样本出现：

```text
ranking changed
but
selected evidence set unchanged
```

### selection change 对 Proxy 极不敏感

在已经发生 evidence replacement 的情况下：

- PHEME prediction change ≈ 0.71%；
- Ma-Weibo ≈ 0.43%。

所以问题不是简单的“动态权重没调好”，而是优化对象选错了：

> 原方法优化的是 individual node score，而最终真正影响 LLM/Proxy 的对象是整个证据集合。

---

# 3. 2025–2026 最新研究对方案设计的启示

本轮调研重点覆盖 ACL/NAACL/EMNLP/COLING 以及相关高水平工作，重点不是泛泛总结谣言检测，而是寻找能够解决当前 E3 failure diagnosis 的方法思想。

## 3.1 谣言检测正在强调 refined social context

Zeng et al., NAACL 2025 的研究直接表明：LLM 在面对大规模、结构化 social context 时表现会下降，moderate/refined context 反而更适合 LLM 推理；其方法通过语义与传播结构协同建模后重构 LLM context。

这与 TC-DSCR 的整体研究假设高度一致：

\[
\text{more social context}
\neq
\text{better LLM context}
\]

真正的问题是：

\[
\text{which evidence set should be exposed to the LLM?}
\]

来源：
Yirong Zeng et al. *Exploring Large Language Models for Effective Rumor Detection on Social Media*. NAACL 2025.
https://aclanthology.org/2025.naacl-long.128/

## 3.2 谣言检测的“动态性”已经从粗粒度 snapshot 转向跨时刻状态更新

FGDGNN（ACL Findings 2025）指出现有 dynamic rumor models 往往只使用粗粒度时间信息，并提出 intra-snapshot + inter-snapshot 动态更新。

SWAM（EMNLP 2025）进一步使用 adaptive sliding window 与 memory-augmented attention 处理长期依赖，说明：

> “传播图动态变化”本身已经成为高水平谣言检测工作的明确研究对象。

来源：
https://aclanthology.org/2025.findings-acl.296/
https://aclanthology.org/2025.emnlp-main.729/

但这两类方法解决的是 representation dynamics，而 TC-DSCR 应避免重新做一个动态图分类模型。TC-DSCR 更有区分度的问题应保持为：

> 动态图已经形成以后，在每个真实时间点怎样精炼 LLM 应读取的社会证据集合。

## 3.3 2026 年 misinformation evaluation 已经明确转向 evolving / incomplete evidence

ACL 2026 的 LiveFact 不再使用传统 static benchmark 逻辑，而是通过 dynamic temporal evidence sets 模拟真实世界的 “fog of war”，强调在 evolving、incomplete evidence 下评测 LLM verification。

来源：
https://aclanthology.org/2026.acl-long.546/

## 3.4 Evidence selection 的研究趋势正在从 ranking 转向 set selection

ACL 2025 的 SetR 明确指出：Individually relevant passages do not necessarily form a collectively useful evidence set，并将传统 reranking 改为 set-wise passage selection。

这正好解释 TC-DSCR E3 的失败：

```text
node rank changed
!=
evidence set improved
```

来源：
https://aclanthology.org/2025.acl-long.861/

## 3.5 Relevance 不等于真实 downstream utility

Uplift-RAG（EMNLP Findings 2025）明确区分 retrieval relevance 与 actual marginal benefit，并使用 uplift/marginal benefit 来判断一条 evidence 是否真的对下游生成产生贡献。

这直接对应 TC-DSCR failure diagnosis 中：

```text
Novelty 很高
但是 Utility 更低
最终 prediction 几乎不变
```

来源：
https://aclanthology.org/2025.findings-emnlp.511/

## 3.6 Evidence sufficiency 比“把 budget 填满”更合理

ECoRAG（ACL Findings 2025）强调 evidentiality 与 evidence sufficiency：不是固定塞入更多 context，而是判断当前 context 是否已经提供足够证据。

SARA（ACL 2026）同样强调在固定 token budget 下保留少量高价值文本并迭代 rerank。

来源：
https://aclanthology.org/2025.findings-acl.1365/
https://aclanthology.org/2026.acl-long.661/

## 3.7 Token budget 应被当作 constrained set optimization

Self-Correcting RAG（ACL Findings 2026）把 context selection 直接建模为 multi-dimensional multiple-choice knapsack problem，在严格 token budget 下最大化 information density 并去除冗余。

来源：
https://aclanthology.org/2026.findings-acl.1052/

## 3.8 Persistence 应由“持续有效”决定，而不是固定奖励历史 evidence

NEST（ACL Industry 2026）使用 Nested Evidence Survival，强调在不同 retrieval scopes 下持续保持 salient 的 evidence，而不是固定给历史 evidence 加 bonus。

因此下一版应将：

\[
+\lambda_p persistence
\]

改为：

> 上一时刻 evidence 在当前 snapshot 中是否仍然具有实际边际贡献。

来源：
https://aclanthology.org/2026.acl-industry.35/

## 3.9 Fake-news LLM 工作也已经开始使用 salience + MMR 而非单一 relevance

ZoFia（ACL Findings 2026）在 fake-news detection 中采用 Salience-Calibrated MMR，说明在 misinformation/fake-news 场景里同时考虑 salience 与 redundancy/diversity 已进入最新工作。

来源：
https://aclanthology.org/2026.findings-acl.1083/

## 3.10 Knowledge selector 的价值取决于 downstream reader

EMNLP Findings 2025 的分析显示，knowledge selection 是否能转化为最终 performance improvement，与 downstream generator 的能力、任务复杂度和数据集特性有关。

因此：

> Proxy 上好的 selector 不应自动等价为 Qwen 上好的 selector。

来源：
https://aclanthology.org/2025.findings-emnlp.218/

---

# 4. 候选 Dynamic V2 方案与审批

## Candidate A — Scale-Normalized Additive Reranking

形式：

\[
d_i=z(u_i)+\alpha z(novelty_i)+\beta persistence_i
\]

或者使用 rank percentile / MMR。

优点是实现成本最低，可以解决 E3 中 utility 与 dynamic bonus 数值尺度不匹配的问题。

缺陷是它仍然是：

```text
individual node score
→ ranking
→ token packing
```

而当前诊断已经证明 ranking change 不一定转化为 set change，set change 也不一定转化为 prediction change。

**审批：不批准作为主方案。**

可保留为 Dynamic V2 的简单 baseline。

## Candidate B — Learnable Temporal Memory Network

借鉴 SWAM / FGDGNN：

```text
previous memory
+
new nodes
→ attention / GRU / Transformer
→ updated temporal representation
```

优点是动态表达能力更强。

但它会把当前研究变成新的 dynamic graph model，重新增加大量训练参数、工程和消融，而且与 TC-DSCR 原本“为 LLM 精炼社会证据”的贡献边界变弱。

**审批：不批准。**

## Candidate C — Marginal-Fidelity Temporal Set Refinement

简称：

# MF-TSR

该方案不再给 novelty/persistence 固定 bonus，而是直接优化：

> 当前 evidence set 是否能够在 token budget 下保留当前完整 causal graph 的判别信息。

**审批：批准。**

---

# 5. 审批后必须加入的优化约束

## 5.1 取消直接 novelty/persistence 加权

原来的：

\[
\lambda_n,\lambda_p
\]

全部取消。

Novelty 只保留为 diagnostic。

Persistence 改为 conditional survival。

## 5.2 Dynamic V2 必须直接优化 evidence set

最终操作单位：

```text
evidence set
```

允许：

```text
ADD
REMOVE
SWAP
```

而不是继续节点逐项加分。

## 5.3 使用当前 causal full encoder 作为 teacher，而不是 gold label

当前时刻：

\[
p_t^{full}
\]

来自当前 \(G_t\) 的 frozen Causal Social Encoder，因此严格 causal，不使用 future，也不使用 gold label。

## 5.4 Static Selector 保留

E2 已证明 Static Utility 有价值。

所以：

```text
Static Utility = initialization / prior
MF-TSR = conditional set refinement
```

## 5.5 Primary budget 固定 1024

Dynamic V1 同时搜索 lambda_n、lambda_p、budget，形成 36 组配置且 validation landscape 极平。

Dynamic V2 Primary experiment 固定：

\[
B=1024
\]

512/2048 只进入后续 ablation。

## 5.6 修复 no-candidate snapshot protocol

正式 V2 前必须删除：

```python
if not cand:
    continue
```

改成：

```text
candidate = empty
selected evidence = empty
z_sel = zero vector
q_t(empty) = Proxy([h_source ; 0])
该 event/cutoff 仍然进入 metrics
```

## 5.7 设置 Static fallback

如果 MF-TSR 最终 evidence set 在 teacher-fidelity objective 下不如 Static set：

```text
fallback to Static
```

## 5.8 必须经过不同 reader 的验证

即使 Proxy 上通过，最终仍必须用 frozen Qwen3-8B 验证 selector transfer。

---

# 6. MF-TSR 最终方法定义

## 6.1 Snapshot 表示

在时刻 \(t\)：

\[
G_t=(V_t,E_t)
\]

Causal Social Encoder：

\[
(H_t,g_t,p_t^{full})=Enc(G_t)
\]

其中 \(H_t=\{h_i^t\}\) 为节点表示，\(p_t^{full}\) 为当前完整 causal snapshot 的分类分布。

## 6.2 Static Utility

保持 corrected E2：

\[
u_i^t=
StaticSelector(
h_i^t,
g_t,
r_i,
struct_i
)
\]

Static baseline：

\[
S_t^{static}=PackByUtility(u^t,B)
\]

---

# 7. Set-Level Fidelity Objective

对于任意 evidence set：

\[
S\subseteq V_t\setminus\{source\}
\]

Proxy distribution：

\[
q_t(S)=softmax(Proxy([h_{source}^t;z_S]))
\]

其中：

\[
z_S=
\begin{cases}
\frac{1}{|S|}\sum_{i\in S}h_i^t,& |S|>0\\
0,& |S|=0
\end{cases}
\]

定义：

\[
D_t(S)=KL(stopgrad(p_t^{full})\parallel q_t(S))
\]

MF-TSR 的目标：

\[
\boxed{\min_S D_t(S)}
\]

subject to：

\[
Cost(S)\le B
\]

其中：

\[
Cost(S)=\sum_{i\in S}TokenCost(EvidenceUnit_i)
\]

---

# 8. 为什么该 Objective 合理

E2 的 Proxy 本来就使用：

\[
KL(p_{full}\parallel p_{selected})
\]

训练。

因此 MF-TSR 没有引入新的陌生 surrogate，而是把原来的 single-node utility 推广到 selected evidence set。

---

# 9. Temporal Memory：Evidence Survival

上一时刻：

\[
M_{t-1}
\]

只作为当前 set refinement 的 warm start，不再自动获得 persistence bonus。

对历史 evidence \(j\)：

\[
R_j^t=D_t(S\setminus\{j\})-D_t(S)
\]

如果：

\[
R_j^t>0
\]

说明删除它会使当前 fidelity 变差，因此应该保留。

如果：

\[
R_j^t\le0
\]

说明它已经无效或有害，可被移除。

---

# 10. 新 Evidence 的 Marginal Gain

对于当前未选 evidence \(i\)：

\[
A_i^t=D_t(S)-D_t(S\cup\{i\})
\]

若：

\[
A_i^t>0
\]

说明 evidence 对当前完整传播信息具有真实边际贡献。

这替代 novelty 作为直接 selection criterion。

---

# 11. Boundary-Aware Evidence Swap

如果 token budget 已满：

\[
S'=S\setminus\{j\}\cup\{i\}
\]

若：

\[
Cost(S')\le B
\]

定义：

\[
X_{i,j}^t=D_t(S)-D_t(S')
\]

只有：

\[
X_{i,j}^t>0
\]

才接受替换。

这一操作直接解决：

```text
ranking changed
but
budget packing masks the change
```

---

# 12. Relative Improvement 与 Temporal Hysteresis

引入唯一 Dynamic V2 控制参数：

\[
\epsilon
\]

定义：

\[
Gain_{rel}
=
\frac{D_t(S)-D_t(S')}{\max(D_t(S),10^{-6})}
\]

只有：

\[
Gain_{rel}\ge\epsilon
\]

才接受 ADD / REMOVE / SWAP。

Primary validation 只搜索：

\[
\epsilon\in\{0,0.01,0.05\}
\]

它同时起到：

- sufficiency threshold；
- temporal hysteresis；
- 防止 meaningless churn；

的作用。

---

# 13. MF-TSR 完整流程

```text
M_previous = empty

for t in [5m, 15m, 30m, 1h, 3h, 6h]:

    build causal G_t

    obtain:
        node representations
        full causal prediction p_full
        static utility u
        token costs

    if no reply candidates:
        S_t = empty
        predict with [h_source ; zero]
        M_previous = empty
        continue

    S_static = PackByUtility(u, B)

    if all evidence fits B:
        S_t = all current evidence
        M_previous = S_t
        continue

    S_memory = feasible(M_previous)

    fill remaining budget in S_memory
        using descending Static Utility

    choose the better initialization between:
        S_static
        S_memory
    according to D_t(S)

    repeat:
        evaluate REMOVE moves
        evaluate feasible ADD moves
        evaluate feasible SWAP moves

        choose move with largest relative fidelity gain

        if gain < epsilon:
            stop

        apply move

    if D(S) > D(S_static):
        S = S_static

    M_t = S
```

---

# 14. Tie-Break

若两个 move 的 fidelity gain 差异小于 \(10^{-6}\)：

1. 优先保留更多上一时刻 evidence；
2. 优先较低 token cost；
3. 优先较高 Static Utility；
4. 最后按 timestamp + original stable order。

Temporal stability 只作为“任务效用近似相同”时的 tie-break，而不是固定加分。

---

# 15. Novelty 在 V2 中的角色

Novelty 不再进入 selection score，但继续报告：

```text
Static selected novelty
MF-TSR selected novelty
newly admitted evidence novelty
```

如果 V2 在没有显式 novelty bonus 的情况下仍选到更新颖且有效的 evidence，说明这是真正 task-relevant novelty。

---

# 16. 计算复杂度

MF-TSR 不增加大型神经网络。

最重部分仍然是 Causal Social Encoder。

ADD / REMOVE / SWAP 只调用 frozen Proxy；Proxy 是小型 MLP，可以 batch。

当前最大节点已限制为 1021，selected evidence 通常远低于该规模，因此工程复杂度属于中等，不属于另起炉灶。

---

# 17. 代码复用关系

继续复用：

```text
PHEME / Ma-Weibo adapters
causal snapshot builder
384D semantic cache
1021D + 3D causal structural features
Random-init Causal Social Encoder
Corrected E2 Static Utility Selector
Proxy Classifier
Reply–Parent Evidence Unit
Qwen tokenizer
token accounting
fold protocol
leakage scanner
```

主要新增：

```text
marginal_set_refiner.py
set_fidelity.py
temporal_survival.py
```

---

# 18. Dynamic V2 前的 Protocol Correction

正式 V2 前先修复 no-candidate metric protocol，并重新计算：

- corrected E2 validation metrics；
- E3 V1 diagnostic metrics。

不需要重新训练 E1/E2 网络。

---

# 19. Dynamic V2 正式验证

Primary budget：

\[
B=1024
\]

唯一 grid：

\[
\epsilon\in\{0,0.01,0.05\}
\]

每 outer fold 只在 validation 选择 epsilon。

Test 完全冻结。

比较：

```text
Static Utility Selection
MF-TSR
```

主时间点：

```text
5m
15m
30m
1h
3h
6h
```

---

# 20. 新有效性 Gate

不再使用“只要 +0.0001 就算 positive”的宽松规则。

## 分类路径

如果：

\[
\Delta MacroF1\ge0.005
\]

且 held-out bootstrap 方向稳定，则认为 Dynamic V2 在分类上有 meaningful effect。

## 稳定性路径

或者：

\[
RelativeFlipReduction\ge10\%
\]

并且：

\[
\Delta MacroF1\ge-0.002
\]

则认为具有 temporal-stability value。

否则 Dynamic V2 不成立。

---

# 21. Mechanism Metrics

必须同时报告：

```text
Static/MF-TSR exact-match
Jaccard
ADD count
REMOVE count
SWAP count
memory survival rate
mean accepted relative gain
set distortion D_t(S)
Static vs MF-TSR KL
token utilization
prediction flip rate
```

---

# 22. Reader-Transfer Gate

即使 MF-TSR 在 Proxy 上通过，也不能直接宣称 LLM context refinement 有效。

最终必须冻结所有 selector/refiner 参数后，用 Qwen3-8B 比较：

```text
Static context
vs
MF-TSR context
```

Qwen 不参与 selection，也不参与 tuning。

---

# 23. 与导师意见的关系

该方案已经明显不同于“用 LLM 再把 Accuracy 提高一点”。

研究贡献可以表述为：

> 在信息逐步到达的真实社交传播过程中，研究如何在严格 temporal causality 和 token budget 下维护一个对当前传播判别信息具有最大边际价值的社会证据集合，并用该集合为 LLM 提供动态、可追溯的 grounding context。

它保留 UMER 的传播图基础和传统 rumor detection task continuity，但扩展到：

```text
evolving evidence
context selection
LLM grounding
token efficiency
temporal causality
context sufficiency
```

---

# 24. 最终贡献结构

### Contribution 1 — Temporally Causal Rumor Context Protocol

构造严格 \(G_t\)，只使用真实当前信息。

### Contribution 2 — Social Evidence Utility Modeling

基于 causal graph representation 学习 Static Utility。

### Contribution 3 — Marginal-Fidelity Temporal Set Refinement

将 dynamic context refinement 建模为：

\[
\text{budget-constrained temporal evidence-set optimization}
\]

通过 conditional survival + marginal ADD/REMOVE/SWAP 维护随传播演化的 evidence set。

### Contribution 4 — LLM Context Grounding

验证精炼社会 evidence set 是否能以更少、更有效的 context 支持 frozen LLM，并分析 temporal stability 与 token efficiency。

---

# 25. 主要风险

## Risk 1 — Proxy objective 过度自洽

MF-TSR 使用 Proxy 计算 set fidelity，因此 Proxy improvement 不保证 Qwen improvement。

解决：必须执行 frozen-reader transfer evaluation。

## Risk 2 — Full encoder teacher 自身可能错误

Teacher 只是 selection guidance，不是 gold truth。

最终 accuracy 仍由 test label 与 Qwen result 判断。

## Risk 3 — Local set search 计算增加

最大节点 1021，Proxy 很小，可 batch，不重新运行 LLM，因此可控。

## Risk 4 — PHEME selection pressure 较弱

PHEME 作为 low-selection-pressure dataset，Ma-Weibo 作为 high-selection-pressure dataset，两者共同解释方法适用边界。

---

# 26. 最终审批结论

| 项目 | 审批 |
|---|---|
| 继续调大 novelty / persistence 权重 | REJECT |
| z-score + additive Dynamic | REJECT AS MAIN |
| GRU/Transformer/SWAM 式新动态图网络 | REJECT FOR CURRENT SCOPE |
| 直接进入 E4 用 LLM 掩盖 Dynamic 问题 | REJECT |
| 保留 Causal Encoder | APPROVE |
| 保留 corrected Static Utility Selector | APPROVE |
| point-wise ranking → set-wise refinement | APPROVE |
| marginal teacher-fidelity objective | APPROVE |
| conditional evidence survival | APPROVE |
| ADD / REMOVE / SWAP under token budget | APPROVE |
| Static fallback | APPROVE |
| fixed 1024 primary budget | APPROVE |
| no-candidate protocol correction | MANDATORY |
| frozen Qwen reader-transfer validation | MANDATORY |

---

# 27. 最终方法冻结建议

\[
\boxed{
\text{TC-DSCR Dynamic V2}
=
\text{MF-TSR}
}
\]

全称：

> Marginal-Fidelity Temporal Set Refinement for Temporally Causal Social Context

核心公式：

\[
\boxed{
D_t(S)
=
KL(
p_t^{full}
\parallel
q_t(S)
)
}
\]

以及：

\[
\boxed{
S_t
=
\arg\min_{Cost(S)\le B}
D_t(S)
}
\]

实际通过：

```text
previous evidence survival
+
new evidence marginal addition
+
budget-aware replacement
+
static fallback
```

近似求解。

---

# Sources

1. Yirong Zeng et al. “Exploring Large Language Models for Effective Rumor Detection on Social Media.” NAACL 2025. https://aclanthology.org/2025.naacl-long.128/

2. Mei Guo et al. “FGDGNN: Fine-Grained Dynamic Graph Neural Network for Rumor Detection on Social Media.” Findings ACL 2025. https://aclanthology.org/2025.findings-acl.296/

3. Mei Guo et al. “SWAM: Adaptive Sliding Window and Memory-Augmented Attention Model for Rumor Detection.” EMNLP 2025. https://aclanthology.org/2025.emnlp-main.729/

4. Cheng Xu et al. “LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection.” ACL 2026. https://aclanthology.org/2026.acl-long.546/

5. Dahyun Lee et al. “Shifting from Ranking to Set Selection for Retrieval Augmented Generation.” ACL 2025. https://aclanthology.org/2025.acl-long.861/

6. Changle Qu et al. “Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation.” Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.511/

7. Yeonseok Jeong et al. “ECoRAG: Evidentiality-guided Compression for Long Context RAG.” Findings ACL 2025. https://aclanthology.org/2025.findings-acl.1365/

8. Shijia Xu et al. “Self-Correcting RAG: Enhancing Faithfulness via MMKP Context Selection and NLI-Guided MCTS.” Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1052/

9. Akshay Verma et al. “NEST: Nested Evidence Survival for Retrieval.” ACL 2026 Industry Track. https://aclanthology.org/2026.acl-industry.35/

10. Yiqiao Jin et al. “SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression.” ACL 2026. https://aclanthology.org/2026.acl-long.661/

11. Lvhua Wu et al. “ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction.” Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1083/

12. Xiangci Li, Jessica Ouyang. “How Does Knowledge Selection Help Retrieval Augmented Generation?” Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.218/
