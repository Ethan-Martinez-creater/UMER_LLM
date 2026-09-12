# TC-DSCR Dynamic V3：Evidence Sufficiency 方法调研、方案审批与最终设计

## 1. 审批结论

基于当前 UMER_LLM 仓库已经完成的 E1 / Corrected E2 / Dynamic V1 / MF-TSR Dynamic V2 实验，以及 2025–2026 年 ACL、NAACL、EMNLP、EACL 等关于 evidence sufficiency、context compression、reader-aware retrieval 和 temporal misinformation reasoning 的最新工作，本轮审批结论如下：

**不再继续优化 `KL(full-encoder || selected-proxy)`，也不再回到 novelty / persistence 的逐节点加权。**

下一版正式建议冻结为：

# MS-TSR — Minimal-Sufficient Temporal Set Refinement
## 最小充分时序社会证据集合精炼

其研究目标不再是：

> Dynamic selection 是否能够再提高 0.x 个点的 rumor detection accuracy？

而是：

> 在严格 temporal causality 下，能否维护一个对当前 rumor judgment 已经“足够”的最小社会证据集合，并在传播继续演化时只在必要时更新它，从而降低 LLM 上下文冗余、干扰与推理成本，同时保持检测能力。

核心优化问题从：

\[
\min_S KL(p_t^{full}\parallel q_t(S))
\]

改为：

\[
\boxed{
\min_S Cost(S)
}
\]

subject to：

\[
\boxed{
S \text{ 对当前判别已经充分}
}
\]

以及：

\[
Cost(S)\le B.
\]

这一本质变化直接来自 Dynamic V2 的失败诊断，而不是单纯更换一个 scoring function。

---

# 2. 当前实验给出的确定事实

最新 MF-TSR validation 已经完成 protocol correction，并纳入所有 no-candidate snapshots。

### No-candidate protocol 修正

PHEME 5min validation snapshot 中约 35.5% 尚无 reply；Ma-Weibo 约 12.7%。这些 snapshot 现在统一执行：

```text
selected evidence = empty
z_sel = zero vector
proxy still produces prediction
```

因此后续所有 V3 实验必须继续使用 all-event protocol。

### Corrected E2 仍然成立

Protocol 修正后：

PHEME Static mean-primary Macro-F1：

\[
0.8567
\]

Ma-Weibo Static：

\[
0.9339
\]

Static social evidence refinement 仍然是当前最可靠的 evidence-selection baseline。

### Dynamic V1 的失败

`utility + novelty + persistence` 几乎不改变最终 evidence set，held-out test 没有真实收益。

### Dynamic V2 的新发现

MF-TSR 确实强烈改变 evidence set：

- PHEME exact-match ≈ 0.29；
- Ma-Weibo exact-match ≈ 0.09。

因此 set-wise refinement 本身已经真正“工作”。

但结果：

| Dataset | Static MF1 | MF-TSR MF1 | Delta | Static tokens | MF-TSR tokens |
|---|---:|---:|---:|---:|---:|
| PHEME | 0.8567 | 0.8596 | +0.0029 | 472.8 | 178.7 |
| Ma-Weibo | 0.9338 | 0.9309 | -0.0029 | 855.6 | 516.3 |

同时 teacher distortion 明显下降。

这说明：

\[
\boxed{
Teacher\ Fidelity \neq Discriminative\ Sufficiency
}
\]

更重要的是，MF-TSR 的搜索被 REMOVE 极端主导：

```text
PHEME:
REMOVE = 173357
ADD = 20
SWAP = 15678

Ma-Weibo:
REMOVE = 123808
ADD = 150
SWAP = 46086
```

因此原 objective 实际学到的是：

> “删除哪些节点以后，Proxy 还能模仿 full encoder？”

而不是：

> “哪些 evidence 对真实 downstream reader 已经足够？”

不过 MF-TSR 同时揭示了一个非常重要的研究机会：

> PHEME 可以减少约 62% context tokens 且分类性能没有下降；Ma-Weibo 可以减少约 40% tokens，而 Macro-F1 只下降约 0.29 percentage point。

因此当前最有实验支撑的下一问题不是继续最大化 classification gain，而是：

\[
\boxed{
Minimal\ Sufficient\ Social\ Context
}
\]

---

# 3. 2025–2026 研究趋势

## 3.1 LLM rumor detection 已经明确暴露“大量 social context 会伤害 reasoning”

Zeng et al., NAACL 2025 发现，在 social-media rumor detection 中，大规模、结构化的 social context 会降低 LLM 推理效果，而 moderate/refined context 更有效。

这说明：

\[
more\ context \neq better\ reasoning
\]

TC-DSCR 的研究重点应继续放在 context refinement，而不是把完整传播图直接转成 prompt。

Source:
Yirong Zeng et al. “Exploring Large Language Models for Effective Rumor Detection on Social Media.” NAACL 2025.
https://aclanthology.org/2025.naacl-long.128/

---

## 3.2 最新 RAG 已经从 top-k 转向“证据是否足够”

S2G-RAG（ACL 2026）显式引入 `Sufficiency Judge`：

```text
current evidence
→ sufficient?
→ yes: stop
→ no: identify evidence gap and continue
```

其核心不是继续提高某个 relevance score，而是判断当前 evidence memory 是否已经支持回答，同时避免无止境积累 redundant / distracting context。

Source:
Minghan Li et al. “S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA.” ACL 2026.
https://aclanthology.org/2026.acl-long.1185/

---

## 3.3 Context compression 正在强调 Minimal Sufficient Information

ACC-RAG（Findings EMNLP 2025）针对 fixed compression rate 的问题，动态选择最小充分信息，并在保持或提高 accuracy 的情况下显著降低推理成本。

这与 MF-TSR 观察到的“传播上下文存在大量冗余”高度一致。

Source:
Shuyu Guo et al. “Enhancing RAG Efficiency with Adaptive Context Compression.” Findings EMNLP 2025.
https://aclanthology.org/2025.findings-emnlp.1307/

---

## 3.4 Reader utility 比 retrieval relevance 更重要

Uplift-RAG（Findings EMNLP 2025）指出 document relevance 与 downstream utility 不等价，并使用 marginal benefit / uplift 判断 evidence 对 reader 的实际价值。

Current MF-TSR 的问题正是：

```text
fidelity objective improved
but downstream classification did not
```

因此下一版必须从 teacher similarity 转为 reader decision utility。

Source:
Changle Qu et al. “Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation.” Findings EMNLP 2025.
https://aclanthology.org/2025.findings-emnlp.511/

---

## 3.5 2026 年 retriever 研究进一步转向 Answer Sufficiency

ARK（ACL 2026）指出传统 query-document similarity 与 downstream answer objective 不一致，并直接通过“evidence 是否足以生成正确答案”定义 positive chunks。

对 TC-DSCR 的启示是：

> evidence 是否应该保留，应由它是否保持 rumor judgment 能力决定，而不是它是否与 full graph representation 足够相似。

Source:
Jiawei Zhou et al. “ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning.” ACL 2026.
https://aclanthology.org/2026.acl-long.371/

---

## 3.6 Noise / redundancy 与 hallucination 直接相关

ReflectiveRAG（EACL Industry 2026）指出大量 irrelevant/redundant context 会导致 grounding 下降、hallucination 和 poor reasoning，并显式进行 sufficiency assessment 与 noise removal。

PrismRAG（EMNLP Industry 2025）也指出 semi-relevant distractors 会降低 RAG factuality。

这使“减少 social evidence 噪声”不仅是效率问题，也可以作为后续 LLM hallucination / grounding 实验的重要切入点。

Sources:
https://aclanthology.org/2026.eacl-industry.27/
https://aclanthology.org/2025.emnlp-industry.53/

---

## 3.7 Knowledge selection 效果取决于 downstream reader

Li & Ouyang（Findings EMNLP 2025）系统分析发现，knowledge selection 对最终 RAG performance 的作用取决于 reader 能力、任务和数据分布。

因此：

\[
Proxy\ sufficiency
\not\Rightarrow
LLM\ sufficiency
\]

必须增加 reader-transfer checkpoint，而不能再像 V1/V2 一样，只在 Proxy 上通过后直接认为方法成立。

Source:
Xiangci Li, Jessica Ouyang. “How Does Knowledge Selection Help Retrieval Augmented Generation?” Findings EMNLP 2025.
https://aclanthology.org/2025.findings-emnlp.218/

---

## 3.8 Fake-news LLM 最新研究正在主动处理 hallucination 与 evidence gap

ZoFia（Findings ACL 2026）直接将 LLM 的 factual hallucination、knowledge cutoff 和 confirmation bias 作为 fake-news detection 的核心问题，通过 salience-calibrated retrieval 与多模型 verification 缓解。

LiveFact（ACL 2026）进一步强调 evolving / incomplete evidence 下的 epistemic humility：早期信息不足时，一个可靠模型不应过度确信。

这为 TC-DSCR 后续扩展到：

```text
sufficient evidence
vs
insufficient evidence
```

提供了最新 misinformation 文献依据。

Sources:
https://aclanthology.org/2026.findings-acl.1083/
https://aclanthology.org/2026.acl-long.546/

---

# 4. Dynamic V3 候选方案

## Candidate A — 继续改 MF-TSR teacher objective

例如：

```text
KL
→ JS divergence
→ cosine representation distance
→ ensemble teacher distance
```

### 优点

修改成本低。

### 缺点

仍然是在优化：

```text
selected context ≈ teacher
```

而当前实验已经证明：

```text
teacher fidelity improvement
does not imply reader utility
```

换 divergence 形式并不能解决根因。

### 审批

**REJECT。**

---

## Candidate B — 直接用 Qwen3-8B 对每个 candidate set 打分

形式：

```text
candidate evidence set
→ Qwen3-8B
→ rumor confidence / sufficiency score
→ set search
```

### 优点

直接 reader-aware。

### 缺点

1. 每个 ADD/REMOVE/SWAP 都调用 LLM，计算量无法接受；
2. selection 与最终 reader 高度耦合；
3. Qwen 自身 hallucination 会反过来影响 selector；
4. 研究会从“利用小模型为 LLM 提供 context”变成“LLM 自己循环挑 context”；
5. 很难保持当前项目的轻量和可复现性。

### 审批

**REJECT AS MAIN METHOD。**

Frozen Qwen 只用于后续 reader-transfer evaluation。

---

## Candidate C — Minimal-Sufficient Reader-Aware Temporal Refinement

简称：

# MS-TSR

核心思想：

> 不要求 selected set 重建 full-graph teacher 的完整概率分布；只要求它保留 Static Selector 已经验证过的 rumor judgment，并保持足够 decision margin。满足判别充分性以后，优先减少 token 与 temporal churn。

### 审批

**APPROVE。**

---

# 5. 为什么 Reference 应从 Full Encoder 改为 Static Context

MF-TSR 使用：

\[
p_t^{full}
\]

作为唯一 teacher。

但实际要交给 LLM 的并不是 full graph representation，而是一个经过 token budget 限制的 textual evidence set。

Corrected E2 已经证明：

\[
S_t^{static}
\]

是目前最可靠、最接近最终应用形态的 evidence baseline。

因此 V3 reference 改为：

\[
q_t^{static}
=
Proxy(S_t^{static})
\]

而 full encoder 不再作为优化 target，只作为第二个独立 decision view。

这样可以避免：

> 为了重建 full encoder 的 probability distribution，把大量 evidence 删除到一个“Proxy mimic set”。

---

# 6. Dual-View Consensus Gate

当前 snapshot \(G_t\)：

Full causal encoder：

\[
p_t^{full}
\]

Static set reader：

\[
q_t^{static}
\]

定义：

\[
\hat y_t^{full}
=
\arg\max p_t^{full}
\]

\[
\hat y_t^{static}
=
\arg\max q_t^{static}
\]

只有：

\[
\boxed{
\hat y_t^{full}
=
\hat y_t^{static}
}
\]

时，允许执行 context compression / refinement。

如果两者不一致：

```text
uncertain snapshot
→ do not compress
→ S_t = S_static
```

原因：

当两个已有判别视角本身都不能达成一致时，不应继续 aggressively 删除 evidence。

该 gate 不需要额外训练参数。

---

# 7. Reader Decision Margin

对于 frozen Proxy logits：

\[
l(S)=[l_0(S),l_1(S)]
\]

以 Static reference label：

\[
y_t^{ref}=\hat y_t^{static}
\]

定义 signed logit margin：

\[
m_t(S)
=
l_{y_t^{ref}}(S)
-
l_{1-y_t^{ref}}(S)
\]

Static margin：

\[
m_t^{static}
=
m_t(S_t^{static})
\]

Static reference 本身应满足：

\[
m_t^{static}>0.
\]

---

# 8. Evidence Sufficiency Definition

对于 candidate set \(S\)，定义：

\[
\boxed{
Sufficient_t(S;\alpha)
}
\]

当且仅当：

### Condition 1 — Decision preservation

\[
\arg\max q_t(S)
=
y_t^{ref}
\]

### Condition 2 — Margin retention

\[
m_t(S)
\ge
\alpha \cdot m_t^{static}
\]

其中：

\[
\alpha
\in
\{0.80,\ 0.90,\ 0.95,\ 1.00\}.
\]

这四个值是 V3 唯一 validation grid。

注意：

V3 不再追求：

\[
q_t(S)\approx q_t^{static}
\]

或：

\[
q_t(S)\approx p_t^{full}.
\]

只要求：

> 相同 rumor judgment + 足够大的 decision margin。

---

# 9. MS-TSR 的正式优化问题

在通过 Dual-View Consensus Gate 的 snapshot 上：

\[
\boxed{
\min_S Cost(S)
}
\]

subject to：

\[
Sufficient_t(S;\alpha)=True
\]

以及：

\[
Cost(S)\le1024.
\]

在多个同样 token-efficient 的 sufficient sets 中，采用 lexicographic tie-break：

1. 最大化与上一时刻 \(M_{t-1}\) 的 overlap；
2. 最大化 Static Utility 总和；
3. timestamp / stable order。

因此 temporal stability 不再作为一个加权 loss，而是在相同 sufficiency 下作为确定性偏好。

---

# 10. Candidate Pool

当前候选 reply nodes：

\[
C_t.
\]

构造：

```text
P_t =
current Static selected nodes
∪
previous MS-TSR memory nodes still visible
∪
top-64 current Static Utility candidates
```

source 永远不进入。

这样：

- Static set 永远在 candidate pool 中；
- previous evidence 可以 survival；
- 新 evidence 可以替换旧 evidence；
- 搜索空间仍然可控。

---

# 11. Forward Sufficiency Construction

每个 snapshot 不再从一个已经很大的 set 开始不断 REMOVE。

初始化：

\[
S=\emptyset.
\]

如果 Dual-View 不一致：

```text
S = S_static
STOP
```

否则逐步 ADD。

对于每个：

\[
i\in P_t\setminus S
\]

计算：

\[
\Delta m_i
=
m_t(S\cup\{i\})-m_t(S)
\]

定义 reader-aware efficiency：

\[
\boxed{
Gain_i
=
\frac{\Delta m_i}{TokenCost(i)}
}
\]

选择 Gain 最大的节点加入。

Tie-break：

1. node 在 \(M_{t-1}\) 中；
2. Static Utility 更高；
3. token cost 更低；
4. stable order。

一旦：

\[
Sufficient_t(S;\alpha)=True
\]

立即停止 ADD。

因此 V3 不会像 MF-TSR 那样为了继续降低 KL 而无止境改变 set。

---

# 12. Backward Redundancy Removal

达到 sufficiency 后进行一次 backward pruning。

对于：

\[
j\in S
\]

若：

\[
S\setminus\{j\}
\]

仍然满足：

\[
Sufficient_t
\]

则该 evidence 当前是 redundant。

在所有可删除 evidence 中，优先：

1. token saving 最大；
2. 非上一时刻 memory node；
3. Static Utility 最低；
4. stable order。

重复直到任何删除都会破坏 sufficiency。

得到：

\[
S_t^{MS}.
\]

---

# 13. 为什么 Primary V3 不使用 SWAP

MF-TSR 已经产生大量 SWAP，但并没有改善 Ma-Weibo。

MS-TSR 的 forward construction 每一步都重新在整个 pool 中计算 marginal decision-margin utility，因此新 evidence 从一开始就可以竞争进入集合。

再增加 SWAP：

- 增加实现复杂度；
- 增加 churn；
- 很难区分收益来自 sufficiency 还是 local-search trick。

因此 Primary V3 固定：

```text
ADD
+
REMOVE
```

不使用 SWAP。

SWAP 最多作为后续 ablation。

---

# 14. Temporal Memory

最终：

\[
M_t=S_t^{MS}.
\]

下一时刻它只作为：

```text
candidate-pool augmentation
+
tie-break preference
```

而不是：

```text
persistence bonus
```

这使 temporal continuity 的语义变成：

> 如果历史 evidence 在当前仍然足够有效，则在等价方案中优先保留。

而不是：

> 历史 evidence 天生更重要。

---

# 15. No-Candidate Snapshot

继续使用已经修复的 all-event protocol：

```text
candidate_count = 0
→ S_static = empty
→ S_MS = empty
→ z_sel = zero vector
→ prediction still generated
→ M_t = empty
```

不得再跳过。

---

# 16. Validation 参数选择

Primary budget：

\[
B=1024
\]

唯一 grid：

\[
\alpha\in\{0.80,0.90,0.95,1.00\}.
\]

每：

```text
dataset × outer fold
```

只使用 validation events。

三个 seeds 聚合。

### 参数选择目标不是最大化 MF1

每个 alpha 先检查 non-inferiority：

\[
\Delta MacroF1
=
F1_{MS}-F1_{Static}
\ge
-0.002.
\]

同时：

\[
RelativeFlipIncrease
\le
5\%.
\]

在满足上述条件的 alpha 中：

\[
\boxed{
选择 token reduction 最大者
}
\]

若没有 alpha 满足：

```text
fold = V3_NOT_SUPPORTED
```

不得放宽 gate。

---

# 17. Dataset-level V3 Gate

V3 的研究目的已经从 accuracy improvement 转为：

> classification-preserving context compression。

因此正式 gate 应改变。

一个 dataset PASS 需要：

### A. Classification Non-Inferiority

\[
\boxed{
\Delta MacroF1\ge -0.002
}
\]

### B. Context Reduction

\[
\boxed{
TokenReduction\ge30\%
}
\]

### C. Temporal Stability

要求：

\[
RelativeFlipIncrease\le5\%.
\]

三者同时满足：

```text
MS_TSR_PROXY_PASS
```

不再要求：

```text
+0.005 Macro-F1
```

因为这与当前研究问题不一致。

---

# 18. 必须报告的新机制指标

除了：

```text
Accuracy
Macro-F1
Weighted-F1
Rumor-F1
FlipRate
tokens
```

必须报告：

### Sufficiency coverage

```text
dual_view_agreement_rate
compression_attempt_rate
sufficiency_success_rate
static_fallback_rate
```

### Compression

```text
mean token reduction
median token reduction
P25/P75
selected unit reduction
```

### Temporal

```text
memory survival rate
set Jaccard between adjacent cutoffs
context churn
```

### Reader Margin

```text
static margin
MS-TSR margin
margin retention ratio
```

### Search

```text
mean ADD count
mean REMOVE count
max candidate-pool size
```

---

# 19. 必须增加 Margin-Stratified Analysis

将 Static margin 按 validation distribution 固定分成：

```text
Low confidence:
bottom 25%

Medium:
25–75%

High:
top 25%
```

分别报告：

```text
token reduction
Delta Macro-F1
fallback rate
```

目的：

验证 aggressive compression 是否主要应该发生在 high-confidence snapshots。

如果 low-margin snapshot 经常造成 classification degradation，则下一步可以基于此研究 uncertainty-aware abstention，而不是继续修改 selector。

---

# 20. Reader-Transfer Checkpoint

这是 V3 相比 V1/V2 必须新增的步骤。

即使：

```text
MS_TSR_PROXY_PASS
```

也不能直接进入 held-out test。

先执行：

# V3-B Frozen Reader Transfer Pilot

只使用 validation events。

Frozen reader：

```text
Qwen3-8B
```

不得 fine-tune。

不得让 Qwen 参与 set selection。

---

# 21. Reader Pilot 采样

为限制成本：

每个 dataset 固定：

```text
300 event-cutoff samples
```

总计：

```text
600
```

使用固定 seed：

```text
3090
```

按以下维度 stratified sampling：

```text
5m / 15m / 30m / 1h / 3h / 6h
low / medium / high selection pressure
```

采样列表必须在第一次 Qwen inference 前写入 manifest 并冻结。

---

# 22. Reader Pilot 输入

对同一个 event-cutoff：

```text
Arm A:
Static social evidence context

Arm B:
MS-TSR minimal sufficient context
```

完全相同：

```text
system prompt
claim/source post
model
temperature
max output
label space
```

唯一差异：

```text
social evidence context
```

---

# 23. Reader Pilot Prompt 原则

Qwen 输出固定 schema：

```json
{
  "label": "RUMOR | NON_RUMOR",
  "confidence": 0.0,
  "evidence_ids": [],
  "reason": "..."
}
```

prompt 必须要求：

```text
只能依据 source post + supplied social evidence 进行判断；
如果 evidence 不支持某项事实，不得自行补充外部事实；
evidence_ids 只能引用提供的 ID。
```

这一设计为后续 hallucination / grounding evaluation 留下接口。

---

# 24. Reader-Transfer Gate

V3-B 不是为了证明 Qwen accuracy 显著提高，而是测试：

> Proxy-defined sufficiency 能否 transfer 到真实 LLM reader。

要求：

### LLM non-inferiority

\[
\Delta MacroF1_{LLM}
\ge
-0.005.
\]

### Context reduction

\[
TokenReduction\ge30\%.
\]

### Prediction disagreement

报告：

```text
Static-Qwen vs MS-Qwen disagreement rate
```

### Evidence citation validity

\[
\frac{
cited\ evidence\ IDs\ that\ exist\ in\ provided\ context
}{
all\ cited\ evidence\ IDs
}
\]

应接近 1。

如果 Proxy pass 但 Reader pilot 明显失败：

```text
PROXY_READER_TRANSFER_FAIL
```

不得进入 held-out test。

---

# 25. 为什么 Reader Pilot 很重要

当前两轮失败已经说明：

```text
selector objective
→ proxy behavior
```

与：

```text
真实 downstream value
```

并非同一件事。

最新 RAG 研究也表明 knowledge selection 的收益与 generator / reader 能力密切相关。

因此 reader-transfer 不再是最终论文才做的附加实验，而是方法有效性的中间 gate。

---

# 26. 与 LLM Hallucination 问题的连接

MS-TSR 本身不声称直接“消灭 hallucination”。

更准确的研究假设是：

\[
\boxed{
redundant/noisy social context
\rightarrow
higher reasoning burden and distractor risk
}
\]

MS-TSR 通过 minimal sufficient context 减少这种风险。

后续 E4 可以进一步增加：

```text
distractor robustness
unsupported evidence citation
abstention / insufficient-evidence behavior
```

特别是 LiveFact 2026 所强调的：

> early evidence insufficient 时，模型需要 epistemic humility。

因此最终课题可以从传统“提高 rumor accuracy”扩展到：

> 在 evolving social context 下，为 LLM 提供足够但不过量的 grounding evidence，并研究其对推理稳定性、上下文效率与 hallucination resilience 的作用。

---

# 27. 候选方案审批表

| Candidate | 结论 | 理由 |
|---|---|---|
| 换 KL 为 JS / cosine 后继续 MF-TSR | REJECT | 仍然是 teacher imitation |
| 调整 REMOVE / SWAP 权重 | REJECT | 属于 symptom-level patch |
| 直接 Qwen-in-the-loop selection | REJECT AS MAIN | 成本高、循环依赖、hallucination 会污染 selector |
| 新训练 GRU/Transformer temporal selector | REJECT | 偏离当前项目、工作量过大 |
| Minimal-Sufficient Set Refinement | APPROVE | 与实验诊断和最新 sufficiency 文献一致 |
| Dual-view consensus gate | APPROVE | uncertainty 时避免 aggressive compression |
| Decision-margin sufficiency | APPROVE | 直接约束 reader decision，而非 teacher distribution |
| Min-token objective | APPROVE | 利用 MF-TSR 已发现的强冗余信号 |
| Temporal continuity as tie-break | APPROVE | 避免 persistence bonus 的错误假设 |
| Validation-only Frozen Qwen reader pilot | MANDATORY | 防止第三次 proxy-reader objective mismatch |

---

# 28. 最终冻结方法

推荐正式冻结：

# MS-TSR
## Minimal-Sufficient Temporal Set Refinement

整体结构：

\[
G_t
\rightarrow
Causal\ Encoder
\rightarrow
Static\ Utility
\rightarrow
S_t^{static}
\]

然后：

\[
(p_t^{full},q_t^{static})
\rightarrow
DualViewConsensus
\]

如果 disagreement：

\[
S_t^{MS}=S_t^{static}
\]

如果 agreement：

\[
P_t
=
S_t^{static}
\cup
M_{t-1}
\cup
Top64(u_t)
\]

随后：

\[
\emptyset
\xrightarrow{reader\ margin\ gain/token}
S_t
\]

直到：

\[
\arg\max q_t(S_t)=y_t^{ref}
\]

且：

\[
m_t(S_t)\ge\alpha m_t^{static}.
\]

最后：

```text
backward remove redundant evidence
→ minimal sufficient set
→ update temporal memory
```

其数学目标：

\[
\boxed{
\min_{S\subseteq P_t} Cost(S)
}
\]

s.t.

\[
\boxed{
\arg\max q_t(S)=y_t^{ref}
}
\]

\[
\boxed{
m_t(S)\ge\alpha m_t^{static}
}
\]

\[
\boxed{
Cost(S)\le1024
}
\]

---

# 29. 预期研究贡献重新定义

如果 MS-TSR 最终成立，TC-DSCR 的主要贡献不应再写成：

> Dynamic selector improves rumor-detection accuracy.

而应写成：

### Contribution 1 — Temporally Causal Social Context Protocol

真实时间点构造因果传播 snapshot，严格排除 future leakage。

### Contribution 2 — Static Social Evidence Utility

利用 propagation representation 学习社会证据 utility。

### Contribution 3 — Minimal-Sufficient Temporal Context

首次将 social-context refinement 建模为：

\[
classification\text{-}preserving
+
budget\text{-}aware
+
temporally\ stable
\]

的最小充分证据集合问题。

### Contribution 4 — Reader Transfer

验证小型 graph/proxy reader 定义的 evidence sufficiency 是否能够迁移到 frozen LLM。

### Contribution 5 — LLM Grounding Efficiency / Robustness

研究减少冗余 social context 后：

```text
token cost
reasoning stability
distractor robustness
evidence grounding
```

如何变化。

这一叙事明显不同于传统：

```text
PHEME / Weibo 上再提高一点 Accuracy
```

并且仍然完整继承当前 UMER / TC-DSCR 工作，而不需要切换到完全不同的 LLM-generated fake-news 任务。

---

# 30. 下一步实验阶段

当前建议不要直接把 MS-TSR 交给执行智能体一次性跑到 test。

下一轮应固定为：

## V3-A — Implementation + Proxy Validation

只执行：

```text
MS-TSR implementation
alpha grid
all-event validation
proxy non-inferiority + context reduction gate
```

禁止 test。

如果：

```text
PHEME PASS
AND
Ma-Weibo PASS
```

或经过研究审批认为 PARTIAL 有明确价值，再进入：

## V3-B — Frozen Qwen Reader Transfer Pilot

仍然只使用 validation。

只有 reader-transfer 也通过，才允许：

## V3-C — Held-out Test

最后再进入完整 E4。

---

# References

1. Zeng, Y., Ding, X., Cai, B., Liu, T., Qin, B. “Exploring Large Language Models for Effective Rumor Detection on Social Media.” NAACL 2025. https://aclanthology.org/2025.naacl-long.128/

2. Li, M., Zou, J., Lv, X., Zhang, C., Zhou, G. “S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA.” ACL 2026. https://aclanthology.org/2026.acl-long.1185/

3. Guo, S., Zhang, S., Ren, Z. “Enhancing RAG Efficiency with Adaptive Context Compression.” Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.1307/

4. Qu, C. et al. “Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation.” Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.511/

5. Zhou, J., Ding, H., Jiang, H. “ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning.” ACL 2026. https://aclanthology.org/2026.acl-long.371/

6. Verma, A. et al. “ReflectiveRAG: Rethinking Adaptivity in Retrieval-Augmented Generation.” EACL Industry 2026. https://aclanthology.org/2026.eacl-industry.27/

7. Kachuee, M. et al. “PrismRAG: Boosting RAG Factuality with Distractor Resilience and Strategized Reasoning.” EMNLP Industry 2025. https://aclanthology.org/2025.emnlp-industry.53/

8. Li, X., Ouyang, J. “How Does Knowledge Selection Help Retrieval Augmented Generation?” Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.218/

9. Wu, L. et al. “ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction.” Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1083/

10. Xu, C. et al. “LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection.” ACL 2026. https://aclanthology.org/2026.acl-long.546/

11. Sun, J. et al. “AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning.” Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1399/

12. Repository result: Ethan-Martinez-creater/UMER_LLM, commit a0a0d31f2f30cfa78f6631758cbd1c4090f87387, TC-DSCR Dynamic V2 MF-TSR validation.
