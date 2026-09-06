# TC-DSCR V2 正式执行计划（执行智能体唯一有效版本）

> 项目名称：**TC-DSCR — Temporally Causal Dynamic Social-Context Refinement for LLM Rumor Detection**
>
> 核心研究问题：
>
> > **在每个真实时间点，只使用当时已经形成的传播图，从中动态选择“最值得 LLM 阅读的社会证据”，并在严格避免 future information leakage 的条件下完成谣言检测。**
>
> 本文档面向执行智能体，是 **TC-DSCR 当前唯一有效的正式执行计划**。
>
> 本文档已经吸收：
>
> - 第一轮 Preflight 验证结果；
> - 第二轮 Delta Preflight 验证结果；
> - 两轮研究审批结论；
> - 数据集从 Weibo22 回退并正式冻结为 Ma-Weibo；
> - 11D user/sentiment 从主路径移除；
> - 正式时间窗口冻结；
> - Primary / Stress / Diagnostic protocol 冻结；
> - Ma-Weibo cleaning/version drift 修正；
> - source timestamp、异常 parent、1021-node cap 等规则冻结。
>
> **旧版 `TC_DSCR_EXPERIMENT_EXECUTION_PLAN.md` 与 `TC_DSCR_PREFLIGHT_VERIFICATION.md` 不再作为执行依据。**
>
> 若本文档与旧文件存在冲突，以本文档为准。

---

# 0. 执行权限与阶段闸门

执行智能体必须严格遵守以下阶段。

## Stage A — Code Complete

当前只批准执行到：

```text
完整代码实现
+ unit tests
+ integration tests
+ tiny smoke run
+ 代码阶段报告
```

本阶段完成后必须：

```text
STOP
→ 提交 GitHub
→ 用户通知研究审批方
→ 等待代码审批
```

**本阶段禁止：**

- 正式五折实验；
- 全量 PHEME 训练；
- 全量 Ma-Weibo 训练；
- 完整 baseline 比较；
- 大规模 Qwen3-8B 推理；
- 正式 ablation；
- test-set 调参；
- 论文结论汇总。

## Stage B — Formal Experiments

只有在研究审批方对 Code Complete 明确批准后才允许启动。

## Stage C — Final Review

正式实验完成后再次 STOP，并提交完整结果等待第二次审批。

执行智能体不得跨阶段执行。

---

# 1. 本轮研究边界

TC-DSCR 只研究：

\[
G_t
\rightarrow
\text{Dynamic Social Evidence Refinement}
\rightarrow
C_t
\rightarrow
\text{Frozen LLM}
\rightarrow
\hat y_t
\]

其中：

- \(G_t\)：时刻 \(t\) 真实可见的传播图；
- \(C_t\)：在 token budget 下选择出的社会证据；
- LLM：只读取 source + selected evidence 并输出最终标签。

本项目不是：

- LLM feature fusion；
- LLM cognitive distillation；
- synthetic reply generation；
- external RAG；
- Agent / RL / POMDP；
- hallucination verifier；
- cross-domain adaptation；
- multimodal rumor detection；
- LLM-generated text detection；
- 新动态图 Transformer 设计；
- 传统 full-event Accuracy 刷榜。

---

# 2. 数据集正式冻结

正式数据集：

```text
PHEME
Ma-Weibo
```

Weibo22 永久移出本轮方案。

## 2.1 PHEME 角色

用于：

- 主开发数据集；
- dynamic refinement 主消融；
- legacy continuity；
- event-level causal evaluation；
- chronological stress test。

## 2.2 Ma-Weibo 角色

用于：

- 中文场景验证；
- 第二独立数据集；
- 更大规模传播图下验证 selector；
- event-internal causal dynamic evaluation。

Ma-Weibo global chronological split 只做 dataset diagnostic，不做正式模型性能比较。

---

# 3. 已冻结的数据事实

## 3.1 PHEME

已验证：

```text
source timestamp coverage = 100%
reply timestamp coverage = 100%
chronology violation = 0
```

parent 互斥统计：

```text
total replies = 98,929

resolved_parent
= 96,414
= 97.46%

missing_parent_field
= 1,017
= 1.03%

external_or_unresolved_parent
= 1,498
= 1.51%
```

处理规则：

```text
resolved
→ 正常构边

missing / external
→ 节点保留
→ 不人工连接到 source
→ evidence packer 中 parent = [UNAVAILABLE]
```

---

## 3.2 Ma-Weibo

已验证：

```text
events = 4,664
posts ≈ 3.8M

original_text coverage ≈ 99.9996%
node timestamp coverage = 100%
source timestamp coverage = 100%

t = absolute Unix seconds

parent resolution ≈ 99.9992%
cycle events = 0
multi-root events = 0
```

历史 frozen `.pt` 文本输入实际为：

```text
clean_text_weibo(original_text)
→ paraphrase-multilingual-MiniLM-L12-v2
```

TC-DSCR 必须显式恢复该流程。

禁止直接使用服务器上已经发生 version drift 的旧 preprocessing 行为。

---

# 4. 正式时间窗口冻结

Primary dynamic trajectory：

```text
SOURCE_ONLY
5m
15m
30m
1h
3h
6h
```

Late diagnostic：

```text
24h
```

## 4.1 SOURCE_ONLY

必须强制只保留 root/source。

禁止用：

```text
timestamp <= source_timestamp
```

来等价替代 SOURCE_ONLY，因为可能存在同 timestamp 节点。

## 4.2 24h

24h：

- 不参与 primary dynamic average；
- 不用于主 selector 超参数选择；
- 只作为 late-context diagnostic。

---

# 5. 数据划分协议正式冻结

## 5.1 Protocol A — Primary

```text
event-level stratified 5-fold
+
within-event strict temporal causality
```

关键规则：

> **先划 event fold，再构造 snapshot。**

禁止：

```text
先生成 snapshots
→ 再随机划 train/test
```

同一 event 的全部：

```text
SOURCE_ONLY
5m
15m
30m
1h
3h
6h
24h
```

必须永远属于同一个 outer fold。

---

## 5.2 Protocol B — PHEME Chronological Stress

PHEME 使用：

```text
70 / 15 / 15
```

按 source absolute time 划分。

该协议只标记为：

```text
temporal + emerging-topic stress test
```

不得将性能下降只归因于 temporal shift。

---

## 5.3 Protocol C — Ma-Weibo Chronological Diagnostic

Ma-Weibo chronological split 已发现：

```text
validation/test rumor ratio collapse to 0
```

因此：

```text
只报告 dataset diagnostic
不训练正式模型
不用于 baseline 比较
```

禁止人为改变时间边界来制造类别平衡。

---

# 6. TC-DSCR 主输入正式冻结

TC-DSCR 主路径：

\[
384D\ semantic
+
1024D\ causal\ structural
\]

即：

```text
node_feat = 384D multilingual semantic
struct_feat = 1021D adjacency signature + 3D structural summary
```

原 UMER：

```text
10D user
1D sentiment
```

不进入新主路径。

---

# 7. 384D semantic feature

两个数据集统一使用：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

embedding dimension：

```text
384
```

## PHEME

使用已恢复并验证的历史 text preprocessing。

## Ma-Weibo

必须：

```python
clean_text_weibo(original_text)
```

然后编码。

禁止：

- 使用 `text` 代替 `original_text`；
- 跳过 cleaning；
- 更换 embedding 模型；
- 使用 LLM embedding；
- 对 test set 单独重新拟合任何 text normalization。

所有 semantic embedding 必须缓存。

---

# 8. 时间基准正式冻结

新 TC-DSCR 概念定义：

\[
t_0 = source/root\ timestamp
\]

对于任意节点：

\[
elapsed_i = timestamp_i - t_0
\]

不再把：

```text
event_min_t
```

作为正式 causal task 的概念性 source time。

Legacy parity 测试可以继续使用旧 historical definition，但正式 TC-DSCR snapshot 必须基于真实 source/root timestamp。

---

# 9. 3D causal structural summary

固定为：

```text
norm_degree
norm_depth
norm_time
```

## 9.1 norm_degree

当前 snapshot 内部计算：

\[
rawDegree_i^t
=
\max(outDegree_i^t - 1,0)
\]

\[
normDegree_i^t=
\begin{cases}
rawDegree_i^t/\max_j rawDegree_j^t,
& \max_j rawDegree_j^t > 0 \\
0,& otherwise
\end{cases}
\]

禁止依赖：

- 240h graph；
- dataset-wide max；
- train-wide max。

## 9.2 norm_depth

固定：

\[
normDepth_i = depth_i/19
\]

19 是固定历史常数，不依赖未来信息。

如果 depth > 19：

```text
记录 overflow
不要 clip
不要自行改常数
```

## 9.3 norm_time

统一：

\[
timeBin_i
=
clip
\left(
\left\lfloor
\frac{elapsed_i}{1800}
\right\rfloor,
0,
479
\right)
\]

\[
normTime_i
=
timeBin_i/480
\]

---

# 10. 时间异常节点规则

## 10.1 timestamp < source_timestamp

节点标记：

```text
TEMPORAL_INVALID_NODE
```

规则：

```text
不进入任何 causal snapshot
保留在 audit log
不修改 timestamp
```

## 10.2 child timestamp < parent timestamp

规则：

```text
child 文本可按 child 自己的 timestamp 出现
parent edge 不提前出现
parent text 不提前提供
parent = [UNAVAILABLE]
```

禁止：

- 修改时间；
- 把 child 延迟到 parent 时间；
- 人工调整拓扑。

---

# 11. 1021D adjacency signature

必须从当前 \(G_t\) 重新生成。

不得：

```text
full graph signature
→ mask future nodes
```

节点顺序必须 deterministic：

优先：

```text
timestamp ascending
```

同 timestamp：

```text
original stable order
```

1021-node cap：

```text
max_nodes = 1021
```

若 snapshot 超过 1021：

```text
保留最早 1021 个节点
```

不能：

- 随机；
- top-degree；
- selector-aware；
- label-aware。

必须记录每个正式 cutoff：

```text
cap_hit_count
cap_hit_rate
mean_nodes_before_cap
P90_nodes_before_cap
max_nodes_before_cap
nodes_removed_total
```

这些统计在 Code Complete 审批时必须提交。

---

# 12. 新目录结构

新工作不得污染冻结 UMER。

创建：

```text
project/tcdscr/
├── __init__.py
├── config/
│   ├── schema.py
│   ├── pheme.yaml
│   └── maweibo.yaml
│
├── data/
│   ├── normalized_schema.py
│   ├── pheme_adapter.py
│   ├── maweibo_adapter.py
│   ├── text_cleaning.py
│   ├── semantic_encoder.py
│   ├── snapshot_builder.py
│   ├── structural_features.py
│   ├── adjacency_signature.py
│   ├── temporal_split.py
│   └── manifests.py
│
├── models/
│   ├── causal_social_encoder.py
│   ├── selector.py
│   ├── selector_proxy.py
│   └── evidence_memory.py
│
├── context/
│   ├── evidence_unit.py
│   ├── token_budget.py
│   ├── packer.py
│   └── prompts.py
│
├── llm/
│   ├── qwen_wrapper.py
│   ├── parser.py
│   └── cache.py
│
├── training/
│   ├── train_encoder.py
│   ├── train_selector.py
│   ├── sampler.py
│   └── checkpointing.py
│
├── evaluation/
│   ├── baselines.py
│   ├── metrics.py
│   ├── temporal_metrics.py
│   ├── evidence_metrics.py
│   ├── leakage_scanner.py
│   └── aggregate.py
│
└── tests/
    ├── test_pheme_adapter.py
    ├── test_maweibo_adapter.py
    ├── test_snapshot_builder.py
    ├── test_time_features.py
    ├── test_structure_features.py
    ├── test_no_future_leakage.py
    ├── test_cap_1021.py
    ├── test_fold_integrity.py
    ├── test_selector.py
    ├── test_memory.py
    ├── test_context_packer.py
    ├── test_llm_parser.py
    └── test_cache.py

scripts/
├── tcdscr_build_cache.py
├── tcdscr_build_snapshots.py
├── tcdscr_train_encoder.py
├── tcdscr_train_selector.py
├── tcdscr_smoke.py
├── tcdscr_run_llm.py
└── tcdscr_summarize.py
```

---

# 13. Module 1 — Dataset Adapters

统一输出 raw event：

```python
{
    "event_id": str,
    "label": int,
    "source_id": str,
    "source_timestamp": int,
    "nodes": [
        {
            "node_id": str,
            "parent_id": str | None,
            "timestamp": int,
            "text": str,
            "original_order": int,
            "status": str,
        }
    ]
}
```

允许 status：

```text
VALID
MISSING_PARENT
EXTERNAL_PARENT
TEMPORAL_INVALID_NODE
```

adapter 不计算模型 feature。

adapter 只能：

- 读 raw；
- 归一化字段；
- 标记异常。

---

# 14. Module 2 — Causal Snapshot Builder

接口：

```python
build_snapshot(event, cutoff_minutes)
```

必须返回：

```python
{
    "event_id",
    "label",
    "cutoff_minutes",
    "source_id",
    "node_ids",
    "texts",
    "timestamps",
    "parent_ids",
    "edge_index",
    "num_nodes_before_cap",
    "num_nodes_after_cap",
    "cap_hit",
}
```

SOURCE_ONLY 单独接口：

```python
build_source_only(event)
```

## 14.1 Snapshot inclusion

节点：

\[
timestamp_i
\le
source\_timestamp + cutoff
\]

同时：

```text
status != TEMPORAL_INVALID_NODE
```

## 14.2 Edge inclusion

只有：

```text
child included
AND parent included
AND parent relation resolved
AND parent_timestamp <= child_timestamp
```

才构边。

---

# 15. Module 3 — Feature Builder

每个 snapshot 独立构造：

```text
384D semantic
1021D adjacency signature
3D structural
```

输出：

```python
{
    "node_feat": Tensor[N,384],
    "struct_feat": Tensor[N,1024],
    "edge_index": Tensor[2,E],
    "node_ids": list[str],
}
```

必须实现 snapshot-level caching。

Cache key：

```text
dataset
event_id
cutoff
preprocess_version
semantic_model_hash
```

---

# 16. Module 4 — Causal Social Encoder

不要复用完整 UMER fusion。

新 encoder 来源：

```text
OriginalGraphBranch
```

但：

```text
node_feat_dim = 384
struct_feat_dim = 1024
```

需要输出：

```text
node contextual embeddings h_i
event embedding g_t
classifier logits p_full
```

接口：

```python
node_repr, event_repr, logits = encoder(
    node_feat,
    struct_feat,
    ...
)
```

## 16.1 UMER checkpoint 初始化

PHEME 必须支持：

```text
random init
UMER init
```

UMER init 时：

旧 node projection：

```text
W_old[:, 0:384]
```

复制到新 384D semantic projection。

以下允许复制：

```text
StructureFeatureEncoder
TransformerEncoder
CLS token
readout
classifier（若维度兼容）
```

不得复制：

- retrieval；
- DeBERTa four-view；
- tri-fusion；
- 11D-dependent node weights。

Ma-Weibo 也必须支持同类型初始化，如果对应历史 graph branch checkpoint 可用。

---

# 17. Module 5 — Static Utility Selector

source 不参与候选排序。

reply node \(i\)：

\[
h_i^t\in R^{768}
\]

event：

\[
g_t\in R^{768}
\]

semantic relevance：

\[
r_i
=
cos(e_i,e_{source})
\]

struct3：

```text
norm_degree
norm_depth
norm_time
```

拼接：

\[
z_i^t
=
[h_i^t;g_t;r_i;struct3_i]
\]

维度：

```text
1540
```

scorer 固定：

```text
LayerNorm(1540)
Linear(1540,256)
GELU
Dropout(0.1)
Linear(256,1)
```

输出：

\[
u_i^t
\]

不得加入：

- attention；
- GNN；
- Transformer；
- LLM features。

---

# 18. Selector Proxy

训练阶段不调用 LLM。

\[
\alpha_i
=
softmax(u_i/\tau)
\]

固定：

```text
tau = 0.5
```

\[
z_{sel}
=
\sum_i \alpha_i h_i
\]

source：

\[
h_s
\]

proxy classifier：

```text
Linear(1536,256)
GELU
Dropout(0.1)
Linear(256,2)
```

## 18.1 Loss

\[
L_{cls}
=
CE(p_{sel},y)
\]

\[
L_{fid}
=
KL(stopgrad(p_{full}) || p_{sel})
\]

\[
L_{div}
=
\sum_{i\neq j}
\alpha_i\alpha_j
\max(0,cos(e_i,e_j))
\]

总损失：

\[
L
=
L_{cls}
+
1.0L_{fid}
+
0.05L_{div}
\]

V2 不增加其他 loss。

---

# 19. Module 6 — Dynamic Evidence Memory

上一时刻：

\[
M_{t-1}
\]

当前：

\[
M_t
\]

novelty：

若 memory 空：

\[
novelty_i=1
\]

否则：

\[
novelty_i
=
1-
\max_{j\in M_{t-1}}
cos(e_i,e_j)
\]

persistence：

\[
persist_i=
1[i\in M_{t-1}]
\]

动态分数：

\[
d_i
=
u_i
+
\lambda_n novelty_i
+
\lambda_p persist_i
\]

允许搜索：

```text
lambda_n ∈ {0, 0.25, 0.5, 1.0}
lambda_p ∈ {0, 0.1, 0.25}
```

只允许 validation 调参。

---

# 20. Evidence Unit

最小单位：

```text
Reply–Parent Pair
```

resolved：

```text
[E_i]
time:
depth:
parent:
<parent text>
reply:
<reply text>
```

unresolved：

```text
parent:
[UNAVAILABLE]
```

source 永远单独提供。

不得：

- full path；
- 自动摘要；
- stance annotation；
- LLM relevance annotation。

---

# 21. Token Budget

候选：

```text
512
1024
2048
```

使用：

```text
Qwen3-8B tokenizer
```

计算。

source + instruction 不计入 evidence budget。

总 prompt token 另行统计。

选择：

```text
dynamic_score descending
```

依次加入完整 evidence unit。

不得截断一个 Reply–Parent Pair。

---

# 22. Context Packer

英文模板：

```text
SOURCE CLAIM
<source text>

CURRENT SNAPSHOT
elapsed_time = ...
observed_replies = ...
max_depth = ...

SELECTED SOCIAL EVIDENCE

[E1]
time = ...
depth = ...
parent:
...
reply:
...

TASK
Using only the source claim and the social evidence shown above,
classify the source claim as RUMOR or NON_RUMOR.
Return exactly one label.
```

中文：

```text
源帖
<source>

当前传播状态
elapsed_time = ...
observed_replies = ...
max_depth = ...

选中的社会证据

[E1]
time = ...
depth = ...
parent:
...
reply:
...

任务
仅依据上面的源帖和社会证据，
判断该源帖为 RUMOR 或 NON_RUMOR。
只返回一个标签。
```

禁止：

- gold；
- future node count；
- final full depth；
- external evidence；
- retrieval label；
- rationale request；
- JSON output。

---

# 23. Module 7 — Qwen3-8B

固定：

```text
Qwen3-8B
frozen
greedy
do_sample=False
```

若推理栈支持：

```text
enable_thinking=False
```

必须关闭 thinking。

允许输出：

```text
RUMOR
NON_RUMOR
```

其他：

```text
INVALID_OUTPUT
```

不得模糊匹配。

invalid rate 后续正式实验必须 < 1%。

---

# 24. LLM Cache

key：

```text
model_id
model_revision
prompt_sha256
generation_config_sha256
dataset
event_id
cutoff
selector_checkpoint_hash
budget
```

同 key 不重复推理。

---

# 25. Baselines

正式阶段固定：

1. Source-only LLM
2. All-current LLM
3. Random-budget
4. Recent-budget
5. Semantic-budget
6. Structural-budget
7. Static Utility Selector
8. TC-DSCR Dynamic Selector

Code Complete 阶段只实现接口和 tiny smoke。

---

# 26. Structural-budget 规则

固定：

\[
score_i
=
z(normDegree_i)
-
0.25z(normDepth_i)
\]

同分：

```text
earlier timestamp first
```

若 std=0：

```text
z = 0
```

---

# 27. 1021-node Cap 统计

必须实现：

```text
results/tcdscr/code_smoke/cap_report.json
```

PHEME / Ma-Weibo 分开。

正式窗口：

```text
5m
15m
30m
1h
3h
6h
24h
```

字段：

```text
events
cap_hits
cap_hit_rate
mean_nodes_before_cap
p90_nodes_before_cap
max_nodes_before_cap
total_nodes_removed
```

---

# 28. 必须实现的 Unit Tests

至少：

## Data

```text
test_pheme_timestamp
test_maweibo_timestamp
test_maweibo_clean_text
test_source_only_exactly_one_node
```

## Causality

```text
test_snapshot_node_monotonicity
test_snapshot_edge_monotonicity
test_no_future_text
test_no_future_topology
test_source_timestamp_as_t0
test_temporal_invalid_node_excluded
test_child_before_parent_edge_removed
```

## Structure

```text
test_norm_degree_snapshot_internal
test_norm_depth_div19
test_norm_time_30min
test_adj_signature_snapshot_only
test_1021_cap_deterministic
```

## Split

```text
test_event_fold_integrity
test_no_event_cross_fold
test_all_snapshots_same_fold
```

## Selector

```text
test_selector_shape
test_no_source_in_candidate_pool
test_novelty_memory_only_past
test_persistence_flag
test_dynamic_score
```

## Context

```text
test_reply_parent_pair
test_unavailable_parent
test_budget_no_partial_pair
test_prompt_no_gold
test_prompt_no_future
```

## LLM

```text
test_exact_parser
test_invalid_parser
test_cache_key_deterministic
```

---

# 29. Integration Tests

至少实现：

```text
PHEME:
raw
→ source-only / 15m / 1h
→ features
→ encoder
→ selector
→ packer

Ma-Weibo:
raw
→ source-only / 15m / 1h
→ features
→ encoder
→ selector
→ packer
```

检查：

```text
finite tensors
correct dimensions
no future leakage
deterministic selection
deterministic prompt
```

---

# 30. Code Complete Tiny Smoke Run

只允许：

```text
PHEME <= 32 events
Ma-Weibo <= 32 events
cutoffs <= 2 dynamic cutoffs + SOURCE_ONLY
encoder = 1 epoch
selector = 1 epoch
LLM <= 20 requests total
```

建议 smoke：

```text
SOURCE_ONLY
15m
1h
```

## 30.1 Smoke 只验证

```text
raw
→ causal snapshot
→ feature
→ encoder
→ selector
→ dynamic memory
→ evidence
→ prompt
→ Qwen
→ parser
→ result
```

不允许：

- 选最佳超参数；
- 对结果做统计显著性；
- 判断方法有效；
- 根据 smoke 修改算法结构。

---

# 31. Code Complete 输出文件

必须生成：

```text
docs/TC_DSCR_V2_IMPLEMENTATION_NOTES.md

results/tcdscr/code_smoke/
├── test_report.txt
├── smoke_manifest.json
├── smoke_predictions.jsonl
├── leakage_report.json
├── cap_report.json
├── prompt_examples.md
├── model_shapes.json
└── known_issues.md
```

---

# 32. Implementation Notes 固定结构

```markdown
# TC-DSCR V2 Implementation Notes

## Git
- commit:

## Implemented Modules
...

## Reused UMER Modules
...

## Dataset Adapters
...

## Causal Snapshot Rules
...

## Semantic Preprocessing
...

## Structural Features
...

## Source Timestamp Handling
...

## 1021 Node Cap
...

## Encoder Initialization
...

## Selector
...

## Dynamic Memory
...

## Evidence Unit
...

## LLM Configuration
...

## Tests
...

## Smoke Run
...

## Deviations from V2 Plan
NONE / list

## Known Issues
...
```

---

# 33. Code Complete STOP 条件

遇到以下任一情况：

```text
snapshot future leakage
feature NaN
Ma-Weibo cleaning parity failure
source timestamp logic mismatch
same event appears across folds
selector accesses gold label
selector uses future memory
prompt contains future text
Qwen wrapper cannot deterministic decode
1021 cap non-deterministic
```

立即停止。

写：

```text
BLOCKER_REPORT.md
```

不得自行重构研究方案。

---

# 34. Code Complete 审批前禁止事项

禁止：

```text
正式 5-fold training
三 seed 正式实验
完整 baseline evaluation
full PHEME Qwen inference
full Ma-Weibo Qwen inference
正式 lambda_n/lambda_p 搜索
正式 token budget 搜索
ablation
chronological stress
论文图表
```

---

# 35. Stage B — 正式实验（仅审批后）

以下章节只用于提前固定后续执行，不代表当前可运行。

---

# 36. Formal E1 — Causal Encoder

每 dataset：

```text
3 seeds:
2000
2001
2002
```

训练：

```text
event-level stratified 5-fold
```

每个 epoch：

> 每个 train event 随机均匀采一个可用 snapshot。

snapshot pool：

```text
5m
15m
30m
1h
3h
6h
```

SOURCE_ONLY 不作为 encoder 主训练 snapshot。

---

# 37. Encoder Initialization

比较：

```text
Random Init
UMER Init
```

报告每 cutoff：

```text
Accuracy
Macro-F1
Weighted-F1
Rumor-F1
```

---

# 38. Formal E2 — Static Selector

训练：

```text
utility scorer
+ proxy classifier
```

readiness gate：

Static selector validation Macro-F1 必须优于：

```text
max(Random-budget, Semantic-budget)
```

至少：

```text
+0.5 Macro-F1 point
```

否则：

```text
SELECTOR_NOT_READY
STOP
```

---

# 39. Formal E3 — Dynamic Memory

validation 搜索：

```text
lambda_n
lambda_p
budget
```

选择优先：

1. mean validation Macro-F1
2. lower flip rate
3. lower turnover
4. smaller hyperparameter sum

若 Dynamic 相对 Static：

```text
Macro-F1 无提升
AND flip 无下降
AND token 无下降
```

三者同时成立：

```text
DYNAMIC_NOT_SUPPORTED
STOP
```

---

# 40. Formal E4 — Frozen LLM

冻结：

```text
encoder checkpoint
selector checkpoint
lambda_n
lambda_p
budget
prompt
```

后才跑 test。

test 后禁止调参。

---

# 41. Formal Baseline Evaluation

对：

```text
SOURCE_ONLY
5m
15m
30m
1h
3h
6h
```

运行 8 个 baseline。

24h 只做 late diagnostic。

---

# 42. Formal Metrics

## Classification

```text
Accuracy
Macro-F1
Weighted-F1
Rumor-F1
```

## Dynamic

```text
Prediction Flip Rate
Evidence Turnover
Persistent/New ratio
```

## Efficiency

```text
mean evidence tokens
median evidence tokens
mean total prompt tokens
P90 total prompt tokens
```

## Context comparison

\[
Gap_t
=
MacroF1_{selected,t}
-
MacroF1_{all-current,t}
\]

---

# 43. Formal Ablations

固定：

```text
- novelty
- persistence
- semantic relevance
- structural3
static vs dynamic
512 / 1024 / 2048
random-init vs UMER-init
```

不得新增大型消融。

---

# 44. PHEME Chronological Stress

最后执行：

```text
70 / 15 / 15 chronological
```

只做关键 baseline：

```text
Source-only
All-current
Static Selector
Dynamic Selector
```

不得做全套超参搜索。

---

# 45. Ma-Weibo Chronological

只生成诊断表：

```text
date distribution
class distribution
rumor ratio
```

不训练模型。

---

# 46. Formal Run 输出

每次：

```text
config_resolved.yaml
run_manifest.json
metrics.json
predictions.jsonl
logs/
checkpoints/
```

`predictions.jsonl`：

```text
dataset
fold
seed
event_id
cutoff
gold
prediction
selected_node_ids
base_score
novelty
persistence
dynamic_score
evidence_tokens
total_prompt_tokens
prompt_sha256
cache_key
```

---

# 47. Final 输出

```text
docs/TC_DSCR_EXPERIMENT_REPORT.md

results/tcdscr/summaries/
├── final_summary.json
├── final_tables.md
├── failure_analysis.md
└── reproducibility_manifest.json
```

最终状态只能：

```text
PASS
PARTIAL
FAIL
NOT_READY
```

---

# 48. 研究成功标准

成功不是：

> “Accuracy 比 UMER 高一点。”

需要同时证明：

1. evolving social context 有明确动态空间；
2. 全量 current context 并非始终最优；
3. static utility selection 有价值；
4. dynamic novelty/persistence 进一步有可测贡献；
5. 更小 token context 能保持或提高检测表现；
6. PHEME 与 Ma-Weibo 方向一致；
7. whole pipeline strict causal；
8. 无 future text / topology leakage；
9. 结果不依赖单一 cutoff；
10. 研究结论围绕 context refinement，而不是传统 leaderboard。

---

# 49. 当前立即执行顺序

执行智能体现在只能执行：

```text
STEP 1
创建 project/tcdscr/ 隔离目录

STEP 2
实现 PHEME / Ma-Weibo adapters

STEP 3
实现 SOURCE_ONLY + causal snapshots

STEP 4
实现 384D semantic cache

STEP 5
实现 1021D + 3D causal features

STEP 6
实现 causal social encoder

STEP 7
实现 static utility selector

STEP 8
实现 selector proxy

STEP 9
实现 dynamic evidence memory

STEP 10
实现 Reply–Parent evidence unit

STEP 11
实现 token budget + context packer

STEP 12
实现 Qwen wrapper/parser/cache

STEP 13
实现全部 unit tests

STEP 14
实现 integration tests

STEP 15
运行 tiny smoke

STEP 16
生成 Code Complete 报告

STEP 17
提交 GitHub

STEP 18
STOP
```

---

# 50. 最终提醒

执行智能体没有方案选择权限。

若出现：

```text
本文档没有规定的模型选择
本文档没有规定的数据处理
本文档没有规定的异常情况
```

正确动作是：

```text
记录
停止对应子任务
写问题报告
```

而不是自行：

```text
换模型
换数据
改 loss
改时间窗口
改 split
加模块
```

**当前唯一目标：按本文档完成 TC-DSCR V2 Code Complete，并在提交 GitHub 后停止。**
