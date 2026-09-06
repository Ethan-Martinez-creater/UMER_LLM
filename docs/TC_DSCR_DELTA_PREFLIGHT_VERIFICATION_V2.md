# TC-DSCR 第二轮前置验证任务（Delta Preflight）

> 项目：**TC-DSCR — Temporally Causal Dynamic Social-Context Refinement for LLM Rumor Detection**
>
> 本文档用于执行第一次 Preflight 审批后要求的**增量验证任务**。  
> 本轮不是重新做第一轮验证，也不是正式实现 TC-DSCR。
>
> 执行智能体必须严格按本文档推进。  
> **不得自行修改研究方向、数据集、模型结构、特征定义、时间窗口、数据划分或实验协议。**
>
> 本轮完成后必须停止，将结果提交仓库，由用户交给研究审批方再次审批。  
> **在第二轮验证获批前，不得进入正式 TC-DSCR 代码实现阶段。**

---

# 1. 本轮验证目标

本轮只回答四个问题：

1. PHEME 的 causal snapshot 原型能否按真实的 **30 分钟 time-bin** 正确重算时间特征？
2. PHEME 中 parent 缺失问题的真实规模是多少，能否按互斥类别准确统计？
3. PHEME 在 `5m / 15m / 30m` 等超早期窗口下是否具有足够动态变化，适合 TC-DSCR？
4. **Ma-Weibo 是否能够作为 TC-DSCR 的第二个正式数据集**，即能否从 raw JSON 中恢复：
   - 动态传播时间；
   - 原始文本；
   - 传播关系；
   - 384D multilingual semantic features；
   - causal 1024D structural features；
   - 无未来信息泄漏的 \(G_t\)。

---

# 2. 本轮已冻结的研究决策

执行智能体不得重新讨论以下决策。

## 2.1 数据集候选

本轮只考虑：

```text
PHEME
Ma-Weibo
```

Weibo22 已从 TC-DSCR 方案中移除。

本轮不得自行切换到：

```text
COVID-Weibo
Twitter15
Twitter16
Weibo22
Weibo21
其他数据集
```

---

## 2.2 TC-DSCR 主路径输入

TC-DSCR 后续主路径暂定为：

\[
384D\ multilingual\ semantic
+
1024D\ causal\ structural
\]

即：

```text
node semantic: 384D
struct feature: 1021D adjacency signature + 3D propagation summary
```

原 UMER 的：

```text
10D user features
1D sentiment feature
```

**不进入 TC-DSCR 主路径。**

本轮无需验证 11D 是否继续保留，也不得自行恢复 11D 到新模型中。

---

## 2.3 norm_degree 规则

正式 causal snapshot 中：

\[
rawDegree_i^t = \max(outDegree_i^t - 1, 0)
\]

\[
normDegree_i^t =
\begin{cases}
rawDegree_i^t / \max_j rawDegree_j^t, & \max_j rawDegree_j^t > 0 \\
0, & otherwise
\end{cases}
\]

即：

> **每个 snapshot 内部独立归一化。**

不得使用：

- 240h full-event 的最大 degree；
- dataset-wide 最大 degree；
- train-set 全局最大 degree。

---

# 3. 输出目录

本轮输出固定写入：

```text
results/tcdscr/delta_preflight/
```

至少必须包含：

```text
README.md
environment_delta.json

pheme_p9_fixed.json
pheme_parent_audit_v2.json
pheme_early_snapshot_distribution.csv
pheme_early_snapshot_summary.json

maweibo_raw_schema_audit.json
maweibo_temporal_audit.json
maweibo_snapshot_distribution.csv
maweibo_feature_pipeline_audit.json
maweibo_full_parity_report.json
maweibo_snapshot_prototype.json
maweibo_manual_snapshot_examples.md

delta_preflight_summary.md
```

所有脚本统一放置：

```text
scripts/tcdscr_delta_preflight/
```

原始数据不得提交仓库。

---

# 4. Step D0 — 环境与仓库状态冻结

记录：

```text
git commit SHA
Python version
PyTorch version
Transformers version
SentenceTransformers version
CUDA version
GPU model
OS

PHEME raw path
PHEME processed path

Ma-Weibo raw path
Ma-Weibo processed path
Ma-Weibo labels path
Ma-Weibo existing UMER checkpoint path（如有）

historical preprocessing source root
```

写入：

```text
results/tcdscr/delta_preflight/environment_delta.json
```

确认：

- 第一轮 Preflight 结果仍存在；
- 不修改第一轮报告；
- 不修改冻结的旧 UMER 实现；
- 不重新运行与本轮无关的大规模任务。

---

# 5. Step D1 — 修复 PHEME P9 的 30 分钟 time-bin

第一轮审批已确认历史 UMER 的真实时间定义是：

```text
time_window_minutes = 30
max_time_steps = 480
```

因此：

\[
timeBin_i =
\left\lfloor
\frac{elapsedSeconds_i}{1800}
\right\rfloor
\]

并裁剪：

\[
timeBin_i \in [0,479]
\]

最终：

\[
normTime_i =
\frac{timeBin_i}{480}
\]

---

## 5.1 必须修复的错误

第一轮原型中错误使用了：

```python
elapsed_seconds // 3600 / 480
```

本轮必须修正为：

```python
time_bin = floor(elapsed_seconds / 1800)
time_bin = min(max(time_bin, 0), 479)
norm_time = time_bin / 480.0
```

---

## 5.2 固定重新验证样本

仍使用：

```text
seed = 3090
10 PHEME events
cutoffs = 1h / 6h / 24h
```

不得换样本。

对每个 snapshot 验证：

```text
included nodes
excluded future nodes
edges
depth
raw degree
norm degree
time bin
norm time
future leakage
```

必须额外报告前 10 个节点：

```text
node_id
elapsed_seconds
expected_time_bin
actual_time_bin
expected_norm_time
actual_norm_time
```

---

## 5.3 D1 通过标准

必须满足：

```text
time_bin exact rate = 100%
norm_time max_abs_diff <= 1e-8
node monotonicity = PASS
edge monotonicity = PASS
future topology leakage = 0
future text leakage = 0
```

输出：

```text
pheme_p9_fixed.json
```

### STOP-D1

若 time-bin 修正后仍无法与历史定义一致：

```text
status = NOT_READY_PHEME_TIME_FEATURE
```

完成当前报告后停止，不得继续正式模型实现。

---

# 6. Step D2 — 修正 PHEME parent / orphan 审计

第一轮 parent 统计存在重叠口径。

本轮必须把所有非 source 节点划分为三个**互斥类别**：

## A. `resolved_parent`

条件：

```text
parent_id 非空
且 parent_id 存在于同 event 节点集合
```

---

## B. `missing_parent_field`

条件：

```text
parent_id is None / empty
```

---

## C. `external_or_unresolved_parent`

条件：

```text
parent_id 非空
但 parent_id 不存在于当前 event 节点集合
```

必须保证：

\[
A+B+C=全部reply/reaction节点
\]

---

## 6.1 必须统计

全数据集报告：

```text
total replies
resolved_parent count / ratio
missing_parent_field count / ratio
external_or_unresolved_parent count / ratio
chronology violation count
```

按 event 额外报告：

```text
events_with_missing_parent
events_with_external_parent
P50/P90/P99 unresolved-parent ratio
```

列出最多 50 个异常例子：

```text
event_id
node_id
parent_id
node_timestamp
parent_resolution_status
```

---

## 6.2 后续处理规则只做审计，不改数据

本轮固定规则：

- resolved parent → 正常构边；
- missing parent → 节点保留，不人工挂到 source；
- external/unresolved parent → 节点保留，不人工挂到 source；
- evidence packer 后续将 unresolved parent 表示为 `[UNAVAILABLE]`。

本轮不需要实现正式 evidence packer。

输出：

```text
pheme_parent_audit_v2.json
```

---

# 7. Step D3 — PHEME 超早期传播窗口统计

第一轮发现：

```text
1h median reply coverage ≈ 80%
3h median reply coverage ≈ 95.8%
```

因此 1h 可能已经过晚。

本轮增加超早期窗口：

```text
0m
5m
15m
30m
1h
3h
6h
24h
```

其中：

```text
0m = source-only cutoff
```

---

## 7.1 每个窗口必须统计

```text
events_total
events_source_only
source_only_ratio

mean_nodes
median_nodes
P75_nodes
P90_nodes
P99_nodes

mean_edges
median_edges

median_depth
P90_depth

rumor_mean_nodes
nonrumor_mean_nodes

reply_coverage_P25
reply_coverage_P50
reply_coverage_P75
reply_coverage_P90
```

---

## 7.2 动态变化额外统计

相邻窗口：

```text
0 -> 5m
5m -> 15m
15m -> 30m
30m -> 1h
1h -> 3h
3h -> 6h
6h -> 24h
```

统计：

```text
events_with_new_nodes
ratio_events_with_new_nodes
median_new_nodes
P90_new_nodes
events_with_new_edges
```

目的是判断：

> 哪些时间段真正存在 context refinement 的动态空间。

---

## 7.3 本轮不得自行冻结正式窗口

执行智能体只输出统计。

不得说：

```text
“因此正式采用 0/15/30/60 分钟”
```

最终窗口由研究审批方根据结果冻结。

输出：

```text
pheme_early_snapshot_distribution.csv
pheme_early_snapshot_summary.json
```

---

# 8. Step D4 — Ma-Weibo 原始数据结构审计

这是本轮最重要部分。

目标不是证明 Ma-Weibo “以前能跑 UMER”，而是证明它适合：

\[
G_t
\rightarrow Dynamic\ Context
\rightarrow LLM
\]

因此必须直接从原始 JSON 审计。

---

# 9. Step D4.1 — Ma-Weibo raw schema

随机固定：

```text
seed = 3090
sample = 100 events
```

同时必须对**全数据集字段覆盖率**做扫描。

逐字段检查：

## Event / source

```text
event_id
label
source text
source node id
source time
source user
```

## Propagation node

```text
node id
text
time / t
parent id
user
repost/reply relation
```

---

## 9.1 对 `t` 字段必须明确回答

不得只写“存在时间戳”。

必须查明：

```text
字段名
数据类型
单位
是绝对时间还是相对时间
相对谁
source 的 t 是多少
是否单调
是否存在负值
是否存在重复值
是否存在缺失
```

例如最终报告必须类似：

```text
t definition:
  relative elapsed seconds from source
```

或：

```text
t definition:
  absolute unix timestamp
```

如果无法从数据、原预处理代码或官方说明确定，标记：

```text
TIME_SEMANTICS_UNRESOLVED
```

不得猜测单位。

---

# 10. Step D4.2 — Ma-Weibo temporal readiness

全数据统计：

```text
events
nodes
source time coverage
node time coverage
text coverage
parent coverage

negative elapsed count
child-before-parent count
duplicate node ids
multi-root events
orphan nodes
source-only events
```

若时间是相对 source：

```text
source t == 0 的比例
```

若时间是绝对时间：

```text
source absolute timestamp coverage
```

---

## 10.1 关键区分

必须分别回答：

### 事件内部动态快照

能否构造：

\[
G_t
\]

只要求：

```text
可靠 relative time
+
传播关系
```

### 全局 chronological split

要求：

```text
event-level absolute source timestamp
```

因此报告：

```text
within_event_temporal_ready = yes/no
global_chronological_ready = yes/no
```

即使第二项为 no，也不自动淘汰 Ma-Weibo。

---

# 11. Step D4.3 — Ma-Weibo propagation relation

必须明确原始数据如何表达树结构：

```text
parent id?
root id?
repost index?
list index?
edge list?
```

不得根据节点顺序自行构树。

报告：

```text
relation_field
edge_direction
root definition
parent resolution rate
cycle count
disconnected nodes
```

验证：

```text
child time >= parent time
```

如果 parent-child 不存在但历史 preprocessing 使用了明确的 deterministic tree construction，也必须定位原实现并说明。

---

# 12. Step D5 — Ma-Weibo 历史特征生成链审计

目标：找到 Ma-Weibo 原 UMER preprocessing 的 authoritative pipeline。

必须定位：

```text
raw adapter
text cleaning
384D embedding
graph builder
1021D adjacency signature
3D structural summary
240h cutoff
max_nodes
node ordering
time bin definition
depth definition
degree definition
```

输出：

```text
feature_name
dimension
definition
source_file
source_function/line
depends_on_future_graph
recomputable_from_snapshot
```

写入：

```text
maweibo_feature_pipeline_audit.json
```

---

# 13. Step D5.1 — 384D semantic feature

必须确认 Ma-Weibo 与 PHEME 历史流程是否都使用：

```text
paraphrase-multilingual-MiniLM-L12-v2
```

验证：

```text
model path
config
tokenizer
embedding dimension
normalization
text cleaning
```

不得自行更换 embedding 模型。

---

# 14. Step D5.2 — 1021D adjacency signature

必须明确：

```text
node ordering
edge direction
self loop
row normalization
padding
truncation
max_nodes
```

并回答：

> 该 signature 是否可以从任意 Ma-Weibo snapshot \(G_t\) 重新生成？

如果依赖 full graph 未来节点，但规则可以因果重算，标记：

```text
recomputable_from_snapshot = yes
depends_on_full_future = yes
```

---

# 15. Step D5.3 — 3D causal structural summary

本轮 TC-DSCR 统一要求：

```text
norm_degree
norm_depth
norm_time
```

但必须首先确认历史 Ma-Weibo 定义。

然后判断能否转换为与 PHEME 一致的 causal 定义。

目标规则：

## norm_degree

snapshot internal max：

\[
normDegree_i^t
\]

按本文件第 2.3 节。

## norm_depth

若历史实现是固定最大 depth 常数归一化：

```text
保留历史定义
```

若使用 full-event max depth：

```text
必须标记 future leakage
```

不得自行选择新规则，提交审批。

## norm_time

如果 Ma-Weibo 原时间单位/窗口与 PHEME 不同：

- 只报告历史定义；
- 不自行强制改成 PHEME 480-bin；
- 由审批方决定是否统一。

---

# 16. Step D6 — Ma-Weibo full-event parity

仅在历史 feature pipeline 已完整定位后执行。

固定：

```text
seed = 3090
sample = 100 events
```

必须覆盖：

```text
rumor
non-rumor
small graph
medium graph
large graph
source-only（如果存在）
```

从 raw JSON 重新构建历史 full / 240h 输入，对比现有 Ma-Weibo `.pt`。

必须比较：

```text
node_ids
num_nodes
edge_index

text384

adjacency signature 1021D
summary3
```

因为 TC-DSCR 已决定不使用 11D，**本轮无需要求 11D parity 作为通过条件**。

但如果现有 `.pt` 中 node_feat 含 11D，可记录，不必重建。

---

## 16.1 通过标准

结构：

```text
node_ids exact = 100%
num_nodes exact = 100%
edge semantic equivalence = 100%
```

若 edge ordering 不一致但集合完全一致：

```text
edge_set_exact = PASS
edge_order_exact = INFO
```

浮点：

```text
text384 max_abs_diff <= 1e-5
adj_signature max_abs_diff <= 1e-5
summary3 max_abs_diff <= 1e-5
```

若失败，必须给出差异来源。

输出：

```text
maweibo_full_parity_report.json
```

### STOP-D6

如果 384D 或 1024D 无法从 raw data 可靠重建：

```text
status = NOT_READY_MAWEIBO_FEATURE_PARITY
```

不得开始正式 TC-DSCR 实现。

---

# 17. Step D7 — Ma-Weibo snapshot distribution

使用与 PHEME 相同的审计窗口：

```text
0m
5m
15m
30m
1h
3h
6h
24h
```

如果 Ma-Weibo 的时间单位无法支持分钟级映射：

- 不得自行近似；
- 只报告可支持的真实单位；
- 标记 `WINDOW_MAPPING_REQUIRES_APPROVAL`。

统计与 PHEME 完全相同：

```text
events_total
events_source_only
source_only_ratio

mean/median/P75/P90/P99 nodes
mean/median edges
median/P90 depth

rumor_mean_nodes
nonrumor_mean_nodes

reply_coverage P25/P50/P75/P90
```

相邻窗口新增：

```text
events_with_new_nodes
ratio_events_with_new_nodes
median_new_nodes
P90_new_nodes
events_with_new_edges
```

输出：

```text
maweibo_snapshot_distribution.csv
```

---

# 18. Step D8 — Ma-Weibo minimal causal snapshot prototype

固定：

```text
seed = 3090
10 events
```

时间点优先：

```text
15m
1h
6h
```

如果 Ma-Weibo 时间语义不能直接映射分钟，则等待审批决定，不自行转换。

每个 snapshot 检查：

```text
included nodes
excluded future nodes
edges
raw degree
norm degree
depth
time feature
future topology leakage
future text leakage
```

必须确保：

```text
G_t1 ⊆ G_t2
```

节点和边都满足单调包含。

输出：

```text
maweibo_snapshot_prototype.json
maweibo_manual_snapshot_examples.md
```

人工示例至少 3 个 event。

---

# 19. Step D9 — Ma-Weibo chronological split 可行性

仅当存在 event-level absolute source timestamp 时执行。

候选：

```text
60/20/20
70/10/20
70/15/15
```

报告：

```text
train/val/test events
rumor/nonrumor count
class ratio
time span
year distribution
source-only ratio
```

如果没有 absolute source timestamp：

```text
global_chronological_ready = false
```

并结束该项。

**不得因此判定 Ma-Weibo 不可用。**

因为 Ma-Weibo 仍可用于：

```text
event-internal causal dynamic evaluation
```

---

# 20. 第二轮验证的统一数据判定标准

## PHEME

本轮目标只是修正并冻结：

```text
time feature
parent audit
early window distribution
```

PHEME 已默认保留，除非发现新的严重 temporal leakage。

---

## Ma-Weibo

判定为可用于 TC-DSCR 的最低条件：

```text
raw text coverage >= 99%
within-event node time coverage >= 99%
传播关系可确定
可以构造 G_t
384D text feature 可重建
1021D adjacency 可重建
3D causal structure 可构造
snapshot 无 future leakage
```

其中：

```text
global chronological timestamp
```

不是强制通过条件。

---

# 21. 本轮最终状态定义

`delta_preflight_summary.md` 必须使用以下状态之一：

## `PASS`

表示：

- PHEME 修正项全部通过；
- Ma-Weibo 满足 TC-DSCR 第二数据集要求；
- 384D + causal 1024D 两个数据集均可构造；
- 无 unresolved blocker；
- 可以建议进入 Code Complete 阶段。

---

## `PARTIAL`

表示：

- 两个数据总体可用；
- 但存在需要研究审批方决定的 protocol 问题，例如：
  - Ma-Weibo time normalization；
  - 正式窗口；
  - chronological split；
  - 某个旧结构定义是否继续保留。

不得自行解决。

---

## `NOT_READY`

表示任一核心条件失败，例如：

```text
Ma-Weibo 无法恢复动态传播
Ma-Weibo 无原始文本
Ma-Weibo 传播关系不可恢复
384D / 1024D 无法重建
PHEME causal time feature 仍不正确
出现 future leakage
```

---

# 22. delta_preflight_summary.md 固定模板

```markdown
# TC-DSCR Delta Preflight Summary

## Overall Status
PASS / PARTIAL / NOT_READY

## PHEME Corrections

### P9 Time-bin Fix
- status:
- definition:
- exact rate:
- leakage:

### Parent Audit V2
- total replies:
- resolved:
- missing parent:
- external/unresolved:
- chronology violations:

### Early Snapshot Dynamics
| cutoff | source-only | median nodes | P90 nodes | median coverage | events with new nodes |
|---|---:|---:|---:|---:|---:|

## Ma-Weibo Raw Schema
- event count:
- raw text coverage:
- time field:
- time semantics:
- parent relation:
- within-event temporal ready:
- global chronological ready:

## Ma-Weibo Feature Pipeline
- 384D encoder:
- 1021D adjacency:
- 3D structural:
- historical cutoff:
- authoritative source files:

## Ma-Weibo Full Parity
- node IDs:
- edges:
- text384:
- adjacency:
- summary3:

## Ma-Weibo Snapshot Dynamics
...

## Leakage Checks
- PHEME:
- Ma-Weibo:

## Blocking Issues
1.
2.

## Questions Requiring Research Approval
1.
2.

## Recommended Status
PASS / PARTIAL / NOT_READY
```

---

# 23. STOP 条件

出现以下任一情况立即停止对应进一步实现：

1. PHEME 30 分钟 time-bin 无法正确恢复；
2. PHEME snapshot 出现 future topology/text leakage；
3. Ma-Weibo `t` 字段语义无法确定；
4. Ma-Weibo 无法形成传播关系；
5. Ma-Weibo 原文覆盖严重不足；
6. Ma-Weibo 384D feature 无法复现；
7. Ma-Weibo 1021D adjacency signature 无法复现；
8. Ma-Weibo 3D causal structural feature 无法合理定义；
9. Ma-Weibo snapshot 不是时间单调包含；
10. 任何步骤需要执行智能体自行“猜一个定义”才能继续。

STOP 后：

```text
写 BLOCKER_REPORT.md
保存当前结果
停止
等待审批
```

---

# 24. 本轮禁止事项

不得：

- 正式训练 TC-DSCR；
- 实现 Dynamic Evidence Selector；
- 调用 Qwen3-8B 做正式推理；
- 修改 UMER 模型；
- 重新引入 11D user/sentiment；
- 自行决定正式时间窗口；
- 自行决定 chronological split；
- 更换 Ma-Weibo；
- 更换 embedding；
- 增加新数据集；
- 根据结果开始调参；
- 删除异常 event 以提高通过率；
- 将 unresolved parent 人工连接到 source；
- 使用 240h future statistics 计算 early snapshot 特征。

---

# 25. 本轮执行顺序

```text
D0  环境冻结
 ↓
D1  修复 PHEME 30min time-bin
 ↓
D2  修正 PHEME parent audit
 ↓
D3  PHEME 5m/15m/30m early statistics
 ↓
D4  Ma-Weibo raw schema / temporal audit
 ↓
D5  Ma-Weibo feature pipeline audit
 ↓
D6  Ma-Weibo full-event parity
 ↓
D7  Ma-Weibo snapshot distribution
 ↓
D8  Ma-Weibo causal snapshot prototype
 ↓
D9  Ma-Weibo chronological feasibility（若 absolute timestamp 可用）
 ↓
生成 delta_preflight_summary.md
 ↓
STOP
 ↓
用户提交 GitHub 结果给研究审批方
```

完成后不得进入正式实现。
