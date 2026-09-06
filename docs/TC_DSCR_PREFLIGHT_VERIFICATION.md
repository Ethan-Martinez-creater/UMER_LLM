# TC-DSCR 实验前置验证任务（必须先执行）

> 本文档只用于判断 **TC-DSCR 是否具备进入代码实现与正式实验的基础条件**。  
> 执行结束后必须停止，把结果交给用户。  
> **不得在本阶段训练正式模型、设计新算法、运行大规模 LLM 推理或自行修复研究方案。**

---

# 1. 验证目标

需要回答六个问题：

1. PHEME 是否能从原始数据严格重建任意时间点的传播子图？
2. Weibo22 是否提供足够的时间、文本和传播关系信息来做同样的工作？
3. 当前 UMER 的 395D `node_feat` 和 1024D `struct_feat` 是否有可找到、可复现的生成逻辑？
4. 能否从 raw full event 重建现有 240h/full-event `.pt`，从而证明 causal snapshot feature builder 不会改变特征定义？
5. PHEME 与 Weibo22 的传播时间分布是否支持合理的多时间快照？
6. 两个数据集是否支持严格 chronological train/validation/test 划分？

任何关键问题不能回答，就不应直接进入正式 TC-DSCR 实验。

---

# 2. 输出目录

所有验证结果写入：

```text
results/tcdscr/preflight/
```

必须生成：

```text
environment.json
repo_inventory.json
pheme_audit.json
weibo22_audit.json
feature_pipeline_audit.json
full_parity_report.json
snapshot_distribution.csv
chronological_split_candidates.csv
preflight_summary.md
```

原始数据不得提交仓库。

---

# 3. Step P0 — 环境与版本冻结

记录：

```text
git commit SHA
Python version
PyTorch version
Transformers version
CUDA version
GPU model
OS
existing PHEME raw path
existing PHEME processed path
existing UMER checkpoint paths
```

写入 `environment.json`，同时确认旧 UMER 基线代码仍可导入/运行。不要修改旧 UMER。

---

# 4. Step P1 — 当前 UMER 特征生成链审计

当前已知 UMER 输入：

```text
node_feat: [N,395]
  384D text semantic
  11D extra node/user

struct_feat: [N,1024]
  1021D adjacency signature
  3D propagation summary
```

在本地工作区、历史脚本、旧工程目录中寻找原始 feature generation 代码。

必须明确找到：

1. 384D text embedding 模型及版本；
2. 文本预处理；
3. 11D 每一维定义；
4. 缺失 user field 的处理；
5. 1021D adjacency signature 构造公式；
6. 节点排序如何映射到 signature；
7. 最大节点数 / 截断规则；
8. 3D propagation summary 定义；
9. depth 计算；
10. relative time 单位和 normalization；
11. 240h cutoff 实现位置；
12. source-only 处理。

输出字段表：

```text
feature_name
dimension
definition
source_file
source_line_or_function
depends_on_future_graph
recomputable_from_snapshot
```

### STOP-P1

若找不到 11D 或 1021D 的真实生成逻辑：

- 不得猜测；
- 不得根据模型代码自行编一套；
- 标记 `NOT_READY_FEATURE_PIPELINE`；
- 完成其余只读审计后停止。

---

# 5. Step P2 — PHEME raw temporal audit

对全部 PHEME event 检查：

```text
source timestamp
reply timestamp
reply parent
child timestamp < parent timestamp
重复 node ID
多个 root
orphan
空 text
```

输出：event/node 总数、timestamp/parent coverage、chronology violation、orphan、source-only、节点数 P50/P90/P99、event duration P50/P90/P99。

通过要求：

```text
source timestamp coverage = 100%
reply timestamp coverage >= 99.9%
parent relation coverage >= 99.9%
```

任何 chronology violation 必须列 ID，不得自动改时间。

---

# 6. Step P3 — 下载与审计 Weibo22

官方仓库：

```text
https://github.com/kkkkk001/KPG
```

Release：

```text
https://github.com/kkkkk001/KPG/releases/tag/v1
```

记录 repository commit SHA、release tag、下载日期和全部下载文件 SHA256。

上游 README 指示解压 `data/Weibo` 中两个 zip，并执行：

```bash
python Process/getWeibograph.py
```

先保留完整 raw 副本，上游图构建结果只能作为参考。

必须审计：

```text
source ID
source text
source absolute timestamp
reply/repost/comment ID
reply/repost/comment text
node timestamp
parent/forward relation
label
user fields
```

特别回答：

1. 是否能确定每个传播节点 absolute timestamp？
2. 是否能确定 parent-child？
3. 如果是 repost chain，图语义是什么？
4. 是否只有顺序而无真实时间？
5. user fields 是否足以复现 UMER 11D？
6. 是否有 source absolute time 做 global chronological split？

### STOP-P3

若无可靠 node timestamp、无足够传播关系或无法形成 \(G_t\)：标记 `NOT_READY_WEIBO22_TEMPORAL`，不得自行换数据集。

---

# 7. Step P4 — Full-event feature parity

仅在 P1 找到完整 feature pipeline 时执行。

固定：

```text
seed = 3090
sample = 100 PHEME events
```

样本覆盖 source-only、小/中/大图、rumor/non-rumor。

比较：

```text
node_ids
num_nodes
edge_index
node_feat[:,:384]
node_feat[:,384:]
struct_feat[:,:1021]
struct_feat[:,-3:]
```

通过标准：

```text
node_ids exact = 100%
edge_index exact = 100%
num_nodes exact = 100%
max_abs_diff <= 1e-5
```

如 embedding 因库版本存在差异，可额外报告 cosine similarity，但不得自行放宽标准。

若无法解释 parity failure，标记 `NOT_READY_PARITY`。

---

# 8. Step P5 — Snapshot distribution audit

仅用于统计的候选窗口：

```text
0h 1h 3h 6h 12h 24h 48h
```

PHEME / Weibo22 分别统计：

```text
events_total
events_source_only
source_only_ratio
mean/median/P90 nodes
mean edges
median/P90 depth
rumor mean_nodes
nonrumor mean_nodes
```

累计 reply coverage：

\[
coverage(t)=\#reply_{<=t}/\#reply_{full}
\]

报告 P25/P50/P75/P90。

不得自行决定正式窗口。

---

# 9. Step P6 — Chronological split feasibility

按 source absolute timestamp 排序，审计候选：

```text
60/20/20
70/10/20
70/15/15
```

输出 train/val/test event 数、类别数/比例、时间跨度、source-only 比例；PHEME 有 event/topic family 时统计 family overlap，Weibo22 统计年份分布。

若所有 chronological split 都导致 validation/test 某类严重不足或没有 absolute time，只报告，不改 random split。

---

# 10. Step P7 — 384D 中文语义兼容性

确认：

1. 当前 PHEME / Ma-Weibo 历史预处理的 384D embedding model；
2. 是否同一模型；
3. 是否 multilingual；
4. Weibo22 是否可合法复用；
5. 权重是否仍存在；
6. tokenizer/version 是否可冻结。

若当前 384D encoder 不支持中文或来源不明，标记：

```text
NEED_TEXT_ENCODER_DECISION
```

不得自行换 BGE/E5/MiniLM。

---

# 11. Step P8 — Weibo22 与 UMER 11D user feature 兼容性

逐维输出：

```text
feature_1 ... feature_11
required_raw_field
PHEME availability
Weibo22 availability
missing_rate
can_compute_exactly
```

禁止缺失直接补 0、近似替代或修改 11D 定义。

若无法完整构造，标记：

```text
NEED_FEATURE_SCHEMA_DECISION
```

---

# 12. Step P9 — 最小 causal snapshot prototype

仅在 P1–P4 不阻塞时执行。

固定抽 10 个 PHEME event（seed=3090），构建：

```text
1h 6h 24h
```

只检查 node/edge inclusion、feature recomputation、monotonicity、future leakage，不训练模型。

每个 snapshot 输出：event_id、cutoff、included/excluded node IDs、edge list、degree/depth/time。

随机 3 个 event 生成人工可读：

```text
results/tcdscr/preflight/manual_snapshot_examples.md
```

---

# 13. 最终报告模板

`preflight_summary.md` 必须包含：

```markdown
# TC-DSCR Preflight Summary

## Overall Status
PASS / PARTIAL / NOT_READY

## PHEME
- temporal readiness:
- parent readiness:
- snapshot readiness:

## Weibo22
- download source:
- temporal readiness:
- propagation readiness:
- global chronology readiness:

## UMER Feature Pipeline
- 384D:
- 11D:
- 1021D:
- 3D:
- source code locations:

## Full-event Parity
- node parity:
- edge parity:
- node_feat parity:
- struct_feat parity:

## Candidate Snapshot Statistics
...

## Chronological Split Candidates
...

## Chinese Semantic Compatibility
...

## Blocking Issues
1.
2.

## Questions Requiring Research Approval
1.
2.

## Files Produced
...
```

---

# 14. PASS 条件

同时满足才可建议 PASS：

1. PHEME 可严格构造 snapshots；
2. Weibo22 可严格构造 snapshots；
3. 两者有 source absolute time；
4. UMER feature generation 逻辑已定位；
5. PHEME full-event parity 通过或差异完全可解释；
6. 1024D struct feature 可从 snapshot 重算；
7. Weibo22 11D 兼容性明确；
8. 384D 中文 embedding 方案明确；
9. 至少一个合理 chronological split 候选；
10. prototype 无 future leakage。

有待审批的模型 schema 问题但数据可行时使用 `PARTIAL`，不得擅自 PASS。

---

# 15. 本阶段绝对禁止

- 正式训练 TC-DSCR；
- 新增算法；
- 改 UMER 结构；
- 换数据集；
- 换 LLM；
- 换 embedding model；
- 修改 11D 定义；
- 自行决定正式时间窗口；
- 自行决定 chronological split；
- 大规模调用 Qwen；
- 为通过 parity 修改旧 `.pt`；
- 把 preflight 结果包装成论文结论。

完成报告后立即停止，等待用户提交研究审批。
