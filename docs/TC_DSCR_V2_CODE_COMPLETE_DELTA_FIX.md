# TC-DSCR V2 Code Complete Delta Fix 任务

> 项目：**TC-DSCR — Temporally Causal Dynamic Social-Context Refinement for LLM Rumor Detection**
>
> 本文档用于执行 **Code Complete 审批后的增量修复**。
>
> 当前状态：
>
> ```text
> TC-DSCR V2 Code Complete = PARTIAL
> Formal Experiments = NOT APPROVED
> ```
>
> 本轮任务不是重新实现 TC-DSCR，也不是进入正式实验。
>
> 执行智能体必须只修复本文档列出的 Code Complete 问题，完成后提交 GitHub 并停止。
>
> **禁止自行修改研究方法、模型结构、数据集、时间窗口、loss 权重、LLM、selector 设计或正式实验协议。**

---

# 1. 本轮目标

本轮只解决以下 5 个阻塞问题：

1. 正式 event-level 5-fold 数据编排未真正接入训练入口；
2. LLM prompt 的 source text 存在跨排序索引错位风险；
3. `recent_budget` baseline 排序方向错误；
4. `all_current` baseline 没有 Qwen context-length 保护；
5. 尚未完成 PHEME / Ma-Weibo 全数据集 1021-node cap 审计。

同时建议修复 2 个非阻塞一致性问题：

6. `L_fid` 日志应改为真正 KL divergence；
7. Ma-Weibo `original_text -> text` fallback 必须增加审计计数。

---

# 2. 当前禁止事项

本轮不得：

```text
运行正式 5-fold 模型训练
运行三 seed 正式实验
运行完整 baseline comparison
运行正式 Qwen 全数据推理
运行正式 lambda_n/lambda_p 搜索
运行 token budget 搜索
运行 ablation
修改 TC-DSCR 方法结构
修改数据集
修改时间窗口
修改 384D + 1024D 输入定义
修改 selector architecture
新增 loss
```

唯一允许的模型运行：

```text
unit tests
integration tests
tiny smoke
```

除此之外只允许做：

```text
full-dataset metadata / snapshot cap audit
```

该审计不得训练模型，不得调用 LLM。

---

# 3. 输出目录

本轮新增：

```text
results/tcdscr/code_delta_fix/
```

至少包含：

```text
test_report.txt
smoke_manifest.json
leakage_report.json
full_cap_report.json
fold_integrity_report.json
all_current_context_report.json
maweibo_text_fallback_report.json
delta_fix_summary.md
```

代码修改继续位于：

```text
project/tcdscr/
scripts/
```

不要创建新的方法分支目录。

---

# 4. Fix-1 — 正式 5-fold 数据编排

这是当前最重要阻塞项。

## 4.1 已发现问题

当前：

```python
event_folds_to_snapshot_folds()
```

实际返回：

```text
snapshot_key -> event_id
```

但应返回：

```text
snapshot_key -> fold_id
```

并且当前：

```text
tcdscr_train_encoder.py
tcdscr_train_selector.py
```

仍然直接加载 events 后训练，没有：

```text
outer fold
train
validation
test
```

的正式隔离。

---

# 5. Fix-1A — 修复 snapshot fold helper

修改：

```text
project/tcdscr/data/temporal_split.py
```

建议接口改为明确形式：

```python
def event_folds_to_snapshot_folds(
    snapshot_event_ids: dict,
    event_folds: dict,
) -> dict:
    return {
        snapshot_key: event_folds[event_id]
        for snapshot_key, event_id in snapshot_event_ids.items()
    }
```

不得继续返回 event_id。

若 event_id 不存在于 `event_folds`：

```text
raise KeyError / ValueError
```

不得静默忽略。

---

# 6. Fix-1B — 实现正式 Outer 5-Fold split

正式 Protocol A 必须保持：

```text
event-level stratified 5-fold
+
within-event strict temporal causality
```

必须：

```text
先划 event
再生成 / 使用该 event 的所有 snapshots
```

不得按 snapshot 划分。

## 6.1 Outer split

使用：

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 3090
```

对于 outer fold k：

```text
outer_test = fold k
outer_train_val = remaining 4 folds
```

## 6.2 Inner validation

从 outer_train_val 中再划：

```text
validation_fraction = 0.10
stratified
random_state = 3090
```

得到：

```text
train
validation
test
```

三个互斥 event 集合。

不得：

```text
对 snapshot 单独 train_test_split
```

## 6.3 输出接口

新增或修正一个正式接口，例如：

```python
build_primary_fold_split(
    event_labels,
    fold_index,
    seed=3090,
    validation_fraction=0.10,
)
```

返回：

```python
{
    "train": [event_ids],
    "validation": [event_ids],
    "test": [event_ids],
}
```

要求：

```text
train ∩ validation = ∅
train ∩ test = ∅
validation ∩ test = ∅

train ∪ validation ∪ test
= all events
```

---

# 7. Fix-1C — 训练入口必须接受 fold

修改：

```text
scripts/tcdscr_train_encoder.py
scripts/tcdscr_train_selector.py
```

正式入口必须至少支持：

```text
--fold 0..4
--partition-seed 3090
```

并且必须区分：

```text
train events
validation events
test events
```

当前 Code Delta Fix 阶段不得真正跑全量训练，但入口必须完成。

## 7.1 Encoder 入口

训练只能读取：

```text
train event IDs
```

validation：

```text
只用于 model selection / early stopping infrastructure
```

test：

```text
训练阶段绝不读取 label 参与 optimization
```

## 7.2 Selector 入口

同样必须：

```text
selector train = train event IDs only
selector validation = validation only
test = completely frozen
```

## 7.3 必须生成 split manifest

建议：

```text
results/.../split_manifest.json
```

字段：

```text
dataset
fold
partition_seed
train_event_ids
validation_event_ids
test_event_ids
train_label_counts
validation_label_counts
test_label_counts
```

---

# 8. Fix-1D — 必须新增真实测试

当前测试不能再自己手工构造 snapshot fold 来“证明”功能正确。

必须直接测试真实 helper。

至少新增：

```text
test_event_folds_to_snapshot_folds_returns_fold_ids
test_all_snapshots_inherit_real_event_fold
test_primary_split_disjoint
test_primary_split_covers_all_events
test_primary_split_stratified
test_train_entry_respects_fold
test_selector_entry_respects_fold
```

关键测试必须验证：

```python
snapshot_folds[(eid, 5)] == event_folds[eid]
snapshot_folds[(eid, 15)] == event_folds[eid]
```

---

# 9. Fix-2 — Source Claim 错位

## 9.1 已发现问题

当前 smoke/inference glue 中存在类似：

```python
src_pos = feat["node_ids"].index(feat["source_id"])
source_text = event["nodes"][src_pos]["text"]
```

这是错误的。

原因：

```text
feat / snapshot 节点顺序
= timestamp sorted

event["nodes"] 顺序
= raw adapter/original order
```

两个列表不能共用 position。

---

# 10. Fix-2A — 统一按 source_id 取 source text

正式规则：

```python
source_pos = snap["node_ids"].index(snap["source_id"])
source_text = snap["texts"][source_pos]
```

或：

```python
source_text = next(
    n["text"]
    for n in event["nodes"]
    if n["node_id"] == event["source_id"]
)
```

推荐优先使用 snapshot 自身：

```text
snap["node_ids"]
snap["texts"]
```

保证 prompt 与当前 snapshot 的 source 完全一致。

---

# 11. Fix-2B — 全局搜索同类错误

必须搜索：

```text
src_pos
source_pos
event["nodes"][...]
source_text
```

检查：

```text
scripts/tcdscr_smoke.py
context packer callers
正式 prompt generator
baseline runner
future Stage B runner
```

所有地方禁止：

> 使用一个数组的 index 去访问另一个排序体系的数组。

---

# 12. Fix-2C — 必须新增测试

构造事件：

```text
raw node order:
reply_A
source
reply_B
```

并让 timestamp 排序产生另一顺序。

验证：

```text
SOURCE CLAIM
```

必须精确等于 `source_id` 对应文本。

新增：

```text
test_prompt_source_bound_by_source_id
test_source_text_correct_when_raw_order_differs
```

---

# 13. Fix-3 — recent_budget 排序方向

## 13.1 当前错误

当前：

```python
return [-u["elapsed_seconds"] for u in units]
```

后续统一：

```text
score descending
```

因此实际优先更早节点。

---

# 14. Fix-3A — 正式修复

修改为：

```python
return [u["elapsed_seconds"] for u in units]
```

然后：

```text
descending
```

即：

```text
最新 reply 优先
```

---

# 15. Fix-3B — tie-break

同一 timestamp：

```text
保持 deterministic snapshot order
```

无需新增新规则。

---

# 16. Fix-3C — 必须新增测试

输入：

```text
reply A = 10s
reply B = 100s
reply C = 1000s
```

要求：

```text
rank:
C
B
A
```

新增：

```text
test_recent_budget_prefers_latest
```

---

# 17. Fix-4 — All-current Context Limit

## 17.1 已发现问题

当前 `all_current`：

```text
无视模型最大 context
把当前所有 evidence unit 全部加入 prompt
```

Ma-Weibo 大事件可能产生数千 reply。

这可能：

```text
超过 Qwen context
OOM
prompt 超长
baseline 不可运行
```

---

# 18. Fix-4A — 冻结正式规则

`All-current` 定义调整为：

> **模型在当前 snapshot 下能够实际读取的全部社会证据。**

按：

```text
snapshot deterministic order
(timestamp ascending, original_order tie)
```

依次加入完整 Reply–Parent Pair。

直到再加入下一个 pair 会导致：

```text
totalPromptTokens + maxNewTokens > modelContextLength
```

则停止。

---

# 19. Fix-4B — context length 获取

优先从模型 / tokenizer config 获取：

```python
model.config.max_position_embeddings
tokenizer.model_max_length
```

必须进行 sanity check。

禁止使用 tokenizer 常见的巨大 placeholder，例如：

```text
1000000000000000019884624838656
```

如果 tokenizer 的值明显无效：

```text
使用 model.config.max_position_embeddings
```

如果两者都不可确定：

```text
STOP
CONTEXT_LENGTH_UNRESOLVED
```

不得自行猜一个 32768 / 131072。

---

# 20. Fix-4C — token accounting

All-current 限制必须使用：

```text
Qwen3-8B tokenizer
```

并基于最终 chat-formatted prompt 计算。

必须预留：

```text
max_new_tokens = 8
```

即：

```text
input_tokens <= context_length - 8
```

---

# 21. Fix-4D — evidence pair 不得截断

仍必须遵守：

```text
Reply–Parent Pair = atomic unit
```

如果一个完整 pair 加入后超限：

```text
整个 pair 不加入
```

不得：

```text
截断 reply
截断 parent
只保留 reply
```

---

# 22. Fix-4E — 必须记录截断

每个 all-current prediction row 增加：

```text
all_current_truncated
units_before_truncation
units_after_truncation
tokens_dropped
context_length
input_tokens
```

---

# 23. Fix-4F — 必须新增测试

模拟：

```text
context_length = 100 tokens
```

构造多个 evidence pair。

验证：

```text
最终 prompt <= context_length - max_new_tokens
pair 不被切半
顺序保持 chronological
truncated = true
```

至少：

```text
test_all_current_respects_context_limit
test_all_current_never_partial_pair
test_all_current_preserves_snapshot_order
```

---

# 24. Fix-5 — 全数据 1021-node Cap Audit

当前 smoke cap report 不足以批准正式实验。

本轮必须对：

```text
完整 PHEME
完整 Ma-Weibo
```

运行 snapshot-level metadata audit。

不训练模型。

不生成 MiniLM embedding。

不调用 Qwen。

---

# 25. Fix-5A — 审计窗口

固定：

```text
5m
15m
30m
1h
3h
6h
24h
```

SOURCE_ONLY 不需要统计 cap。

---

# 26. Fix-5B — 每 dataset × cutoff 报告

```text
events
cap_hits
cap_hit_rate

mean_nodes_before_cap
median_nodes_before_cap
p90_nodes_before_cap
p95_nodes_before_cap
p99_nodes_before_cap
max_nodes_before_cap

mean_nodes_after_cap

total_nodes_removed
mean_removed_per_hit_event
p90_removed_per_hit_event
max_removed
```

---

# 27. Fix-5C — 额外 class breakdown

分别统计：

```text
rumor cap_hit_rate
nonrumor cap_hit_rate
```

目的是检查：

> 1021 cap 是否可能对某个类别产生系统性不对称。

---

# 28. Fix-5D — 事件级异常列表

输出 cap 最严重的前 50 个事件：

```text
dataset
event_id
label
cutoff
nodes_before
nodes_after
nodes_removed
source_timestamp
```

---

# 29. Fix-5E — 输出

写入：

```text
results/tcdscr/code_delta_fix/full_cap_report.json
```

同时 summary 中生成表：

```markdown
| dataset | cutoff | cap hit rate | P90 | P99 | max | removed |
```

---

# 30. Fix-5F — 本轮不得改变 1021 cap

即使发现 cap hit 很高：

```text
不要修改 max_nodes
不要改为 top-k
不要改为 sampling
不要扩到 2048
```

只报告。

由研究审批方决定下一步。

---

# 31. Optional Fix-6 — L_fid 改为真正 KL

## 31.1 当前实现

目前：

```text
-sum p_full * log p_sel
```

是 cross entropy。

虽然与：

```text
KL(p_full || p_sel)
```

对 selector 的梯度等价，因为 `p_full` stop-gradient，但日志值并非 KL。

---

# 32. Fix-6A — 推荐实现

可使用：

```python
p_ref = F.softmax(p_full.detach(), dim=-1)
log_p_ref = F.log_softmax(p_full.detach(), dim=-1)
log_q = F.log_softmax(p_sel, dim=-1)

l_fid = (p_ref * (log_p_ref - log_q)).sum()
```

或合法的：

```python
F.kl_div(...)
```

必须保证：

```text
target direction = KL(p_full || p_sel)
```

---

# 33. Fix-6B — 测试

新增：

```text
test_fidelity_loss_matches_manual_kl
```

验证数值，而不只是梯度。

---

# 34. Optional Fix-7 — Ma-Weibo text fallback 审计

当前允许：

```text
original_text 缺失
→ fallback 到 text
```

继续保留。

但不能静默。

---

# 35. Fix-7A — Adapter audit

增加计数：

```text
total_nodes
original_text_used
text_fallback_count
empty_text_count
fallback_rate
```

全数据输出：

```text
maweibo_text_fallback_report.json
```

---

# 36. 必须补充的 Tests

本轮完成后 pytest 至少增加以下测试：

```text
test_event_folds_to_snapshot_folds_returns_fold_ids
test_all_snapshots_inherit_real_event_fold
test_primary_split_disjoint
test_primary_split_covers_all_events
test_primary_split_stratified

test_prompt_source_bound_by_source_id
test_source_text_correct_when_raw_order_differs

test_recent_budget_prefers_latest

test_all_current_respects_context_limit
test_all_current_never_partial_pair
test_all_current_preserves_snapshot_order

test_fidelity_loss_matches_manual_kl

test_maweibo_text_fallback_count
```

若正式训练入口增加 parser / split helper：

还要增加：

```text
test_train_encoder_fold_argument
test_train_selector_fold_argument
```

---

# 37. Integration Test 必须增加的场景

新增一个非理想 synthetic event：

```text
raw order != timestamp order
source not raw index 0
same-timestamp replies
one unresolved parent
one future node
```

走完整：

```text
raw
→ snapshot
→ feature
→ encoder
→ selector
→ memory
→ evidence
→ prompt
```

验证：

```text
source text correct
no future leakage
same event fold consistent
recent baseline correct
all-current context bounded
```

---

# 38. Tiny Smoke 重跑要求

修复后重新跑 tiny smoke：

```text
PHEME <= 32 events
Ma-Weibo <= 32 events
SOURCE_ONLY + 15m + 1h
encoder 1 epoch
selector 1 epoch
Qwen <= 20 requests total
```

smoke 不能用于：

```text
性能判断
调参
方法修改
```

---

# 39. Smoke 必须新增检查

输出中明确报告：

```text
prompt_source_binding_failures = 0
fold_integrity_failures = 0
recent_baseline_order_failures = 0
context_overflow_failures = 0
future_leakage_failures = 0
```

---

# 40. Fold Integrity Report

生成：

```text
results/tcdscr/code_delta_fix/fold_integrity_report.json
```

至少包含两个数据集：

```text
events_total
fold_0_size
fold_1_size
fold_2_size
fold_3_size
fold_4_size

per_fold_rumor_ratio

cross_fold_event_count
snapshot_fold_mismatch_count
```

要求：

```text
cross_fold_event_count = 0
snapshot_fold_mismatch_count = 0
```

---

# 41. All-current Context Report

生成：

```text
all_current_context_report.json
```

可使用有限 smoke + synthetic stress，不需全数据 Qwen inference。

至少报告：

```text
context_length_detected
max_new_tokens
stress_cases
overflow_failures
partial_pair_failures
```

必须：

```text
overflow_failures = 0
partial_pair_failures = 0
```

---

# 42. delta_fix_summary.md 固定模板

```markdown
# TC-DSCR V2 Code Complete Delta Fix Summary

## Overall Status
PASS / PARTIAL / NOT_READY

## Git
- commit:

## Fix 1 — Five-Fold Protocol
- helper fixed:
- train entry fold-aware:
- selector entry fold-aware:
- cross-fold events:
- snapshot fold mismatches:

## Fix 2 — Source Binding
- source lookup:
- adversarial raw-order test:
- failures:

## Fix 3 — Recent Baseline
- scoring direction:
- ordering test:

## Fix 4 — All-current Context Limit
- context length:
- max_new_tokens:
- overflow failures:
- partial-pair failures:

## Fix 5 — Full Dataset Cap Audit

### PHEME
| cutoff | cap hit rate | P90 | P99 | max | total removed |
|---|---:|---:|---:|---:|---:|

### Ma-Weibo
| cutoff | cap hit rate | P90 | P99 | max | total removed |
|---|---:|---:|---:|---:|---:|

## KL Fix
- implemented:
- numeric test:

## Ma-Weibo Text Fallback
- total nodes:
- fallback count:
- fallback rate:

## Tests
- total:
- passed:
- failed:

## Tiny Smoke
- PHEME:
- Ma-Weibo:
- LLM requests:
- future leakage failures:
- source binding failures:
- fold failures:
- context overflow failures:

## Blocking Issues
1.
2.

## Recommended Status
PASS / PARTIAL / NOT_READY
```

---

# 43. PASS 条件

本轮可标记 PASS 仅当：

```text
5-fold helper 语义正确
训练入口 fold-aware
train/val/test event 严格隔离
snapshot fold mismatch = 0

source prompt binding 按 source_id
source binding test = PASS

recent_budget 最新优先
recent test = PASS

all_current 不超过模型 context
partial pair = 0

full PHEME cap audit 完成
full Ma-Weibo cap audit 完成

全部 pytest 通过
tiny smoke 通过
future leakage = 0
```

---

# 44. STOP 条件

出现任一情况立即停止，不得进入 Formal Experiments：

```text
train/test event overlap
snapshot cross-fold
source prompt text 仍错位
all-current context length 无法可靠获取
all-current 仍 overflow
future leakage
full-data cap audit 无法完成
测试失败
```

输出：

```text
BLOCKER_REPORT.md
```

---

# 45. 当前执行顺序

执行智能体必须按顺序：

```text
STEP 1
修复 temporal_split fold helper

STEP 2
实现正式 5-fold + inner validation split

STEP 3
让 encoder / selector entry points fold-aware

STEP 4
修复 source text binding

STEP 5
修复 recent_budget

STEP 6
实现 all_current context limit

STEP 7
修复真正 KL 日志

STEP 8
增加 Ma-Weibo fallback audit

STEP 9
补 unit tests

STEP 10
补 integration tests

STEP 11
运行完整 pytest

STEP 12
运行 PHEME / Ma-Weibo full cap audit

STEP 13
重跑 tiny smoke

STEP 14
生成 code_delta_fix 所有报告

STEP 15
提交 GitHub

STEP 16
STOP
```

---

# 46. 最终限制

本轮结束后：

```text
不要启动 Formal E1
不要跑正式五折
不要跑正式 Qwen baseline
```

即使：

```text
delta_fix_summary = PASS
```

也必须先：

```text
提交 GitHub
停止
等待研究审批
```

**本轮唯一目标：把 Code Complete 修到可以安全进入 Formal Experiments 的状态。**
