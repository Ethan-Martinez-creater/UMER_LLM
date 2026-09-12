基于当前最新仓库提交，正式执行：

# TC-DSCR V3-B — Frozen Qwen3-8B Reader-Transfer Pilot

当前研究状态已经冻结：

```text
E1 Random-init Causal Encoder = VALID
Corrected E2 Static Utility Selector = VALID
Dynamic V1 = REJECTED
MF-TSR V2 = REJECTED AS FINAL METHOD
MS-TSR V3-A = PASS

V3-A commit:
656c611af7598295c3323f867f2929b89ec363fe

V3-B Reader Transfer Pilot = APPROVED
Held-out test = NOT APPROVED
Full E4 = NOT APPROVED
```

本轮唯一目标是：

> 验证 V3-A 中由 frozen Proxy 定义的 Minimal-Sufficient Social Context，是否能够迁移到一个完全独立的 frozen Qwen3-8B reader。

换句话说，本轮要回答：

```text
Static social context
vs
MS-TSR compressed social context
```

在真实 LLM reader 上：

```text
检测能力是否基本保持？
context token 是否显著降低？
prediction / confidence 是否出现系统性退化？
grounding / evidence citation 是否仍然有效？
```

本轮禁止修改 MS-TSR、禁止调 alpha、禁止读取 held-out test、禁止 fine-tune Qwen。

---

## 1. 冻结 V3-A 配置

所有 fold：

```text
alpha = 0.8
budget = 1024
```

不得：

```text
重新搜索 alpha
比较 0.8 / 0.9 / 0.95 / 1.0 后选择 Qwen 最优值
根据 Qwen 结果改 alpha
根据 dataset 单独改 alpha
修改 candidate pool
修改 MS-TSR objective
```

正式 MS-TSR context 必须直接来自当前 V3-A 冻结结果。

---

## 2. 数据范围

仅使用：

```text
validation split
```

数据集：

```text
PHEME
Ma-Weibo
```

不得读取：

```text
outer test labels
outer test predictions
outer test events for inference
```

如果 runner 发现 test 被读取：

```text
TEST_PROTOCOL_VIOLATION
STOP
```

---

## 3. Pilot 样本量

每个 dataset 固定：

```text
300 unique event-cutoff samples
```

因此总共：

```text
PHEME = 300
Ma-Weibo = 300

total paired samples = 600
```

每个 sample 同时生成：

```text
Arm A = Static Context
Arm B = MS-TSR Context
```

所以 Qwen inference records 总量为：

```text
600 × 2 = 1200 arm evaluations
```

不要把两个 arm 当成独立采样。

它们必须严格 paired。

---

## 4. Sampling seed

固定：

```text
seed = 3090
```

采样 manifest 在第一次 Qwen inference 前必须写入磁盘并冻结。

生成：

```text
results/tcdscr/dynamic_v3_reader/
sampling_manifest.json
```

manifest 一旦生成：

```text
不得根据 Qwen 结果修改
不得补换“难样本”
不得删除表现不好的样本
```

---

## 5. Sampling source

样本必须来自当前 V3-A：

```text
validation predictions / selected context artifacts
```

每个 sample 至少包含：

```text
dataset
fold
seed
event_id
cutoff
gold label

candidate_count
selection_pressure

static_selected_node_ids
ms_selected_node_ids

static_evidence_tokens
ms_evidence_tokens

dual_view_agree
fallback_to_static
static_margin
ms_margin
```

---

## 6. 去重规则

Pilot sampling unit：

```text
event-cutoff
```

同一个：

```text
dataset + event_id + cutoff
```

只能出现一次。

不能因为三个 E2 seeds 都产生同一个 event-cutoff 就重复采样三次。

如果多个 seed 对同一个 event-cutoff 的 MS-TSR set 不完全一致：

固定选择：

```text
seed = 2000
```

作为 Reader Pilot 的 context generation source。

本轮不要把不同 E2 seeds 当成 3 个 LLM 样本。

原因：

> LLM reader pilot 的 sampling unit 是真实 event-cutoff，而不是模型 seed。

---

## 7. Stratified Sampling — cutoff

每个 dataset 的 300 samples 必须覆盖：

```text
5m
15m
30m
1h
3h
6h
```

尽可能均衡：

```text
约 50 samples / cutoff
```

如果某 cutoff 可用 validation event 不足：

1. 全部保留该 cutoff；
2. 剩余额度按其他 cutoff 均匀补齐；
3. 记录 allocation deviation。

不得人为删除 5m / 15m。

---

## 8. Stratified Sampling — selection pressure

继续沿用已经冻结的定义。

对每个 event-cutoff：

```text
saturation =
static_selected_count / candidate_count
```

当：

```text
candidate_count = 0
```

单独标记：

```text
NO_CANDIDATE
```

非空情况下：

```text
Low selection pressure:
saturation >= 0.8

Medium:
0.4 <= saturation < 0.8

High:
saturation < 0.4
```

sampling 时尽量保证：

```text
Low
Medium
High
```

都有代表性。

但：

```text
不要为了比例平衡过采极少见 bin
```

优先保证：

```text
cutoff coverage
+
真实数据分布
```

并在 manifest 中记录实际分层数量。

---

## 9. No-candidate samples

允许出现在 Pilot 中。

对于：

```text
candidate_count = 0
```

Static 与 MS context 均为空。

仍然执行 Qwen inference：

```text
source post only
```

目的：

> 检查早期没有任何 social evidence 时 LLM 的行为。

这些样本：

```text
不计入 context compression gain
```

但计入：

```text
overall detection
early-stage behavior
```

---

## 10. Qwen model

固定使用项目已经计划的：

```text
Qwen3-8B
```

必须使用当前服务器中既定的同一 checkpoint。

在 run manifest 中记录：

```text
model path / model id
model config
dtype
device
checkpoint hash if available
```

禁止：

```text
fine-tune
LoRA
adapter
gradient update
```

模型必须：

```text
eval()
torch.no_grad() / inference_mode()
```

---

## 11. Inference configuration

Static 和 MS 两个 arm 必须完全一致。

固定：

```text
temperature = 0
do_sample = false
```

如果当前 transformers/Qwen API 需要额外设置：

```text
top_p
top_k
```

则在 deterministic inference 下禁用或设置为不生效。

固定：

```text
same max_new_tokens
same system prompt
same user prompt template
same source post
same evidence formatting
same ordering policy
```

唯一差异：

```text
social evidence set
```

---

## 12. Evidence ordering

Arm A Static：

```text
使用 Static Selector 的最终顺序
```

Arm B MS-TSR：

不得按 node id 随机排序。

固定排序：

```text
按照当前 snapshot 中 evidence timestamp ascending
```

若 timestamp 相同：

```text
original stable order
```

Static arm 也统一采用同样的：

```text
timestamp ascending
```

作为最终 prompt presentation order。

原因：

> Reader Pilot 比较的是 evidence content/set，不应混入排序差异。

Selection algorithm 内部顺序保持原算法，不修改；这里只规定最终 prompt rendering。

---

## 13. Reply–Parent Evidence Unit

继续使用：

```text
Reply–Parent Pair
```

不得改变 evidence atomic unit。

每条 evidence 在 prompt 中至少包含：

```text
Evidence ID
Reply text
Parent text
relative timestamp / cutoff-visible time information
```

不要把隐藏未来信息写进 prompt。

---

## 14. Evidence ID

每个 evidence unit 必须分配稳定 ID，例如：

```text
E1
E2
E3
...
```

同时保存映射：

```text
prompt evidence ID
→ original node_id
```

Qwen 输出只能引用 prompt 中存在的 evidence ID。

---

## 15. Prompt 设计原则

本轮 prompt 必须固定一次后冻结。

不要根据结果迭代 prompt。

System prompt 的任务含义应为：

```text
You are evaluating whether a social-media source post is a rumor.

You must base the judgment only on:
1. the source post
2. the supplied social evidence

Do not use external knowledge.
Do not invent facts that are not supported by the supplied evidence.

If the supplied evidence is weak, incomplete, or conflicting,
reflect that uncertainty in confidence.

Return only the required JSON object.
```

中文数据可保留原始中文文本，不翻译。

英文 PHEME 保留英文。

---

## 16. User Prompt 固定结构

建议：

```text
Source Post:
<source text>

Observed Social Evidence up to <cutoff>:
[E1]
Reply: ...
Parent: ...

[E2]
Reply: ...
Parent: ...

...

Task:
Determine whether the Source Post should be classified as:

RUMOR
or
NON_RUMOR

Return JSON only.
```

不要在 prompt 中透露：

```text
gold label
Static prediction
MS-TSR prediction
full encoder prediction
fold
seed
confidence stratum
```

---

## 17. Qwen 输出 schema

固定要求：

```json
{
  "label": "RUMOR",
  "confidence": 0.73,
  "evidence_ids": ["E2", "E5"],
  "reason": "..."
}
```

允许 label 只有：

```text
RUMOR
NON_RUMOR
```

confidence：

```text
0.0 <= confidence <= 1.0
```

evidence_ids：

```text
只能引用 prompt 中实际存在的 evidence ID
```

reason：

```text
简洁即可
```

---

## 18. JSON parsing

实现 robust parser。

如果第一次输出不能解析：

允许：

```text
同一输入再进行 1 次 deterministic formatting retry
```

retry prompt 只能要求：

```text
Return valid JSON matching the required schema.
```

不得增加关于 gold label 或正确答案的提示。

最多：

```text
1 retry
```

仍失败则标记：

```text
PARSE_FAILURE
```

不得人工修结果。

---

## 19. 两个 arm 的执行顺序

避免固定顺序可能造成系统性 runtime/cache 差异。

对于每个 sample：

通过固定 hash：

```text
hash(dataset,event_id,cutoff,3090)
```

决定：

```text
Static first
or
MS first
```

但两个 arm 都必须执行。

最终统计仍恢复成 paired Static/MS。

---

## 20. Prompt context token accounting

正式统计：

```text
source_tokens
static_social_tokens
ms_social_tokens

static_total_input_tokens
ms_total_input_tokens
```

Primary compression metric：

```text
social_context_token_reduction
=
1 - ms_social_tokens / static_social_tokens
```

当 Static social tokens = 0：

```text
compression metric = N/A
```

不要设成 0 后混入 compression mean。

同时额外报告：

```text
total_prompt_token_reduction
```

以区分 source post 固定开销。

---

## 21. Primary detection metrics

分别对 Static-Qwen 和 MS-Qwen 计算：

```text
Accuracy
Macro-F1
Weighted-F1
Rumor-F1
```

overall：

```text
PHEME 300
Ma-Weibo 300
```

另外按 cutoff 报告：

```text
5m
15m
30m
1h
3h
6h
```

---

## 22. Primary Reader-Transfer Delta

定义：

```text
Delta Macro-F1
=
MF1_MS_Qwen
-
MF1_Static_Qwen
```

本轮 frozen gate：

```text
Delta Macro-F1 >= -0.005
```

即最多允许：

```text
0.5 percentage point
```

绝对下降。

不得根据结果修改 threshold。

---

## 23. Context reduction gate

只在：

```text
static_social_tokens > 0
```

的 paired samples 上计算。

要求：

```text
mean social context token reduction >= 30%
```

如果低于：

```text
30%
```

Reader Transfer FAIL。

---

## 24. Paired prediction analysis

对每个 paired sample 分类：

```text
Static correct / MS correct
Static correct / MS wrong
Static wrong / MS correct
Static wrong / MS wrong
```

正式报告：

```text
wrong_to_correct
correct_to_wrong
net_correction_gain
```

其中：

```text
net_correction_gain
=
wrong_to_correct - correct_to_wrong
```

---

## 25. Prediction disagreement

报告：

```text
Static-Qwen vs MS-Qwen disagreement rate
```

overall + per cutoff。

同时分解：

```text
label changed but both wrong
label changed Static→correct
label changed Static→wrong
```

---

## 26. Confidence analysis

分别统计：

```text
mean confidence Static
mean confidence MS

confidence delta
```

还必须分析：

### Correct predictions

```text
mean confidence
```

### Wrong predictions

```text
mean confidence
```

目的：

> 检查 context compression 是否导致 overconfidence。

---

## 27. Confidence calibration diagnostic

本轮不要求完整 calibration paper-level analysis，但至少报告：

```text
ECE
```

Static 与 MS 各一份。

固定：

```text
10 equal-width bins
```

confidence 使用 Qwen 输出值。

这只是 diagnostic，不作为 V3-B gate。

---

## 28. Evidence citation validity

对于 Qwen 输出：

```text
evidence_ids
```

统计：

```text
valid_citation_rate
=
引用的 evidence IDs 中实际存在于该 arm prompt 的数量
/
全部引用 evidence ID 数量
```

如果没有引用任何 evidence：

单独统计：

```text
no_citation_rate
```

不得把 empty citation 自动计为 100% valid。

---

## 29. Unsupported citation

如果 Qwen 引用：

```text
不存在的 evidence ID
```

记录：

```text
unsupported_citation_count
unsupported_citation_rate
```

这是当前 hallucination / grounding 的第一层 proxy metric。

---

## 30. Evidence coverage

报告：

```text
mean cited evidence count
```

以及：

```text
cited evidence / supplied evidence
```

但仅作为 diagnostic。

---

## 31. Reason grounding heuristic

本轮不要引入新的 LLM judge。

只做确定性检查：

如果：

```text
reason 中显式出现 [E#] 或 E#
```

则检查是否属于 supplied evidence IDs。

报告：

```text
reason_invalid_evidence_reference_rate
```

不要做 semantic hallucination judge。

---

## 32. Paired bootstrap

必须做 event-cutoff level paired bootstrap。

每 dataset：

```text
iterations = 10000
seed = 3090
sampling unit = paired event-cutoff sample
```

必须保留 bootstrap multiplicity。

不得再出现此前：

```text
list -> dict
```

导致重复 sample 被折叠的问题。

计算：

```text
Delta Macro-F1
95% percentile CI
```

Primary interpretation：

```text
point estimate
+
95% CI
```

---

## 33. McNemar test

对 Static vs MS paired correctness：

构造：

```text
Static correct / MS wrong
Static wrong / MS correct
```

执行 exact McNemar test 或 binomial equivalent。

报告：

```text
p-value
```

只作为统计补充。

不要仅凭 p-value 决策。

---

## 34. Stratified reader analysis — cutoff

分别报告：

```text
5m
15m
30m
1h
3h
6h
```

每个 cutoff：

```text
n
Static MF1
MS MF1
delta
token reduction
disagreement
```

---

## 35. Stratified reader analysis — selection pressure

按 sampling manifest 的：

```text
NO_CANDIDATE
LOW
MEDIUM
HIGH
```

报告：

```text
n
Static accuracy/MF1
MS accuracy/MF1
delta
token reduction
```

尤其重点检查：

```text
HIGH selection pressure
```

因为这是 MS-TSR 最应该发挥作用的场景。

---

## 36. Stratified reader analysis — Proxy margin

使用 V3-A Static margin。

在 sampling population 内固定分：

```text
Low = bottom 25%
Medium = 25–75%
High = top 25%
```

报告：

```text
Static Qwen MF1
MS Qwen MF1
Delta
token reduction
disagreement
```

不得重新基于 Qwen confidence 分组选择方案。

---

## 37. Fallback vs compressed samples

必须分别分析：

```text
fallback_to_static = true
fallback_to_static = false
```

在 fallback samples：

两个 context 应完全相同。

必须 assertion：

```text
Static prompt evidence IDs
==
MS prompt evidence IDs
```

对应 Qwen deterministic inference 理论上也应相同。

若同 context、同 prompt、temperature=0 时输出不同：

标记：

```text
DETERMINISM_WARNING
```

并报告数量。

---

## 38. MS context integrity

每个 MS arm：

必须核对：

```text
prompt evidence IDs
==
V3-A selected evidence IDs
```

不得在 prompt 构造时：

```text
重新排序后丢失 evidence
自动截断 evidence
增加 Static evidence
去掉 MS evidence
```

如果模型最大 context 限制导致 prompt 超长：

```text
不要自行截断
标记 CONTEXT_OVERFLOW
STOP FOR THAT SAMPLE
```

但在当前 1024 social-token budget 下正常不应发生。

---

## 39. Prompt source integrity

Source post：

Static/MS 两个 arm 必须：

```text
byte/text identical
```

同一个 paired sample：

```text
source text
cutoff
task instructions
```

必须完全一致。

只有 social evidence block 不同。

---

## 40. Qwen 不参与 selection

禁止：

```text
先让 Qwen 评价 evidence
再选择 evidence

让 Qwen rerank

用 Qwen confidence 修改 MS set

根据 Qwen 输出补 evidence
```

Qwen 是：

```text
frozen external reader
```

仅用于 evaluation。

---

## 41. 本轮 Gate

Dataset-level V3-B PASS 必须同时满足：

### A. Reader non-inferiority

```text
Delta Macro-F1 >= -0.005
```

### B. Social-context reduction

```text
mean token reduction >= 30%
```

### C. Grounding integrity

要求：

```text
unsupported citation rate <= 1%
```

如果没有 citation：

```text
不要因此自动 FAIL
```

但必须报告 no-citation rate。

---

## 42. CLEAR_TRANSFER / WEAK_TRANSFER / FAIL 标签

在 Gate 基础上增加解释标签。

### CLEAR_TRANSFER

```text
Delta MF1 >= -0.005
AND
bootstrap CI lower bound >= -0.01
AND
token reduction >= 30%
AND
unsupported citation <= 1%
```

### WEAK_TRANSFER

满足 dataset gate：

```text
Delta MF1 >= -0.005
token reduction >= 30%
citation integrity pass
```

但 bootstrap CI 较宽。

### TRANSFER_FAIL

任一：

```text
Delta MF1 < -0.005
or
token reduction < 30%
or
unsupported citation > 1%
```

---

## 43. Overall decision

如果：

```text
PHEME PASS
AND
Ma-Weibo PASS
```

则：

```text
Overall V3-B = PASS
Recommendation = START_V3_C_HELD_OUT_TEST
```

如果：

```text
only one dataset PASS
```

则：

```text
Overall = PARTIAL
Recommendation = STOP_FOR_RESEARCH_REVIEW
```

如果：

```text
both FAIL
```

则：

```text
Overall = FAIL
Recommendation = PROXY_READER_TRANSFER_FAIL
```

不得自动进入下一阶段。

---

## 44. 不得把 Pilot 当作最终论文结果

这 600 samples 来自：

```text
validation
```

所以 V3-B 只回答：

```text
transfer feasibility
```

不得写成：

```text
final held-out performance
```

不得与 test result 混合。

---

## 45. 输出目录

新建：

```text
results/tcdscr/dynamic_v3_reader/
```

建议结构：

```text
dynamic_v3_reader/
├── sampling_manifest.json
├── prompts/
│   ├── pheme/
│   └── maweibo/
├── raw_generations/
├── parsed/
├── statistics/
├── diagnostics/
├── V3_B_READER_TRANSFER_REPORT.md
├── reader_transfer_summary.json
└── reader_transfer_verify.json
```

---

## 46. Sampling manifest

至少：

```json
{
  "seed": 3090,
  "dataset": "pheme",
  "event_id": "...",
  "cutoff": "30",
  "fold": 0,
  "context_source_seed": 2000,
  "gold": 1,
  "candidate_count": 20,
  "pressure_bin": "HIGH",
  "static_margin": 3.1,
  "static_selected_node_ids": [],
  "ms_selected_node_ids": [],
  "static_social_tokens": 600,
  "ms_social_tokens": 120,
  "fallback_to_static": false
}
```

---

## 47. Raw generation artifact

每个 arm 保存：

```text
sample_id
dataset
event_id
cutoff
arm
prompt_hash
input_token_count
social_context_token_count
raw_output
parse_status
retry_used
latency if available
```

---

## 48. Parsed artifact

保存：

```text
sample_id
arm

gold
parsed_label
correct

confidence

evidence_ids
valid_evidence_ids
invalid_evidence_ids

reason

parse_failure
```

---

## 49. Verifier

新增：

```text
scripts/tcdscr_verify_v3_reader_transfer.py
```

至少检查：

```text
exactly 300 unique paired samples per dataset

exactly one Static and one MS inference per sample

sampling manifest created before generations

seed = 3090

validation only
no test events

alpha = 0.8
budget = 1024

context source seed = 2000

MS evidence IDs exactly match frozen V3-A artifact

Static evidence IDs exactly match Static frozen artifact

paired source text identical

paired prompts differ only in social evidence block

same model/config

temperature = 0
do_sample = false

Qwen weights frozen

no Qwen output used for selection

bootstrap preserves multiplicity

fallback pair contexts identical

unsupported evidence IDs correctly counted
```

要求：

```text
issues = 0
```

---

## 50. Tests

至少增加：

```text
test_reader_sampling_unique_event_cutoff

test_reader_sampling_uses_validation_only

test_reader_sampling_seed_fixed

test_static_ms_pair_same_source

test_static_ms_pair_only_context_differs

test_ms_prompt_matches_frozen_v3_selection

test_fallback_prompts_identical

test_json_parser_valid_output

test_json_parser_one_retry_only

test_invalid_evidence_id_detected

test_bootstrap_preserves_multiplicity

test_context_token_reduction_excludes_zero_static_context

test_qwen_not_used_for_selection

test_alpha_frozen_08

test_reader_gate_logic
```

---

## 51. V3_B_READER_TRANSFER_REPORT

生成：

```text
results/tcdscr/dynamic_v3_reader/
V3_B_READER_TRANSFER_REPORT.md
```

固定结构至少：

```markdown
# TC-DSCR V3-B — Frozen Qwen Reader Transfer

## Protocol
- model:
- checkpoint:
- validation only:
- samples:
- sampling seed:
- alpha:
- budget:

## PHEME

### Detection
| Arm | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 |
|---|---:|---:|---:|---:|
| Static | | | | |
| MS-TSR | | | | |

Delta Macro-F1:
Bootstrap 95% CI:
McNemar p:

### Compression
Static social tokens:
MS social tokens:
Reduction:

Static total prompt tokens:
MS total prompt tokens:
Reduction:

### Paired Outcomes
wrong -> correct:
correct -> wrong:
both correct:
both wrong:

### Disagreement
...

### Confidence / Calibration
...

### Grounding
valid citation rate:
unsupported citation rate:
no-citation rate:

### Stratified
- cutoff
- selection pressure
- proxy margin

### Gate
PASS / FAIL
CLEAR_TRANSFER / WEAK_TRANSFER / TRANSFER_FAIL

## Ma-Weibo
...

## Determinism Audit
...

## Verifier
issues =

## Overall
PASS / PARTIAL / FAIL

## Recommendation
START_V3_C_HELD_OUT_TEST
STOP_FOR_RESEARCH_REVIEW
PROXY_READER_TRANSFER_FAIL
```

---

## 52. 执行顺序

严格执行：

```text
STEP 1
确认 V3-A commit 和 frozen alpha=0.8

STEP 2
构建 validation sampling population

STEP 3
生成并冻结 300+300 sampling_manifest

STEP 4
实现 evidence-to-prompt formatter

STEP 5
实现 deterministic Qwen inference runner

STEP 6
实现 JSON parser + single retry

STEP 7
增加 unit/integration tests

STEP 8
执行 Static/MS paired inference
总计 1200 arm generations

STEP 9
计算 detection / compression / grounding metrics

STEP 10
执行 paired bootstrap + McNemar

STEP 11
执行 cutoff / pressure / margin stratified analysis

STEP 12
运行 reader-transfer verifier

STEP 13
生成 V3_B_READER_TRANSFER_REPORT

STEP 14
git commit
git push

STEP 15
STOP
```

---

## 53. 本轮禁止事项

不得：

```text
修改 MS-TSR

重新运行 alpha grid 并根据 Qwen 选择 alpha

改 budget

改 Static Selector

重新训练 E1/E2

fine-tune Qwen

LoRA Qwen

让 Qwen 参与 evidence selection

修改 prompt 后重复实验直到结果变好

读取 held-out test

运行 V3-C test

运行完整 E4

加入新的 LLM baseline
```

如果 V3-B FAIL：

```text
如实报告
STOP
```

不得自动修改方法继续实验。

---

完成并提交 GitHub 后，最终只需汇报：

```text
pytest

Sampling:
PHEME n=
Ma-Weibo n=
manifest hash=

PHEME:
Static Qwen Macro-F1
MS Qwen Macro-F1
Delta
bootstrap 95% CI
social-context token reduction
wrong_to_correct
correct_to_wrong
disagreement rate
unsupported citation rate
status

Ma-Weibo:
同上

determinism warnings
parse failures
verifier issues

latest commit SHA

Recommendation
```

本轮最核心的问题只有一个：

> **V3-A 中 Proxy 认为“充分”的极简社会上下文，是否真的能够在完全独立的 frozen Qwen3-8B reader 上保持判断能力，同时显著减少上下文 token？**
