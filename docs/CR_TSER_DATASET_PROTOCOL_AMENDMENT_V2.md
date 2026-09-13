# CR-TSER Dataset Protocol Amendment V2

## 1. Amendment status

本文件是对现有 `CR_TSER_FEASIBILITY_PILOT_PLAN.md` 的正式数据集协议修订。

原 V1 文件及已经产生的 Weibo22 P0 evidence 必须保留，不得覆盖或删除。

本修订只改变：

- primary / secondary dataset assignment；
- P0 dataset-readiness definition；
- dataset-specific orchestration；
- dataset-specific artifact / verifier contract。

本修订不改变 CR-TSER 的：

- research question；
- intervention utility definition；
- intervention family；
- SRC definition；
- BiTTE；
- B0/B1/B3 predictor definition；
- shared / residual decomposition；
- three frozen readers；
- leave-one-reader-out protocol；
- S1–S5 selector definition；
- statistical protocol；
- P1–P4 thresholds；
- GO / NO-GO scientific criteria。

因此本文件属于 `DATASET PROTOCOL AMENDMENT`，而不是新的 CR-TSER 方法版本。

---

## 2. Reason for amendment

CR-TSER V1 将 Weibo22 定义为 mandatory primary dataset，并设置 P0 检查其 strict temporal feasibility。

该 P0 已真实执行并得到：

```text
P0_FAIL
WEIBO22_TEMPORAL_UNAVAILABLE
```

对官方 KPG Weibo22 public release 的审计确认：

```text
event/source ID       available
binary label          available
node ID               available
parent ID             available

source raw text       unavailable
reply/repost raw text unavailable
source timestamp      unavailable
per-node timestamp    unavailable
```

公开 propagation representation 仅包含：

```text
event ID
parent index
node index
bag-of-words index:frequency
```

因此无法合法构造严格的：

```text
15m
1h
6h
```

causal snapshots。

禁止使用：

```text
row order
node index
tree position
original_order
```

替代真实时间。

因此：

```text
Weibo22 public release
        ↓
REJECTED AS CR-TSER PRIMARY PILOT DATASET
```

该结论只否定 `Weibo22-as-primary`，不否定 CR-TSER 的 scientific hypothesis。

---

## 3. Revised dataset roles

### 3.1 Primary dataset

```text
Ma-Weibo
```

Ma-Weibo 决定：

```text
P1
P2
P3
P4
primary GO / NO-GO
```

### 3.2 Secondary dataset

```text
PHEME
```

PHEME 用于：

- English-language comparison；
- cross-dataset robustness；
- dataset-dependence analysis；
- continuity with TC-DSCR；
- secondary non-catastrophic requirement。

PHEME 单独不能产生 FULL_GO。

### 3.3 Weibo22

Weibo22 不再进入正式 CR-TSER Pilot。

冻结状态：

```text
REJECTED_PRIMARY_CANDIDATE
PUBLIC_RELEASE_TEMPORAL_UNAVAILABLE
NOT_USED_IN_P1_P4
```

现有 `results/cr_tser/p0/weibo22_*` 必须保留为 V1 dataset-feasibility evidence。

如果未来获得作者提供的、能够证明属于 Weibo22 的真实 raw text + timestamp 数据，可以作为：

```text
future external validation dataset
```

但不得自动重新取代 Ma-Weibo primary，也不得改变已经冻结的 V2 Pilot decision。

---

## 4. Scientific justification

CR-TSER 的核心研究问题是：

> 在真实时间演化的 social propagation graph 中，结构有效的 evidence intervention 是否产生能够从 propagation structure/time 预测，并且在不同 frozen LLM readers 之间具有稳定成分的 utility，从而支持对 unseen reader 的 robust context selection？

这一研究问题要求的是：

```text
real social propagation tree
+
real evidence arrival time
+
real source / reply text
+
valid structural interventions
+
multiple frozen readers
+
unseen-reader transfer
```

而不要求数据集必须是 Weibo22。

Ma-Weibo 已经满足这一研究问题所需的数据结构。

因此 `Weibo22 → Ma-Weibo` 属于实验载体修正，而不是科学问题修改。

---

## 5. Ma-Weibo source of record

正式 CR-TSER V2 必须从 Ma-Weibo original/raw JSON 构造数据。

Source of record 必须同时冻结：

```text
Ma-Weibo raw event JSON directory
+
Ma-Weibo label file
```

不得使用：

- TC-DSCR processed tensors 代替 raw source；
- UMER cached graph 代替 raw source；
- historical prediction files；
- historical Static Utility artifacts；
- synthetic timestamps；
-重新生成的传播关系。

每个 raw post 至少应提供：

```text
mid / id
parent
t
original_text / text
```

时间只允许来自：

```text
t
```

不得由：

```text
original_order
array index
node index
```

构造时间。

Source manifest 必须同时记录：

```text
raw JSON directory fingerprint
label-file fingerprint
adapter version / code commit
```

后续 label generation、training、selection 和 evaluation 必须验证同一 source identity。

---

## 6. Ma-Weibo normalized event contract

继续使用现有 normalized causal event abstraction：

```text
event_id
label
source_id
source_timestamp

nodes:
    node_id
    parent_id
    timestamp
    text
    original_order
    status
```

其中：

```text
timestamp = raw post["t"]
```

`original_order` 只能作为 deterministic tie-break。

不得进入 temporal inclusion rule。

Strict snapshot：

\[
V_t=\{v_i\mid timestamp_i\le source\_timestamp+t\}
\]

保持不变。

---

## 7. Invalid-node handling

V2 不要求通过人工修补把 raw Ma-Weibo 变成 100% 完美数据。

任何异常必须：

```text
audit
→ explicit status
→ deterministic filtering
```

而不是：

```text
imputation
heuristic repair
manual parent attachment
synthetic text
synthetic time
```

至少区分：

```text
VALID
EMPTY_TEXT
MISSING_PARENT
EXTERNAL_PARENT
TEMPORAL_INVALID_NODE
DUPLICATE_ID
```

处理原则：

### VALID

可以进入：

```text
snapshot
evidence unit
intervention
utility labeling
```

### EMPTY_TEXT

节点可保留在 raw audit / topology accounting 中。

不得进入需要 textual evidence 的 CR-TSER evidence unit。

### MISSING_PARENT / EXTERNAL_PARENT

节点保留在 audit。

不得人工挂到 source。

不得用于：

```text
Reply–Parent evidence unit
parent-child intervention
structured matched-pair analysis
```

### TEMPORAL_INVALID_NODE

不得进入合法 causal snapshot。

### DUPLICATE_ID

必须 fail closed 或使该 event 被判 invalid。

不得自动重新编号。

---

## 8. Event viability filtering

正式 split 必须在 viability filtering 后构造。

一个 Ma-Weibo event 至少需要：

```text
valid binary label
exactly one valid source/root
valid source timestamp
non-empty source text
unique source/node IDs
no graph cycle among valid nodes
```

并且至少在：

```text
15m
1h
6h
```

中的一个 snapshot 内存在一个可构造的 valid Reply–Parent Evidence Unit。

三个 cutoff 仍全部尝试。

某个 cutoff 没有 eligible reply：

```text
retain snapshot in audit
generate no intervention rows
```

不得因为一个 cutoff 为 zero-reply 就删除整个 event。

After filtering：

```text
viable event count >= 170
```

是构造当前 Pilot split 的必要条件。

若不足 170：

```text
P0_FAIL
STOP_FOR_RESEARCH_REVIEW
```

不得缩小 split 来强行继续。

---

## 9. Revised P0

P0 修订为：

# P0 — Ma-Weibo Data Integrity + Reader Readiness

P0 必须验证三个部分。

### P0-A — Ma-Weibo source integrity

至少报告：

```text
raw event count
label distribution
source-text coverage
reply-text coverage
timestamp coverage
parent-resolution coverage
duplicate IDs
cycle count
multi-root event count
missing-parent count/rate
external-parent count/rate
temporal-invalid node count
events with >=1 valid Reply–Parent unit
events viable at 15m
events viable at 1h
events viable at 6h
total viable events
```

历史审计结果可以作为 expected reference，但不能代替本次正式 audit。

尤其不得假定历史的：

```text
99.9996% text coverage
99.9992% parent resolution
0 cycles
```

仍然成立。

必须重新从当前 source of record 得到。

### P0-B — PHEME smoke

PHEME 必须可以：

```text
load raw events
recover source/reply text
recover timestamps
recover parent relation
build 15m / 1h / 6h snapshots
show zero future leakage
```

PHEME 不决定 primary P1–P4，但 FULL_GO 所需 secondary evaluation 必须有可执行的数据入口。

### P0-C — Frozen readers

保持 V1 完全不变。

必须加载：

```text
Qwen/Qwen3-8B
zai-org/glm-4-9b-chat-hf
internlm/internlm3-8b-instruct
```

记录：

```text
model ID
local path
weight hash
tokenizer hash
chat-template hash
dtype
transformers version
device config
```

并完成 A/B teacher-forced sequence-scoring sanity：

```text
boundaries_ok = true
identical_predictions = true
```

三个 reader 全部满足才允许 P0 PASS。

---

## 10. Revised P0 decision

P0 PASS 要求：

```text
Ma-Weibo source integrity PASS
AND
Ma-Weibo viable events >= 170
AND
PHEME smoke PASS
AND
all three frozen readers load successfully
AND
all A/B boundary checks PASS
AND
all repeated-scoring determinism checks PASS
```

否则：

```text
P0_FAIL
```

并立即停止。

---

## 11. Pilot split

保持 V1 不变。

使用：

```text
seed = 7319
```

在 Ma-Weibo viability-filtered pool 中构造：

```text
foundation_train: 80
utility_train:    50
utility_dev:      15
utility_eval:     25
```

合计：

```text
170 events
```

要求：

```text
event-disjoint
approximately label-balanced
```

不得恢复历史 TC-DSCR folds。

不得根据 P1–P4 结果重新抽 split。

不得因为历史 TC-DSCR 中某些 Ma-Weibo event 被使用过而修改 seed。

---

## 12. Temporal cutoffs

保持 V1：

```text
15m
1h
6h
```

不进行新的 cutoff search。

---

## 13. P1 — Reader heterogeneity

Primary dataset 改为：

```text
Ma-Weibo utility_eval
```

除此以外定义与阈值完全不变。

Atomic intervention active condition：

```text
abs(u_r) >= 0.05
OR correctness changes
```

Pass：

```text
MeanDisagreement >= 0.10
```

且至少 2/3 reader pairs：

```text
Disagreement >= 0.05
```

---

## 14. P2 — Structured social interaction

Primary dataset 改为：

```text
Ma-Weibo
```

定义与阈值保持不变。

Primary：

\[
\Delta_{edge}
=
E|I(parent-child)|
-
E|I(matched-nonadjacent)|
\]

Pass：

```text
Delta_edge >= 0.02
```

且：

```text
event-bootstrap 95% CI lower bound > 0
```

只有拥有真实 resolved parent-child relation 的 evidence 才能进入 P2。

---

## 15. P3 — Structural utility prediction increment

Primary dataset 改为：

```text
Ma-Weibo utility_eval
```

保持：

```text
B0 = Text Only
B1 = Text + Scalar Structure/Time
B3 = BiTTE
```

Pass：

```text
MacroF1(B3)
-
max(MacroF1(B0), MacroF1(B1))
>= 0.02
```

并要求：

```text
event-bootstrap 95% CI lower bound > 0
```

以及：

```text
B3 continuous Spearman
>= strongest B0/B1
```

---

## 16. P4 — Unseen-reader transfer

Primary dataset 改为：

```text
Ma-Weibo
```

三个 LORO rotation 完全不变：

```text
Qwen + GLM       → hold InternLM
Qwen + InternLM  → hold GLM
GLM + InternLM   → hold Qwen
```

Primary selector：

```text
S5 — Cross-Reader Robust
```

比较：

```text
bestSimple = max(S1 Random-TM, S2 Semantic-TM)
```

Pass 继续要求：

```text
Mean(delta_r) >= +0.01
>= 2/3 rotations positive
worst rotation >= -0.005
same maximum token target
```

---

## 17. PHEME secondary condition

保持 V1 不变。

要求：

```text
Mean Delta(S5 - bestSimple) >= -0.005
```

PHEME 不决定 primary P1–P4。

但 FULL_GO 仍需要 PHEME secondary evidence complete 且 non-catastrophic。

---

## 18. TC-DSCR historical isolation

因为 Ma-Weibo 已经参与过历史：

```text
UMER
TC-DSCR E1/E2/E3
MF-TSR
MS-TSR
reader-transfer analysis
```

所以 CR-TSER V2 必须禁止历史 Ma-Weibo learned artifacts 进入 primary pipeline。

禁止在 Ma-Weibo P1–P4 中使用：

```text
old UMER checkpoint
old CausalSocialEncoder checkpoint
old Static Utility checkpoint
old Proxy
old MF-TSR outputs
old MS-TSR outputs
historical utility scores
historical reader-selection labels
historical test predictions
```

尤其禁止新增：

```text
Ma-Weibo B2
Ma-Weibo S6
```

B2/S6 继续严格：

```text
PHEME only
diagnostic only
```

CR-TSER 的 Ma-Weibo utility supervision 必须重新来自：

```text
frozen real-reader interventions
```

---

## 19. Historical experimental information firewall

历史 TC-DSCR Ma-Weibo 结果可以用于：

```text
research motivation
failure analysis
data feasibility justification
```

不得用于：

```text
P1–P4 hyperparameter selection
seed selection
cutoff selection
event selection
reader selection
threshold tuning
selector tuning
utility_eval tuning
```

V2 protocol 批准后，`utility_eval outcomes` 仍然不得用于方法修改。

---

## 20. Selection arms

保持：

```text
S0 — SRC Full
S1 — Random Token-Matched
S2 — Semantic Token-Matched
S3a — Single-Reader Utility A
S3b — Single-Reader Utility B
S4 — Shared-Only Utility
S5 — Cross-Reader Robust
```

PHEME additionally：

```text
S6 — frozen TC-DSCR Static Utility-TM
```

Ma-Weibo：

```text
NO S6
```

---

## 21. Final GO / NO-GO

### FULL_GO

要求：

```text
V2 P0 PASS
Ma-Weibo P1 PASS
Ma-Weibo P2 PASS
Ma-Weibo P3 PASS
Ma-Weibo P4 PASS
PHEME secondary non-catastrophic
```

Recommendation：

```text
START_FULL_CR_TSER_METHOD_DEVELOPMENT
```

### PARTIAL_GO

核心证据总体成立，但有一个边界条件 narrowly missed。

Recommendation：

```text
STOP_FOR_RESEARCH_REVIEW
```

### NO_GO

以下任一核心前提失败：

```text
Ma-Weibo primary data unusable
reader heterogeneity weak
structured interaction absent
BiTTE has no structural increment
robust selection fails unseen-reader transfer
```

Recommendation：

```text
STOP_CR_TSER
```

`WEIBO22_TEMPORAL_UNAVAILABLE` 不再属于 V2 CR-TSER NO_GO 条件。

它只属于：

```text
V1 DATASET CANDIDATE REJECTION
```

---

## 22. Artifact namespace

不得覆盖 V1 artifacts。

V1 保留：

```text
results/cr_tser/
```

V2 固定建议写入：

```text
results/cr_tser_v2/
```

至少：

```text
results/cr_tser_v2/
├── p0/
├── manifests/
│   ├── maweibo/
│   └── pheme/
├── utility_labels/
├── predictors/
├── unseen_reader/
├── reports/
└── verifier/
```

现有：

```text
results/cr_tser/p0/weibo22_*
```

保持历史只读。

---

## 23. Code-freeze migration boundary

当前 V1 frozen baseline：

```text
b5cfa168245c46347e0f12833a4ef1017e93e94f
```

批准 V2 后，仅重新开放：

```text
dataset configuration
dataset adapters / bridge
dataset registry
source fingerprint
manifest orchestration
P0 audit
dataset naming in evaluation/report
verifier
dataset-specific tests
```

核心研究代码继续冻结。

禁止修改：

```text
project/cr_tser/models/bitte.py
utility head mathematical definitions
intervention utility equation
structured-intervention definitions
reader scoring contract
B0/B1/B3 architectures
training loss weights
optimizer protocol
LORO definition
robust selector equation
bootstrap protocol
P1–P4 thresholds
```

只有发现与 Ma-Weibo 接入直接相关、且无法通过 adapter/orchestration 层解决的真实 implementation bug 时，必须停止并上报研究审批，不得自行修改核心模块。

---

## 24. Recommended Ma-Weibo implementation strategy

优先复用已经经过 TC-DSCR 审计的：

```text
project/tcdscr/data/maweibo_adapter.py
```

不要重新实现一套 raw parser。

允许建立：

```text
project/cr_tser/data/maweibo_bridge.py
```

其职责仅为：

```text
reuse audited TC-DSCR adapter
→ enforce CR-TSER eligibility rules
→ expose normalized event
```

不得：

```text
repair raw data
change timestamps
change parent relations
infer missing text
```

---

## 25. Source fingerprint amendment

因为 Ma-Weibo source of record 包含：

```text
raw JSON directory
+
label file
```

source manifest 必须生成 composite fingerprint。

至少包含：

```text
dataset = maweibo

raw_json:
    absolute/resolved path
    sha256
    n_files
    bytes

label_file:
    path
    sha256
    bytes

combined_source_sha256
```

任何后续 stage 如发现 source identity 改变：

```text
FAIL CLOSED
```

---

## 26. V2 execution checkpoints

V2 不允许直接从现有 P0 FAIL 跳到正式 Pilot。

### Checkpoint V2-M0 — protocol migration

只完成：

```text
dataset/orchestration migration
tests
verifier
```

不得运行正式 P0。

提交并审批。

### Checkpoint V2-P0 — Ma-Weibo + PHEME + readers preflight

完成：

```text
Ma-Weibo source/data audit
Ma-Weibo snapshot viability
PHEME smoke
3-reader load/hash
A/B teacher-forced sanity
```

然后：

```text
COMMIT
PUSH
STOP
```

等待审批。

### Checkpoint V2-P1

只有 V2-P0 获批后才能：

```text
build/freeze manifests
generate reader intervention labels
run P1/P2 prerequisite evidence generation
```

具体是否进一步拆分 P1/P2/P3/P4，由研究审批决定。

执行智能体不得自动连续运行全部 Pilot。

---

## 27. Required V2 documentation changes

新增：

```text
docs/CR_TSER_DATASET_PROTOCOL_AMENDMENT_V2.md
```

保留：

```text
docs/CR_TSER_FEASIBILITY_PILOT_PLAN.md
```

并在 V1 顶部增加非破坏性说明：

```text
SUPERSEDED FOR DATASET PROTOCOL BY
CR_TSER_DATASET_PROTOCOL_AMENDMENT_V2.md
```

不得删除 V1 内容。

---

## 28. Research interpretation

V1 P0 的最终科学结论应保留为：

> The public KPG Weibo22 release cannot support the strict temporal/textual intervention protocol required by CR-TSER.

不得写成：

> CR-TSER is infeasible.

V2 的正式解释是：

> The candidate primary dataset failed feasibility audit; therefore the pilot dataset protocol was amended before any CR-TSER intervention labels or P1–P4 outcomes were generated.

因为 dataset amendment 发生在：

```text
formal intervention generation = NOT RUN
P1 = NOT RUN
P2 = NOT RUN
P3 = NOT RUN
P4 = NOT RUN
```

所以不存在根据 CR-TSER experiment outcome 更换数据集的问题。

---

## 29. Frozen V2 research chain

```text
TC-DSCR
    ↓
Proxy–real-reader transfer mismatch
    ↓
CR-TSER hypothesis
    ↓
V1 candidate dataset audit
    ↓
Weibo22 rejected before scientific Pilot
    ↓
V2 primary = Ma-Weibo
secondary = PHEME
    ↓
P0 Data + Reader Readiness
    ↓
P1 Reader Heterogeneity
    ↓
P2 Structured Social Interaction
    ↓
P3 Predictable Structural Utility
    ↓
P4 Unseen-Reader Transfer
```

只有 P1–P4 联合得到支持，才能启动 full CR-TSER method development。

---

## 30. V2 approval decision

本 amendment 冻结：

```text
PRIMARY_DATASET = Ma-Weibo
SECONDARY_DATASET = PHEME
WEIBO22_STATUS = REJECTED_PRIMARY_CANDIDATE
```

同时冻结：

```text
P1–P4 scientific definitions = unchanged
P1–P4 thresholds = unchanged
reader set = unchanged
temporal cutoffs = 15m / 1h / 6h
split seed = 7319
split sizes = 80 / 50 / 15 / 25
```

以及：

```text
B2/S6 = PHEME only

historical Ma-Weibo TC-DSCR learned artifacts
MUST NOT participate in CR-TSER primary training or evaluation
```

下一步只能执行：

```text
V2-M0 dataset/orchestration migration
```

完成后：

```text
COMMIT
PUSH
STOP
```

等待新的代码审批。

不得直接运行 V2 P0 或正式 Pilot。
