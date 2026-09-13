# CR-TSER V2-P0 Execution Plan

## 1. Purpose

本轮任务是执行 **CR-TSER V2-P0 — Ma-Weibo Data Integrity + PHEME Smoke + Frozen Reader Preflight**。

已批准的代码基线：

```text
fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f
```

当前协议状态：

```text
CR_TSER_V2_M0 = APPROVED
V2_PROTOCOL_MIGRATION = CODE_FROZEN
PRIMARY_DATASET = Ma-Weibo
SECONDARY_DATASET = PHEME
WEIBO22 = REJECTED_PRIMARY_CANDIDATE
```

本轮目标仅是回答：

> 当前真实数据和 reader 环境是否满足启动 CR-TSER V2 正式 Pilot 的全部 P0 prerequisites？

本轮不是正式 Pilot，也不是新的方法开发阶段。

---

## 2. Authoritative protocol

执行前必须完整阅读：

```text
docs/CR_TSER_FEASIBILITY_PILOT_PLAN.md
docs/CR_TSER_DATASET_PROTOCOL_AMENDMENT_V2.md
docs/CR_TSER_IMPLEMENTATION_NOTES.md
```

当 V1 与 V2 dataset protocol 冲突时，以 `CR_TSER_DATASET_PROTOCOL_AMENDMENT_V2.md` 为准。

以下科学内容继续冻结，不得修改：

- BiTTE；
- intervention utility equation；
- intervention families；
- reader scoring contract；
- B0/B1/B3 architecture；
- shared/residual decomposition；
- LORO rotations；
- robust selector；
- bootstrap protocol；
- P1–P4 thresholds；
- temporal cutoffs；
- split seed / split sizes；
- reader set。

---

## 3. Hard prohibitions

本轮禁止执行：

```text
formal manifest freeze
formal intervention generation
formal utility labels
predictor training
single-reader training
shared/residual training
Stage-A subset freeze
held-out reader evaluation
P1
P2
P3
P4
formal Pilot report
```

禁止调整任何研究阈值、更换 reader、使用量化或 API 替代模型、修改 15m/1h/6h cutoff、使用 historical Ma-Weibo TC-DSCR learned artifacts、将 B2/S6 引入 Ma-Weibo、根据 P0 结果修改方法，或因 prerequisite 失败自行设计 fallback。

如果 P0 失败，正确行为是记录真实 failure evidence 并 STOP。

---

## 4. Checkpoint P0-0 — Frozen baseline verification

首先执行：

```bash
git status
git rev-parse HEAD
```

必须确认：

```text
HEAD = fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f
working tree clean
```

如不满足则 STOP。

然后运行：

```bash
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python -m compileall project/cr_tser scripts
```

要求 tests PASS、V2 code verifier issues=0、compileall clean。

同时记录 git commit、Python、torch、transformers、CUDA 和 GPU 信息，进入 P0 evidence package。

---

## 5. Checkpoint P0-A — Ma-Weibo source-of-record resolution

配置：

```text
CRTSER_MAWEIBO_RAW
CRTSER_MAWEIBO_LABELS
```

source of record 必须是 Ma-Weibo raw event JSON directory + label file。

禁止使用 historical tensors、processed graph cache、UMER cache、TC-DSCR predictions/checkpoints 或 pseudo-time。

至少记录 resolved paths、raw file count、raw bytes、raw sha256、label sha256 和 combined_source_sha256。

若 source 缺失：

```text
P0_FAIL
reason = MAWEIBO_SOURCE_MISSING
STOP after evidence package
```

---

## 6. Checkpoint P0-B — Ma-Weibo full integrity audit

必须调用冻结 V2 audit 路径，不得手工编辑结果。

至少核对：

```text
raw_event_count
label_distribution
source_text_coverage
reply_text_coverage
source_timestamp_coverage
timestamp_coverage
parent_resolution_coverage
reply_node_count
duplicate_ids
cycle_count
multi_root_event_count
missing_parent_count
missing_parent_rate
external_parent_count
external_parent_rate
temporal_invalid_node_count
node_status_counts
reply_status_counts
events_with_ge1_valid_reply_parent_unit
events_viable_15m
events_viable_1h
events_viable_6h
total_viable_events
```

Ma-Weibo timestamp 必须直接来自 raw `post["t"]`，不得由 original_order、array index、node id 或 tree position 推导。

Duplicate ID 不得自动重编号；multi-root event invalid；cycle event 不得进入 viable pool。

异常节点必须沿 V2 contract 显式标注：

```text
VALID
EMPTY_TEXT
MISSING_PARENT
EXTERNAL_PARENT
TEMPORAL_INVALID_NODE
DUPLICATE_ID
```

不得人工修复。

---

## 7. Checkpoint P0-C — Ma-Weibo viability validation

一个 event 只有满足 V2 viability rule 才能进入 candidate pool。

要求 valid binary label、exactly one source/root、valid source timestamp、non-empty source text、unique node IDs、no valid-node cycle，并且至少在 15m / 1h / 6h 中一个 snapshot 存在合法 Reply–Parent Evidence Unit。

合法 unit 必须同时满足：

```text
reply.status == VALID
reply text non-empty
parent exists
parent.status == VALID
parent text non-empty
parent.timestamp <= reply.timestamp
```

重新统计三个 cutoff 的 viability 和 `total_viable_events`。

P0 最低要求：

```text
total_viable_events >= 170
```

若不足 170：

```text
P0_FAIL
STOP_FOR_RESEARCH_REVIEW
```

不得缩小 80/50/15/25 split。本轮不得创建正式 event split。

---

## 8. Checkpoint P0-D — Causal snapshot integrity

从真实 Ma-Weibo 抽取一组 events 覆盖 15m / 1h / 6h，验证：

```text
source always present
all included timestamps <= source_timestamp + cutoff
no future node leakage
no 1021-node cap
parent edge only exists when parent visible
parent.timestamp <= child.timestamp
original_order only tie-breaks equal timestamps
```

记录 sample event IDs、per-cutoff node counts、valid evidence-unit counts、future-leak count、cap_hit 和 unreachable_count。

某 cutoff 没有 reply 时，只记录 zero-reply，不删除整个 event。

---

## 9. Checkpoint P0-E — PHEME smoke

配置：

```text
CRTSER_PHEME_RAW
```

使用冻结 V2 PHEME smoke。

至少一个真实 PHEME event 必须满足：

```text
raw event loads
source text recoverable
reply text recoverable
timestamps recoverable
at least one resolved parent relation exists
15m snapshot builds
1h snapshot builds
6h snapshot builds
future leakage = 0
```

少量 missing/external parent 可存在。

若没有 event 满足完整 smoke contract：

```text
PHEME_SMOKE_FAIL
P0_FAIL
```

不得因为 PHEME 是 secondary 而忽略。

---

## 10. Checkpoint P0-F — Frozen reader environment

必须且只能使用：

```text
Qwen/Qwen3-8B
zai-org/glm-4-9b-chat-hf
internlm/internlm3-8b-instruct
```

逐个 reader 顺序加载，不要同时常驻显存。

每个 reader 记录：

```text
reader key
expected model id
resolved model path
weight hash
tokenizer hash
chat-template hash
reader identity hash
dtype
device
transformers version
model_path_exists
loaded
load_error if any
```

任意 reader 无法加载则 P0_FAIL，不得替换模型继续。

---

## 11. Checkpoint P0-G — Teacher-forced A/B scoring sanity

使用冻结 teacher-forced sequence scoring。

禁止 generate()、free-form generation、generated confidence、self-reported probability 和 answer parsing。

至少使用冻结默认 >=20 sanity prompts / reader，每个 prompt 重复评分两次。

记录：

```text
A continuation token ids
B continuation token ids
prompt token count
continuation boundary check
logprob(A)
logprob(B)
normalized P(A)
normalized P(B)
prediction
```

检查 finite log probabilities、无 NaN/Inf、无 empty continuation、无 boundary mismatch。

每个 reader 必须满足：

```text
boundaries_ok = true
identical_predictions = true
identity_rate = 1.0
```

任一失败则 P0_FAIL。

---

## 12. Formal V2 P0 execution

输入准备完成后执行冻结正式入口，例如：

```bash
python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity
```

不得手工修改 `results/cr_tser_v2/p0/p0_readiness.json`。

P0 verdict 必须由冻结代码产生。

---

## 13. P0 PASS criteria

只有以下全部成立：

```text
Ma-Weibo source integrity PASS
Ma-Weibo viable events >= 170
PHEME smoke = OK
Qwen loaded
GLM loaded
InternLM loaded
all boundaries_ok
all repeated scoring deterministic
```

才能得到：

```text
P0_PASS
```

即使 PASS，也必须 STOP，不得继续 manifest 或正式 Pilot。

---

## 14. P0 FAIL criteria

任何 prerequisite 失败即：

```text
P0_FAIL
```

典型原因：

```text
MAWEIBO_SOURCE_MISSING
MAWEIBO_INTEGRITY_FAIL
MAWEIBO_VIABLE_EVENTS_LT_170
PHEME_RAW_MISSING
PHEME_SMOKE_FAIL
QWEN_MODEL_MISSING
GLM_MODEL_MISSING
INTERNLM_MODEL_MISSING
READER_LOAD_ERROR
AB_BOUNDARY_FAIL
AB_NONDETERMINISTIC
```

失败后不得修改 scientific protocol、reader、split size 或继续 P1–P4。

---

## 15. Required P0 evidence package

正式输出：

```text
results/cr_tser_v2/p0/
```

至少保留：

```text
p0_readiness.json
P0_READINESS.md
maweibo_audit.json
maweibo_source_fingerprint.json   # 或等价 source identity artifact
pheme_smoke.json
reader_audit.json
label_scoring_sanity.json
```

如果现有代码采用拆分文件，也可保持现有格式，但必须完整包含 Ma-Weibo source identity / integrity / viability、PHEME smoke、3 readers identity/load、3 readers A/B boundary 和 deterministic scoring、exact P0 verdict 及 failure reason。

---

## 16. P0 report requirements

`P0_READINESS.md` 至少说明：

1. 执行 git commit；
2. Ma-Weibo source of record；
3. composite fingerprint；
4. 数据完整性；
5. viable event count；
6. 15m / 1h / 6h viability；
7. PHEME smoke；
8. 三 reader 是否真实加载；
9. A/B token boundaries；
10. repeated scoring deterministic status；
11. P0 PASS / FAIL；
12. FAIL 时唯一明确 blocker；
13. 按协议允许的下一步。

---

## 17. Verification

P0 后运行：

```bash
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2
```

P1–P4 缺失应保持 pending，不得制造 fake artifacts。

---

## 18. Historical namespace protection

整个 V2-P0 期间：

```text
results/cr_tser/
```

必须保持只读。

所有 V2 outputs 只进入：

```text
results/cr_tser_v2/
```

建议执行前后记录 V1 historical directory hash。

---

## 19. Allowed code changes during V2-P0

原则上：

```text
NO CODE CHANGE
```

若发现纯 runtime bug 且阻止冻结 P0 正常执行：

```text
STOP
document bug
do not patch automatically
```

等待研究审批。

不得边跑 P0 边改代码再继续。

---

## 20. Final stopping rule

本轮只有两个合法终点：

```text
P0_PASS
```

或：

```text
P0_FAIL
```

无论哪种：

```text
git status
git add <P0 evidence artifacts only>
git commit
git push
STOP
```

建议 commit message：

PASS：

```text
CR-TSER V2-P0 preflight complete — prerequisites PASS, formal pilot NOT RUN
```

FAIL：

```text
CR-TSER V2-P0 preflight complete — prerequisites FAIL, formal pilot NOT RUN
```

提交后立即停止，等待下一轮研究审批。
