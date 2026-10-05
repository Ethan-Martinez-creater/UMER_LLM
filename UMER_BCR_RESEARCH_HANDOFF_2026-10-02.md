# UMER / CR-TSER / BCR-Utility 研究任务交接手册
## 给新执行智能体的完整研究与运行环境说明

**日期：2026-10-02**  
**GitHub：** `https://github.com/Ethan-Martinez-creater/UMER_LLM`  
**当前远端 HEAD：** `b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12`  
**当前研究主线：** `BCR-Utility / bcr_v1`  
**当前状态：** `M1-E 已完成，等待研究评审；M2 未授权`

---

# 1. 接手时必须先知道的结论

1. **CR-TSER 已正式 NO-GO，历史研究线结束。**
2. CR-TSER 中成立的是 **reader-specific evidence utility heterogeneity**；未成立的是“传播结构存在足够强且稳定的非加性 utility interaction”。
3. 新主线 BCR-Utility 原计划研究：
   `behavioral reader fingerprint -> unseen-reader signed evidence utility transfer`。
4. **M1 的 ZERO-TOUCH behavioral fingerprint 机制在 Ma-Weibo 上失败。**
5. M1 的 LIGHT-TOUCH 版本在 Ma-Weibo 上通过，但 M1-E 进一步证明：
   - 真正带来收益的是 `E3` 的 reader source-state / evidence-familiarity；
   - behavioral fingerprint 在 E3 已存在时没有正增量；
   - fingerprint 在 PHEME 上明显有害。
6. 因此**不要直接执行原 M2 六-reader 扩展计划**。当前应先做研究评审，决定是否 pivot 到：
   `query/source-specific reader state + evidence familiarity -> reader-specific utility`。

---

# 2. GitHub 与当前版本

仓库：

```text
https://github.com/Ethan-Martinez-creater/UMER_LLM
```

当前 GitHub HEAD：

```text
b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12
```

开始任何工作前执行：

```bash
git status
git rev-parse HEAD
git log --oneline -10
```

关键历史 commits：

```text
518a821d8d839fb7857be99acb91fde71b80c846
CR-TSER V2R1 failure closure / NO-GO
```

```text
6ac2c29c078ff889deec40c03bb63ab0034629af
BCR M0 complete
```

```text
fef08cd581fab3b2208cc5b55c49cccb5e9915d4
BCR M1 complete / M1_CONDITIONAL_GO
```

```text
b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12
BCR M1-E complete / SERVER verifier 49 checks, 0 issues, 0 pending
```

---

# 3. 本机环境

本机 conda 环境：

```text
pytorch
```

最近记录：

```text
Python 3.9.18
torch 2.3.1+cu121
```

本机职责：

```text
代码实现
Git
pytest
compileall
static verifier
synthetic tests
小型 CPU 数据处理
报告
```

禁止本机承担：

```text
7B/8B/9B reader 正式加载
正式 LLM utility inference
正式 GPU 实验
```

最近实际本机仓库路径：

```text
E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER
```

接手时仍应使用：

```bash
git rev-parse --show-toplevel
```

确认，不要只依赖旧路径。

---

# 4. 服务器环境

服务器总根目录：

```text
/data/jyz/next/llm/
```

实际 Git 工作区：

```text
/data/jyz/next/llm/cr_tser_ws
```

注意：

```text
/data/jyz/next/llm/
```

本身不是 Git 仓库。

服务器 conda：

```text
DGPA
```

最近冻结环境：

```text
Python 3.11.13
torch 2.7.1+cu128
GPU = NVIDIA RTX 4090 24GB
```

正式 reader inference、feature extraction、训练和正式评估都在 SERVER/DGPA。

---

# 5. Transformers compatibility overlay

InternLM3 曾与 DGPA 原 Transformers 版本发生 `LossKwargs` API 兼容问题。

最终没有修改 DGPA 全局环境，而使用：

```text
/data/jyz/next/llm/.cr_tser_v2p0/tf453
```

正式 reader 运行时：

```text
transformers = 4.53.3
tokenizers   = 0.21.4
```

原则：

```text
不要修改 DGPA site-packages
不要 patch 官方模型代码
新 reader 若依赖不同版本，建立独立 overlay
```

未来可使用：

```text
/data/jyz/next/llm/.bcr_env/<reader>/
```

---

# 6. 连接服务器的唯一方式

服务器命令：

```text
next/.codex_tmp_ssh_run.cmd
```

文件传输：

```text
next/.codex_tmp_scp_run.cmd
```

不要自行切换到其他 SSH/SCP 方案。

如果出现：

```text
connection refused
server refused
tunnel unavailable
connection closed due to tunnel/server
```

必须：

```text
STOP
保存原始错误
告知用户
等待用户恢复隧道
用户确认后再重试
```

禁止：

```text
改 host / port / user
换 SSH client
换 tunnel
连续 hammer retry
把 SERVER 工作转 LOCAL
把 tunnel failure 当实验失败
```

统一记录：

```text
INFRASTRUCTURE_PAUSE
```

M1-E 已真实发生过一次该情况，证据在：

```text
results/bcr_utility_v1/m1e/INFRASTRUCTURE_PAUSE.json
```

---

# 7. 模型资产

## Qwen3-8B

```text
Qwen/Qwen3-8B
```

当前使用路径：

```text
/data/jyz/next/llm/model/qwen3-8b
```

历史 authoritative path：

```text
/data/jyz/next/model/qwen3-8b
```

不要重复下载。

## Mistral-7B-Instruct-v0.3

```text
mistralai/Mistral-7B-Instruct-v0.3
```

路径：

```text
/data/jyz/next/llm/model/mistral-7b-instruct-v0.3
```

当前正式 reader。  
它替换了显存占用过高的 GLM。BF16 preflight 峰值约 13.9 GiB。

不要重复下载。

## InternLM3-8B-Instruct

```text
internlm/internlm3-8b-instruct
```

路径：

```text
/data/jyz/next/llm/model/internlm3-8b-instruct
```

必须继续使用已验证的 4.53.3 compatibility overlay。

不要重新下载，不要 patch remote code。

## 历史 GLM

```text
zai-org/glm-4-9b-chat-hf
```

路径：

```text
/data/jyz/next/llm/model/glm-4-9b-chat-hf
```

状态：

```text
retired historical reader
```

不能重新加入当前 `bcr_v1` reader panel，除非新的正式协议明确批准。

## Qwen3-0.6B

服务器历史上已经部署并使用过。

状态：

```text
historically present
exact current path NOT reliably frozen
```

如未来需要：

```text
SEARCH SERVER FIRST
```

不要猜路径，不要先重下。

## NLI feature extractor

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

路径：

```text
/data/jyz/next/llm/model/mdeberta-v3-base-mnli-xnli
```

主要权重：

```text
pytorch_model.bin
557692715 bytes

sha256 =
345f880b8390c64336cb9fd2907544fc02418bc4e97a74d5952b26b82bfb6f74
```

用于 E1 NLI features，不是 rumor reader。

---

# 8. 模型下载规则

已存在的模型：

```text
禁止重复下载
```

新模型优先：

```text
SERVER/DGPA 直接下载
-> /data/jyz/next/llm/model/
-> server-side verify
```

服务器下载因网络失败时才允许：

```text
LOCAL/pytorch
-> 非 C 盘
-> next/.codex_tmp_scp_run.cmd
-> SERVER
-> server-side verify
```

绝对禁止：

```text
大模型权重下载到本机 C:
```

---

# 9. 数据集

当前 primary：

```text
Ma-Weibo
```

当前 secondary：

```text
PHEME
```

Weibo22：

```text
REJECTED_PRIMARY_CANDIDATE
```

不要重新启用 Weibo22 作为当前 primary，除非用户正式修订研究协议。

冻结 split：

```text
foundation_train = 80
utility_train    = 50
utility_dev      = 15
utility_eval     = 25
seed             = 7319
```

时间 cutoffs：

```text
15 min
60 min
360 min
```

Ma-Weibo viable events：

```text
4591
```

PHEME viable events：

```text
5669
```

Ma-Weibo composite source fingerprint：

```text
b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d
```

---

# 10. 历史 frozen utility caches

历史结果根目录：

```text
results/cr_tser_v2r1/
```

Ma-Weibo：

```text
utility_labels/maweibo/labels.jsonl
rows = 9606
bytes = 16015182
sha256 =
6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3
```

每 reader：

```text
3202
```

I1 atomic：

```text
2549 unique keys
7647 reader rows
```

PHEME：

```text
utility_labels/pheme/labels.jsonl
rows = 8637
bytes = 14457787
sha256 =
773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15
```

每 reader：

```text
2879
```

I1 atomic：

```text
2107 unique keys
6321 reader rows
```

---

# 11. 永久只读历史目录

以下目录不可修改：

```text
results/cr_tser/
results/cr_tser_v2/
results/cr_tser_v2r1/
```

禁止：

```text
改旧 manifest
覆盖旧 cache
删除旧 GLM rows
把历史 glm 改成 mistral
改历史 gate
改 closure 结论
```

新 BCR 结果：

```text
results/bcr_utility_v1/
```

---

# 12. CR-TSER 历史研究结论

CR-TSER 原假设：

```text
传播结构 + 时间
-> structured social interaction
-> shared + reader residual utility
-> unseen-reader robust selection
```

正式 P1：

```text
PASS
MeanDisagreement = 0.4082
```

说明：

```text
同一 evidence 对不同 reader 的 help/harm utility 明显不同
```

正式 P2：

```text
FAIL

Delta_edge = 0.012150
required = 0.02

95% CI =
(-0.002342, 0.027196)
```

subtree：

```text
0.016013
CI (-0.003599, 0.035486)
```

正确解释：

> 结构化交互方向为正，但效应未达到预注册阈值，且 event-bootstrap CI 跨 0。

禁止写成：

```text
social structure has no effect
```

最终 closure：

```text
P1 = PASS
P2 = FAIL
P3 = NOT RUN
P4 = NOT RUN

CR_TSER_V2R1_FEASIBILITY = NO_GO
reason = STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED
```

commit：

```text
518a821d8d839fb7857be99acb91fde71b80c846
```

因此 CR-TSER 不再继续。

---

# 13. BCR-Utility 当前研究问题

BCR 从 CR-TSER 的正结果出发：

```text
reader-specific evidence utility exists
```

原始 BCR 问题：

> 不使用目标 reader utility labels，仅通过 lightweight behavioral fingerprint，
> 能否预测 unseen reader 对 temporal social evidence 的 HELPFUL / NEUTRAL / HARMFUL utility？

protocol：

```text
bcr_v1
```

代码：

```text
project/bcr_utility/
scripts/bcr_*.py
```

结果：

```text
results/bcr_utility_v1/
```

---

# 14. BCR utility 定义

只使用：

```text
I1_atomic
```

utility：

```text
u_r(e) =
p_r(gold | SRC)
-
p_r(gold | SRC without e)
```

sign：

```text
u >= +0.05 -> HELPFUL
u <= -0.05 -> HARMFUL
otherwise  -> NEUTRAL
```

BCR 不把 CR-TSER 的：

```text
I2 parent-child
I3 matched non-adjacent
I4 subtree
I5 matched disconnected
```

作为 supervision。

---

# 15. Reader scoring contract

reader task：

```text
A = RUMOR
B = NON_RUMOR
```

正式 scoring：

```text
teacher-forced candidate log probability
```

禁止替换成：

```text
free-form generate()
generated confidence
让模型输出自然语言解释后再 parse
```

Mistral 曾暴露 tokenizer text-retokenization boundary 问题，已通过 tokenizer-native autoregressive boundary audit 正式解决。

不要重新把旧：

```text
tokenize(prompt + candidate) ==
tokenize(prompt) + tokenize(candidate)
```

作为所有 tokenizer 的硬 gate。

---

# 16. BCR M0

M0 完成：

```text
historical read-only importer
I1 atomic index
reader geometry
72-item probe manifest
BCR verifier
```

probe：

```text
36 Ma-Weibo
36 PHEME
12 per cutoff per dataset
foundation_train only
```

与：

```text
utility_train/dev/eval
```

交集：

```text
0
```

reader×evidence variance share：

```text
Ma-Weibo ≈ 0.643
PHEME    ≈ 0.650
```

这是 diagnostic，只说明 reader×evidence interaction 很强，不证明可 transfer。

---

# 17. BCR M1 probe 与 features

M1 probe：

```text
3 readers × 72 items × 4 contexts
= 864 / 864
```

contexts：

```text
P0 source only
P1 highest semantic relevance evidence
P2 lowest relevance among SRC-selected
P3 full SRC
```

behavioral fingerprint：

```text
216D / reader
sha256 =
2a5d7cdd1c532c2b34979edac90b7868b3c20c9d53e07076f4e03579f9c4afc5
```

features：

```text
E0 = semantic/text/relevance/time
E1 = multilingual NLI
E2 = tokenizer compatibility
E3 = reader-forward compatibility
```

E3：

```text
source_only_margin
source_only_entropy
source_nll
evidence_nll_per_token
conditional_evidence_nll_per_token
nll_gap
```

---

# 18. M1 ZERO-TOUCH 结果

B4：

```text
E0 + E1 + E2 + behavioral fingerprint
```

Ma-Weibo held-out：

```text
qwen      Δ = +0.0112
mistral   Δ = -0.0978
internlm  Δ = +0.0751
```

aggregate：

```text
mean Δ = -0.0038
95% CI = [-0.0360, +0.0255]
```

结论：

```text
ZERO-TOUCH = FAIL
```

因此不要声称 behavioral fingerprint 已解决 unseen-reader transfer。

---

# 19. M1 LIGHT-TOUCH 结果

B5：

```text
B4 + E3
```

Ma-Weibo：

```text
qwen      +0.1247
mistral   +0.0469
internlm  +0.0131
```

aggregate：

```text
mean Δ = +0.0616
95% CI = [+0.0068, +0.1084]
3/3 positive
```

正式 verdict：

```text
M1_CONDITIONAL_GO
```

PHEME diagnostic：

```text
ZERO-TOUCH  +0.0312
LIGHT-TOUCH -0.1063
```

PHEME 不决定 gate，但暴露出显著跨数据集不稳定性。

---

# 20. M1-E：当前最新结论

M1-E：

```text
COMPLETED
post-hoc diagnostic only
defines_new_gate = false
M1_CONDITIONAL_GO unchanged
```

feature groups：

```text
Z = E0 + E1 + E2
F = behavioral fingerprint
S = source-state E3
C = evidence-familiarity E3
```

variants：

```text
D0 = Z
D1 = Z+F            = B4
D2 = Z+F+S
D3 = Z+F+C
D4 = Z+F+S+C        = B5
D5 = Z+S+C          = no fingerprint
```

Ma-Weibo vs frozen B0：

```text
D0 -0.0471
D1 -0.0038
D2 +0.0482
D3 +0.0185
D4 +0.0616
D5 +0.0710
```

关键：

```text
D5 WITHOUT fingerprint
=
+0.0710
CI [+0.0448, +0.0974]
3/3 positive
worst +0.0409
```

fingerprint incremental：

```text
D4 - D5
Ma-Weibo = -0.0094
CI [-0.0675, +0.0435]

PHEME = -0.1340
CI [-0.1582, -0.1000]
```

结论：

```text
behavioral fingerprint 在 E3 存在时无正增量
PHEME 上显著有害
```

因此当前最强机制不是 static fingerprint，而是：

```text
query/source-specific reader state
+
evidence familiarity
```

---

# 21. M1-E dataset shift

Ma-Weibo -> PHEME 的 E3 分布明显移动。

典型：

```text
Qwen source-only margin at 15m:
Ma-Weibo = -6.17
PHEME    = +9.15
Wasserstein ≈ 15.73
```

所有 reader 还看到：

```text
conditional evidence NLL 在 PHEME 上升
NLL gap 缩小
```

只能说与 PHEME LIGHT-TOUCH failure 共现，不能声称因果。

---

# 22. M1-E concentration

Ma-Weibo B5-B0：

按 reader：

```text
qwen      +0.1247
mistral   +0.0469
internlm  +0.0131
```

按 cutoff：

```text
15m   +0.0692
60m   +0.0511
360m  +0.0601
```

不是单一 cutoff artifact。

按 class：

```text
HELPFUL +0.0467
NEUTRAL +0.0348
HARMFUL +0.1031
```

收益最集中在：

```text
HARMFUL
```

这使 reader-specific harmful evidence / susceptibility 成为值得后续考虑的子方向。

---

# 23. B3 baseline

B3 看起来很强（Macro-F1 约 0.396），但 M1-E 已确认：

```text
B3 = same-evidence cross-reader transfer
```

它会直接复制 fingerprint 最近的 training reader 对**同一个 evidence key** 的真实 utility label。

所以 B3 需要：

```text
该 evidence 已经至少有一个 existing-reader utility oracle label
```

它不等价于真正新事件、无 utility oracle 的部署场景。

不要降低或隐藏 B3；但也不要错误把它当完全公平的 oracle-free baseline。

---

# 24. 已经得到支持的内容

```text
reader-specific evidence utility exists
```

```text
reader × evidence interaction is strong
```

```text
structure-centric strong non-additive utility hypothesis was not established
```

```text
LIGHT-TOUCH reader-state/evidence-familiarity is promising on Ma-Weibo
```

```text
HARMFUL prediction appears to benefit strongly on Ma-Weibo
```

---

# 25. 已经失败或不应继续原样推进的内容

```text
CR-TSER P3/P4
```

```text
通过改 P2 threshold 救回 CR-TSER
```

```text
重新启用 Weibo22 作为当前 primary
```

```text
static behavioral fingerprint 已经解决 unseen-reader utility transfer
```

```text
ZERO-TOUCH BCR 已被证明有效
```

```text
直接进入旧 M2 reader-panel expansion
```

---

# 26. 当前真正未解决的问题

1. 是否正式 pivot 到：

```text
query-specific reader state
+
evidence familiarity
-> reader-specific utility
```

2. PHEME domain shift 的主要来源是什么：
   - language/domain
   - calibration
   - source prior
   - evidence familiarity
   - tokenizer
   - stance distribution

3. 是否还需要继续研究 static fingerprint，还是将其降级为 diagnostic。

4. 是否应该将新主线更聚焦于：

```text
reader-specific harmful evidence / susceptibility
```

5. 是否值得扩到更多 reader。  
   原 M2 是为了验证 fingerprint transfer；该机制已被 M1-E 明显削弱，因此必须先研究评审。

---

# 27. BCR 代码结构

```text
project/bcr_utility/
```

重要模块：

```text
config/protocol.py

data/historical_import.py
data/atomic_manifest.py

probes/probe_manifest.py
probes/probe_contexts.py
probes/fingerprint.py

features/evidence_features.py
features/nli_features.py
features/tokenizer_features.py
features/compatibility_features.py

models/baselines.py
models/conditioned_utility.py

evaluation/reader_geometry.py
evaluation/utility_metrics.py
evaluation/bootstrap.py
evaluation/unseen_reader.py

attribution/feature_ablation.py
attribution/evidence_pins.py
attribution/dataset_shift.py
attribution/concentration.py
attribution/b3_contract.py

verifier.py
verifier_m1.py
verifier_m1e.py
```

脚本：

```text
scripts/bcr_build_protocol.py
scripts/bcr_import_historical_atomic.py
scripts/bcr_probe_scan.py
scripts/bcr_build_probe_manifest.py
scripts/bcr_verify.py

scripts/bcr_run_reader_probes.py
scripts/bcr_extract_features.py
scripts/bcr_run_mechanism_pilot.py

scripts/bcr_run_attribution.py
scripts/bcr_audit_m1e.py
```

---

# 28. 结果目录

```text
results/bcr_utility_v1/
```

M0：

```text
bootstrap/
M0_REPORT.md
```

M1：

```text
m1/
M1_REPORT.md
M1_VERDICT.json
probe_responses/
fingerprints/
features/
evaluation/
```

M1-E：

```text
m1e/
M1E_REPORT.md
M1E_VERDICT.json
m1_evidence_pins.json
dataset_shift.json
concentration.json
b3_contract.json
comparisons_vs_B0.json
attribution/
```

新智能体最先读：

```text
results/bcr_utility_v1/m1/M1_REPORT.md
results/bcr_utility_v1/m1/M1_VERDICT.json
results/bcr_utility_v1/m1e/M1E_REPORT.md
results/bcr_utility_v1/m1e/M1E_VERDICT.json
```

---

# 29. Verifier 与测试

当前 verifier modes：

```text
m0
m1
m1e
```

最新 SERVER authoritative：

```text
mode = m1e
49 checks
issues = 0
pending = 0
```

接手前至少运行：

```bash
python -m pytest project/bcr_utility/tests -q
python -m pytest project/cr_tser/tests -q
python -m compileall project/bcr_utility scripts/bcr_*.py
```

最近 M1-E 开发记录：

```text
BCR tests ≈ 216 passed
CR-TSER regression = 219 passed
```

SERVER 还应执行：

```bash
python scripts/bcr_verify.py --mode m1e
```

---

# 30. CR-TSER 可复用依赖

BCR 只读复用以下 CR-TSER contract：

```text
project/cr_tser/config/pilot_config.py
project/cr_tser/data/snapshot_bridge.py
project/cr_tser/intervention/evidence_units.py
project/cr_tser/intervention/semantic_reference.py
project/cr_tser/evaluation/bootstrap.py
project/cr_tser/readers/sequence_scorer.py
project/cr_tser/readers/base_reader.py
project/cr_tser/models/utility_heads.py
scripts/cr_tser_common.py
```

不要为了 BCR 方便直接修改历史 CR-TSER 实现。

如 scientific contract 必须变化：

```text
STOP
研究评审
正式协议修订
```

---

# 31. GPU 共享资源规则

RTX 4090 是共享卡。

历史 GLM OOM 与共享 GPU 显存有关。

运行大型模型前：

```bash
nvidia-smi
```

禁止：

```text
kill 其他用户进程
偷偷量化
降低 token budget
改 dtype
改 scoring contract
```

如果资源不足：

```text
RESOURCE_CONTENTION / STOP
```

不要用工程妥协污染科学协议。

---

# 32. Artifact 纪律

正式 artifact 应记录：

```text
path
bytes
SHA256
schema/protocol version
environment
commit
```

大型 checkpoint 可 server-only。

不要未经批准：

```text
引入 Git LFS
改变 artifact retention policy
为了 push 而重写/压缩正式 cache
```

---

# 33. Fail-closed 纪律

发现任何：

```text
source drift
hash mismatch
cache identity mismatch
manifest mismatch
reader identity drift
schema mismatch
unexpected row count
split overlap
unexpected model substitution
```

必须：

```text
STOP
保存证据
报告
```

禁止：

```text
强行改 fingerprint
删除旧 cache 重跑
自动降低 verifier
为了通过测试修改科学标准
```

---

# 34. 协作方式

历史工作模式：

```text
执行一轮
-> commit
-> push
-> STOP
-> 研究审批
-> 下一轮
```

GO 不能自动进入下一阶段。  
FAIL 不能自动修改方案。

新智能体继续遵守。

---

# 35. 当前正式状态

```text
GitHub HEAD =
b62010b07f8fa8fbc9d2604b4e8fdf8cbe6a8d12

CR-TSER =
CLOSED / NO-GO

BCR M0 =
COMPLETE

BCR M1 =
M1_CONDITIONAL_GO
ZERO-TOUCH FAIL
LIGHT-TOUCH PASS on Ma-Weibo

BCR M1-E =
COMPLETED

behavioral fingerprint incremental contribution =
NOT SUPPORTED

reader-state + evidence-familiarity =
PROMISING ON MA-WEIBO

cross-dataset robustness =
NOT ESTABLISHED

M2 =
NOT ENTERED
NOT AUTHORIZED

next =
AWAIT RESEARCH REVIEW
```

---

# 36. 新智能体接手后的第一步

不要立即启动模型或新实验。

执行：

```text
1. fetch/pull GitHub
2. HEAD 对齐 b62010b07...
3. 阅读本交接手册
4. 阅读 M1_REPORT / M1_VERDICT
5. 阅读 M1E_REPORT / M1E_VERDICT
6. 阅读 project/bcr_utility/config/protocol.py
7. LOCAL 跑 BCR tests + CR-TSER regression + compileall
8. SERVER 跑 bcr_verify --mode m1e
9. 确认历史 namespaces/hash 未漂移
10. 等待新的研究决策
```

不要自动：

```text
部署 Qwen3-0.6B
部署 Phi
部署 Gemma
扩 reader panel
生成新 utility labels
进入 M2
```

---

# 37. 下一阶段最可能的研究 pivot

当前更符合证据的新假设是：

> Reader-specific evidence utility 主要与 reader 在当前 source/query 上的即时 epistemic state 和对 evidence 的 familiarity 有关，而不是由一个跨任务稳定的静态 behavioral fingerprint 决定。

形式上更像：

```text
U(e, r, q, t)
=
f(
evidence semantics,
reader source-state,
reader evidence familiarity,
time
)
```

而不是：

```text
U(e,r) = f(fingerprint(r), evidence)
```

这是**候选下一阶段方向**，尚未正式注册成新 protocol。

---

# 38. 对外表述边界

可以说：

```text
reader-specific evidence utility clearly exists
```

```text
strong structure-centric interaction was not established
```

```text
zero-touch fingerprint conditioning failed the Ma-Weibo pilot
```

```text
light-touch reader-state/familiarity signals improved Ma-Weibo
```

```text
fingerprint added no positive value once E3 was available
```

```text
PHEME revealed substantial domain instability
```

不能说：

```text
BCR 已经跨数据集泛化
behavioral fingerprint 已解决 unseen-reader transfer
LIGHT-TOUCH universally works
PHEME confirms the method
social structure is useless
```

---

# 39. 路径速查

```text
LOCAL env:
pytorch
```

```text
LOCAL repo:
E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER
```

```text
SERVER root:
/data/jyz/next/llm/
```

```text
SERVER repo:
/data/jyz/next/llm/cr_tser_ws
```

```text
SERVER env:
DGPA
```

```text
compat overlay:
/data/jyz/next/llm/.cr_tser_v2p0/tf453
```

```text
SSH:
next/.codex_tmp_ssh_run.cmd
```

```text
SCP:
next/.codex_tmp_scp_run.cmd
```

```text
Qwen3-8B:
/data/jyz/next/llm/model/qwen3-8b
```

```text
Mistral:
/data/jyz/next/llm/model/mistral-7b-instruct-v0.3
```

```text
InternLM:
/data/jyz/next/llm/model/internlm3-8b-instruct
```

```text
retired GLM:
/data/jyz/next/llm/model/glm-4-9b-chat-hf
```

```text
NLI:
/data/jyz/next/llm/model/mdeberta-v3-base-mnli-xnli
```

```text
Qwen3-0.6B:
historically present; search server first
```

---

# 40. 最重要的一句话

> **新智能体不要继续执行旧 M2。当前最新实验已经说明 static behavioral fingerprint 不是 Ma-Weibo LIGHT-TOUCH 增益来源；下一步必须先进行研究假设评审，决定是否将主线正式 pivot 到 query-specific reader state / evidence familiarity，再制定新的实验协议。**
