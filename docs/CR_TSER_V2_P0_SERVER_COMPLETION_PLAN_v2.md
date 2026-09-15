# CR-TSER V2-P0 Server Completion Execution Plan

## 1. Purpose

This round completes the unfinished **CR-TSER V2-P0 frozen-reader preflight** on the GPU server.

Latest reviewed repository commit:

```text
b5f91e162ed0d6d2431add410760c155c3f23c39
```

Frozen scientific-code baseline:

```text
fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f
```

The reviewed delta from `fa8ddbb` to `b5f91e` contains only documentation and P0 evidence artifacts. It does **not** modify the frozen CR-TSER implementation under `project/cr_tser` or `scripts`.

Current V2-P0 status:

```text
Ma-Weibo data integrity        PASS
Ma-Weibo viability             PASS (4591 >= 170)
Causal snapshot integrity      PASS
PHEME smoke                    PASS
Frozen reader load             NOT COMPLETED on suitable GPU server
Teacher-forced A/B sanity      NOT COMPLETED on suitable GPU server
Overall P0                     P0_FAIL solely because reader prerequisites were unresolved locally
Formal Pilot                   NOT RUN
```

The only objective of this round is:

> Run the frozen three-reader P0-F/P0-G checks on the designated server, then re-run the frozen V2-P0 entrypoint to obtain the authoritative P0 verdict.

Even if the result is `P0_PASS`, this round must stop. It must not start the formal Pilot.

---

## 2. Mandatory execution-environment contract

This contract applies to this plan and to all subsequent CR-TSER plans/prompts unless explicitly revised by the user.

### 2.1 Local machine

Python virtual environment:

```text
pytorch
```

If Conda is used:

```bash
conda activate pytorch
```

The local machine is for **lightweight development and verification only**:

- repository synchronization / Git inspection;
- source-code review;
- unit tests that do not require large frozen readers;
- code verifier;
- `compileall`;
- static artifact/schema checks;
- generation/review of execution plans and scripts;
- small deterministic smoke tests that fit the local environment.

The local machine must **not** be used for:

- loading Qwen3-8B / GLM-4-9B-Chat-HF / InternLM3-8B-Instruct;
- frozen-reader A/B scoring;
- formal utility-label generation;
- large-model inference;
- predictor/full training intended as a formal experiment;
- Stage-A/Stage-B formal held-out-reader evaluation;
- P1-P4 formal experiments.

### 2.2 Server

Server Python virtual environment:

```text
DGPA
```

Activate before any Python command:

```bash
conda activate DGPA
```

Server storage/work root:

```text
/data/jyz/next/llm/
```

All newly created server-side project files, caches intentionally created for this task, checkpoints, logs, and result artifacts must remain under this root.

Existing authoritative raw datasets or model resources may be referenced at their existing audited locations; do not silently copy or relocate a source of record merely to satisfy this directory rule.

The server is the required environment for:

- downloading the official frozen model weights whenever server network access permits;
- validating downloaded model files / checkpoint completeness;
- frozen 8B/9B reader loading;
- reader identity/hash audit;
- teacher-forced A/B log-probability sanity;
- GPU-dependent formal preflight;
- later large-model inference;
- formal utility-label generation;
- training;
- held-out-reader experiments;
- P1-P4 formal experiments.

### 2.3 Model-weight acquisition rule

The three frozen reader checkpoints must ultimately reside and be validated on the server under:

```text
/data/jyz/next/llm/
```

Preferred path:

```text
SERVER/DGPA downloads model
→ verify files on server
→ load/test on server
```

If and only if the server cannot download a required checkpoint because of network/access restrictions, a fallback transfer is allowed:

```text
LOCAL/pytorch downloads exact official checkpoint
→ store only on a NON-C drive temporary directory
→ verify local file set / source
→ upload/copy to /data/jyz/next/llm/ on server
→ re-verify model identity/hash on server
→ actual load and A/B tests run on SERVER/DGPA
```

The local fallback must obey all of the following:

- **never download model weights to the Windows C: drive**;
- use an existing non-C drive workspace (for example a directory on D:, E:, or another non-system drive);
- do not treat a local successful download as a valid reader preflight;
- do not load or test the 8B/9B models on the local machine;
- do not convert, quantize, merge, prune, rename into another model identity, or substitute checkpoints;
- preserve the exact official model/revision files;
- after upload, the server-side copy is the authoritative checkpoint used for hashing and all P0 tests;
- temporary local weights may be removed only after confirming the server copy is complete.

Every future execution report must explicitly record whether each command was run on **LOCAL/pytorch** or **SERVER/DGPA**.

---

## 3. Frozen research protocol

Do not modify:

- primary dataset: Ma-Weibo;
- secondary dataset: PHEME;
- reader set:
  - `Qwen/Qwen3-8B`
  - `zai-org/glm-4-9b-chat-hf`
  - `internlm/internlm3-8b-instruct`
- 15m / 1h / 6h temporal cutoffs;
- split seed / split sizes;
- BiTTE;
- utility equation;
- intervention definitions;
- B0/B1/B3;
- shared/residual decomposition;
- LORO rotations;
- robust selector;
- bootstrap protocol;
- P1-P4 thresholds.

Do not substitute a smaller model, quantized checkpoint, API model, or another model family.

Do not use historical TC-DSCR learned artifacts as a replacement for V2 data or reader evidence.

---

## 4. Local checkpoint L0 — lightweight verification only

Run on:

```text
HOST = LOCAL
ENV  = pytorch
```

### L0.1 Repository state

Synchronize to the latest approved submission:

```text
b5f91e162ed0d6d2431add410760c155c3f23c39
```

Record:

```bash
git status
git rev-parse HEAD
git log -1 --oneline
```

Working tree must be clean before verification.

### L0.2 Lightweight regression checks

Run:

```bash
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python -m compileall project/cr_tser scripts
```

Expected from the latest submission:

```text
tests: PASS
code verifier: issues = 0
compileall: clean
```

If these fail because the current checkout differs from the reviewed submission, stop and report the discrepancy.

### L0.3 Prohibited local actions

Do not attempt to make P0 pass by downloading/loading the three frozen readers on the local GPU.

Do not run formal reader sanity locally.

Do not change code in response to a local resource limitation.

---

## 5. Server checkpoint S0 — environment and workspace freeze

Run on:

```text
HOST = SERVER
ENV  = DGPA
ROOT = /data/jyz/next/llm/
```

Before any experiment:

```bash
conda activate DGPA
which python
python -V
echo "$CONDA_DEFAULT_ENV"
echo "$CONDA_PREFIX"
python - <<'PY'
import torch, transformers
print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_version =", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
nvidia-smi
```

Record the exact output in the new P0 evidence package.

Required:

```text
CONDA_DEFAULT_ENV = DGPA
CUDA available = true
```

If the server is not using `DGPA`, stop before any formal P0 run.

---

## 6. Server checkpoint S1 — repository synchronization

The synchronized repository/worktree used for this task must live under:

```text
/data/jyz/next/llm/
```

Use the existing synchronized repository location under this root. Do not create a second working copy elsewhere.

Record:

```bash
pwd
git status
git rev-parse HEAD
git log -1 --oneline
```

Target repository commit:

```text
b5f91e162ed0d6d2431add410760c155c3f23c39
```

Also verify that the frozen implementation remains equivalent to the scientific baseline `fa8ddbb`:

```bash
git diff --name-only fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f..b5f91e162ed0d6d2431add410760c155c3f23c39
```

The reviewed difference should be documentation/evidence only. If `project/cr_tser` or `scripts` unexpectedly differs, stop and report.

---

## 7. Server checkpoint S2 — acquire and resolve exact frozen reader resources

The code consumes:

```text
CRTSER_QWEN_MODEL
CRTSER_GLM_MODEL
CRTSER_INTERNLM_MODEL
```

Required exact readers:

```text
Qwen/Qwen3-8B
zai-org/glm-4-9b-chat-hf
internlm/internlm3-8b-instruct
```

First search existing server resources under:

```text
/data/jyz/next/llm/
```

If a required exact checkpoint is missing, **attempt the official download on SERVER/DGPA first** and store it under `/data/jyz/next/llm/`.

If the server download fails specifically because of network/access restrictions, use the allowed fallback:

1. on LOCAL, activate `pytorch`;
2. choose an existing **non-C drive** temporary model directory;
3. download the exact official checkpoint there;
4. do not load/test the model locally;
5. transfer the complete checkpoint directory to `/data/jyz/next/llm/` on the server;
6. on SERVER/DGPA, verify the uploaded file set, model identity, tokenizer, chat template, and hashes;
7. only the server-side copy may be assigned to `CRTSER_*_MODEL` and used for P0-F/P0-G.

For each reader, record:

- acquisition mode: `server_existing`, `server_download`, or `local_nonC_download_then_upload`;
- official source/model ID;
- local temporary path if fallback was used (must not be on C:);
- absolute server path;
- whether the server path exists;
- checkpoint/config identity;
- tokenizer identity;
- chat-template identity.

Then export only the server paths:

```bash
export CRTSER_QWEN_MODEL="<server path to exact Qwen3-8B>"
export CRTSER_GLM_MODEL="<server path to exact GLM-4-9B-Chat-HF>"
export CRTSER_INTERNLM_MODEL="<server path to exact InternLM3-8B-Instruct>"
```

Do **not** guess a path and do not substitute another checkpoint.

If an exact frozen reader cannot be obtained by either the server download or the permitted local-non-C-drive transfer route, record the missing resource and finish with `P0_FAIL`.

Forbidden:

- downloading model weights to the local C: drive;
- quantized substitutes;
- alternate checkpoints;
- API substitutes;
- locally loading/testing these 8B/9B readers.

---

## 8. Server checkpoint S3 — resolve V2 data inputs

Use the already-audited source-of-record paths.

Required environment variables:

```text
CRTSER_MAWEIBO_RAW
CRTSER_MAWEIBO_LABELS
CRTSER_PHEME_RAW
```

Do not change the underlying dataset identity.

Before the formal run, verify that the resolved Ma-Weibo composite fingerprint matches the reviewed P0 evidence:

```text
combined_source_sha256 =
b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d
```

If it differs, stop and report source drift rather than silently continuing.

---

## 9. Server checkpoint S4 — frozen reader load audit

Run P0-F on the server only.

Rules:

- use `DGPA`;
- GPU required;
- load readers sequentially;
- never keep all three readers resident simultaneously;
- default frozen dtype/device unless the code itself specifies otherwise;
- no quantization;
- no API fallback.

For each reader the evidence must include:

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
load_error
```

Any failure => `P0_FAIL`.

---

## 10. Server checkpoint S5 — teacher-forced A/B sanity

Run P0-G on the server only.

Use the frozen teacher-forced sequence-scoring implementation.

Forbidden:

```text
generate()
free-form generation
generated confidence
self-reported probability
answer parsing
```

Use at least 20 frozen sanity prompts per reader, each scored twice.

Each reader must satisfy:

```text
boundaries_ok = true
identical_predictions = true
identity_rate = 1.0
finite log probabilities
no NaN / Inf
no empty continuation
```

Any failure => `P0_FAIL`.

---

## 11. Server checkpoint S6 — authoritative V2-P0 rerun

After all required paths are exported, run the frozen formal entrypoint on:

```text
SERVER / DGPA
```

Command:

```bash
python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity
```

Do not manually edit the generated verdict.

The formal run may recompute the already-passed data-side checks; that is acceptable because the authoritative P0 verdict must be emitted by the frozen entrypoint.

All V2 result artifacts must remain under the synchronized repository/work root beneath:

```text
/data/jyz/next/llm/
```

The historical V1 namespace:

```text
results/cr_tser/
```

is read-only and must remain byte-identical.

---

## 12. Server checkpoint S7 — verification

After the P0 run:

```bash
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2
```

Expected:

- code verifier `issues = 0`;
- if P0 passes, later P1-P4 artifacts may still be pending because they are intentionally not run in this round;
- do not fabricate missing downstream artifacts.

---

## 13. Required outputs

Update/regenerate the V2 P0 evidence under:

```text
results/cr_tser_v2/p0/
```

At minimum:

```text
p0_readiness.json
P0_READINESS.md
P0_EVIDENCE_PACKAGE.md
environment.json
maweibo_audit.json
maweibo_source_fingerprint.json
snapshot_integrity.json
pheme_smoke.json
reader_audit.json
label_scoring_sanity.json
v1_namespace_baseline.json
```

The report must additionally state:

```text
local_env = pytorch
server_env = DGPA
server_root = /data/jyz/next/llm/
local_tasks = [...]
server_tasks = [...]
```

Do not hand-edit scientific metrics or the P0 verdict.

---

## 14. Stop rules

There are only two valid endpoints.

### P0_PASS

All of the following must hold:

```text
Ma-Weibo source integrity PASS
Ma-Weibo viable events >= 170
PHEME smoke OK
Qwen loaded
GLM loaded
InternLM loaded
all A/B boundaries OK
all repeated scoring deterministic
```

Then:

```text
P0_PASS
```

Commit the evidence and **STOP**.

Do not create formal manifests.
Do not generate formal utility labels.
Do not train predictors.
Do not run Stage A/B.
Do not run P1-P4.

### P0_FAIL

If any prerequisite fails:

```text
P0_FAIL
```

Record exactly one or more concrete blockers, preserve raw error evidence, commit the evidence, and **STOP**.

Do not change the scientific protocol to make the check pass.

---

## 15. Git submission requirements

Before commit:

```bash
git status
git diff --stat
```

Only expected evidence/report changes for this round may be committed.

Suggested commit messages:

PASS:

```text
CR-TSER V2-P0 server preflight complete — prerequisites PASS, formal pilot NOT RUN
```

FAIL:

```text
CR-TSER V2-P0 server preflight complete — prerequisites FAIL, formal pilot NOT RUN
```

Push to GitHub and stop for research review.

The final agent response must report:

1. exact commit SHA pushed;
2. LOCAL/pytorch checks executed and outcomes;
3. SERVER/DGPA checks executed and outcomes;
4. server repository root used under `/data/jyz/next/llm/`;
5. acquisition mode and exact server path for each of the three readers, plus any non-C local temporary path used for fallback transfer;
6. confirmation that all actual reader load/A-B tests ran on SERVER/DGPA;
7. exact P0 verdict;
7. code verifier / pilot verifier status;
8. files changed;
9. confirmation that formal Pilot and P1-P4 were not run.
