# CR-TSER V2-P1A Execution Plan
## Formal Manifest Freeze + Frozen-Reader Utility Label Generation

### Approved baseline

```text
HEAD = c4773a3ec605e40d6aa26cd2244a41ccb51b998a
V2-P0 = P0_PASS
PRIMARY = Ma-Weibo
SECONDARY = PHEME
```

This round starts the first formal expensive stage after P0 approval.

Scope:

```text
1. build and validate formal manifests
2. freeze Ma-Weibo/PHEME source + split + snapshot/intervention manifests
3. generate formal frozen-reader utility labels
4. audit label-cache integrity
5. commit/push evidence
6. STOP
```

Do **not** run predictor training, Stage A/B selection, held-out evaluation, P1-P4 gates, or the final Pilot report.

---

## 1. Permanent execution contract

### LOCAL

```text
env = pytorch
role = lightweight verification only
```

Allowed locally:

- Git checks;
- unit tests;
- code verifier;
- compileall;
- static/schema review.

Do not load the 8B/9B readers locally.

### SERVER

```text
env = DGPA
root = /data/jyz/next/llm/
role = manifests, large-model inference, formal experiment work
```

All reader inference and formal utility-label generation must run on SERVER/DGPA.

### Server access

Use only the wrappers under the local `next` directory:

```text
next/.codex_tmp_ssh_run.cmd
next/.codex_tmp_scp_run.cmd
```

If SSH/SCP reports an explicit server/tunnel refusal:

```text
STOP
preserve the exact error
tell the user the tunnel/server refused the connection
wait for the user to restore the tunnel
retry only after user confirmation
```

Do not change host/port/user/connection method and do not continue the server stage locally.

---

## 2. Existing server model resources

Do not redownload models that are already present.

Formal V2 readers already available:

```text
Qwen3-8B:
  /data/jyz/next/llm/model/qwen3-8b
  historical authoritative target:
  /data/jyz/next/model/qwen3-8b

GLM-4-9B-Chat-HF:
  /data/jyz/next/llm/model/glm-4-9b-chat-hf

InternLM3-8B-Instruct:
  /data/jyz/next/llm/model/internlm3-8b-instruct
```

Historical server resource also known:

```text
Qwen3-0.6B
```

It is not a CR-TSER V2 reader and must not be substituted into this experiment.

If some *other* required model resource is genuinely absent, deploy only that missing resource. Prefer server download. If server download fails, LOCAL/pytorch may download the exact resource to a **non-C drive** and transfer it with `.codex_tmp_scp_run.cmd`. Actual validation/use remains server-side.

---

## 3. Frozen compatibility runtime

DGPA itself remains unchanged.

For every server process that imports/loads the frozen readers, use the approved isolated compatibility overlay:

```text
/data/jyz/next/llm/.cr_tser_v2p0/tf453
```

Required effective runtime:

```text
transformers = 4.53.3
tokenizers   = 0.21.4
```

Example:

```bash
export PYTHONPATH=/data/jyz/next/llm/.cr_tser_v2p0/tf453${PYTHONPATH:+:$PYTHONPATH}
```

Before formal label generation, verify that the effective Python process reports Transformers 4.53.3 and all three frozen readers remain loadable.

Do not patch model remote code and do not modify DGPA site-packages.

---

## 4. LOCAL checkpoint

Run in LOCAL/pytorch from commit `c4773a3...`:

```bash
git status
git rev-parse HEAD
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python -m compileall project/cr_tser scripts
```

Expected baseline:

```text
139 tests pass
code verifier issues = 0
compileall clean
```

If baseline fails, stop before server formal work.

---

## 5. SERVER checkpoint — environment and source identity

Connect with `.codex_tmp_ssh_run.cmd`.

Use:

```text
SERVER/DGPA
workspace under /data/jyz/next/llm/
HEAD = c4773a3ec605e40d6aa26cd2244a41ccb51b998a
```

Activate DGPA and the approved Transformers overlay.

Resolve the already-approved data sources:

```text
CRTSER_MAWEIBO_RAW
CRTSER_MAWEIBO_LABELS
CRTSER_PHEME_RAW
CRTSER_SEMANTIC_MODEL
CRTSER_CANONICAL_TOKENIZER
CRTSER_QWEN_MODEL
CRTSER_GLM_MODEL
CRTSER_INTERNLM_MODEL
CRTSER_OUT_ROOT
```

Ma-Weibo must reproduce the approved composite identity:

```text
combined_source_sha256 =
b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d
```

Any source drift => STOP. Do not rebuild against a different dataset.

All V2 formal outputs stay under the synchronized server workspace's:

```text
results/cr_tser_v2/
```

V1 `results/cr_tser/` remains immutable.

---

## 6. Formal manifest freeze

Build manifests for both datasets:

```bash
python scripts/cr_tser_build_manifests.py --dataset maweibo
python scripts/cr_tser_build_manifests.py --dataset pheme
```

Do not use `--smoke`.

Do not use `--force` unless the formal manifest directory already exists **and** no formal label cache exists; if unexpected formal artifacts already exist, inspect and report before overwriting anything.

Before any reader utility call, audit the generated manifests.

Required Ma-Weibo split:

```text
foundation_train = 80
utility_train    = 50
utility_dev      = 15
utility_eval     = 25
seed             = 7319
event-disjoint   = true
```

Verify for both datasets:

- source fingerprint frozen;
- snapshot manifest exists;
- intervention manifest exists;
- hashes.json exists;
- 15m / 1h / 6h cutoffs are the only cutoffs;
- no future-node leakage;
- no 1021-node truncation;
- no unexpected dataset mixing;
- reader IDs/hashes match the approved frozen readers.

Record the complete manifest SHA256/fingerprints before label generation.

Once the first formal utility-label row exists, manifests are immutable.

---

## 7. Formal utility-label generation

Run only on SERVER/DGPA with the Transformers 4.53.3 overlay.

Use one reader at a time so GPU memory is released between readers and the cache remains resumable.

Recommended order:

```text
Ma-Weibo:
  qwen
  glm
  internlm

PHEME:
  qwen
  glm
  internlm
```

Commands:

```bash
python scripts/cr_tser_generate_labels.py --dataset maweibo --reader qwen
python scripts/cr_tser_generate_labels.py --dataset maweibo --reader glm
python scripts/cr_tser_generate_labels.py --dataset maweibo --reader internlm

python scripts/cr_tser_generate_labels.py --dataset pheme --reader qwen
python scripts/cr_tser_generate_labels.py --dataset pheme --reader glm
python scripts/cr_tser_generate_labels.py --dataset pheme --reader internlm
```

Do not use `--smoke` or `--limit-snapshots` for the formal run.

The generator is resumable. If execution is interrupted, resume the same reader/dataset cache. Do not delete valid completed rows.

Any `CacheIdentityMismatch` is fail-closed: STOP and report. Do not rewrite fingerprints to force reuse.

---

## 8. Label-cache audit

After each reader/dataset completes, verify at minimum:

```text
dataset
reader
row count
unique cache-key count
duplicate-key count = 0
NaN/Inf count = 0
missing required fields = 0
reader identity/hash matches approved reader
source fingerprint matches frozen manifest
cutoffs only 15/60/360
formal namespace only
```

After all six runs, produce a concise utility-label summary containing:

```text
per dataset:
  total rows
  rows per reader
  rows per cutoff
  rows per intervention type
  events represented in train/dev/eval
  label/sign distribution
  cache file path
  file size
  sha256

global:
  identity mismatch count
  duplicate count
  invalid numeric count
```

Do not tune thresholds or reinterpret labels based on these distributions.

---

## 9. Verification and stop point

Run:

```bash
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2
```

Pilot verifier may still report downstream P1-P4 artifacts as pending. Do not fabricate them.

This round ends after manifests + formal utility labels are complete and audited.

Do **not** run:

```text
cr_tser_train_predictors.py
formal Stage-A subset freeze
held-out-reader scoring
P1/P2/P3/P4 final gates
cr_tser_run_pilot.py final report
```

---

## 10. Submission

Do not change the scientific protocol.

If a genuine runtime/code defect blocks this frozen stage:

```text
STOP
preserve evidence
report the defect
do not patch and continue automatically
```

Before Git commit, inspect artifact sizes and follow the repository's existing artifact-retention policy. Do not invent Git LFS/compression or alter `.gitignore` simply to force large caches into Git.

Commit/push the formal manifests, required evidence/summaries, and any artifacts already intended to be repository-tracked. Keep any heavyweight server-only artifacts at their authoritative server location if that is the existing project policy, and record their path + SHA256 in the committed summary.

Final report must state:

1. new commit SHA;
2. LOCAL/pytorch checks;
3. SERVER/DGPA workspace;
4. connection/transfer wrappers used;
5. effective Transformers/tokenizers versions;
6. exact data/model paths;
7. Ma-Weibo source fingerprint;
8. manifest hashes and split counts;
9. utility-label row counts/hashes per dataset/reader;
10. verifier status;
11. any interruption/resume;
12. confirmation that predictor training, Stage A/B and P1-P4 were NOT run.

Push and STOP for approval.
