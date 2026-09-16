# CR-TSER V2 Reader Protocol Amendment R1
## GLM → Mistral-7B-Instruct-v0.3

### Status

Approved research decision:

```text
REPLACE:
  zai-org/glm-4-9b-chat-hf

WITH:
  mistralai/Mistral-7B-Instruct-v0.3
```

Reason:

```text
GLM-4-9B-Chat-HF requires ~19.3 GiB before the final scoring peak
and repeatedly OOMs on the shared RTX 4090 under the available sustained
memory window.

The substitution is made before predictor training, Stage A/B and P1-P4.
No P1-P4 scientific outcome has been observed.
```

Current repository baseline:

```text
6da025cbf60c0990efcc301b06fa2efef21e2fbb
```

Current formal V2 artifacts under:

```text
results/cr_tser_v2/
```

are historical and must not be rewritten.

This amendment creates a new reader-protocol namespace:

```text
protocol = v2r1
results  = results/cr_tser_v2r1/
```

---

# 1. Scientific scope of the amendment

This amendment changes only the third-party frozen reader set.

Unchanged:

```text
Primary dataset       = Ma-Weibo
Secondary dataset     = PHEME
cutoffs               = 15m / 1h / 6h
split seed            = 7319
split sizes           = 80 / 50 / 15 / 25
SRC definition
intervention family
utility equation
utility threshold
BiTTE
B0/B1/B3 definitions
shared/residual decomposition
training losses
bootstrap protocol
selection arms
P1-P4 thresholds
PHEME secondary condition
```

The research hypothesis remains:

```text
cross-reader social-evidence utility
+
shared/residual utility structure
+
unseen-reader transfer
```

The dataset protocol V2 remains authoritative.

---

# 2. New frozen reader set

Freeze:

```text
qwen:
  Qwen/Qwen3-8B

mistral:
  mistralai/Mistral-7B-Instruct-v0.3

internlm:
  internlm/internlm3-8b-instruct
```

Retire from all new v2r1 formal execution:

```text
glm:
  zai-org/glm-4-9b-chat-hf
```

GLM remains only as historical V2 evidence.

Do not reuse the key `glm` for Mistral.

New reader keys:

```text
("qwen", "mistral", "internlm")
```

New LORO rotations:

```text
qwen + mistral       -> hold internlm
qwen + internlm      -> hold mistral
mistral + internlm   -> hold qwen
```

No other LORO semantics change.

---

# 3. Why Mistral-7B-Instruct-v0.3

Official model:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

Required form:

```text
full BF16 / non-quantized official checkpoint
```

Do not use:

```text
4-bit
8-bit
GGUF
AWQ
GPTQ
community-converted checkpoint
```

The official checkpoint uses `MistralForCausalLM`, BF16 and a vocabulary of
32768, with approximately 14.5 GB of safetensor weights. This materially
reduces both model-weight memory and the full-vocabulary scoring peak compared
with the blocked GLM reader.

---

# 4. Artifact-history rule

Existing:

```text
results/cr_tser_v2/
```

is immutable historical evidence.

It contains:

```text
frozen V2 manifests
completed Qwen/InternLM utility labels
partial GLM utility labels
GLM OOM evidence
```

Do not:

```text
delete GLM rows
rewrite old manifests
change old hashes.json
rename glm to mistral inside old files
overwrite old label caches
```

New work must use:

```text
results/cr_tser_v2r1/
```

---

# 5. Reuse policy for Qwen / InternLM labels

The completed Qwen and InternLM utility labels are scientifically reusable
because their scores depend only on:

```text
frozen source
frozen event/cutoff/intervention
frozen prompt
their own reader identity
```

and not on which third reader participates in LORO.

However reuse is permitted only after strict mechanical verification.

The v2r1 stage must later prove that:

```text
source fingerprint              identical
event_split                     identical
snapshot_manifest               identical
intervention_manifest           identical
Qwen reader identity            identical
InternLM reader identity        identical
prompt/context fingerprints     identical
row cache keys                  identical
```

No GLM row may migrate.

The migration must be implemented as a fail-closed, tested operation.

Do not perform the formal cache migration during the current R1-M0 round.

---

# 6. Permanent execution environment

## LOCAL

```text
env = pytorch
role = lightweight development / tests only
```

Use for:

```text
Git
unit tests
code verifier
compileall
static/synthetic tests
```

Do not load 7B/8B models locally.

## SERVER

```text
env = DGPA
root = /data/jyz/next/llm/
role = model deployment, GPU preflight and formal experiments
```

Reader-loading processes continue to use the approved compatibility overlay
unless the new Mistral preflight proves a separate requirement:

```text
/data/jyz/next/llm/.cr_tser_v2p0/tf453

transformers = 4.53.3
tokenizers   = 0.21.4
```

Do not modify DGPA site-packages.

---

# 7. Server connection contract

All server command execution must use the local wrapper:

```text
next/.codex_tmp_ssh_run.cmd
```

All file transfer must use:

```text
next/.codex_tmp_scp_run.cmd
```

If the server/tunnel explicitly refuses the connection:

```text
STOP
preserve exact error
notify user
wait for tunnel restoration
retry only after user confirmation
```

Do not change host/port/user/connection mechanism.

A refused connection is an infrastructure pause, not a research failure.

---

# 8. Existing model resources

Do not redownload existing models:

```text
Qwen3-8B
/data/jyz/next/llm/model/qwen3-8b

historical Qwen3-8B:
/data/jyz/next/model/qwen3-8b

InternLM3-8B-Instruct
/data/jyz/next/llm/model/internlm3-8b-instruct

GLM-4-9B-Chat-HF
/data/jyz/next/llm/model/glm-4-9b-chat-hf
(historical only after this amendment)
```

The server also historically contains Qwen3-0.6B; it is not a CR-TSER reader.

New model to deploy:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

Target:

```text
/data/jyz/next/llm/model/mistral-7b-instruct-v0.3
```

Download on SERVER first.

If server download fails because of network/access restrictions:

```text
LOCAL/pytorch
-> download exact official model to NON-C drive
-> transfer with next/.codex_tmp_scp_run.cmd
-> /data/jyz/next/llm/model/mistral-7b-instruct-v0.3
-> server-side verification
```

Never download model weights to local C:.

Record:

```text
official model id
resolved upstream revision/commit if available
all weight shard hashes
tokenizer hash
config hash
chat-template hash
final model_weight_hash
reader_identity_hash
```

---

# 9. R1-M0 implementation boundary

Current round:

```text
V2-R1-M0
Reader Protocol Migration + Mistral Preflight
```

Allowed code changes:

```text
reader configuration
reader registry
PilotPaths / environment wiring
Mistral reader adapter
LORO reader names
protocol/version namespace
verifier
reader-specific tests
cache-migration utility + synthetic tests
documentation
```

Forbidden changes:

```text
sequence-scoring mathematics
candidate A/B definition
prompt semantics
SRC
interventions
utility equation
BiTTE
predictor architectures
training losses
selection algorithm
P1-P4 thresholds
dataset split
cutoffs
```

---

# 10. Required configuration migration

Introduce an explicit reader-amendment identity, e.g.:

```text
DATASET_PROTOCOL_VERSION = "v2"
READER_PROTOCOL_VERSION  = "r1"
PROTOCOL_VERSION         = "v2r1"
```

Formal output root:

```text
results/cr_tser_v2r1
```

Freeze:

```text
READER_KEYS = ("qwen", "mistral", "internlm")
```

and the new model IDs / LORO rotations.

Remove GLM from all **v2r1 execution loops**, while retaining the ability to
read historical V2 evidence.

Add:

```text
mistral_model
CRTSER_MISTRAL_MODEL
```

to resolved paths.

Do not alias `CRTSER_GLM_MODEL` to Mistral.

---

# 11. Mistral reader implementation

Add a dedicated reader implementation such as:

```text
project/cr_tser/readers/mistral_reader.py
```

It must use the same shared:

```text
SYSTEM_PROMPT
USER_TEMPLATE
build_messages()
candidate_logprobs_hf()
ab_scores()
reader identity hashing
```

as the other readers.

Required formal properties:

```text
AutoTokenizer
AutoModelForCausalLM
trust_remote_code = False
dtype = bfloat16
device = cuda
thinking = False
teacher-forced scoring only
```

Do not introduce model-specific free-form generation.

---

# 12. Cache migration utility

Implement a dedicated fail-closed migration utility for the *next* round.

Purpose:

```text
V2 historical labels
-> retain only qwen/internlm
-> verify every retained row
-> write to empty v2r1 cache
```

Rules:

```text
source must match
split/manifests must match
reader identity must match
prompt/context hashes must remain unchanged
duplicates forbidden
GLM rows forbidden
target cache must be empty before first migration
```

After migration, running the ordinary label generator for Qwen and InternLM
against the v2r1 cache must result in:

```text
0 new rows
all expected rows reused
0 CacheIdentityMismatch
```

Do not execute this formal migration during R1-M0.

Only implement + synthetic-test it.

---

# 13. LOCAL/pytorch verification

Run:

```bash
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2r1
python -m compileall project/cr_tser scripts
```

Tests must cover at least:

```text
reader keys exactly qwen/mistral/internlm
model IDs exact
LORO rotations exact
GLM absent from v2r1 execution loops
Mistral path/env resolution
Mistral reader uses shared teacher-forced scorer
v2 historical namespace immutable
v2r1 namespace separate
Qwen/InternLM migration accepts valid synthetic cache
migration rejects GLM rows
migration rejects identity/source/manifest drift
migration rejects duplicate keys
```

---

# 14. SERVER/DGPA Mistral preflight

After code passes locally and is pushed/synced:

1. connect through the required SSH wrapper;
2. activate DGPA;
3. activate the 4.53.3 compatibility overlay;
4. deploy Mistral if absent;
5. verify checkpoint identity;
6. load only Mistral;
7. run the same frozen A/B teacher-forced sanity protocol.

Required:

```text
model loaded = true
dtype = bfloat16
device = cuda
boundaries_ok = true
identical_predictions = true
identity_rate = 1.0
finite logprobs
no NaN/Inf
no empty continuation
```

Use at least:

```text
20 sanity prompts
2 repeated scoring passes
```

Also record:

```text
peak allocated GPU memory
peak reserved GPU memory
free memory before load
```

This is to confirm that the replacement solves the resource blocker.

Do not generate formal utility labels in this round.

---

# 15. Reader-set provenance report

Generate a concise report under:

```text
results/cr_tser_v2r1/reader_amendment/
```

containing:

```text
old reader set
new reader set
reason for amendment
timing of amendment (before predictor/P1-P4)
Mistral official model identity
server path
weight/tokenizer/chat-template hashes
A/B sanity result
GPU memory evidence
local/server environments
confirmation old v2 artifacts were untouched
```

---

# 16. R1-M0 stopping rule

After:

```text
code migration PASS
LOCAL tests PASS
Mistral server deployment PASS
Mistral load PASS
Mistral A/B sanity PASS
verifier PASS
```

commit/push and STOP.

Do not:

```text
build new formal v2r1 manifests
migrate formal Qwen/InternLM caches
generate Mistral utility labels
train predictors
run Stage A/B
run P1-P4
```

Those operations require approval of the reader-amendment implementation.

If Mistral itself cannot satisfy the BF16 server preflight:

```text
STOP
preserve evidence
do not choose a fourth model autonomously
return for research review
```

---

# 17. Final execution report

Report only:

```text
commit SHA
LOCAL/pytorch test status
SERVER/DGPA workspace
connection wrapper used
Mistral acquisition mode
Mistral server path
upstream revision if available
weight/tokenizer/chat-template hashes
Mistral load result
A/B sanity result
GPU peak memory
v2r1 code verifier result
confirmation results/cr_tser_v2 was untouched
confirmation no formal v2r1 labels / training / P1-P4 ran
```
