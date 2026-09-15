# CR-TSER V2-P0 Server Completion — Execution Report

> **Historical record.** This report documents the server-completion round,
> which ended in `P0_FAIL` because InternLM3 could not load under transformers
> 4.57.6. The following **V2-P0 Compatibility Closure** round resolved that
> (transformers 4.53.3) and reached `P0_PASS` — see
> `COMPATIBILITY_CLOSURE_REPORT.md` and `P0_EVIDENCE_PACKAGE.md`.

Round: **V2-P0 Server Completion** (`docs/CR_TSER_V2_P0_SERVER_COMPLETION_PLAN_v2.md`)
Repository commit carrying this evidence: `3d76af20e04eb5e04a9ad152133478a01b45efba`
Frozen scientific baseline: `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`
Previous approved submission: `b5f91e162ed0d6d2431add410760c155c3f23c39`

Full report of the round that completed the frozen-reader preflight on the GPU
server. The companion machine-readable evidence lives beside this file in
`results/cr_tser_v2/p0/`.

## 1. Execution-environment contract

```text
LOCAL  env  = pytorch
SERVER env  = DGPA
server root = /data/jyz/next/llm/
```

| | LOCAL | SERVER |
|---|---|---|
| env name | `pytorch` | `DGPA` |
| interpreter | `E:\anaconda\envs\pytorch\python.exe` | `/data/jyz/envs/DGPA/bin/python` |
| Python | 3.9.18 | 3.11.13 |
| torch | 2.3.1+cu121 | 2.7.1+cu128 |
| transformers | 4.44.0 | 4.57.6 |
| CUDA | 12.1 | 12.8 |
| GPU | GTX 1050 Ti, 4.00 GiB | RTX 4090, 23.52 GiB |
| repository/worktree | `E:\...\UMER` | `/data/jyz/next/llm/cr_tser_ws` |

```text
local_env   = pytorch
server_env  = DGPA
server_root = /data/jyz/next/llm/

local_tasks = [
  git state check,
  pytest project/cr_tser/tests -q,
  cr_tser_verify_pilot.py --mode code --protocol v2,
  compileall project/cr_tser scripts,
  evidence packaging / review
]

server_tasks = [
  S0 environment and workspace freeze,
  S1 repository synchronization + frozen-code diff check,
  S2 reader acquisition + official checkpoint hash verification,
  S3 data-input resolution + Ma-Weibo composite fingerprint check,
  S4 P0-F frozen reader load audit,
  S5 P0-G teacher-forced A/B sanity,
  S6 authoritative V2-P0 entrypoint run,
  S7 code/pilot verifier runs
]
```

No reader was loaded and no A/B scoring was executed on the local machine.

## 2. LOCAL/pytorch checks and outcomes

```text
git rev-parse HEAD                       -> b5f91e1 (tracked tree clean)
python -m pytest project/cr_tser/tests -q -> 128 passed
cr_tser_verify_pilot.py --mode code --protocol v2 -> issues = 0
python -m compileall project/cr_tser scripts      -> clean
```

## 3. SERVER/DGPA checks and outcomes

| checkpoint | command / probe | outcome |
|---|---|---|
| S0 | `conda activate DGPA`; `which python`; `python -V`; torch/transformers/CUDA/GPU probe; `nvidia-smi` | `CONDA_DEFAULT_ENV=DGPA`, `CONDA_PREFIX=/data/jyz/envs/DGPA`, Python 3.11.13, torch 2.7.1+cu128, transformers 4.57.6, CUDA 12.8, `cuda_available=True`, RTX 4090 23.52 GiB |
| S1 | `git clone` + `git checkout b5f91e1`; `git diff --name-only fa8ddbb..b5f91e1` | workspace `/data/jyz/next/llm/cr_tser_ws`, HEAD `b5f91e1`; diff = docs + `results/cr_tser_v2/p0/*` only → `project/cr_tser` and `scripts` unchanged |
| S2 | search, download, official hash verification | all three readers resolved and **ALL_MATCH** (see §4) |
| S3 | composite source fingerprint | `b982076d…` reproduced exactly → no source drift |
| S4 | P0-F reader load audit | qwen loaded, glm loaded, **internlm load failed** |
| S5 | P0-G A/B sanity | qwen and glm pass; internlm not evaluable |
| S6 | `python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity` | **P0_FAIL** |
| S7 | `verify_pilot.py --mode code\|pilot --protocol v2` | code `issues=3`, pilot `issues=0, pending=1` |

Endpoint reachability from the server: `huggingface.co` **timeout (rc=124)**,
`hf-mirror.com` **HTTP 200**, `modelscope.cn` **HTTP 302**. All checkpoints
were therefore downloaded directly on the server.

## 4. Reader acquisition mode and server paths

| reader | acquisition mode | absolute server path |
|---|---|---|
| qwen | `server_existing` | `/data/jyz/next/llm/model/qwen3-8b` (symlink → `/data/jyz/next/model/qwen3-8b`) |
| glm | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/glm-4-9b-chat-hf` |
| internlm | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/internlm3-8b-instruct` |

The existing authoritative Qwen copy was referenced at its audited location
and exposed under the llm root by a symlink rather than copied, so no source
of record was duplicated or relocated.

**Local fallback transfer: not used.** The server could download every
checkpoint itself, so there is no non-C-drive local temporary path and no
model weight was ever placed on `C:`.

### Official integrity verification

Method: LFS files are compared as content sha256 against the mirror's
`x-linked-etag`; non-LFS files as git blob sha1 against the HF API `blobId`.
Full records: `official_verify_*.txt` in this directory.

| reader | weight shards (sha256, official = local) | tokenizer |
|---|---|---|
| Qwen3-8B | `31d6a825…`, `5991236c…`, `c5185c47…`, `b5ee7de7…`, `20c2d636…` | `tokenizer.json` `aeb13307…`; all non-LFS files `ALL_NON_LFS_MATCH True` |
| GLM-4-9B-Chat-HF | `36b42739…`, `029329f3…`, `b5e6131e…`, `02e9f256…` | `tokenizer.json` `8a7269d6…` |
| InternLM3-8B-Instruct | `9a18eb70…`, `e324110d…` | `tokenizer.model` `bcacff32…` |

All three readers returned `ALL_MATCH True`. GLM initially landed without
`model-00003-of-00004.safetensors` (dropped by the mirror during the first
snapshot fetch); the exact official shard was re-fetched and verified
(`b5e6131e…`), completing the 4-shard set.

## 5. P0-F — frozen reader load audit

| reader | expected model id | weight_hash | tokenizer_hash | chat_template_hash (after load) | loaded |
|---|---|---|---|---|---|
| qwen | `Qwen/Qwen3-8B` | `345a676964219bec…` | `26a5805b76938647…` | `a55ee1b1660128b7…` | **true** |
| glm | `zai-org/glm-4-9b-chat-hf` | `50b8b4f31d1ce5d7…` | `263851b177e6d7a0…` | `27288f957f8364c9…` | **true** |
| internlm | `internlm/internlm3-8b-instruct` | `3ac547bf153173e2…` | `c0ed5fe5c9f2d713…` | — | **false** |

```text
internlm load_error:
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
  (/data/jyz/envs/DGPA/lib/python3.11/site-packages/transformers/utils/__init__.py)
```

Root cause (diagnosed on the server): transformers 4.57.6 has **no**
`LossKwargs` in `transformers.utils` or `transformers.utils.generic`, and
ships no built-in `internlm3` model type. The official
`modeling_internlm3.py` line 40 imports `LossKwargs`, so the frozen reader's
`trust_remote_code=True` path cannot execute. The `reader_identity_hash`
differs before/after load for qwen and glm purely because
`chat_template_hash` is only populated once a tokenizer is attached.

## 6. P0-G — teacher-forced A/B sanity

Frozen contract: `scoring_mode = teacher_forced_logprob_sum`,
`generated_text_used = false`, `generated_confidence_used = false`;
20 prompts per reader, each scored twice.

| reader | boundaries_ok | identical_predictions | identity_rate | label distribution |
|---|---|---|---|---|
| qwen | **true** | **true** | **1.0** | A=20, B=0 |
| glm | **true** | **true** | **1.0** | A=0, B=20 |
| internlm | not evaluable | not evaluable | — | — |

Explicit A/B tokenization was recorded per reader (prompt token count,
candidate token ids, continuation boundary check). Example (qwen): prompt 70
tokens, candidate A → `[32]`, candidate B → `[33]`, `joined_len` 71 for both,
`all_boundaries_ok = true`. Example (glm): prompt 60 tokens, candidates
`[32]` / `[33]`, `all_boundaries_ok = true`. No `generate()`, free-form
generation, generated confidence, self-reported probability or answer parsing
was used anywhere.

## 7. Data side — PASS (re-verified on the server)

```text
CRTSER_MAWEIBO_RAW    = /data/jyz/next/llm/data/maweibo_raw
CRTSER_MAWEIBO_LABELS = /data/jyz/next/llm/data/maweibo_labels.txt
CRTSER_PHEME_RAW      = /data/jyz/next/llm/data/pheme_raw
```

| fingerprint | value |
|---|---|
| raw JSON files / bytes | 4664 / 3,995,877,128 |
| raw sha256 | `c6afd1c50a6cde8c27137d367d8e420b4a3afc43e017e3b19846e8b001ca1a27` |
| label bytes / sha256 | 64,463,295 / `032f10e2175fa461203bdba77ef2a492e0ebde1cdd24b9bcd2100e307bc6b8e4` |
| **combined_source_sha256** | `b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d` |

Identical to the reviewed evidence — no source drift.

Integrity `MAWEIBO_READY`: 4664 parsed / 0 invalid; labels 2351×0, 2313×1;
source-text 1.0, timestamp 1.0, parent-resolution 0.9999918; 0 duplicate ids,
0 cycles, 0 multi-root.

Viability (V2 strict Reply–Parent rule): 15m **4296** / 1h **4486** /
6h **4591** → `total_viable_events = 4591 >= 170`.

Snapshot integrity (8 viable events × 3 cutoffs, re-sampled on the server):
source present, 0 future leakage, `cap_hit=false` (`MAX_NODES_CAP=None`),
parent visible only when present, `parent.ts <= child.ts`, ordering exactly
`(timestamp, original_order)`, `unreachable_count=0`. Sample `10031994215`
(941 nodes at 6h) exceeds the retired 1021 cap without truncation.

PHEME smoke: `OK`.

## 8. Authoritative P0 verdict

```text
P0_FAIL
```

**Single blocker**: `internlm/internlm3-8b-instruct` cannot be loaded under
the mandated DGPA environment (transformers 4.57.6 removed `LossKwargs` and
provides no built-in internlm3, while the official remote code requires it).

This is an official-checkpoint ↔ mandated-environment incompatibility — not a
data, protocol or method finding.

P0 PASS criteria vs actual:

| criterion (amendment §10 / plan §14) | result |
|---|---|
| Ma-Weibo source integrity PASS | ✅ `MAWEIBO_READY` |
| Ma-Weibo viable events >= 170 | ✅ 4591 |
| PHEME smoke = OK | ✅ |
| Qwen loaded | ✅ |
| GLM loaded | ✅ |
| InternLM loaded | ❌ `LossKwargs` ImportError |
| all A/B boundaries OK | ⚠️ qwen/glm true; internlm not evaluable |
| all repeated scoring deterministic | ⚠️ qwen/glm `identity_rate=1.0`; internlm not evaluable |

The verdict was emitted by the frozen entrypoint; no artifact was
hand-edited. No model was substituted, no threshold changed, and no patch of
the official remote code was applied.

## 9. code / pilot verifier results

```text
verify_pilot.py --mode code  --protocol v2 -> issues = 3, pending = 0
verify_pilot.py --mode pilot --protocol v2 -> issues = 0, pending = 1
```

The pilot `pending = 1` is the expected `pilot_artifacts` entry (no manifests;
P1–P4 not run). No downstream artifact was fabricated.

The three code-verifier failures are **line-ending artifacts of the local
Windows checkout, not content changes**:

```text
[fail] v1_historical_immutable:code_verify.json  sha256=134b32e2… expected=edc76e50…
[fail] v1_historical_immutable:pilot_verify.json  sha256=c81000a9… expected=a45dc514…
[fail] v1_verifier_dir_immutable                 sha256=aba9f516… expected=efae6c5a…
```

| file | local worktree sha256 / bytes | server worktree sha256 / bytes | **stored git blob (both hosts)** |
|---|---|---|---|
| `results/cr_tser/verifier/code_verify.json` | `edc76e50…` / 8604 | `134b32e2…` / 8298 | `0d0e601e292481020b979ff8f5531c1920b59812` |
| `results/cr_tser/verifier/pilot_verify.json` | `a45dc514…` / 298 | `c81000a9…` / 287 | `587d163f5c973d60c4875c7388130308fa57cb5f` |

The local checkout has `core.autocrlf=true` (8604 vs 8298 bytes = one `\r`
per line); the Linux checkout keeps LF. The frozen verifier expectations were
computed on the CRLF worktree, so they cannot match LF bytes. On both hosts
the V1 directory is byte-stable across the whole round:

```text
SERVER v1_pre  = b1b348b83c181aefd7ec411844b00badc22f14c8aafcdffcc989c64442fbca21
SERVER v1_post = b1b348b83c181aefd7ec411844b00badc22f14c8aafcdffcc989c64442fbca21
unchanged = true
```

## 10. Files changed in this round

17 files, +1318 / −369:

```text
docs/CR_TSER_IMPLEMENTATION_NOTES.md                     (new section 19)
docs/CR_TSER_V2_P0_SERVER_COMPLETION_PLAN_v2.md          (round plan)
results/cr_tser_v2/p0/P0_EVIDENCE_PACKAGE.md
results/cr_tser_v2/p0/P0_READINESS.md
results/cr_tser_v2/p0/p0_readiness.json
results/cr_tser_v2/p0/environment.json
results/cr_tser_v2/p0/maweibo_audit.json
results/cr_tser_v2/p0/maweibo_source_fingerprint.json
results/cr_tser_v2/p0/snapshot_integrity.json
results/cr_tser_v2/p0/pheme_smoke.json
results/cr_tser_v2/p0/reader_audit.json
results/cr_tser_v2/p0/label_scoring_sanity.json
results/cr_tser_v2/p0/v1_namespace_baseline.json
results/cr_tser_v2/p0/official_verify_qwen3-8b.txt
results/cr_tser_v2/p0/official_verify_qwen3-8b_nonLFS.txt
results/cr_tser_v2/p0/official_verify_glm-4-9b-chat-hf.txt
results/cr_tser_v2/p0/official_verify_internlm3-8b-instruct.txt
results/cr_tser_v2/verifier/code_verify.json
results/cr_tser_v2/verifier/pilot_verify.json
```

`project/cr_tser` and `scripts` are byte-identical to the frozen baseline —
**no scientific code was modified**.

## 11. Confirmation of what was NOT run

```text
formal Pilot                   NOT RUN
formal manifest freeze         NOT RUN
formal intervention generation NOT RUN
formal utility labels          NOT RUN
predictor training             NOT RUN
single-reader training         NOT RUN
shared/residual training       NOT RUN
Stage A                        NOT RUN
Stage B                        NOT RUN
held-out reader evaluation     NOT RUN
P1                             NOT RUN
P2                             NOT RUN
P3                             NOT RUN
P4                             NOT RUN
formal pilot report            NOT RUN
```

No threshold, reader, cutoff, split size, eligibility rule or method was
changed, and no fallback was designed in response to the failure. The
historical namespace `results/cr_tser/` (V1) stayed read-only throughout.

## 12. Allowed next step

The only admissible remedy is an environment/research decision — **not** a
code, model or protocol change:

1. provide a runtime for `internlm/internlm3-8b-instruct` that still exposes
   `LossKwargs` (the ≤ 4.5x transformers line) while still satisfying the DGPA
   contract, or
2. obtain an explicit research waiver to substitute the R3 reader.

Neither may be decided inside a P0 round. Until one of them is approved, no
V2 manifest, utility label, training or evaluation stage may start.
