# CR-TSER V2-P0 — Preflight Evidence Package (server completion)

Repository commit: `b5f91e162ed0d6d2431add410760c155c3f23c39`
Frozen scientific baseline: `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`
Round: **V2-P0 Server Completion** (`docs/CR_TSER_V2_P0_SERVER_COMPLETION_PLAN_v2.md`).
No scientific code was changed in this round.

## 1. Verdict

```text
P0_FAIL
```

**Single blocker**: `internlm/internlm3-8b-instruct` cannot be loaded under the
mandated DGPA environment. Its official remote implementation
(`modeling_internlm3.py`) does `from transformers.utils import LossKwargs`,
but transformers **4.57.6 removed `LossKwargs`** entirely (it is absent from
both `transformers.utils` and `transformers.utils.generic`), and this
transformers version ships no built-in `internlm3` model type:

```text
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
  (/data/jyz/envs/DGPA/lib/python3.11/site-packages/transformers/utils/__init__.py)
```

The frozen `InternLMReader` loads with `trust_remote_code=True`, so the
official `auto_map` (`configuration_internlm3` / `modeling_internlm3`) is
executed and fails. This is an official-checkpoint ↔ mandated-environment
incompatibility, not a data, protocol or method finding.

**Both other frozen readers passed completely** (load + identity + A/B
sanity): `Qwen/Qwen3-8B` and `zai-org/glm-4-9b-chat-hf`.

Per plan §14 the failure is recorded with the raw error preserved; no model
was substituted, no protocol was changed, no local patch of the official
remote code was applied.

## 2. Dual-environment execution contract

| | LOCAL | SERVER |
|---|---|---|
| env | `pytorch` | `DGPA` |
| interpreter | `E:\anaconda\envs\pytorch\python.exe` (3.9.18) | `/data/jyz/envs/DGPA/bin/python` (3.11.13) |
| torch | 2.3.1+cu121 | 2.7.1+cu128 |
| transformers | 4.44.0 | 4.57.6 |
| CUDA | 12.1 | 12.8 |
| GPU | GTX 1050 Ti, 4.00 GiB | RTX 4090, 23.52 GiB |
| root | `E:\...\UMER` | `/data/jyz/next/llm/cr_tser_ws` |

```text
local_env  = pytorch
server_env = DGPA
server_root = /data/jyz/next/llm/
```

**LOCAL/pytorch tasks executed** (lightweight only):

* git state check (`b5f91e1`), working tree clean for tracked files;
* `python -m pytest project/cr_tser/tests -q` → **128 passed**;
* `scripts/cr_tser_verify_pilot.py --mode code --protocol v2` → **issues = 0**;
* `python -m compileall project/cr_tser scripts` → **clean**;
* read-only evidence review and packaging.

No reader was loaded and no A/B scoring was run locally.

**SERVER/DGPA tasks executed**:

* S0 environment/workspace freeze (`DGPA`, CUDA true, RTX 4090);
* S1 repository synchronization to `b5f91e1` + frozen-code diff check;
* S2 acquisition, official-hash verification and resolution of the three
  frozen readers;
* S3 data-input resolution + Ma-Weibo composite fingerprint check;
* S4 frozen reader load audit (P0-F);
* S5 teacher-forced A/B sanity (P0-G);
* S6 authoritative `cr_tser_p0_audit.py --protocol v2 --readers --sanity`;
* S7 `verify_pilot.py --mode code|pilot --protocol v2`.

All actual reader loading, identity hashing and A/B scoring ran on
**SERVER/DGPA** only.

## 3. Repository synchronization (S1)

```text
server workspace : /data/jyz/next/llm/cr_tser_ws
HEAD             : b5f91e162ed0d6d2431add410760c155c3f23c39
```

`git diff --name-only fa8ddbb..b5f91e1` returned documentation and
`results/cr_tser_v2/p0/*` evidence files only — `project/cr_tser` and
`scripts` are unchanged from the frozen scientific baseline.

## 4. Reader acquisition and identity (S2)

| reader | official model id | acquisition mode | absolute server path | official hash check |
|---|---|---|---|---|
| qwen | `Qwen/Qwen3-8B` | `server_existing` | `/data/jyz/next/llm/model/qwen3-8b` (symlink → `/data/jyz/next/model/qwen3-8b`) | **ALL_MATCH** |
| glm | `zai-org/glm-4-9b-chat-hf` | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/glm-4-9b-chat-hf` | **ALL_MATCH** |
| internlm | `internlm/internlm3-8b-instruct` | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/internlm3-8b-instruct` | **ALL_MATCH** |

No local (non-C-drive) transfer was needed: `huggingface.co` is unreachable
from the server, but `hf-mirror.com` returns HTTP 200, so all checkpoints were
downloaded **directly on the server** into `/data/jyz/next/llm/`. No fallback
download to the local machine was used, so there is no local temporary path.

Official integrity verification (methods and full records in
`official_verify_*.txt`): LFS files are compared as content sha256 against the
mirror's `x-linked-etag`; non-LFS files as git blob sha1 against the API
`blobId`. Lowercase `local=` vs `official=` values are the actual digests.

| reader | weight shards (sha256, official = local) | tokenizer |
|---|---|---|
| Qwen3-8B | `31d6a825…`, `5991236c…`, `c5185c47…`, `b5ee7de7…`, `20c2d636…` | `tokenizer.json` `aeb13307…` (+ all git-blob files MATCH) |
| GLM-4-9B-Chat-HF | `36b42739…`, `029329f3…`, `b5e6131e…`, `02e9f256…` | `tokenizer.json` `8a7269d6…` |
| InternLM3-8B-Instruct | `9a18eb70…`, `e324110d…` | `tokenizer.model` `bcacff32…` |

GLM initially landed with `model-00003-of-00004.safetensors` missing
(hf-mirror dropped it during the first snapshot fetch); the exact shard was
re-fetched from the official repo and verified, so the checkpoint set is
complete: `{model-00001, 00002, 00003, 00004}-of-00004`.

## 5. P0-F — frozen reader load audit

From the authoritative `reader_audit.json`:

| reader | weight_hash | tokenizer_hash | chat_template_hash (after load) | loaded |
|---|---|---|---|---|
| qwen | `345a676964219bec…` | `26a5805b76938647…` | `a55ee1b1660128b7…` | **true** |
| glm | `50b8b4f31d1ce5d7…` | `263851b177e6d7a0…` | `27288f957f8364c9…` | **true** |
| internlm | `3ac547bf153173e2…` | `c0ed5fe5c9f2d713…` | — | **false** |

```text
internlm load_error:
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```

## 6. P0-G — teacher-forced A/B sanity

Frozen contract: `scoring_mode = teacher_forced_logprob_sum`,
`generated_text_used = false`, `generated_confidence_used = false`,
20 prompts per reader scored twice.

| reader | boundaries_ok | identical_predictions | identity_rate | label distribution |
|---|---|---|---|---|
| qwen | **true** | **true** | **1.0** | A=20, B=0 |
| glm | **true** | **true** | **1.0** | A=0, B=20 |
| internlm | not evaluable | not evaluable | — | — |

Explicit A/B tokenization was recorded per reader (prompt token count,
candidate ids, continuation boundary check). Example (qwen): prompt 70 tokens,
candidate A → `[32]`, candidate B → `[33]`, `joined_len` 71 for both,
`all_boundaries_ok = true`.

## 7. Data side — PASS (unchanged, re-verified on the server)

Source of record (audited locations, referenced not relocated):

```text
CRTSER_MAWEIBO_RAW    = /data/jyz/next/llm/data/maweibo_raw
CRTSER_MAWEIBO_LABELS = /data/jyz/next/llm/data/maweibo_labels.txt
CRTSER_PHEME_RAW      = /data/jyz/next/llm/data/pheme_raw
```

Composite fingerprint **matches the reviewed P0 evidence exactly**:

| field | value |
|---|---|
| raw JSON files / bytes | 4664 / 3,995,877,128 |
| raw sha256 | `c6afd1c50a6cde8c27137d367d8e420b4a3afc43e017e3b19846e8b001ca1a27` |
| label bytes / sha256 | 64,463,295 / `032f10e2175fa461203bdba77ef2a492e0ebde1cdd24b9bcd2100e307bc6b8e4` |
| **combined_source_sha256** | `b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d` |

No source drift.

Integrity: `MAWEIBO_READY`; 4664 parsed / 0 invalid; labels 2351×0, 2313×1;
source-text 1.0, timestamp 1.0, parent-resolution 0.9999918; 0 duplicate ids,
0 cycles, 0 multi-root.

Viability (V2 strict Reply–Parent rule): 15m **4296** / 1h **4486** /
6h **4591** → `total_viable_events = 4591 >= 170`.

Snapshot integrity (re-sampled on the server, 8 viable events × 3 cutoffs):
source present, 0 future leakage, `cap_hit=false` (`MAX_NODES_CAP=None`),
parent visible only when present, `parent.ts <= child.ts`, ordering exactly
`(timestamp, original_order)`. Sample `10031994215` (941 nodes at 6h) exceeds
the retired 1021 cap without truncation.

PHEME smoke: `OK`.

## 8. P0 criteria

| criterion (amendment §10 / plan §14) | result |
|---|---|
| Ma-Weibo source integrity PASS | ✅ `MAWEIBO_READY` |
| Ma-Weibo viable events >= 170 | ✅ 4591 |
| PHEME smoke = OK | ✅ |
| Qwen loaded | ✅ |
| GLM loaded | ✅ |
| InternLM loaded | ❌ `LossKwargs` ImportError (transformers 4.57.6) |
| all A/B boundaries OK | ⚠️ qwen/glm true; internlm not evaluable |
| all repeated scoring deterministic | ⚠️ qwen/glm `identical_predictions=true`, `identity_rate=1.0`; internlm not evaluable |

→ `P0_FAIL`, emitted by the frozen entrypoint
(`python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity`).
No artifact was hand-edited.

## 9. Verification (S7) and the V1 line-ending note

```text
verify_pilot.py --mode code  --protocol v2 → issues = 3, pending = 0
verify_pilot.py --mode pilot --protocol v2 → issues = 0, pending = 1
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

Evidence that the V1 content is untouched:

| file | local worktree sha256 / bytes | server worktree sha256 / bytes | **stored git blob** |
|---|---|---|---|
| `results/cr_tser/verifier/code_verify.json` | `edc76e50…` / 8604 | `134b32e2…` / 8298 | `0d0e601e292481020b979ff8f5531c1920b59812` (identical) |
| `results/cr_tser/verifier/pilot_verify.json` | `a45dc514…` / 298 | `c81000a9…` / 287 | `587d163f5c973d60c4875c7388130308fa57cb5f` (identical) |

The local checkout has `core.autocrlf=true` (8604 vs 8298 bytes = one `\r`
per line), while the Linux checkout keeps LF. The frozen verifier expectations
were computed on the CRLF worktree, so they cannot match LF bytes. On both
hosts the V1 directory is byte-stable across the whole round:

```text
SERVER v1_pre  = b1b348b83c181aefd7ec411844b00badc22f14c8aafcdffcc989c64442fbca21
SERVER v1_post = b1b348b83c181aefd7ec411844b00badc22f14c8aafcdffcc989c64442fbca21
unchanged = true
```

## 10. Historical namespace protection

`results/cr_tser/` (V1) is read-only and byte-identical before/after the whole
server run (hash above). All V2 output lives under `results/cr_tser_v2/`,
inside the server root `/data/jyz/next/llm/`.

## 11. Evidence files

```text
results/cr_tser_v2/p0/
  p0_readiness.json                  (frozen entrypoint product)
  P0_READINESS.md                    (frozen entrypoint product)
  P0_EVIDENCE_PACKAGE.md             (this report)
  SERVER_COMPLETION_REPORT.md        (full round execution report)
  environment.json                   (SERVER/DGPA, resolved paths)
  maweibo_audit.json
  maweibo_source_fingerprint.json
  snapshot_integrity.json
  pheme_smoke.json
  reader_audit.json
  label_scoring_sanity.json
  official_verify_qwen3-8b.txt
  official_verify_qwen3-8b_nonLFS.txt
  official_verify_glm-4-9b-chat-hf.txt
  official_verify_internlm3-8b-instruct.txt
  v1_namespace_baseline.json
results/cr_tser_v2/verifier/
  code_verify.json, pilot_verify.json, v1_code_verify.json
```

## 12. Not run

formal manifest freeze, formal intervention generation, formal utility labels,
predictor / single-reader / shared-residual training, Stage-A freeze,
held-out reader evaluation, P1, P2, P3, P4, formal pilot report.

No threshold, reader, cutoff, split size, eligibility rule or method was
changed; no fallback method was designed in response to this failure.

## 13. Allowed next step (requires research/environment decision)

The only admissible remedy is an environment decision — **not** a code,
model or protocol change:

1. obtain an InternLM3-Series-Chat-compatible runtime for
   `internlm/internlm3-8b-instruct` (a transformers release that still
   exposes `LossKwargs`, i.e. ≤ the 4.5x line, in an environment that still
   satisfies the DGPA contract), or
2. obtain an explicit research waiver to substitute the R3 reader.

Neither may be decided inside the P0 round. Until one of them is approved, no
V2 manifest, utility label, training or evaluation stage may start.
