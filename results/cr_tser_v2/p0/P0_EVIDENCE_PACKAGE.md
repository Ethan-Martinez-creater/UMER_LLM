# CR-TSER V2-P0 — Preflight Evidence Package

Repository commit carrying this evidence:
`808fd83e61f71b0935a5a54543859519e6d0473d` (code fix) plus the commit that
adds this package.
Frozen scientific baseline: `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`

## 1. Verdict

```text
P0_PASS
```

All P0 prerequisites hold, emitted by the frozen entrypoint
`python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity`
(exit 0) on SERVER/DGPA. No artifact was hand-edited.

Round history:

| round | reader prerequisite | verdict |
|---|---|---|
| V2-P0 preflight (local, `b5f91e1`) | weights absent locally | `P0_FAIL` |
| V2-P0 server completion (`3d76af2`) | Qwen ok, GLM ok, InternLM `LossKwargs` ImportError under transformers 4.57.6 | `P0_FAIL` |
| **V2-P0 compatibility closure (`808fd83`)** | **all three load under transformers 4.53.3** | **`P0_PASS`** |

## 2. Environment

```text
local_env   = pytorch   (lightweight only)
server_env  = DGPA
server_root = /data/jyz/next/llm/
workspace   = /data/jyz/next/llm/cr_tser_ws
```

Server environment actually used for the authoritative run:

```text
CONDA_DEFAULT_ENV = DGPA
CONDA_PREFIX      = /data/jyz/envs/DGPA
PYTHONPATH        = /data/jyz/next/llm/.cr_tser_v2p0/tf453
python 3.11.13 / torch 2.7.1+cu128 / transformers 4.53.3 / tokenizers 0.21.4
CUDA 12.8 available, NVIDIA GeForce RTX 4090 23.52 GiB
```

Compatibility closure: DGPA's own `site-packages` was left untouched
(pip-freeze snapshot in `dgpa_pip_freeze_before.txt`); transformers 4.53.3 is
injected from an isolated target directory because 4.57.6 removed
`LossKwargs`, which InternLM3's official remote code imports. Rollback is
`unset PYTHONPATH`. See `COMPATIBILITY_CLOSURE_REPORT.md` §2.

## 3. Data side — PASS

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

`MAWEIBO_READY`; 4664 parsed / 0 invalid; labels 2351×0, 2313×1;
source-text 1.0, timestamp 1.0, parent-resolution 0.9999918; 0 duplicate ids,
0 cycles, 0 multi-root.

Viability (V2 strict Reply–Parent rule): 15m 4296 / 1h 4486 / 6h **4591** →
`total_viable_events = 4591 >= 170`.

Snapshot integrity (8 viable events × 3 cutoffs): source present, 0 future
leakage, `cap_hit=false` (`MAX_NODES_CAP=None`), parent visible only when
present, `parent.ts <= child.ts`, ordering exactly
`(timestamp, original_order)`.

PHEME smoke: `OK`.

## 4. Reader acquisition and identity

| reader | official model id | acquisition mode | server path |
|---|---|---|---|
| qwen | `Qwen/Qwen3-8B` | `server_existing` | `/data/jyz/next/llm/model/qwen3-8b` (symlink → `/data/jyz/next/model/qwen3-8b`) |
| glm | `zai-org/glm-4-9b-chat-hf` | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/glm-4-9b-chat-hf` |
| internlm | `internlm/internlm3-8b-instruct` | `server_download` (hf-mirror) | `/data/jyz/next/llm/model/internlm3-8b-instruct` |

No local fallback transfer was used; no model weight was ever placed on `C:`.
Official integrity verified with LFS content sha256 against the mirror's
`x-linked-etag` and git blob sha1 against the HF API `blobId`
(`official_verify_*.txt`): **all three ALL_MATCH**.

## 5. P0-F — frozen reader load audit

| reader | weight_hash | tokenizer_hash | transformers | loaded |
|---|---|---|---|---|
| qwen | `345a676964219bec…` | `26a5805b76938647…` | 4.53.3 | **true** |
| glm | `50b8b4f31d1ce5d7…` | `263851b177e6d7a0…` | 4.53.3 | **true** |
| internlm | `3ac547bf153173e2…` | `c0ed5fe5c9f2d713…` | 4.53.3 | **true** |

Load proof: `reader_load_test_transformers-4.53.3.json` (`ALL_LOADED True`).

## 6. P0-G — teacher-forced A/B sanity

Frozen contract: `teacher_forced_logprob_sum`, `generated_text_used=false`,
`generated_confidence_used=false`; 20 prompts per reader, each scored twice.

| reader | boundaries_ok | identical_predictions | identity_rate | A/B distribution |
|---|---|---|---|---|
| qwen | **true** | **true** | **1.0** | A=20, B=0 |
| glm | **true** | **true** | **1.0** | A=0, B=20 |
| internlm | **true** | **true** | **1.0** | A=17, B=3 |

`ALL_SANITY_OK True` (`label_scoring_sanity.json`, plus the standalone
compatibility record `label_scoring_sanity_transformers-4.53.3.json`).

## 7. P0 criteria

| criterion (amendment §10 / plan §14) | result |
|---|---|
| Ma-Weibo source integrity PASS | ✅ `MAWEIBO_READY` |
| Ma-Weibo viable events >= 170 | ✅ 4591 |
| PHEME smoke = OK | ✅ |
| Qwen loaded | ✅ |
| GLM loaded | ✅ |
| InternLM loaded | ✅ (transformers 4.53.3) |
| all A/B boundaries OK | ✅ |
| all repeated scoring deterministic | ✅ `identity_rate = 1.0` |

## 8. Verification

```text
SERVER/DGPA  cr_tser_verify_pilot.py --mode code  --protocol v2 -> issues = 0, pending = 0
SERVER/DGPA  cr_tser_verify_pilot.py --mode pilot --protocol v2 -> issues = 0, pending = 1
LOCAL/pytorch cr_tser_verify_pilot.py --mode code --protocol v2 -> issues = 0
LOCAL/pytorch python -m pytest project/cr_tser/tests -q         -> 139 passed
LOCAL/pytorch python -m compileall project/cr_tser scripts      -> clean
```

The pilot `pending = 1` is the expected `pilot_artifacts` entry (no
manifests; P1–P4 not run). No downstream artifact was fabricated.

## 9. Historical namespace protection

`results/cr_tser/` (V1) is read-only. Verifier identity is now the
cross-platform **canonical LF digest**, so the same repository state passes on
both the CRLF Windows checkout and the LF Linux checkout while any real edit
still fails. Both hosts agree:

```text
code_verify.json   134b32e2f00765ab221ee710a8def67ede91b009b97da95beecfac4463e22f0e
pilot_verify.json  c81000a97ed82c323c1953049d83fa864f9eba2b83640c6211af810076a061af
verifier dir       aba9f5169d63b39c7e423d886e01ec5e40547f43aa0f46a5c937bf73df4d90c2
```

## 10. Evidence files

```text
results/cr_tser_v2/p0/
  p0_readiness.json                  (frozen entrypoint product)
  P0_READINESS.md                    (frozen entrypoint product)
  P0_EVIDENCE_PACKAGE.md             (this file)
  COMPATIBILITY_CLOSURE_REPORT.md    (this round's full report)
  SERVER_COMPLETION_REPORT.md        (previous round's full report)
  environment.json                   (SERVER/DGPA, resolved paths + closure)
  dgpa_pip_freeze_before.txt         (DGPA rollback snapshot)
  maweibo_audit.json
  maweibo_source_fingerprint.json
  snapshot_integrity.json
  pheme_smoke.json
  reader_audit.json
  label_scoring_sanity.json
  reader_load_test_transformers-4.53.3.json
  label_scoring_sanity_transformers-4.53.3.json
  official_verify_qwen3-8b.txt
  official_verify_qwen3-8b_nonLFS.txt
  official_verify_glm-4-9b-chat-hf.txt
  official_verify_internlm3-8b-instruct.txt
  server_verification_run.log
  v1_namespace_baseline.json
results/cr_tser_v2/verifier/
  code_verify.json, pilot_verify.json, v1_code_verify.json
```

## 11. Not run

formal manifest freeze, formal intervention generation, formal utility labels,
predictor / single-reader / shared-residual training, Stage-A freeze,
held-out reader evaluation, P1, P2, P3, P4, formal pilot report.

No threshold, reader, cutoff, split size, eligibility rule or method was
changed in this round.

## 12. Follow-up for the research owner

`P0_PASS` depends on the transformers 4.53.3 `PYTHONPATH` override. Every
later stage that loads the three readers must use it, or making 4.53.3 the
DGPA default — either way an environment decision, not a protocol change.
