# BCR-Utility — Phase M0 Report

**Protocol:** `bcr_v1` · **Baseline commit:** `518a821d8d839fb7857be99acb91fde71b80c846`
**Scope:** protocol migration and historical bootstrap only (plan §11, §30).
**Status:** M0 completion gate satisfied — commit/push, then STOP. M1 was not started.

```text
CR_TSER_V2R1_FEASIBILITY = NO_GO   (historical, unchanged)
BCR_UTILITY_V1 = M0 COMPLETE, NOT YET EVALUATED
```

---

## 1. What M0 did

Created the new research line without spending any new LLM inference:

1. `project/bcr_utility/` package + `scripts/bcr_*.py` + `results/bcr_utility_v1/`;
2. frozen `bcr_v1` configuration (`bootstrap/protocol.json`);
3. historical **read-only** importer with fail-closed SHA256 pins
   (`bootstrap/historical_identity.json`);
4. `I1_atomic` bootstrap index, `I0`/`I2`–`I5` counted and excluded
   (`bootstrap/atomic_index.json`);
5. reader-utility geometry diagnostics (`bootstrap/geometry_diagnostic.json`);
6. 72-item frozen behavioral probe manifest (`bootstrap/probe_manifest.json`)
   from a data-side availability scan (`bootstrap/probe_availability.json`);
7. BCR verifier + 117 synthetic unit tests.

## 2. Execution environments

| | environment | role in M0 |
|---|---|---|
| LOCAL | conda `pytorch` (Python 3.9.18, torch 2.3.1+cu121) | implementation, unit tests, compileall, probe-manifest freeze, verifier |
| SERVER | `DGPA` @ `/data/jyz/next/llm/cr_tser_ws`, Python 3.11.13, transformers 4.53.3 overlay | read-only historical bootstrap + atomic index + geometry + probe availability scan |

Server access used only `next/.codex_tmp_ssh_run.cmd` / `next/.codex_tmp_scp_run.cmd`.
Server material was read (frozen caches, raw datasets, MiniLM encoder, Qwen
tokenizer) and written to `results/bcr_utility_v1/`; **no reader was loaded and
no GPU inference ran**. Wall clock: stage A ≈ 30 s, stage B 109 s (Ma-Weibo) +
9.4 s (PHEME), stage C (verifier) ≈ 5 s.

## 3. Historical input audit (fail-closed)

| input | value | frozen | match |
|---|---|---|---|
| Ma-Weibo labels SHA256 | `6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3` | same | ✅ |
| Ma-Weibo labels rows / bytes | 9606 / 16015182 | same | ✅ |
| PHEME labels SHA256 | `773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15` | same | ✅ |
| PHEME labels rows / bytes | 8637 / 14457787 | same | ✅ |
| v2r1 manifests (10 files) | canonical LF digests re-hashed in the worktree | pinned | ✅ |
| split seed / sizes | 7319, 80/50/15/25 per dataset | pinned | ✅ |
| reader contract | qwen / mistral / internlm, no GLM | pinned | ✅ |

`results/cr_tser/`, `results/cr_tser_v2/`, `results/cr_tser_v2r1/` were not
modified: the verifier fails on any *tracked* change there, and the label-cache
digests were recomputed on SERVER during stage C.

Reused CR-TSER modules (read-only imports, digest-pinned in
`historical_identity.json`): `config/pilot_config.py`, `data/snapshot_bridge.py`,
`intervention/evidence_units.py`, `intervention/semantic_reference.py`,
`evaluation/bootstrap.py`, `readers/sequence_scorer.py`,
`readers/base_reader.py`, `models/utility_heads.py`,
`scripts/cr_tser_common.py`.

## 4. Atomic evidence index

| dataset | I1_atomic keys | expected | I1 rows | indexed events | cutoffs |
|---|---|---|---|---|---|
| Ma-Weibo | **2549** | 2549 | 7647 | 90 | 15 / 60 / 360 |
| PHEME | **2107** | 2107 | 6321 | 90 | 15 / 60 / 360 |

Every key carries all three readers; rows outside `I1_atomic` are counted and
excluded from BCR supervision. Excluded rows (all readers pooled):
Ma-Weibo `I0_base` 774, `I2` 303, `I3` 300, `I4` 303, `I5` 279; PHEME
`I0_base` 783, `I2` 390, `I3` 381, `I4` 393, `I5` 369. Each row's cached `sign`
is re-derived with the frozen tri-class rule and must agree, otherwise the
import fails.

## 5. Frozen 72-item probe manifest

| dataset | items | 15m | 1h | 6h | label balance | candidate pool (15/60/360) |
|---|---|---|---|---|---|---|
| Ma-Weibo | 36 | 12 | 12 | 12 | 18 / 18 | 75 / 78 / 80 |
| PHEME | 36 | 12 | 12 | 12 | 18 / 18 | 71 / 75 / 80 |

* source split: `foundation_train` only; the three cutoff groups are event-disjoint;
* every item has ≥ 1 valid visible evidence unit **and** ≥ 1 SRC-selected unit;
* gold labels were used for sampling balance only and are **not** stored per item;
* **overlap audit: `utility_train` 0, `utility_dev` 0, `utility_eval` 0** for
  both datasets;
* selection seed 7319; the manifest is byte-reproducible from
  `probe_availability.json`.

Contexts P0–P3 are implemented and frozen (selection reader-independent, no
utility outcome consulted) but **were not executed** in M0.

## 6. Reader-utility geometry (diagnostic only — not a gate)

Intervals are event-level bootstraps (10000 iterations, seed 7319).

| dataset | active rate (qwen / mistral / internlm) | jointly active pairs | sign-agreement pairs | interaction share [95% CI] |
|---|---|---|---|---|
| Ma-Weibo | 437 / 444 / 448 of 2549 | 81 / 123 / 79 | 0.494 / 0.585 / 0.696 | **0.643** [0.617, 0.667] |
| PHEME | 405 / 292 / 600 of 2107 | 99 / 138 / 70 | 0.525 / 0.688 / 0.657 | **0.650** [0.620, 0.673] |

Two-way decomposition of the keys × readers utility matrix:

* reader main effect share ≈ 0.002;
* evidence main effect share ≈ 0.35;
* **reader × evidence interaction share ≈ 0.64**.

Ordinal magnitude geometry is weak (Spearman on |u| between 0.015 and 0.272),
while signed direction is only partly shared. Applied to the synthetic suite
this diagnostic separates the two quantities explicitly: an identical
magnitude ordering with an opposite sign yields `spearman_abs = 1.0` and
`sign_agreement = 0.0`, which is exactly why the report keeps ordinal geometry
and conditional signed direction apart (plan §1.2).

**Sample-size caveat.** Pairwise signed and ordinal statistics are computed only
on the jointly active keys (79–138 keys out of 2107–2549), because that is the
frozen activity definition. The interaction-share decomposition uses all keys,
with a bootstrap over 90 events. Both are descriptive, so nothing beyond the
observed values should be inferred from them.

**What this does and does not mean.** The interaction share says that most of
the variance in these records is reader-specific rather than shared across
evidence items — a *motivation* for the BCR question, consistent with the
historical CR-TSER P1 finding. It is not evidence for or against BCR
predictability, it is not a GO/NO-GO, and it must not be read as "social
structure has no effect" or as transfer between readers.

## 7. Verification

| check | result |
|---|---|
| LOCAL `pytest project/bcr_utility/tests` | **117 passed** |
| LOCAL `pytest project/cr_tser/tests` (regression) | 219 passed |
| LOCAL `compileall project/bcr_utility scripts/bcr_*.py` | clean |
| BCR verifier — LOCAL | issues **0**, pending 1 (label caches are SERVER-side) |
| BCR verifier — SERVER | issues **0**, pending **0** (22 checks) |
| historical CR-TSER namespaces | unchanged (tracked-clean; caches re-hashed on SERVER) |

## 8. Explicitly NOT run

No reader inference (Qwen / Mistral / InternLM), no model download, no new
utility labels, no predictor training, no behavior-conditioned model training,
no reader-panel expansion, no Phi/Gemma deployment, no M1–M5 work, no probe
execution. The verifier fails if a `probe_responses/`, `utility_labels/`,
`models/` or similar namespace appears under `results/bcr_utility_v1/`.

## 9. Artifacts

| file | bytes | SHA256 |
|---|---|---|
| `bootstrap/protocol.json` | 4635 | `6b7f0b0876f9a6dc7293c4a11921f317cfcfd3fe85f03f3e2054907287b61fb0` |
| `bootstrap/historical_identity.json` | 18467 | `2758856ef16bfea62ae8a297ec6cf54dcb9595ad17664d2aca5c39ff47188bae` |
| `bootstrap/atomic_index.json` | 2878957 | `f09e0487200dbdf905c2483e68b53d99cc405e45fa6617b1c5c584ff16d07477` |
| `bootstrap/geometry_diagnostic.json` | 20598 | `fae46c8b36a5af0e58ddddc0decbd86f5b47970743aa57dee579d5d4e770c84d` |
| `bootstrap/probe_availability.json` | 69167 | `751a79d8b8379d911b83b56a0ebee7dffaefc69a2cded8099c1958aeb91953cb` |
| `bootstrap/probe_manifest.json` | 13517 | `48fb3ce44e27330667afedd1c6db1f0aa9a91d72c2cd04c956f7b414ac6c249c` |

`verifier/bcr_verify.json` (LOCAL) and `verifier/bcr_verify_server.json`
(SERVER) hold the full check logs.

## 10. Stop rule

M0 is complete. Per plan §11 and §30 the line stops here: **no automatic entry
into M1**. Reader probing, feature extraction and any training require the next
approval.
