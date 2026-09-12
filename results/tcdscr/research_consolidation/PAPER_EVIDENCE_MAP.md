# TC-DSCR Paper Evidence Map

Every paper section, the artifacts it may cite, the numbers it may use, the commit they come from, and the split type. Split types are strictly separated: **TRAIN / VALIDATION / HELD_OUT_TEST / DIAGNOSTIC**. Validation and pilot numbers must never be placed in a held-out test table.

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

---

## Introduction

| Claim | Evidence | Split | Commit | Paper placement | Caveat |
|---|---|---|---|---|---|
| Early detection must be evaluated causally at real cutoffs | Protocol manifest + leakage verifier | VALIDATION | db3c269 / a0a0d31 | Intro + Setup | protocol requirement, not an effect |
| Social context is large and redundant | V3-A token statistics | VALIDATION | 656c611 | Intro motivation | proxy-level redundancy |

## Related Work

No TC-DSCR artifact is cited here. The related-work section may only position the work; it must not use any number from this repository as a comparison against prior methods (no such comparison exists).

## Method

| Component | Evidence | Split | Commit | Caveat |
|---|---|---|---|---|
| Causal snapshot construction (all events, real cutoff) | `results/tcdscr/dynamic_v2_protocol/protocol_manifest.json` | VALIDATION | a0a0d31 | — |
| Static Evidence Utility Selector | `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json` | VALIDATION (all-event) | a0a0d31 | dataset-dependent effect; the held-out comparison (Gap A) does not exist yet |
| MS-TSR minimal-sufficient compression | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` | VALIDATION | 656c611 | present as compression component only; ΔMacro-F1 = 0 is structural |
| Reader prompt (v3b-2) and parser | `results/tcdscr/dynamic_v3_reader/prompts/`, `run_manifest.json` | VALIDATION | 063f0df | prompt frozen once |

## Experimental Setup

| Item | Evidence | Split | Commit |
|---|---|---|---|
| Datasets, folds, cutoffs, budget | `results/tcdscr/dynamic_v3_reader/run_manifest.json`, `sampling_manifest.json` | VALIDATION | 063f0df |
| Encoder init comparison | `results/tcdscr/formal_e1/formal_e1_summary.json` | TRAIN (fold-readout on validation folds) | db3c269 |
| No-candidate coverage | `results/tcdscr/dynamic_v2_protocol/no_candidate_audit.json` | VALIDATION | a0a0d31 |
| Frozen model manifest (Qwen3-8B, decoding settings, no fine-tuning) | `results/tcdscr/dynamic_v3_reader/run_manifest.json` | VALIDATION | 063f0df |

## Main Results

| Claim | Numbers | Artifact | Split | Commit |
|---|---|---|---|---|
| Static utility beats simple baselines | all-event: PHEME 0.85669 / 0.85482 / 0.85382; Ma-Weibo 0.93387 / 0.91357 / 0.90893 | `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json` | VALIDATION | a0a0d31 |
| Dynamic V1 does not improve | Δ +0.0001037 / +0.0000120, CIs cross zero | `results/tcdscr/formal_e3_test/e3_test_summary.json`, `results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json` | CONDITIONAL_HELD_OUT (candidate-conditioned; see scope audit) | f7ec0ac / 1d97fd1 |
| MS-TSR compresses heavily with no proxy decision change | 57.9% / 82.7% token reduction, prediction change 0 | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` | VALIDATION | 656c611 |

Main-results tables must not contain any "our method improves" statement from held-out data: the only held-out run is candidate-conditioned (`CONDITIONAL_HELD_OUT`) and its verdict is negative. The historical candidate-conditioned E2 readiness values must not appear as absolute metrics; use the all-event values above. Artifact for historical readiness (audit only): `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json`.

## Ablation

| Ablation | Evidence | Split | Commit | Role |
|---|---|---|---|---|
| Point-wise temporal reranking (Dynamic V1) | `results/tcdscr/formal_e3_test/e3_test_summary.json` + `results/tcdscr/e3_failure_diagnosis/` | HELD_OUT_TEST / DIAGNOSTIC | f7ec0ac / 1d97fd1 | design evolution |
| Set-level fidelity refinement (MF-TSR) | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` | VALIDATION | a0a0d31 | negative finding |
| Compression severity as a risk predictor | `results/tcdscr/v3b_failure_diagnosis/compression_severity_analysis.json` | DIAGNOSTIC | c96ccbd | rule not triggered |

## Reader Transfer

| Claim | Numbers | Artifact | Split | Commit |
|---|---|---|---|---|
| Proxy sufficiency does not automatically transfer | PHEME +0.01974 (CI [-0.02192, 0.06144]); Ma-Weibo -0.00933 (CI [-0.04010, 0.02129]); overall PARTIAL | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` | **VALIDATION PILOT** | 063f0df |
| Grounding hygiene held | valid citation rate 1.0, unsupported 0.0 | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` | VALIDATION PILOT | 063f0df |
| Reader confidence falls while ECE worsens | Δconf -0.0587 / -0.0567; ECE 0.1777->0.1937 / 0.1900->0.2407 | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` | VALIDATION PILOT | 063f0df |

**Placement rule.** This section must always be labeled "validation-only reader pilot". It must not appear in the main test table, and no significance claim may be made for PHEME.

| Reader section | Status | Evidence |
|---|---|---|
| Current | VALIDATION PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Future closure | Gap B + Gap C (one combined fold-local held-out reader run) | `NEXT_EXPERIMENT_GAPS.md` |

Contribution D is therefore reported as `VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING`.

## Analysis

| Claim | Numbers | Artifact | Split | Commit |
|---|---|---|---|---|
| Bonus scale vs utility scale | ratio 0.0392 / 0.0190 | `results/tcdscr/e3_failure_diagnosis/score_scale_analysis.json` | DIAGNOSTIC | 1d97fd1 |
| Budget masking | ranking-changed-but-selection-same 0.1990 / 0.4865 | `results/tcdscr/e3_failure_diagnosis/budget_masking_analysis.json` | DIAGNOSTIC | 1d97fd1 |
| Teacher fidelity != sufficiency | gate FAIL both datasets | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` | VALIDATION | a0a0d31 |
| Proxy margin mismatch | mean retention 1.2986 with ΔMacro-F1 -0.00933 | `results/tcdscr/v3b_failure_diagnosis/proxy_reader_margin_transfer.json` | DIAGNOSTIC | c96ccbd |
| Dataset-specific context need | static units 6.05 / 9.83 vs MS units ~1.1 both | `results/tcdscr/v3b_failure_diagnosis/cross_dataset_comparison.json` | DIAGNOSTIC | c96ccbd |
| Static-utility -> citation alignment | AUC 0.4370 / 0.6158 | `results/tcdscr/v3b_failure_diagnosis/utility_reader_alignment.json` | DIAGNOSTIC | c96ccbd |
| Paired outcome groups | CC/CW/WC/WW 172/17/23/88 and 252/13/10/25 | `results/tcdscr/v3b_failure_diagnosis/paired_outcome_groups.json` | DIAGNOSTIC | c96ccbd |

## Limitations

| Limitation | Evidence | Split |
|---|---|---|
| Reader pilot is validation-only, one reader, one prompt | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` | VALIDATION PILOT |
| Reader pilot mixes outer-fold pools (contamination risk) | `results/tcdscr/v3b_failure_diagnosis/cross_fold_development_audit.json` | DIAGNOSTIC |
| Static utility effect is dataset-dependent (PHEME below readiness threshold) | `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json` | VALIDATION |
| Pre-fix structural statistics were degenerate | `PROTOCOL_CORRECTIONS.md` P03, `results/tcdscr/v3b_failure_diagnosis/structural_role_analysis.json` | DIAGNOSTIC |
| No all-event held-out evaluation exists (the held-out run is candidate-conditioned) | `results/tcdscr/research_consolidation/E3_NO_CANDIDATE_SCOPE_AUDIT.md` | DIAGNOSTIC |
| Static utility has no held-out baseline comparison | `NEXT_EXPERIMENT_GAPS.md` Gap A | — |
| Reader transfer has no fold-local held-out evaluation | `NEXT_EXPERIMENT_GAPS.md` Gap B | — |
| Reader comparison has no token-matched compression baselines | `NEXT_EXPERIMENT_GAPS.md` Gap C | — |

## Prohibited combinations

- V3-B reader numbers in a Main Results (test) table.
- MS-TSR ΔMacro-F1 = 0 presented as an empirical improvement.
- Any claim of statistical significance for the PHEME reader delta.
- Any citation that resolves only through `results/tcdscr/formal_e2/` (INVALID).
- Any pre-correction (candidate-conditioned) absolute value as canonical.
- E3 held-out absolute Macro-F1 presented as all-event canonical (it is `CONDITIONAL_HELD_OUT`).
- Validation all-event diagnostics presented as a substitute for the held-out result.
- The historical candidate-conditioned E2 readiness values presented as the paper's absolute metric.
