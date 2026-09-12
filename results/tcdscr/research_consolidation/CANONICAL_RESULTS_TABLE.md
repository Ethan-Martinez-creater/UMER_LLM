# TC-DSCR Canonical Results Table

Only numbers listed here may be cited as canonical results. Every row carries its **split**, **dataset**, **method**, **metric**, **value** and **status** (CANONICAL / CANONICAL-PILOT). Anything marked DEPRECATED in `DEPRECATED_RESULTS.md` must not be used as a canonical result.

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

## 1. E1 — Causal Encoder (split: TRAIN)

Validation-fold readout of the trained causal encoder; `n_runs = 60` (2 datasets x 5 folds x 3 seeds x 2 inits). Macro-F1 is the per-cutoff, per-seed mean.

| Dataset | Method | Metric | Cutoff | Value | Status | Artifact |
|---|---|---|---|---|---|---|
| PHEME | UMER init | macro_f1 | SOURCE_ONLY | 0.8527 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| PHEME | Random init | macro_f1 | SOURCE_ONLY | 0.8464 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| PHEME | UMER init | macro_f1 | 5m | 0.8597 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| PHEME | UMER init | macro_f1 | 6h | 0.8654 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| Ma-Weibo | UMER init | macro_f1 | SOURCE_ONLY | 0.7194 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| Ma-Weibo | Random init | macro_f1 | SOURCE_ONLY | 0.7066 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |
| Ma-Weibo | UMER init | macro_f1 | 6h | 0.9474 | CANONICAL | `results/tcdscr/formal_e1/formal_e1_summary.json` |

Key deltas (UMER - random macro_f1): PHEME +0.00632 / +0.01184 / +0.01477 at SOURCE_ONLY / 5m / 6h; Ma-Weibo +0.01285 / +0.00992 / +0.00586.

## 2. Corrected E2 — Static Utility Selector (split: VALIDATION)

`n_runs = 30` (2 datasets x 5 folds x 3 seeds). Primary metric = mean macro-F1 over the six primary cutoffs.

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | Static selector | macro_f1 (primary) | 0.85591 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |
| PHEME | Random top-k | macro_f1 (primary) | 0.85372 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |
| PHEME | Semantic top-k | macro_f1 (primary) | 0.85254 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |
| Ma-Weibo | Static selector | macro_f1 (primary) | 0.93503 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |
| Ma-Weibo | Random top-k | macro_f1 (primary) | 0.91408 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |
| Ma-Weibo | Semantic top-k | macro_f1 (primary) | 0.90908 | CANONICAL | `results/tcdscr/formal_e2_corrected/readiness/e2_summary.json` |

Readiness: Ma-Weibo pass (+0.02095), PHEME fail (+0.00218 against a 0.005 threshold). Overall PARTIAL, recommendation STOP_FOR_RESEARCH_REVIEW. Artifact: `results/tcdscr/formal_e2_corrected/E2_CORRECTED_READINESS_REPORT.md`.

## 3. All-event protocol correction (split: VALIDATION)

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | Static selector, all events | macro_f1 (primary) | 0.85669 | CANONICAL | `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json` |
| PHEME | No-candidate coverage | no_candidate_rate @5m | 0.35525 | CANONICAL | `results/tcdscr/dynamic_v2_protocol/no_candidate_audit.json` |
| Ma-Weibo | No-candidate coverage | no_candidate_rate @5m | 0.12674 | CANONICAL | `results/tcdscr/dynamic_v2_protocol/no_candidate_audit.json` |

## 4. Dynamic V1 — point-wise novelty + persistence (split: HELD_OUT_TEST)

`n_runs = 30` frozen fold configs; 94,104 (PHEME) and 79,173 (Ma-Weibo) event-cutoff rows. Paired bootstrap 10,000 iterations, seed 3090, multiplicity preserved.

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | Dynamic V1 | macro_f1 | 0.84653 | CANONICAL | `results/tcdscr/formal_e3_test/e3_test_summary.json` |
| PHEME | Static selector | macro_f1 | 0.84643 | CANONICAL | `results/tcdscr/formal_e3_test/e3_test_summary.json` |
| PHEME | Dynamic V1 - Static | delta macro_f1 | +0.0001037 | CANONICAL | `results/tcdscr/formal_e3_test/e3_test_summary.json` |
| PHEME | Dynamic V1 - Static | delta macro_f1 95% CI | [-0.0000969, +0.0003214] | CANONICAL | `results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json` |
| Ma-Weibo | Dynamic V1 | macro_f1 | 0.92679 | CANONICAL | `results/tcdscr/formal_e3_test/e3_test_summary.json` |
| Ma-Weibo | Static selector | macro_f1 | 0.92678 | CANONICAL | `results/tcdscr/formal_e3_test/e3_test_summary.json` |
| Ma-Weibo | Dynamic V1 - Static | delta macro_f1 95% CI | [-0.0001399, +0.0001614] | CANONICAL | `results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json` |

Verdict: WEAK_POSITIVE, recommendation REVIEW_DYNAMIC_DESIGN; both CIs cross zero, so Dynamic V1 is REJECTED as a method.

## 5. Dynamic V2 / MF-TSR (split: VALIDATION)

`n_runs = 10` (2 datasets x 5 folds), all-event protocol, budget 1024.

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | MF-TSR | macro_f1 | 0.85962 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |
| PHEME | Static selector | macro_f1 | 0.85669 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |
| PHEME | MF-TSR | evidence tokens | 178.69 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |
| Ma-Weibo | MF-TSR | macro_f1 | 0.93095 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |
| Ma-Weibo | Static selector | macro_f1 | 0.93381 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |
| Ma-Weibo | MF-TSR | evidence tokens | 516.33 | CANONICAL | `results/tcdscr/dynamic_v2/mf_tsr_summary.json` |

Gate FAIL on both datasets; recommendation REDESIGN_OR_REMOVE_DYNAMIC.

## 6. Dynamic V3-A / MS-TSR (split: VALIDATION)

`n_runs = 30`, 319,680 event-cutoff rows, alpha = 0.8 in every fold, budget 1024.

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | MS-TSR | macro_f1 | 0.85669 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| PHEME | MS-TSR | mean token reduction | 0.57859 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| PHEME | MS-TSR | dual-view agreement rate | 0.94730 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| PHEME | MS-TSR | prediction change rate | 0.0 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| Ma-Weibo | MS-TSR | macro_f1 | 0.93381 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| Ma-Weibo | MS-TSR | mean token reduction | 0.82738 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| Ma-Weibo | MS-TSR | dual-view agreement rate | 0.97323 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |
| Ma-Weibo | MS-TSR | prediction change rate | 0.0 | CANONICAL | `results/tcdscr/dynamic_v3/ms_tsr_summary.json` |

Overall MS_TSR_PROXY_PASS, recommendation START_V3_B_READER_TRANSFER_PILOT. `delta macro_f1 = 0` is structural (condition C1 + static fallback), not an empirical gain.

## 7. Dynamic V3-B — Frozen Qwen3-8B reader pilot (split: **VALIDATION PILOT**)

300 paired event-cutoffs per dataset (600 pairs, 1,200 arm evaluations), sampling seed 3090, context source seed 2000, alpha 0.8, budget 1024, prompt v3b-2, greedy decoding.

| Dataset | Method | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|
| PHEME | MS context (Qwen3-8B) | macro_f1 | 0.64755 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| PHEME | Static context (Qwen3-8B) | macro_f1 | 0.62781 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| PHEME | MS - Static | delta macro_f1 | +0.01974 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| PHEME | MS - Static | delta macro_f1 95% CI | [-0.02192, +0.06144] | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| PHEME | MS context | social token reduction | 0.68700 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| PHEME | both arms | unsupported citation rate | 0.0 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | MS context (Qwen3-8B) | macro_f1 | 0.87319 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | Static context (Qwen3-8B) | macro_f1 | 0.88252 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | MS - Static | delta macro_f1 | -0.00933 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | MS - Static | delta macro_f1 95% CI | [-0.04010, +0.02129] | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | MS context | social token reduction | 0.85322 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |
| Ma-Weibo | both arms | unsupported citation rate | 0.0 | CANONICAL-PILOT | `results/tcdscr/dynamic_v3_reader/reader_transfer_summary.json` |

Labels: PHEME WEAK_TRANSFER, Ma-Weibo TRANSFER_FAIL; overall PARTIAL, recommendation STOP_FOR_RESEARCH_REVIEW.

**This table is a validation pilot. It must not be placed in a held-out test table.**

## 8. V3-B failure diagnosis (split: VALIDATION, DIAGNOSTIC)

| Dataset | Group | n | Metric | Value | Status | Artifact |
|---|---|---|---|---|---|---|
| PHEME | CC / CW / WC / WW | 172 / 17 / 23 / 88 | paired outcome counts | — | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |
| Ma-Weibo | CC / CW / WC / WW | 252 / 13 / 10 / 25 | paired outcome counts | — | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |
| PHEME | MS-TSR | mean reader evidence loss rate | 0.67300 | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |
| Ma-Weibo | MS-TSR | mean reader evidence loss rate | 0.84813 | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |
| PHEME | Static utility -> Qwen citation | AUC | 0.43702 | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |
| Ma-Weibo | Static utility -> Qwen citation | AUC | 0.61583 | CANONICAL-PILOT | `results/tcdscr/v3b_failure_diagnosis/diagnosis_summary.json` |

Root causes: PROXY_READER_MARGIN_MISMATCH + DATASET_SPECIFIC_CONTEXT_NEED. Recommendation: MS_TSR_COMPRESSION_ONLY. Finalization confirmed the structural-bug fix changed no root cause and no recommendation.
