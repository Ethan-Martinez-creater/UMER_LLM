# TC-DSCR Formal E2 — Static Selector

## Overall Status
PARTIAL

## Git
- base commit: `b8a3121` (E2 first submission (invalidated))
- E2 commit: (recorded in the follow-up commit after this submission)

## Encoder
- type: Random-init TC-DSCR Causal Social Encoder
- historical UMER checkpoint used: NO
- encoder frozen: YES
- checksum mismatches: 0 (recorded per run in run_manifest)

## Run Completeness
- expected runs: 30 (PHEME 15, Ma-Weibo 15)
- completed: 30 (per-run manifests present)
- failed: 0

## PHEME Validation
| method | mean primary Macro-F1 | delta vs best baseline |
|---|---:|---:|
| static | 0.8559 ± 0.0012 | +0.0022 |
| random | 0.8537 ± 0.0014 |  |
| semantic | 0.8525 ± 0.0019 |  |

Readiness: FAIL

## Ma-Weibo Validation
| method | mean primary Macro-F1 | delta vs best baseline |
|---|---:|---:|
| static | 0.9350 ± 0.0014 | +0.0210 |
| random | 0.9141 ± 0.0041 |  |
| semantic | 0.9091 ± 0.0050 |  |

Readiness: PASS

## Per-cutoff Validation Results
Full per-cutoff Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1 tables for both datasets are in `readiness/e2_tables.md` and per run in `validation_metrics.json`.

## Selector Diagnostics
- entropy / selected units / evidence tokens / score stats / semantic-static Jaccard / L_cls / L_fid / L_div means: per dataset x cutoff in `readiness/e2_summary.json#diagnostics`

## Proxy Sanity vs E1
- pheme: E1 full-encoder mean primary Macro-F1 = 0.8597; static proxy = 0.8559; PROXY_SANITY_WARNING = no
- maweibo: E1 full-encoder mean primary Macro-F1 = 0.9302; static proxy = 0.9350; PROXY_SANITY_WARNING = no

## Leakage Audit
- future leakage failures: 0 (selection restricted to the current snapshot; verified by `scripts/tcdscr_verify_e2.py`)
- fold mismatch: 0 (E1/E2 split parity exact match on all 30 runs)
- encoder checksum mismatch: 0

## Test Results
Not run: readiness was not PASS on both datasets (or E2-B was not executed), so the test split was never evaluated.

## Blocking Issues
1. readiness gate not met on: pheme

## Recommendation
STOP_FOR_RESEARCH_REVIEW
