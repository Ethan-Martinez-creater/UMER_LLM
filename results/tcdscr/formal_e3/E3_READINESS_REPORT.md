# TC-DSCR Formal E3 — Dynamic Evidence Memory

## Overall Status
PASS

## Git
- base commit: `851b509`
- E3 commit: `80dc1e51b71554fc102d3a8dffbc0695fbc270ce`

## Frozen Components
- E1 encoder: Random-init TC-DSCR Causal Social Encoder
- corrected E2 selector: formal_e2_corrected best_selector.pt
- proxy: corrected E2 frozen proxy
- trainable E3 params: NONE
- checksum match (encoder/selector/proxy): all 30 runs true

## Hyperparameter Grid
- lambda_n: 0.0, 0.25, 0.5, 1.0
- lambda_p: 0.0, 0.1, 0.25
- budgets: 512, 1024, 2048 (36 configurations)

## PHEME

### Fold-selected configurations
| fold | lambda_n | lambda_p | budget |
|---|---:|---:|---:|
| 0 | 0.5 | 0.1 | 512 |
| 1 | 1.0 | 0.1 | 512 |
| 2 | 0.0 | 0.25 | 512 |
| 3 | 0.25 | 0.1 | 2048 |
| 4 | 1.0 | 0.0 | 512 |

### Validation comparison (5 folds x 3 seeds)
| method | Macro-F1 | FlipRate | Evidence Tokens | Turnover |
|---|---:|---:|---:|---:|
| Static | 0.8567 | 0.0132 | 439.9 | 0.2478 |
| Dynamic | 0.8568 ± 0.0013 | 0.0144 | 439.9 | 0.2868 |

delta Macro-F1: +0.0002
delta FlipRate: -0.0012
delta tokens: +0.0
effect label: MODERATE
readiness: PASS

### Selection Saturation (selected/candidates)
| cutoff | saturation |
|---:|---:|
| 5m | 0.8781 |
| 15m | 0.7255 |
| 30m | 0.6341 |
| 60m | 0.5559 |
| 180m | 0.4880 |
| 360m | 0.4598 |

## Per-cutoff Results
Full per-cutoff Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1 and temporal metrics are in `readiness/e3_summary.json` and per fold in `temporal_metrics.json`.

## Novelty Diagnostics
per-cutoff mean/median/P25/P75 selected novelty in `readiness/e3_summary.json#datasets.<ds>.dynamics`.

## Persistence Diagnostics
- memory retention rate and persistent/new ratios: per fold in `evidence_dynamics.json`, dataset means in `readiness/e3_summary.json`.

## Memory Leakage Audit
- future memory leakage: 0 (verified by `scripts/tcdscr_verify_e3.py`)
- future candidate leakage: 0

## Verification
- issues: 0

## Test Split
NOT EVALUATED

## Ma-Weibo

### Fold-selected configurations
| fold | lambda_n | lambda_p | budget |
|---|---:|---:|---:|
| 0 | 0.25 | 0.25 | 1024 |
| 1 | 1.0 | 0.25 | 1024 |
| 2 | 1.0 | 0.0 | 1024 |
| 3 | 0.25 | 0.1 | 1024 |
| 4 | 0.0 | 0.1 | 1024 |

### Validation comparison (5 folds x 3 seeds)
| method | Macro-F1 | FlipRate | Evidence Tokens | Turnover |
|---|---:|---:|---:|---:|
| Static | 0.9350 | 0.0280 | 901.3 | 0.4641 |
| Dynamic | 0.9352 ± 0.0016 | 0.0279 | 901.3 | 0.4679 |

delta Macro-F1: +0.0002
delta FlipRate: +0.0001
delta tokens: -0.0
effect label: MODERATE
readiness: PASS

### Selection Saturation (selected/candidates)
| cutoff | saturation |
|---:|---:|
| 5m | 0.2769 |
| 15m | 0.1361 |
| 30m | 0.0941 |
| 60m | 0.0702 |
| 180m | 0.0478 |
| 360m | 0.0418 |

## Per-cutoff Results
Full per-cutoff Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1 and temporal metrics are in `readiness/e3_summary.json` and per fold in `temporal_metrics.json`.

## Novelty Diagnostics
per-cutoff mean/median/P25/P75 selected novelty in `readiness/e3_summary.json#datasets.<ds>.dynamics`.

## Persistence Diagnostics
- memory retention rate and persistent/new ratios: per fold in `evidence_dynamics.json`, dataset means in `readiness/e3_summary.json`.

## Memory Leakage Audit
- future memory leakage: 0 (verified by `scripts/tcdscr_verify_e3.py`)
- future candidate leakage: 0

## Verification
- issues: 0

## Test Split
NOT EVALUATED

## Recommendation
START_E3_TEST
