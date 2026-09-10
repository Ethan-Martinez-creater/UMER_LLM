# TC-DSCR Formal E3 Held-out Test

## Git
- base commit: `d87c990` (Formal E3 validation)
- E3-B commit: `f7ec0ac396ab8d5502ed91a65ea6c0137f47a65a`

## Overall Status
WEAK_POSITIVE

## Frozen Validation Configs
Fold-specific best_config.json from `results/tcdscr/formal_e3/` (authoritative; never re-searched at test time).

## Run Completeness
- expected = 30, completed = 30, failed = 0

## PHEME

Static mean-primary Macro-F1: 0.8464
Dynamic mean-primary Macro-F1: 0.8465 ± 0.0006
Delta: +0.0001

Static FlipRate: 0.0130
Dynamic FlipRate: 0.0137
Delta: +0.0007

Bootstrap Macro-F1 95% CI: [-0.0001, +0.0003]
Bootstrap FlipRate 95% CI: [+0.0005, +0.0009]

Positive folds: 3, Negative folds: 1

Per-seed deltas:
- seed2000: -0.0001
- seed2001: +0.0002
- seed2002: +0.0002

Per-cutoff (pooled):
| cutoff | static MF1 | dynamic MF1 |
|---|---:|---:|
| 5m | 0.8428 | 0.8428 |
| 15m | 0.8452 | 0.8453 |
| 30m | 0.8468 | 0.8465 |
| 60m | 0.8468 | 0.8471 |
| 180m | 0.8485 | 0.8488 |
| 360m | 0.8486 | 0.8489 |

Selection saturation (PHEME):
- 5m: 0.8779
- 15m: 0.7189
- 30m: 0.6250
- 60m: 0.5519
- 180m: 0.4833
- 360m: 0.4570

Dynamic Diagnostics:
- turnover: {'5': 0.0, '15': 0.4793323592155267, '30': 0.2939425428984408, '60': 0.25898656904382555, '180': 0.26668193761987785, '360': 0.15994687491942777}
- retention / novelty / persistent-new ratios: `e3_test_summary.json#datasets.pheme.dynamics`

status: WEAK_POSITIVE

## Ma-Weibo

Static mean-primary Macro-F1: 0.9268
Dynamic mean-primary Macro-F1: 0.9268 ± 0.0033
Delta: +0.0000

Static FlipRate: 0.0311
Dynamic FlipRate: 0.0311
Delta: -0.0000

Bootstrap Macro-F1 95% CI: [-0.0001, +0.0001]
Bootstrap FlipRate 95% CI: [-0.0002, +0.0001]

Positive folds: 3, Negative folds: 2

Per-seed deltas:
- seed2000: +0.0001
- seed2001: -0.0001
- seed2002: +0.0000

Per-cutoff (pooled):
| cutoff | static MF1 | dynamic MF1 |
|---|---:|---:|
| 5m | 0.8946 | 0.8946 |
| 15m | 0.9218 | 0.9217 |
| 30m | 0.9291 | 0.9291 |
| 60m | 0.9338 | 0.9337 |
| 180m | 0.9390 | 0.9391 |
| 360m | 0.9425 | 0.9426 |

Selection saturation (Ma-Weibo):
- 5m: 0.2817
- 15m: 0.1391
- 30m: 0.0958
- 60m: 0.0710
- 180m: 0.0486
- 360m: 0.0424

Cap-aware Diagnostic (Ma-Weibo):
- 180m capped=False: n=12942 static MF1=0.9370 dynamic MF1=0.9371
- 180m capped=True: n=741 static MF1=0.9695 dynamic MF1=0.9695
- 360m capped=False: n=12693 static MF1=0.9402 dynamic MF1=0.9404
- 360m capped=True: n=1083 static MF1=0.9657 dynamic MF1=0.9647

Dynamic Diagnostics:
- turnover: {'5': 0.0, '15': 0.6925252651717811, '30': 0.48824934879961024, '60': 0.43484828762414707, '180': 0.4824952875443752, '360': 0.2731024805487675}
- retention / novelty / persistent-new ratios: `e3_test_summary.json#datasets.maweibo.dynamics`

status: WEAK_POSITIVE

## Leakage / Protocol Audit
- issues: 0
- future memory leakage: 0
- config mismatch: 0
- checksum mismatch: 0

## Research Interpretation
- PHEME: positive point estimate but CI crosses 0
- Ma-Weibo: positive point estimate but CI crosses 0

## Recommendation
REVIEW_DYNAMIC_DESIGN
