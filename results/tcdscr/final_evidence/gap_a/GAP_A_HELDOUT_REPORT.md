# Gap A — Static Utility All-Event Held-out Report

Protocol: ALL_EVENT outer test split, budget 1024, frozen corrected-E2 checkpoints, 30 runs (2 datasets x 5 folds x 3 seeds), paired event bootstrap 10000 / seed 4096, fold-stratified.

## PHEME

- Static: 0.84701
- Random: 0.84610
- Semantic: 0.84612
- best simple baseline: semantic
- Static - Random: +0.00091 CI [+0.00019, +0.00165]
- Static - Semantic: +0.00089 CI [+0.00000, +0.00178]
- **Status: STRONG_SUPPORT** (meets historical 0.005 threshold: False)

| cutoff | Static | Random | Semantic |
|---|---:|---:|---:|
| 5m | 0.8428 | 0.8431 | 0.8429 |
| 15m | 0.8458 | 0.8458 | 0.8459 |
| 30m | 0.8477 | 0.8474 | 0.8475 |
| 60m | 0.8479 | 0.8473 | 0.8472 |
| 180m | 0.8490 | 0.8467 | 0.8466 |
| 360m | 0.8489 | 0.8463 | 0.8465 |

| fold | Static | Random | Semantic |
|---|---:|---:|---:|
| 0 | 0.8415 | 0.8426 | 0.8429 |
| 1 | 0.8660 | 0.8663 | 0.8660 |
| 2 | 0.8292 | 0.8260 | 0.8268 |
| 3 | 0.8405 | 0.8391 | 0.8382 |
| 4 | 0.8570 | 0.8555 | 0.8559 |

## Ma-Weibo

- Static: 0.92446
- Random: 0.90733
- Semantic: 0.90497
- best simple baseline: random
- Static - Random: +0.01713 CI [+0.01395, +0.02032]
- Static - Semantic: +0.01949 CI [+0.01624, +0.02275]
- **Status: STRONG_SUPPORT** (meets historical 0.005 threshold: True)

| cutoff | Static | Random | Semantic |
|---|---:|---:|---:|
| 5m | 0.8937 | 0.8921 | 0.8860 |
| 15m | 0.9180 | 0.9096 | 0.9066 |
| 30m | 0.9255 | 0.9094 | 0.9091 |
| 60m | 0.9312 | 0.9099 | 0.9092 |
| 180m | 0.9368 | 0.9088 | 0.9089 |
| 360m | 0.9416 | 0.9143 | 0.9101 |

| fold | Static | Random | Semantic |
|---|---:|---:|---:|
| 0 | 0.9236 | 0.9137 | 0.9121 |
| 1 | 0.9279 | 0.9054 | 0.9014 |
| 2 | 0.9285 | 0.9222 | 0.9145 |
| 3 | 0.9375 | 0.9168 | 0.9149 |
| 4 | 0.9048 | 0.8783 | 0.8818 |

