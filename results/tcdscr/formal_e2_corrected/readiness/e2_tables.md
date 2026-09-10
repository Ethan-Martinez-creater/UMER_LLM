# Formal E2 — Static Selector (validation readiness)

30 runs (2 datasets x 5 folds x 3 seeds). Budget: 1024 evidence tokens, Qwen3-8B tokenizer, atomic Reply-Parent pairs. Readiness delta = static - max(random, semantic) >= 0.005 (0.5 percentage point) on the pooled 3-seed mean primary Macro-F1.

## pheme validation

| method | mean primary Macro-F1 (pooled, 3 seeds) | delta vs best baseline |
|---|---:|---:|
| static | 0.8559 ± 0.0012 | +0.0022 |
| random | 0.8537 ± 0.0014 |  |
| semantic | 0.8525 ± 0.0019 |  |

Readiness: FAIL (threshold +0.005); best baseline = 0.8537

| cutoff | static mF1 | random mF1 | semantic mF1 |
|---|---:|---:|---:|
| 5 | 0.8509 | 0.8511 | 0.8511 |
| 15 | 0.8533 | 0.8516 | 0.8513 |
| 30 | 0.8542 | 0.8531 | 0.8508 |
| 60 | 0.8581 | 0.8549 | 0.8538 |
| 180 | 0.8589 | 0.8555 | 0.8537 |
| 360 | 0.8600 | 0.8560 | 0.8545 |

Selector diagnostics (validation, 15-run mean):

| cutoff | entropy | selected | evidence tokens | Jaccard static∩semantic |
|---|---:|---:|---:|---:|
| 5 | 0.2951 | 3.88 | 291.0 | 0.9938 |
| 15 | 0.4317 | 6.11 | 469.3 | 0.9499 |
| 30 | 0.4991 | 7.27 | 565.4 | 0.8943 |
| 60 | 0.5623 | 8.13 | 637.4 | 0.8334 |
| 180 | 0.6331 | 8.97 | 708.4 | 0.7619 |
| 360 | 0.6555 | 9.26 | 731.6 | 0.7343 |

Proxy sanity vs E1 full encoder (validation, Macro-F1):

| cutoff | E1 full | static | random | semantic |
|---|---:|---:|---:|---:|
| 5 | 0.8525 | 0.8509 | 0.8511 | 0.8511 |
| 15 | 0.8595 | 0.8533 | 0.8516 | 0.8513 |
| 30 | 0.8591 | 0.8542 | 0.8531 | 0.8508 |
| 60 | 0.8615 | 0.8581 | 0.8549 | 0.8538 |
| 180 | 0.8626 | 0.8589 | 0.8555 | 0.8537 |
| 360 | 0.8633 | 0.8600 | 0.8560 | 0.8545 |

E1 full mean primary: 0.8597; proxy sanity warning: no

## maweibo validation

| method | mean primary Macro-F1 (pooled, 3 seeds) | delta vs best baseline |
|---|---:|---:|
| static | 0.9350 ± 0.0014 | +0.0210 |
| random | 0.9141 ± 0.0041 |  |
| semantic | 0.9091 ± 0.0050 |  |

Readiness: PASS (threshold +0.005); best baseline = 0.9141

| cutoff | static mF1 | random mF1 | semantic mF1 |
|---|---:|---:|---:|
| 5 | 0.8967 | 0.8950 | 0.8837 |
| 15 | 0.9338 | 0.9210 | 0.9139 |
| 30 | 0.9379 | 0.9210 | 0.9143 |
| 60 | 0.9423 | 0.9145 | 0.9145 |
| 180 | 0.9492 | 0.9183 | 0.9139 |
| 360 | 0.9504 | 0.9146 | 0.9142 |

Selector diagnostics (validation, 15-run mean):

| cutoff | entropy | selected | evidence tokens | Jaccard static∩semantic |
|---|---:|---:|---:|---:|
| 5 | 0.2949 | 8.94 | 760.4 | 0.6110 |
| 15 | 0.3489 | 10.37 | 866.7 | 0.4259 |
| 30 | 0.3716 | 10.70 | 902.4 | 0.3547 |
| 60 | 0.4037 | 10.96 | 926.5 | 0.2950 |
| 180 | 0.4521 | 11.31 | 959.8 | 0.2170 |
| 360 | 0.4803 | 11.45 | 974.0 | 0.1866 |

Proxy sanity vs E1 full encoder (validation, Macro-F1):

| cutoff | E1 full | static | random | semantic |
|---|---:|---:|---:|---:|
| 5 | 0.8858 | 0.8967 | 0.8950 | 0.8837 |
| 15 | 0.9251 | 0.9338 | 0.9210 | 0.9139 |
| 30 | 0.9345 | 0.9379 | 0.9210 | 0.9143 |
| 60 | 0.9407 | 0.9423 | 0.9145 | 0.9145 |
| 180 | 0.9470 | 0.9492 | 0.9183 | 0.9139 |
| 360 | 0.9484 | 0.9504 | 0.9146 | 0.9142 |

E1 full mean primary: 0.9302; proxy sanity warning: no

## Overall

Status: **PARTIAL**
Recommendation: STOP_FOR_RESEARCH_REVIEW

