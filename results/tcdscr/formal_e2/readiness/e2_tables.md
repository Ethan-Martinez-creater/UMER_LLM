# Formal E2 — Static Selector (validation readiness)

30 runs (2 datasets x 5 folds x 3 seeds). Budget: 1024 evidence tokens, Qwen3-8B tokenizer, atomic Reply-Parent pairs. Readiness delta = static - max(random, semantic) >= 0.005 (0.5 percentage point) on the pooled 3-seed mean primary Macro-F1.

## pheme validation

| method | mean primary Macro-F1 (pooled, 3 seeds) | delta vs best baseline |
|---|---:|---:|
| static | 0.7209 ± 0.0320 | -0.0003 |
| random | 0.7212 ± 0.0338 |  |
| semantic | 0.7199 ± 0.0333 |  |

Readiness: FAIL (threshold +0.005); best baseline = 0.7212

| cutoff | static mF1 | random mF1 | semantic mF1 |
|---|---:|---:|---:|
| 5 | 0.7166 | 0.7166 | 0.7166 |
| 15 | 0.7218 | 0.7211 | 0.7205 |
| 30 | 0.7217 | 0.7202 | 0.7189 |
| 60 | 0.7208 | 0.7213 | 0.7205 |
| 180 | 0.7213 | 0.7236 | 0.7212 |
| 360 | 0.7234 | 0.7246 | 0.7217 |

Selector diagnostics (validation, 15-run mean):

| cutoff | entropy | selected | evidence tokens | Jaccard static∩semantic |
|---|---:|---:|---:|---:|
| 5 | 0.3253 | 3.88 | 290.9 | 0.9937 |
| 15 | 0.4751 | 6.11 | 469.2 | 0.9491 |
| 30 | 0.5500 | 7.27 | 565.4 | 0.8927 |
| 60 | 0.6112 | 8.13 | 637.4 | 0.8313 |
| 180 | 0.6780 | 8.98 | 708.1 | 0.7591 |
| 360 | 0.7039 | 9.27 | 731.4 | 0.7312 |

## maweibo validation

| method | mean primary Macro-F1 (pooled, 3 seeds) | delta vs best baseline |
|---|---:|---:|
| static | 0.7125 ± 0.0321 | -0.0120 |
| random | 0.7246 ± 0.0338 |  |
| semantic | 0.7144 ± 0.0352 |  |

Readiness: FAIL (threshold +0.005); best baseline = 0.7246

| cutoff | static mF1 | random mF1 | semantic mF1 |
|---|---:|---:|---:|
| 5 | 0.7182 | 0.7196 | 0.7180 |
| 15 | 0.7192 | 0.7318 | 0.7263 |
| 30 | 0.7180 | 0.7303 | 0.7209 |
| 60 | 0.7141 | 0.7287 | 0.7159 |
| 180 | 0.7026 | 0.7186 | 0.7027 |
| 360 | 0.7031 | 0.7182 | 0.7028 |

Selector diagnostics (validation, 15-run mean):

| cutoff | entropy | selected | evidence tokens | Jaccard static∩semantic |
|---|---:|---:|---:|---:|
| 5 | 0.2643 | 8.85 | 760.1 | 0.6428 |
| 15 | 0.3251 | 10.16 | 867.1 | 0.4635 |
| 30 | 0.3547 | 10.41 | 902.4 | 0.3841 |
| 60 | 0.3828 | 10.63 | 926.5 | 0.3256 |
| 180 | 0.4302 | 10.89 | 959.6 | 0.2476 |
| 360 | 0.4509 | 10.98 | 973.8 | 0.2167 |

## Overall

Status: **FAIL**
Recommendation: DO_NOT_START_E3

