# TC-DSCR Dynamic V3 — MS-TSR Validation (Stage V3-A)

## Implementation
- encoder frozen: True (corrected E2 provenance SHA-checked)
- selector frozen: True
- proxy frozen: True
- new trainable parameters: NONE
- budget: 1024 (frozen)
- alpha grid: [0.8, 0.9, 0.95, 1.0] (only validation grid)
- moves: ADD + REMOVE only (no SWAP)

## Protocol
- all validation events at all 6 cutoffs (no-candidate snapshots included)
- test split never read; Qwen never called in V3-A

## PHEME

### Fold alpha
| fold | alpha |
|---|---:|
| 0 | 0.8 |
| 1 | 0.8 |
| 2 | 0.8 |
| 3 | 0.8 |
| 4 | 0.8 |

### Validation
| method | Macro-F1 | FlipRate | tokens |
|---|---:|---:|---:|
| Static | 0.8567 | 0.0135 | 472.8 |
| MS-TSR | 0.8567 | 0.0135 | 78.8 |

delta Macro-F1: +0.00000
token reduction: 0.5786
relative flip increase: +0.0000
gate: MS_TSR_PROXY_PASS ({'classification_non_inferiority': True, 'context_reduction': True, 'temporal_stability': True})

### Mechanism
- dual-view agreement rate: 0.9473
- compression attempt rate: 0.7710
- sufficiency success rate: 1.0000
- static fallback rate: 0.2819
- prediction change rate (MS-TSR vs Static): 0.000000
- mean token reduction: 0.5786 (median 0.8058, P25 0.0000, P75 0.9231)
- unit reduction: 0.8194
- memory survival rate: 0.4716
- adjacent set Jaccard: 0.4299
- context churn: 0.5701
- margin retention ratio: 1.3228
- mean ADD / REMOVE per snapshot: 0.75 / 0.00
- max candidate-pool size: 81

### Margin-stratified (§19)
| stratum | n | token reduction | Delta MF1 | fallback |
|---|---:|---:|---:|---:|
| low_confidence | 11567 | 0.3241 | +0.0000 | 0.5932 |
| medium | 23130 | 0.6421 | +0.0000 | 0.1605 |
| high_confidence | 11563 | 0.7061 | +0.0000 | 0.0016 |

### Stage-wise
| stage | Static MF1 | MS-TSR MF1 | delta | token reduction |
|---|---:|---:|---:|---:|
| Very Early | 0.8526 | 0.8526 | +0.0000 | 0.4267 |
| Early | 0.8584 | 0.8584 | +0.0000 | 0.6173 |
| Mid | 0.8594 | 0.8594 | +0.0000 | 0.6918 |

## Ma-Weibo

### Fold alpha
| fold | alpha |
|---|---:|
| 0 | 0.8 |
| 1 | 0.8 |
| 2 | 0.8 |
| 3 | 0.8 |
| 4 | 0.8 |

### Validation
| method | Macro-F1 | FlipRate | tokens |
|---|---:|---:|---:|
| Static | 0.9338 | 0.0283 | 855.6 |
| MS-TSR | 0.9338 | 0.0283 | 81.3 |

delta Macro-F1: +0.00000
token reduction: 0.8274
relative flip increase: +0.0000
gate: MS_TSR_PROXY_PASS ({'classification_non_inferiority': True, 'context_reduction': True, 'temporal_stability': True})

### Mechanism
- dual-view agreement rate: 0.9732
- compression attempt rate: 0.9258
- sufficiency success rate: 1.0000
- static fallback rate: 0.0782
- prediction change rate (MS-TSR vs Static): 0.000000
- mean token reduction: 0.8274 (median 0.9453, P25 0.8717, P75 0.9641)
- unit reduction: 0.8875
- memory survival rate: 0.2993
- adjacent set Jaccard: 0.2850
- context churn: 0.7150
- margin retention ratio: 1.6460
- mean ADD / REMOVE per snapshot: 0.95 / 0.05
- max candidate-pool size: 105

### Margin-stratified (§19)
| stratum | n | token reduction | Delta MF1 | fallback |
|---|---:|---:|---:|---:|
| low_confidence | 8416 | 0.6513 | +0.0000 | 0.2713 |
| medium | 16830 | 0.8763 | +0.0000 | 0.0128 |
| high_confidence | 8414 | 0.9057 | +0.0000 | 0.0000 |

### Stage-wise
| stage | Static MF1 | MS-TSR MF1 | delta | token reduction |
|---|---:|---:|---:|---:|
| Very Early | 0.9143 | 0.9143 | +0.0000 | 0.7301 |
| Early | 0.9388 | 0.9388 | +0.0000 | 0.8449 |
| Mid | 0.9484 | 0.9484 | +0.0000 | 0.9072 |

## Reading of the proxy result

- MS-TSR changed the Static decision on 0 of 79920 snapshots. Sufficiency condition C1 requires argmax q(S) == y_ref for every successfully compressed snapshot, and fallback rows keep the Static set verbatim; at the proxy layer Delta Macro-F1 and FlipRate therefore match Static by construction rather than by measurement.
- The gate's classification non-inferiority and temporal stability conditions consequently carry no evidential weight on the proxy reader; only compression, sufficiency, margin and temporal-set metrics are informative here.
- The proxy is additionally insensitive to evidence substitution (the Dynamic V2 diagnosis measured a 0.4-0.7% prediction-change rate under set replacement), so a proxy-level 'no loss' must not be read as a real-reader 'no loss'.
- Fold-local alpha collapsed to the loosest grid point (0.8) in every fold, i.e. the C2 margin constraint never bound; the achieved compression is driven by the budget and the greedy forward construction rather than by the sufficiency margin.
- Resolving exactly this gap is the purpose of Stage V3-B (frozen Qwen reader transfer), which is the required next step.

## Verifier
issues = 0 (runs=30, rows=319680, no_candidate_rows=41568)

## Recommendation
Overall = MS_TSR_PROXY_PASS
Recommendation = START_V3_B_READER_TRANSFER_PILOT
