# TC-DSCR Dynamic V2 — MF-TSR Validation

## Protocol Correction
- scope: all validation events at every cutoff (candidate_count = 0 included)
- corrected E2 impact: see results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.md
- old E3 V1 all-event diagnostic: see results/tcdscr/dynamic_v2_protocol/e3_v1_all_event_diagnostic.json

## Implementation
- encoder frozen: True (corrected E2 provenance SHA-checked)
- selector frozen: True
- proxy frozen: True
- new trainable parameters: NONE
- budget: 1024 (frozen)

## PHEME

### Fold epsilon
| fold | epsilon |
|---|---:|
| 0 | 0.01 |
| 1 | 0.01 |
| 2 | 0.01 |
| 3 | 0.05 |
| 4 | 0.01 |

### Validation
| method | Macro-F1 | FlipRate | tokens |
|---|---:|---:|---:|
| Static | 0.8567 | 0.0135 | 472.8 |
| MF-TSR | 0.8596 | 0.0135 | 178.7 |

delta: +0.00293
relative flip reduction: 0.0019
gate: FAIL (MF_TSR_NOT_SUPPORTED)
effect: mean-primary Macro-F1 over 15 runs, all-event protocol

### Mechanism
- exact match: 0.2915
- Jaccard: 0.5065
- ADD: 20
- REMOVE: 173357
- SWAP: 15678
- fallback rate: 0.0000
- distortion reduction: +0.02126
- mean moves/snapshot: 4.09
- memory survival rate: 0.7940

## Ma-Weibo

### Fold epsilon
| fold | epsilon |
|---|---:|
| 0 | 0.01 |
| 1 | 0.05 |
| 2 | 0.01 |
| 3 | 0.01 |
| 4 | 0.05 |

### Validation
| method | Macro-F1 | FlipRate | tokens |
|---|---:|---:|---:|
| Static | 0.9338 | 0.0283 | 855.6 |
| MF-TSR | 0.9309 | 0.0281 | 516.3 |

delta: -0.00286
relative flip reduction: 0.0063
gate: FAIL (MF_TSR_NOT_SUPPORTED)
effect: mean-primary Macro-F1 over 15 runs, all-event protocol

### Mechanism
- exact match: 0.0893
- Jaccard: 0.3315
- ADD: 150
- REMOVE: 123808
- SWAP: 46086
- fallback rate: 0.0000
- distortion reduction: +0.02462
- mean moves/snapshot: 5.05
- memory survival rate: 0.6897

## Stage-wise
### PHEME
| stage | Static MF1 | MF-TSR MF1 | delta | set-change |
|---|---:|---:|---:|---:|
| Very Early | 0.8525 | 0.8558 | +0.0033 | 0.5610 |
| Early | 0.8582 | 0.8611 | +0.0029 | 0.7506 |
| Mid | 0.8593 | 0.8619 | +0.0026 | 0.8139 |

### Ma-Weibo
| stage | Static MF1 | MF-TSR MF1 | delta | set-change |
|---|---:|---:|---:|---:|
| Very Early | 0.9143 | 0.9086 | -0.0057 | 0.8300 |
| Early | 0.9388 | 0.9372 | -0.0015 | 0.9266 |
| Mid | 0.9484 | 0.9470 | -0.0014 | 0.9755 |

## Selection-pressure
### PHEME
| bin | Static MF1 | MF-TSR MF1 | delta | set-change |
|---|---:|---:|---:|---:|
| low | 0.8629 | 0.8626 | -0.0003 | 0.0763 |
| medium | 0.8542 | 0.8534 | -0.0008 | 0.9974 |
| high | 0.8529 | 0.8598 | +0.0069 | 1.0000 |

### Ma-Weibo
| bin | Static MF1 | MF-TSR MF1 | delta | set-change |
|---|---:|---:|---:|---:|
| low | 0.8659 | 0.8659 | +0.0000 | 0.0957 |
| medium | 0.9243 | 0.9243 | -0.0001 | 0.9802 |
| high | 0.9351 | 0.9317 | -0.0034 | 0.9978 |

## Verifier
issues = 0 (runs=30, rows=239760, no_candidate_rows=31176)

## Recommendation
Overall = FAIL
Recommendation = REDESIGN_OR_REMOVE_DYNAMIC
