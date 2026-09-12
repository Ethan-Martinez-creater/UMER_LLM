# TC-DSCR V3-B — Frozen Qwen Reader Transfer

## Protocol
- model: /data/jyz/next/model/qwen3-8b
- checkpoint: config_sha256=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30, weight_shards_sha256=b26df35264664d35515faa952273e0bb7f40de410c96059a8fc2e1ce4c66530f
- dtype/device: torch.bfloat16 / cuda:0
- decoding: do_sample=False, num_beams=1, max_new_tokens=256, temperature=0.0
- validation only: True (no test events read)
- scope: validation-only transfer-feasibility pilot; this is not held-out test performance (§44)
- samples: PHEME n=300, Ma-Weibo n=300
- sampling seed: 3090 (context source seed 2000)
- alpha: 0.8 (frozen V3-A)
- budget: 1024
- prompt version: v3b-2
- manifest samples_sha256: 2ad549e2adc6513412464aacf9c73b15ff6baa4df006cf2d9015fb9902c71124

## PHEME

### Detection
| Arm | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 |
|---|---:|---:|---:|---:|
| Static | 0.6300 | 0.6278 | 0.6350 | 0.5993 |
| MS-TSR | 0.6500 | 0.6476 | 0.6550 | 0.6182 |

delta Macro-F1: +0.01974
bootstrap 95% CI: [-0.0219, 0.0614] (10000 iters, seed 3090, multiplicity preserved)
McNemar p: 0.4296 (b=17, c=23)
effective pairs: 300 / 300 (parse failures static=0, ms=0)

### Compression
Static social tokens: 530.7
MS social tokens: 95.6
Reduction: 0.6870
Static total prompt tokens: 760.9
MS total prompt tokens: 325.9
Reduction: 0.4704
excluded zero-Static-context samples: 48
max total input tokens (all samples): 1236
context overflow (CONTEXT_OVERFLOW): 0 (flagged, never truncated)

### Paired Outcomes
wrong -> correct: 23
correct -> wrong: 17
both correct: 172
both wrong: 88
net correction gain: 6

### Disagreement
rate: 0.1333
label changed, both wrong: 0
label changed Static->correct: 23
label changed Static->wrong: 17

### Confidence / Calibration
mean confidence Static: 0.6150
mean confidence MS: 0.5563
confidence delta: -0.0587
Static confidence correct/wrong: 0.6294 / 0.5905
MS confidence correct/wrong: 0.5638 / 0.5424
ECE Static / MS (10 bins): 0.1777 / 0.1937

### Grounding
| metric | Static | MS-TSR |
|---|---:|---:|
| valid_citation_rate | 1.0000 | 1.0000 |
| unsupported_citation_rate | 0.0000 | 0.0000 |
| no_citation_rate | 0.1600 | 0.1767 |
| mean_cited_evidence_count | 4.6600 | 1.1300 |
| cited_over_supplied | 0.7702 | 0.9549 |
| reason_invalid_evidence_reference_rate | 0.0000 | 0.0000 |

unsupported citations (MS): 0

### Stratified — cutoff
| cutoff | n | Static MF1 | MS MF1 | delta | token reduction | disagreement |
|---|---:|---:|---:|---:|---:|---:|
| 5m | 50 | 0.6753 | 0.6940 | +0.01868 | 0.4899 | 0.0600 |
| 15m | 50 | 0.5572 | 0.6566 | +0.09940 | 0.5931 | 0.1000 |
| 30m | 50 | 0.6795 | 0.6532 | -0.02628 | 0.7321 | 0.2200 |
| 1h | 50 | 0.6599 | 0.6999 | +0.04002 | 0.7206 | 0.1600 |
| 3h | 50 | 0.6377 | 0.5994 | -0.03832 | 0.7515 | 0.1200 |
| 6h | 50 | 0.5536 | 0.5758 | +0.02219 | 0.7583 | 0.1400 |

### Stratified — selection pressure
| pressure | n | Static MF1 | MS MF1 | delta | token reduction |
|---|---:|---:|---:|---:|---:|
| NO_CANDIDATE | 48 | 0.7083 | 0.7083 | +0.00000 | 0.0000 |
| LOW | 207 | 0.6154 | 0.6531 | +0.03764 | 0.6412 |
| MEDIUM | 40 | 0.5990 | 0.5747 | -0.02426 | 0.8921 |
| HIGH | 5 | 0.3750 | 0.2857 | -0.08929 | 0.9425 |

### Stratified — proxy margin
| stratum | n | Static MF1 | MS MF1 | delta | token reduction | disagreement |
|---|---:|---:|---:|---:|---:|---:|
| low_confidence | 76 | 0.5649 | 0.5983 | +0.03341 | 0.5018 | 0.0789 |
| medium | 150 | 0.6162 | 0.6186 | +0.00247 | 0.7684 | 0.1333 |
| high_confidence | 74 | 0.6415 | 0.6795 | +0.03805 | 0.6642 | 0.1892 |

### Fallback vs compressed
| subset | n | Static MF1 | MS MF1 | delta | token reduction |
|---|---:|---:|---:|---:|---:|
| fallback | 64 | 0.6246 | 0.6246 | +0.00000 | 0.0000 |
| compressed | 236 | 0.6272 | 0.6522 | +0.02506 | 0.7336 |

### Gate — PASS / WEAK_TRANSFER
conditions: {'reader_non_inferiority': True, 'context_reduction': True, 'grounding_integrity': True}

## Ma-Weibo

### Detection
| Arm | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 |
|---|---:|---:|---:|---:|
| Static | 0.8833 | 0.8825 | 0.8833 | 0.8923 |
| MS-TSR | 0.8733 | 0.8732 | 0.8735 | 0.8774 |

delta Macro-F1: -0.00933
bootstrap 95% CI: [-0.0401, 0.0213] (10000 iters, seed 3090, multiplicity preserved)
McNemar p: 0.6776 (b=13, c=10)
effective pairs: 300 / 300 (parse failures static=0, ms=0)

### Compression
Static social tokens: 872.4
MS social tokens: 87.1
Reduction: 0.8532
Static total prompt tokens: 1141.3
MS total prompt tokens: 356.0
Reduction: 0.6477
excluded zero-Static-context samples: 14
max total input tokens (all samples): 1351
context overflow (CONTEXT_OVERFLOW): 0 (flagged, never truncated)

### Paired Outcomes
wrong -> correct: 10
correct -> wrong: 13
both correct: 252
both wrong: 25
net correction gain: -3

### Disagreement
rate: 0.0767
label changed, both wrong: 0
label changed Static->correct: 10
label changed Static->wrong: 13

### Confidence / Calibration
mean confidence Static: 0.7213
mean confidence MS: 0.6647
confidence delta: -0.0567
Static confidence correct/wrong: 0.7253 / 0.6914
MS confidence correct/wrong: 0.6616 / 0.6855
ECE Static / MS (10 bins): 0.1900 / 0.2407

### Grounding
| metric | Static | MS-TSR |
|---|---:|---:|
| valid_citation_rate | 1.0000 | 1.0000 |
| unsupported_citation_rate | 0.0000 | 0.0000 |
| no_citation_rate | 0.0467 | 0.0733 |
| mean_cited_evidence_count | 7.6533 | 1.0567 |
| cited_over_supplied | 0.7783 | 0.9577 |
| reason_invalid_evidence_reference_rate | 0.0000 | 0.0000 |

unsupported citations (MS): 0

### Stratified — cutoff
| cutoff | n | Static MF1 | MS MF1 | delta | token reduction | disagreement |
|---|---:|---:|---:|---:|---:|---:|
| 5m | 50 | 0.8750 | 0.8768 | +0.00185 | 0.7276 | 0.0400 |
| 15m | 50 | 0.8599 | 0.8586 | -0.00136 | 0.8575 | 0.1200 |
| 30m | 50 | 0.8390 | 0.8199 | -0.01904 | 0.8618 | 0.1000 |
| 1h | 50 | 0.7971 | 0.7792 | -0.01787 | 0.8815 | 0.0600 |
| 3h | 50 | 0.9800 | 0.9199 | -0.06012 | 0.8896 | 0.1000 |
| 6h | 50 | 0.9388 | 0.9798 | +0.04100 | 0.8881 | 0.0400 |

### Stratified — selection pressure
| pressure | n | Static MF1 | MS MF1 | delta | token reduction |
|---|---:|---:|---:|---:|---:|
| NO_CANDIDATE | 14 | 1.0000 | 1.0000 | +0.00000 | 0.0000 |
| LOW | 65 | 0.7771 | 0.6962 | -0.08099 | 0.6714 |
| MEDIUM | 35 | 0.8849 | 0.9132 | +0.02828 | 0.8722 |
| HIGH | 186 | 0.8790 | 0.8754 | -0.00361 | 0.9132 |

### Stratified — proxy margin
| stratum | n | Static MF1 | MS MF1 | delta | token reduction | disagreement |
|---|---:|---:|---:|---:|---:|---:|
| low_confidence | 76 | 0.7787 | 0.7173 | -0.06142 | 0.7996 | 0.1447 |
| medium | 150 | 0.8995 | 0.9200 | +0.02053 | 0.8541 | 0.0733 |
| high_confidence | 74 | 0.9309 | 0.9167 | -0.01418 | 0.8971 | 0.0135 |

### Fallback vs compressed
| subset | n | Static MF1 | MS MF1 | delta | token reduction |
|---|---:|---:|---:|---:|---:|
| fallback | 21 | 0.9443 | 0.9443 | +0.00000 | 0.0000 |
| compressed | 279 | 0.8777 | 0.8674 | -0.01031 | 0.8746 |

### Gate — FAIL / TRANSFER_FAIL
conditions: {'reader_non_inferiority': False, 'context_reduction': True, 'grounding_integrity': True}

## Determinism Audit
fallback pairs: 85
identical prompt hash: 85
identical raw output: 85
DETERMINISM_WARNING: 0

## Verifier
issues = 0 (samples=600, generations=1200, determinism_warnings=0)

## Overall
PARTIAL

## Recommendation
STOP_FOR_RESEARCH_REVIEW
