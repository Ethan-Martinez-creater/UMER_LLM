# TC-DSCR Final Evidence Closure

## 1. Frozen Research State

- research freeze: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`
- consolidation baseline: `bb063317ccb026c5cb0dee2da5f33747e5ea5972`
- E1 causal encoder = VALID; corrected E2 static utility = VALID (all-event canonical, validation)
- Dynamic V1 = NOT SUPPORTED historically, previous held-out result CONDITIONAL_HELD_OUT
- Dynamic V2 / MF-TSR = NOT SUPPORTED AS FINAL METHOD
- Dynamic V3 / MS-TSR = MS_TSR_COMPRESSION_ONLY
- V3-B Qwen reader pilot = VALIDATION PILOT (PHEME WEAK_TRANSFER, Ma-Weibo TRANSFER_FAIL)
- Contribution D = VALIDATION-PILOT SUPPORTED, FINAL HELD-OUT EVIDENCE PENDING

This round is the FINAL MAIN EXPERIMENT ROUND for the current TC-DSCR line: frozen-model inference only, no training, no test-time tuning.

## 2. Documentation Correction

The over-strong claim that identical no-candidate predictions cannot change the Macro-F1 delta sign was removed from the consolidation artifacts and replaced with: Dynamic V1 remains a supported historical negative finding based on its negligible candidate-conditioned held-out effect, corrected bootstrap intervals crossing zero, and consistent all-event validation diagnostics; however, because Macro-F1 is nonlinear, the final all-event held-out effect is not inferred from identical no-candidate predictions and must be established by Gap F.

## 3. Gap A — Static Held-out Evidence

Protocol: ALL_EVENT outer test, budget 1024, frozen corrected-E2 checkpoints, 30 runs (2 datasets x 5 folds x 3 seeds).

### PHEME

Static:   0.8470
Random:   0.8461
Semantic: 0.8461

Best simple baseline: semantic
Static - Random:   +0.0009 CI [+0.0002, +0.0016]
Static - Semantic: +0.0009 CI [+0.0000, +0.0018]
Primary (Static - best): +0.0009 CI [+0.0000, +0.0018]
Status: **STRONG_SUPPORT** (meets historical 0.005 threshold: False)

### Ma-Weibo

Static:   0.9245
Random:   0.9073
Semantic: 0.9050

Best simple baseline: random
Static - Random:   +0.0171 CI [+0.0140, +0.0203]
Static - Semantic: +0.0195 CI [+0.0162, +0.0227]
Primary (Static - best): +0.0171 CI [+0.0140, +0.0203]
Status: **STRONG_SUPPORT** (meets historical 0.005 threshold: True)

## 4. Gap F — Dynamic V1 All-Event Held-out Closure

Protocol: ALL_EVENT outer test, frozen per-fold best_config, 30 runs, no test-time search.

### PHEME

Static:  0.8466
Dynamic: 0.8467
Delta:   +0.0001 CI [-0.0001, +0.0003]
FlipRate delta: +0.0005 CI [+0.0003, +0.0008]
Status: **DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED**

### Ma-Weibo

Static:  0.9244
Dynamic: 0.9244
Delta:   +0.0000 CI [-0.0001, +0.0002]
FlipRate delta: -0.0000 CI [-0.0002, +0.0002]
Status: **DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED**

## 5. Final Reader Protocol

- Qwen: `/data/jyz/next/model/qwen3-8b`
- model hash match vs V3-B: **True**
- prompt: v3b-2; temperature 0; do_sample false
- sample seed: 4096
- context source seed: 2000; alpha 0.8; budget 1024
- prior-reader exclusion: every V3-B validation pilot event id
- samples: 600 (300/dataset unless a deterministic shortage is recorded)
- arms: STATIC_FULL, UTILITY_TOKEN_MATCHED, RANDOM_TOKEN_MATCHED, MS_TSR

## 6. Reader Results — PHEME

| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |
|---|---:|---:|---:|---:|
| STATIC_FULL | 0.6433 | 0.6433 | 0.6397 | 471.897 |
| UTILITY_TOKEN_MATCHED | 0.6659 | 0.6667 | 0.6815 | 79.167 |
| RANDOM_TOKEN_MATCHED | 0.6692 | 0.6700 | 0.6857 | 78.507 |
| MS_TSR | 0.6724 | 0.6733 | 0.6899 | 80.107 |

MS vs Static: delta +0.0291 CI [-0.0147, +0.0732]
  gate: **HELD_OUT_TRANSFER_PASS** / **WEAK_TRANSFER**
  MS social-token reduction vs Static Full: 0.714
  unsupported citation rate (MS): 0.0059

MS vs Utility-TM: delta +0.0065 CI [-0.0102, +0.0236]
  interpretation: **MS_SELECTION_ADVANTAGE**

paired McNemar (MS vs Static) exact p = 0.2430
paired McNemar (MS vs Utility-TM) exact p = 0.7266

## 7. Reader Results — Ma-Weibo

| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |
|---|---:|---:|---:|---:|
| STATIC_FULL | 0.8462 | 0.8467 | 0.8380 | 850.643 |
| UTILITY_TOKEN_MATCHED | 0.8366 | 0.8367 | 0.8328 | 90.210 |
| RANDOM_TOKEN_MATCHED | 0.8200 | 0.8200 | 0.8176 | 87.223 |
| MS_TSR | 0.8332 | 0.8333 | 0.8288 | 90.750 |

MS vs Static: delta -0.0130 CI [-0.0463, +0.0201]
  gate: **HELD_OUT_TRANSFER_FAIL**
  MS social-token reduction vs Static Full: 0.854
  unsupported citation rate (MS): 0.0000

MS vs Utility-TM: delta -0.0034 CI [-0.0169, +0.0100]
  interpretation: **PRACTICALLY_SIMILAR**

paired McNemar (MS vs Static) exact p = 0.5716
paired McNemar (MS vs Utility-TM) exact p = 1.0000

## 8. Token / Latency

- PHEME: mean latency 1.242 s, median 1.182 s, total input tokens 490853, total generated tokens 69830
  - utility token gap mean 0.940 (P90 1.100), random token gap mean 1.600 (P90 6.000); never overshoot: True
- Ma-Weibo: mean latency 1.449 s, median 1.268 s, total input tokens 656922, total generated tokens 80814
  - utility token gap mean 0.540 (P90 2.000), random token gap mean 3.527 (P90 11.100); never overshoot: True

- monetary cost: no price is fabricated; only token/latency cost is reported

## 9. Grounding / Citation

- PHEME STATIC_FULL: valid 0.9986, unsupported 0.0014, no-citation 0.173, mean cited 4.850
- PHEME UTILITY_TOKEN_MATCHED: valid 0.9941, unsupported 0.0059, no-citation 0.197, mean cited 1.127
- PHEME RANDOM_TOKEN_MATCHED: valid 0.9939, unsupported 0.0061, no-citation 0.197, mean cited 1.093
- PHEME MS_TSR: valid 0.9941, unsupported 0.0059, no-citation 0.197, mean cited 1.127
- Ma-Weibo STATIC_FULL: valid 1.0000, unsupported 0.0000, no-citation 0.043, mean cited 8.547
- Ma-Weibo UTILITY_TOKEN_MATCHED: valid 1.0000, unsupported 0.0000, no-citation 0.067, mean cited 1.193
- Ma-Weibo RANDOM_TOKEN_MATCHED: valid 1.0000, unsupported 0.0000, no-citation 0.067, mean cited 1.133
- Ma-Weibo MS_TSR: valid 1.0000, unsupported 0.0000, no-citation 0.067, mean cited 1.193

Only evidence-id grounding is measured; natural-language hallucination is not claimed.

## 10. Fold-wise Results

### PHEME

| fold | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 0 | 0.6663 | 0.6329 | 0.6000 | 0.6329 |
| 1 | 0.7321 | 0.7483 | 0.7306 | 0.7306 |
| 2 | 0.5469 | 0.5823 | 0.5982 | 0.6157 |
| 3 | 0.6317 | 0.6997 | 0.7333 | 0.7000 |
| 4 | 0.6181 | 0.6411 | 0.6557 | 0.6557 |

### Ma-Weibo

| fold | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 0 | 0.8326 | 0.8653 | 0.8316 | 0.8479 |
| 1 | 0.8996 | 0.8661 | 0.8316 | 0.8825 |
| 2 | 0.7991 | 0.7442 | 0.7442 | 0.7442 |
| 3 | 0.8971 | 0.8806 | 0.8806 | 0.8806 |
| 4 | 0.7943 | 0.8154 | 0.7991 | 0.7980 |

## 11. Cutoff-wise Results

### PHEME

| cutoff | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 5m | 0.7159 | 0.7196 | 0.6999 | 0.7391 |
| 15m | 0.6604 | 0.7083 | 0.7083 | 0.7083 |
| 30m | 0.6198 | 0.6900 | 0.6900 | 0.6716 |
| 60m | 0.5994 | 0.6162 | 0.6162 | 0.6162 |
| 180m | 0.6599 | 0.6800 | 0.6989 | 0.6999 |
| 360m | 0.5758 | 0.5484 | 0.5716 | 0.5659 |

### Ma-Weibo

| cutoff | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 5m | 0.8377 | 0.8390 | 0.8390 | 0.8390 |
| 15m | 0.8798 | 0.8199 | 0.8400 | 0.8199 |
| 30m | 0.8193 | 0.8199 | 0.7596 | 0.8000 |
| 60m | 0.8400 | 0.8199 | 0.8199 | 0.8199 |
| 180m | 0.8397 | 0.8390 | 0.8193 | 0.8397 |
| 360m | 0.8572 | 0.8782 | 0.8377 | 0.8782 |

## 12. Statistical Summary

- reader bootstrap: 10000 iterations, seed 4096, stratified by fold x cutoff cells, multiplicity preserved
- Gap A bootstrap: {'unit': 'event', 'iterations': 10000, 'seed': 4096, 'stratified': 'within outer fold', 'multiplicity': 'preserved'}
- Gap F bootstrap: {'unit': 'event', 'iterations': 10000, 'seed': 4096, 'stratified': 'within outer fold', 'multiplicity': 'preserved'}
- verifier issues: 0

## 13. Evidence Closure Status

Gap A:
CLOSED

Gap F:
CLOSED

Gap B:
CLOSED

Gap C:
CLOSED

## 14. Research Implication

Based only on the measured results:

- PHEME Gap A: static - best baseline = +0.0009 (STRONG_SUPPORT).
- PHEME Gap F: dynamic - static = +0.0001 (DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED).
- PHEME reader: MS vs Static = +0.0291 (HELD_OUT_TRANSFER_PASS), MS vs Utility-TM = +0.0065 (MS_SELECTION_ADVANTAGE).
- Ma-Weibo Gap A: static - best baseline = +0.0171 (STRONG_SUPPORT).
- Ma-Weibo Gap F: dynamic - static = +0.0000 (DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED).
- Ma-Weibo reader: MS vs Static = -0.0130 (HELD_OUT_TRANSFER_FAIL), MS vs Utility-TM = -0.0034 (PRACTICALLY_SIMILAR).

No new method is proposed here; claim adjustments are left to the review stage.

## 15. Final Recommendation

**READY_FOR_PAPER_CONSOLIDATION**
