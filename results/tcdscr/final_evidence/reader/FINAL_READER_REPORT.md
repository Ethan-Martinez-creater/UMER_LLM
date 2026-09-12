# Final Fold-Local Frozen-Qwen Reader Report

- model: `/data/jyz/next/model/qwen3-8b` (hash match vs V3-B: True)
- prompt: v3b-2, temperature 0, do_sample false
- sampling seed 4096, context source seed 2000, alpha 0.8, budget 1024
- samples: 600 (300 per dataset), four arms: STATIC_FULL, UTILITY_TOKEN_MATCHED, RANDOM_TOKEN_MATCHED, MS_TSR

## PHEME

| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |
|---|---:|---:|---:|---:|
| STATIC_FULL | 0.6433 | 0.6433 | 0.6397 | 471.90 |
| UTILITY_TOKEN_MATCHED | 0.6659 | 0.6667 | 0.6815 | 79.17 |
| RANDOM_TOKEN_MATCHED | 0.6692 | 0.6700 | 0.6857 | 78.51 |
| MS_TSR | 0.6724 | 0.6733 | 0.6899 | 80.11 |

- MS vs Static: +0.02910 CI [-0.01471, +0.07322] -> **HELD_OUT_TRANSFER_PASS** / **WEAK_TRANSFER**
- MS vs Utility-TM: +0.00646 CI [-0.01018, +0.02359] -> **MS_SELECTION_ADVANTAGE**
- MS social-token reduction vs Static Full: 0.7144
- unsupported citation rate (MS): 0.0059
- token matching never overshoots: True
- exact McNemar p (MS vs Static): 0.2430

| cutoff | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 5m | 0.7159 | 0.7196 | 0.6999 | 0.7391 |
| 15m | 0.6604 | 0.7083 | 0.7083 | 0.7083 |
| 30m | 0.6198 | 0.6900 | 0.6900 | 0.6716 |
| 60m | 0.5994 | 0.6162 | 0.6162 | 0.6162 |
| 180m | 0.6599 | 0.6800 | 0.6989 | 0.6999 |
| 360m | 0.5758 | 0.5484 | 0.5716 | 0.5659 |

## Ma-Weibo

| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |
|---|---:|---:|---:|---:|
| STATIC_FULL | 0.8462 | 0.8467 | 0.8380 | 850.64 |
| UTILITY_TOKEN_MATCHED | 0.8366 | 0.8367 | 0.8328 | 90.21 |
| RANDOM_TOKEN_MATCHED | 0.8200 | 0.8200 | 0.8176 | 87.22 |
| MS_TSR | 0.8332 | 0.8333 | 0.8288 | 90.75 |

- MS vs Static: -0.01301 CI [-0.04628, +0.02011] -> **HELD_OUT_TRANSFER_FAIL**
- MS vs Utility-TM: -0.00336 CI [-0.01687, +0.01003] -> **PRACTICALLY_SIMILAR**
- MS social-token reduction vs Static Full: 0.8545
- unsupported citation rate (MS): 0.0000
- token matching never overshoots: True
- exact McNemar p (MS vs Static): 0.5716

| cutoff | STATIC_FULL | UTILITY_TOKEN_MATCHED | RANDOM_TOKEN_MATCHED | MS_TSR |
|---|---:|---:|---:|---:|
| 5m | 0.8377 | 0.8390 | 0.8390 | 0.8390 |
| 15m | 0.8798 | 0.8199 | 0.8400 | 0.8199 |
| 30m | 0.8193 | 0.8199 | 0.7596 | 0.8000 |
| 60m | 0.8400 | 0.8199 | 0.8199 | 0.8199 |
| 180m | 0.8397 | 0.8390 | 0.8193 | 0.8397 |
| 360m | 0.8572 | 0.8782 | 0.8377 | 0.8782 |

