# Corrected E2 — All-Event Validation Metrics (budget 1024)

old metric: conditioned on candidate_count > 0

new metric: all events at the real cutoff (no-candidate snapshots classify with the zero evidence vector)

## PHEME

| method | old mean-primary Macro-F1 | new all-event mean-primary Macro-F1 | delta |
|---|---:|---:|---:|
| random | 0.8535 | 0.8548 | +0.0013 |
| semantic | 0.8523 | 0.8538 | +0.0015 |
| static | 0.8557 | 0.8567 | +0.0010 |

## Ma-Weibo

| method | old mean-primary Macro-F1 | new all-event mean-primary Macro-F1 | delta |
|---|---:|---:|---:|
| random | 0.9139 | 0.9136 | -0.0004 |
| semantic | 0.9090 | 0.9089 | -0.0001 |
| static | 0.9351 | 0.9339 | -0.0012 |

