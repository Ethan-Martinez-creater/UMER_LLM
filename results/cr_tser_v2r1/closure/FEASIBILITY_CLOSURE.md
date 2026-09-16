# CR-TSER V2R1 — feasibility closure

Baseline commit: `052cf13b41f4a68e9849644614fe5541944ce919`  
Protocol: `v2r1`  
Readers: qwen, mistral, internlm  
Primary decision dataset: `maweibo`  
Secondary (diagnostic only): `pheme`

## Final verdict

```text
P1 = PASS
P2 = FAIL
P3 = NOT RUN
P4 = NOT RUN

CR_TSER_V2R1_FEASIBILITY = NO_GO
reason = STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED
```

The frozen pilot did not establish the required structural-interaction effect: the observed edge gap was positive but below the pre-registered threshold and its event-bootstrap confidence interval crossed zero.

The observed direction was positive — parent-child removal cost more than matched non-adjacent removal — but the frozen pilot did not establish the pre-registered structural-interaction effect, so no structural claim is made.

## P1 — reader heterogeneity (Ma-Weibo, primary) = PASS

- mean disagreement **0.4082** against a threshold of 0.10
- pairs at or above 0.05: 3 of 3, with 2 required

| pair | jointly active | disagreement | utility Spearman |
|---|---|---|---|
| qwen + mistral | 81 | **0.5062** | 0.2392 |
| qwen + internlm | 123 | **0.4146** | 0.2051 |
| mistral + internlm | 79 | **0.3038** | 0.2364 |

- active counts: qwen 437/2549, mistral 444/2549, internlm 448/2549
- coverage: 90/90 labelled events, 258 event×cutoff pairs, cutoffs [15, 60, 360]
- artifact: `gates/p1_maweibo.json` (`8c8e661107d17bdd5dd17026e3618a642c5ae69e56d29cc933a05910b64c6c34`)

## P2 — structured interaction (Ma-Weibo, primary) = FAIL

- **Δedge = 0.012150** against a pre-registered threshold of 0.02
- 95% event-bootstrap CI = **(-0.002342, 0.027196)** over 49 events and 297 matched (reader, snapshot) pairs
- the CI lower bound is not above 0, so the second condition of the frozen gate also fails
- confirmatory Δsubtree = 0.016013, 95% CI (-0.003599, 0.035486) over 47 events

| slot | role | matched pairs | mean \|Interaction\| |
|---|---|---|---|
| I2 parent-child | treatment | 300 | 0.05509 |
| I3 matched-nonadjacent | control | 300 | 0.04287 |
| I4 subtree | treatment | 300 | 0.07819 |
| I5 matched-disconnected | control | 276 | 0.06279 |

- artifact: `gates/p2_maweibo.json` (`8de1f10b644cbbae80af5dd54065d3908afa1952c2a186dfd5bd154c0e1585f0`)

## PHEME — diagnostic only

- `diagnostic_only = true`, `decides_primary_gate = false`; PHEME decided no gate
- P1 mean disagreement 0.3764; P2 Δedge 0.003835, 95% CI (-0.010822, 0.018171), Δsubtree 0.005818
- artifacts: `gates/p1_pheme_diagnostic.json`, `gates/p2_pheme_diagnostic.json`

## Frozen protocol (unchanged by this round)

```text
utility_threshold        = 0.05
p1_mean_disagreement_min = 0.1
p1_pair_disagreement_min = 0.05
p1_pairs_required        = 2
p2_edge_delta_min        = 0.02
bootstrap                = event-level, 10000 iterations, seed 7319
matching_unit            = reader_x_snapshot
cutoffs                  = [15, 60, 360]
partition_seed           = 7319
split_sizes              = {'foundation_train': 80, 'utility_train': 50, 'utility_dev': 15, 'utility_eval': 25}
```

## Immutability

| object | status |
|---|---|
| `utility_labels/maweibo/labels.jsonl` | `6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3` (unchanged) |
| `utility_labels/pheme/labels.jsonl` | `773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15` (unchanged) |
| 10 v2r1 manifest files | unchanged |
| `results/cr_tser_v2/` | untouched (read-only history) |

## Not run

No predictor training, B0/B1/B3 training, Stage A/B, held-out-reader evaluation, P3, P4, final selector experiment or final pilot report was run, and no P3/P4 artifact exists. No threshold, reader, dataset, split, cutoff, intervention group or utility label was changed in response to the P2 result.
