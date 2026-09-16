# CR-TSER V2R1 — P1 / P2 formal gate evaluation

Baseline: `daf82b642c85fcbdd9607d11149de06947bd311d`
Protocol: `v2r1` (primary = Ma-Weibo, secondary = PHEME)
Readers: `qwen` (Qwen3-8B), `mistral` (Mistral-7B-Instruct-v0.3), `internlm`
(InternLM3-8B-Instruct)

**Verdicts**

| gate | dataset | role | verdict |
|---|---|---|---|
| P1 reader heterogeneity | Ma-Weibo | primary decision | **P1_PASS** |
| P2 structured interaction | Ma-Weibo | primary decision | **P2_FAIL** |
| P1 reader heterogeneity | PHEME | diagnostic only | `DIAGNOSTIC_ONLY` |
| P2 structured interaction | PHEME | diagnostic only | `DIAGNOSTIC_ONLY` |

This round computed the two gates from the frozen caches only. No model was
loaded, no label was generated or rewritten, no intervention group was
re-matched, no threshold was touched and no research-plan change is proposed in
response to the P2 result.

## 1. Frozen inputs actually used

The stage refuses to run unless every input is the frozen one, so these digests
are the identity of the evidence behind both verdicts.

| input | sha256 |
|---|---|
| `utility_labels/maweibo/labels.jsonl` (9606 rows) | `6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3` |
| `utility_labels/pheme/labels.jsonl` (8637 rows) | `773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15` |

All ten v2r1 manifest digests are re-checked inside the stage and are unchanged
from the P1A freeze (`source.json` / `event_split.json` / `hashes.json` /
`snapshot_manifest.jsonl` / `intervention_manifest.jsonl`, per dataset).

Coverage: 90 labelled-pool events per dataset, all 90 carrying labels, no
missing event; 258 (Ma-Weibo) and 261 (PHEME) event-cutoff pairs; cutoffs
15/60/360 only; split 80/50/15/25 seed 7319; Ma-Weibo source
`b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d`.

## 2. P1 — reader heterogeneity (Ma-Weibo, primary)

Frozen statistic: sign disagreement on jointly active atomic interventions
(`|u| >= 0.05` or a correctness transition), `Disagree(a,b) =
P[sign(u_a) != sign(u_b)]`.

| pair | jointly active | disagreement | sign contingency (`s_a:s_b`) | utility Spearman |
|---|---|---|---|---|
| qwen + mistral | 81 | **0.5062** | `-1:1` 28, `-1:-1` 14, `1:-1` 13, `1:1` 26 | 0.2392 |
| qwen + internlm | 123 | **0.4146** | `1:-1` 22, `-1:-1` 30, `-1:1` 29, `1:1` 42 | 0.2051 |
| mistral + internlm | 79 | **0.3038** | `1:1` 39, `-1:1` 17, `-1:-1` 16, `1:-1` 7 | 0.2364 |

* MeanDisagreement (macro mean over the three pairs) = **0.4082**
* Active intervention counts per reader (of 2549 atomic evidence keys each):
  qwen 437, mistral 444, internlm 448 — the atomic cache has 2549 distinct
  evidence keys with all three readers present.
* Frozen threshold: mean ≥ 0.10 and ≥ 2/3 pairs ≥ 0.05.
  Observed: 0.4082 ≥ 0.10 and 3/3 pairs above 0.05.

**P1 = PASS.** The three readers disagree on roughly 30–50% of the
interventions they are both sensitive to, which is the heterogeneity the plan
requires before any reader-robust selector is worth training. The pairwise
utility Spearman coefficients (0.21–0.24) point the same way: the readers
correlate, but only weakly.

## 3. P2 — structured social interaction (Ma-Weibo, primary)

Frozen statistic: `Interaction_r(A) = u_r(A) - Σ u_r({e_i})`, matched
reader- and snapshot-specifically, with the event-level paired bootstrap
(10 000 iterations, seed 7319).

| slot | role | matched pairs | mean \|Interaction\| |
|---|---|---|---|
| I2 parent-child | treatment | 300 | 0.05509 |
| I3 matched-nonadjacent | control | 300 | 0.04287 |
| I4 subtree | treatment | 300 | 0.07819 |
| I5 matched-disconnected | control | 276 | 0.06279 |

**Edge (primary condition)**

* `Delta_edge` = **0.012150**
* event-bootstrap 95% CI = **(-0.002342, 0.027196)**, 49 events, 297 matched
  (reader, snapshot) pairs
* frozen threshold: `Delta_edge ≥ 0.02` **and** CI lower bound > 0
* observed: 0.012150 < 0.02, and the CI lower bound is negative

**Subtree (confirmatory)**

* `Delta_subtree` = **0.016013**, 95% CI = (-0.003599, 0.035486), 47 events,
  276 matched pairs

**P2 = FAIL.** The structured interventions do point in the expected direction
— both `Delta_edge` and `Delta_subtree` are positive, and removing a
parent-child pair costs more than removing a matched non-adjacent pair — but
the effect is about 0.012 against a 0.02 threshold and its bootstrap interval
straddles zero, so the primary condition is not met. The confirmatory subtree
condition behaves the same way (0.016, CI crossing zero).

The finding is recorded as measured. No threshold, reader, intervention or
dataset was adjusted in response, and no re-run with different settings was
performed.

## 4. PHEME diagnostic (never a gate)

| | value |
|---|---|
| P1 mean disagreement | 0.3764 (pairs 0.4747 / 0.3116 / 0.3429) |
| P1 active counts (of 2107 keys each) | qwen 405, mistral 292, internlm 600 |
| P2 `Delta_edge` | 0.003835, 95% CI (-0.010822, 0.018171), 52 events, 381 pairs |
| P2 `Delta_subtree` | 0.005818, 95% CI (-0.013563, 0.025113), 53 events, 369 pairs |
| P2 slot means | pc 0.07508, na 0.06272, sub 0.10333, disc 0.09245 |

Both PHEME artifacts carry `diagnostic_only = true` and
`decides_primary_gate = false`; their computed gate values are stored under
`diagnostic_gate` and never enter a primary decision. The PHEME pattern is the
same shape as Ma-Weibo: a positive but smaller edge gap with an interval
crossing zero.

## 5. Implementation

`scripts/cr_tser_eval_p1_p2.py` is a thin stage entrypoint. It

* verifies the frozen manifest and cache digests and refuses otherwise
  (`StageRefused`, exit code 3),
* calls the existing frozen
  `heterogeneity_report` / `gate_p1` and `structural_interaction_report` /
  `gate_p2`,
* reuses the frozen data-reading semantics of `cr_tser_run_pilot`
  (`load_unit_table`, `load_interaction_records`),
* serialises the artifacts, and computes only descriptive per-slot means and
  coverage counts.

Nothing in the evaluation mathematics is reimplemented, and the script carries
no threshold literal of its own; the verifier checks both properties
(`p1_p2_stage_reuses_frozen_statistics`) together with the artifact contract
(`p1_p2_primary_and_diagnostic_contract`,
`p1_p2_bootstrap_protocol_unchanged`), which assert protocol `v2r1`, the exact
reader set, Ma-Weibo as the decision dataset, the frozen cache digests, PHEME
as diagnostic-only, no retired reader and the unchanged bootstrap protocol.

`project/cr_tser/tests/test_v2r1_p1_p2_stage.py` covers this on synthetic data:
the frozen report/gate values are reproduced exactly (P1 macro 2/3 with two of
three pairs disagreeing; P2 `Delta_edge` 0.45 and `Delta_subtree` 0.6 on a
hand-computable construction), PHEME is diagnostic-only, and a drifted
manifest, a missing or mismatched cache, an inexact reader set and a retired
reader in the cache each fail closed.

## 6. Immutability

| artifact | before | after |
|---|---|---|
| `utility_labels/maweibo/labels.jsonl` | `6c6591ea…` | `6c6591ea…` (unchanged) |
| `utility_labels/pheme/labels.jsonl` | `773bee3e…` | `773bee3e…` (unchanged) |
| all 10 v2r1 manifest files | — | unchanged |

The evaluation is read-only with respect to every frozen input, and
`results/cr_tser_v2/` was not touched by this round.

## 7. Artifacts

| file | sha256 |
|---|---|
| `p1_maweibo.json` | `8c8e661107d17bdd5dd17026e3618a642c5ae69e56d29cc933a05910b64c6c34` |
| `p2_maweibo.json` | `8de1f10b644cbbae80af5dd54065d3908afa1952c2a186dfd5bd154c0e1585f0` |
| `p1_pheme_diagnostic.json` | `4790dd286cdde59c75f9c295249ca9fb9abef272554a7a57304011e412c6bb12` |
| `p2_pheme_diagnostic.json` | `412cdb424b6e29cb14ebefc39ec271599551e2b1dbd05b948677f935ef5b1c56` |

## 8. Not run

No predictor training, no B0/B1/B3 training, no Stage A/B, no held-out-reader
evaluation, no P3, no P4 and no final Pilot report. The P2 result is reported
as measured and the round stops for research review.
