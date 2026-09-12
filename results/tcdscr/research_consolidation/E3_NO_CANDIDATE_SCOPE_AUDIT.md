# E3 Held-out No-Candidate Scope Audit

Read-only audit of whether the E3 held-out test runner keeps event-cutoffs with `candidate_count == 0`.

## Runner behavior

- entry point: `scripts/tcdscr_run_e3_test.py`
- item build: scripts/tcdscr_run_e3_test.py main(): for ev in events['test']: for c in PRIMARY_CUTOFFS: items.append(build_light_item(...)) -- one item per event-cutoff, no candidate filter
- row build: scripts/tcdscr_run_e3.py collect_event_data()
- trajectory: scripts/tcdscr_run_e3.py run_event_trajectory(): appends one row per (event, cutoff) carrying n_candidates = len(cand_ids); no early continue
- current on-disk source keeps zero-candidate rows: **True** (The on-disk collect_event_data now carries the Dynamic V2 protocol correction ('no-candidate snapshots are kept in the trajectory'); this correction was applied to the same file AFTER the E3 held-out artifacts were produced.)
- frozen artifacts skipped zero-candidate rows: **True** (The frozen prediction rows contain no row with n_candidates == 0 and are short by exactly the no-candidate fraction, so the run that produced these artifacts skipped them.)

## Row-count evidence

| Dataset | runs | test events | expected rows | actual rows | zero-candidate rows | zero-candidate rate |
|---|---|---|---|---|---|---|
| pheme | 15 | 19275 | 115650 | 94104 | 0 | 0.000000 |
| maweibo | 15 | 13992 | 83952 | 79173 | 0 | 0.000000 |
| **total** | 30 | 33267 | 199602 | 173277 | 0 | |

- expected event x cutoff rows: 199602
- actual rows: 173277
- missing rows: 26325
- rows with n_candidates == 0: 0

## Cross-check against the validation no-candidate rate

| Dataset | validation mean no-candidate rate | held-out missing-row rate | abs gap | consistent |
|---|---|---|---|---|
| pheme | 0.187743 | 0.186304 | 0.001440 | True |
| maweibo | 0.050713 | 0.056925 | 0.006212 | True |

## Verdict

- scope verdict: **CASE_B_ZERO_CANDIDATE_SKIPPED**
- E3 held-out canonical status: **CONDITIONAL_HELD_OUT**

The held-out runner dropped event-cutoffs with no candidate. The missing rows match the validation no-candidate rate, so the lost rows are no-candidate snapshots rather than a generic row loss.

Consequences (no re-run, per protocol):

- E3 held-out absolute Macro-F1 values cannot remain ALL-EVENT canonical; they are classified CONDITIONAL_HELD_OUT / DEPRECATED_ABSOLUTE.
- Dynamic V1 remains a supported historical negative finding based on its negligible candidate-conditioned held-out effect, corrected bootstrap intervals crossing zero, and consistent all-event validation diagnostics.
- However, because Macro-F1 is nonlinear, the final all-event held-out effect is not inferred from identical no-candidate predictions and must be established by Gap F.
- An all-event held-out evaluation is a missing evidence cell and is recorded in NEXT_EXPERIMENT_GAPS.md.
- Validation all-event diagnostics are NOT a substitute for the held-out result.
