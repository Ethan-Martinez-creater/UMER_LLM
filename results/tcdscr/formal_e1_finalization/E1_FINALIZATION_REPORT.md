# TC-DSCR Formal E1 Finalization Report

## Overall Status
PARTIAL

(E1 training itself completed 60/60 with zero failures and the full
per-cutoff result set was delivered in E1_REPORT.md; the finalization audit
found that the historical UMER fold partition and the TC-DSCR Protocol A
partition assign *different event sets* to every outer fold, which blocks
the use of UMER-init for E2. See Section 1.)

## Git
- base commit: `db3c269` (TC-DSCR Formal E1, with SHA-backfill commit `2cbcdb6`)
- final commit: `26029af` (TC-DSCR Formal E1 finalization; SHA recorded by
  the immediately following backfill commit — a commit cannot contain its
  own hash)

## 1. UMER / TC-DSCR Fold-ID Parity

Restoration paths (used verbatim from the historical tree):
- gold: `original_model_optimization/optimization_rounds/
  round_006_node_deberta/screen_fold.py` — `ordered_labels`
  (`splits/all_event_ids.txt` file order + `labels.csv` manifest) and
  `strict_indices` (outer `StratifiedKFold(5, shuffle, random_state=3090)`,
  inner `train_test_split(test_size=0.10, random_state=3090, shuffle,
  stratify=)`), exactly as the round_032_sam runner
  (`--partition-seed 3090 --inner-split-strategy label`) invoked them.
  Data dirs: PHEME `/data/jyz/next/modified/common/pheme_240h_data`,
  Ma-Weibo `/data/jyz/next/llm/data/maweibo_240h_data`.
- new: TC-DSCR `build_primary_fold_split` over `sorted(registry)`
  (StratifiedKFold + StratifiedShuffleSplit `random_state=3090`; the two
  sklearn paths are the same algorithm — verified `gold_vs_sim_identical`
  = true for all 10 folds).

Dataset-level checks: event sets identical (PHEME 6425/6425, Ma-Weibo
4664/4664), label mismatches 0, no id only-on-one-side.

### PHEME
| fold | train exact | val exact | test exact | old train ∩ new test | old train ∩ new val |
|---|---:|---:|---:|---:|---:|
| 0 | false | false | false | 936 | 364 |
| 1 | false | false | false | 938 | 372 |
| 2 | false | false | false | 947 | 374 |
| 3 | false | false | false | 945 | 358 |
| 4 | false | false | false | 953 | 372 |

(old validation ∩ new test: 91 / 105 / 99 / 89 / 107.)

### Ma-Weibo
| fold | train exact | val exact | test exact | old train ∩ new test | old train ∩ new val |
|---|---:|---:|---:|---:|---:|
| 0 | false | false | false | 668 | 267 |
| 1 | false | false | false | 679 | 264 |
| 2 | false | false | false | 681 | 264 |
| 3 | false | false | false | 666 | 271 |
| 4 | false | false | false | 675 | 271 |

(old validation ∩ new test: 72 / 66 / 77 / 78 / 68.)

### Conclusion
UMER_INIT_SAFE = **false**

The two protocols use the same sklearn calls, seeds and sizes, but the old
side orders events by the historical `all_event_ids.txt` file order while
TC-DSCR uses the sorted event registry. Stratified folds are therefore
assigned to *different events*: ~72% of every new test set (666–953 events
per fold) appeared in the old UMER training set. The historical UMER
checkpoints were trained under a different fold partition, so initializing
the E1 encoder from "the same fold" checkpoint carries training-history
overlap with the E1 test events. Full per-fold detail (counts, only-in
sets, first-50 diff ids) in `fold_parity_audit.json` / `fold_parity_table.md`.

Status: **UMER_INIT_FOLD_PARITY_FAIL**. No split was modified, no E1 run
was rerun.

## 2. Ma-Weibo Cap-aware Diagnostic

Post-hoc, descriptive only: `max_nodes = 1021` unchanged, no sampling /
top-k / cap modification. Per-event cap status rebuilt with the frozen
snapshot builder (3h: 247 / 4664 hit, 6h: 360 / 4664 hit — identical to
the earlier full cap audit). Predictions reused from the E1 UMER-init runs;
UMER-init results are flagged **UMER_INIT_RESULT_NOT_YET_APPROVED** because
of Section 1. Metrics below are 3-seed mean ± std over per-seed pooled
(5 folds merged) scores.

### 3h
| subset | n | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 | rumor ratio |
|---|---:|---:|---:|---:|---:|---:|
| uncapped | 4417 | 0.9421 ± 0.0026 | 0.9421 ± 0.0026 | 0.9421 ± 0.0026 | 0.9435 ± 0.0027 | 0.5065 |
| capped | 247 | 0.9811 ± 0.0023 | 0.9776 ± 0.0028 | 0.9810 ± 0.0023 | 0.9687 ± 0.0040 | 0.3077 |

(nonrumor ratio: uncapped 0.4935, capped 0.6923.)

### 6h
| subset | n | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 | rumor ratio |
|---|---:|---:|---:|---:|---:|---:|
| uncapped | 4304 | 0.9445 ± 0.0018 | 0.9445 ± 0.0018 | 0.9445 ± 0.0018 | 0.9463 ± 0.0020 | 0.5088 |
| capped | 360 | 0.9815 ± 0.0016 | 0.9793 ± 0.0017 | 0.9814 ± 0.0016 | 0.9726 ± 0.0023 | 0.3417 |

(nonrumor ratio: uncapped 0.4912, capped 0.6583.)

Capped events are far more often non-rumors (consistent with the full cap
audit's class rate asymmetry); per-seed and per-fold tables are in
`maweibo_cap_aware_e1.json` / `maweibo_cap_aware_e1.md`. Because the parent
UMER-init results are not approved, no selection conclusion is drawn from
these numbers.

## 3. Epoch Recording Fix

- old meaning: `hparams.epochs_run` in the 60 historical manifests recorded
  the *configured* `max_epochs` (60), even though training early-stopped.
- new meaning: `scripts/tcdscr_run_e1.py` now records
  `"epochs_run": len(history)` — the true number of epochs actually run
  (validated: history length monotonically equals training progress and
  never exceeds the configured max, verified on all 60 stored runs).
  `max_epochs`, `patience`, `best_epoch` remain recorded as before.
- historical runs retrained: **NO**; no historical manifest was rewritten.
- post-hoc audit (read-only over stored artifacts, `epochs_posthoc_audit
  .json` / `.md`): actual epochs run = `len(history.json)`:
  - pheme random: mean 20.4 (min 14, max 36)
  - pheme umer: mean 16.7 (min 10, max 29)
  - maweibo random: mean 17.9 (min 10, max 29)
  - maweibo umer: mean 20.3 (min 9, max 30)

## 4. Documentation Correction

- old wording: "冻结超参（UMER 历史五折训练配方…）" and script docstring
  "Hyperparameters follow the frozen UMER five-fold training recipe".
  The historical UMER optimization was a two-stage recipe (stage-1 lr 1e-4
  × 5 epochs, stage-2 lr 3e-5 × 55 epochs, SAM, layer-wise decay etc.) —
  not the E1 setting.
- new wording (E1_REPORT.md §2 and `scripts/tcdscr_run_e1.py`): the
  hyperparameters are the **TC-DSCR E1 frozen causal-encoder training
  recipe**, plus the explicit note: "UMER Init refers only to parameter
  initialization from the corresponding historical UMER fold checkpoint;
  TC-DSCR E1 uses its own frozen causal-encoder optimization recipe."
- No E1 number or result was changed.

## Blocking Issues

1. **UMER_INIT_FOLD_PARITY_FAIL** — old-UMER and TC-DSCR folds assign
   different event sets (0/10 folds exact-match), with old-train events
   overlapping the new test sets by 666–953 per fold. UMER-init E1 numbers
   (and the UMER-vs-random comparison) carry init-history overlap with the
   test splits and cannot be used as evidence for E2 selection.
2. Ma-Weibo UMER-init cap-aware numbers are therefore diagnostic only
   (UMER_INIT_RESULT_NOT_YET_APPROVED).

## Recommendation for E2

RANDOM_INIT

(Fixed rule: fold parity not satisfied ⇒ recommended encoder = RANDOM_INIT,
irrespective of the UMER-init scores.)