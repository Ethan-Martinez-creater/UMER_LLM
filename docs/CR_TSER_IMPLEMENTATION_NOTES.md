# CR-TSER Feasibility Pilot — implementation notes

Scope of this commit: **Checkpoint 1 (P0 data/reader plumbing) and
Checkpoint 2 (code complete)** of
[`CR_TSER_FEASIBILITY_PILOT_PLAN.md`](CR_TSER_FEASIBILITY_PILOT_PLAN.md).
The formal pilot (Checkpoint 3) has **not** been executed; no reader utility
label, no predictor training and no unseen-reader evaluation was run.

## 1. What was implemented

### Package `project/cr_tser/` (plan §28)

| Area | Files |
|---|---|
| Frozen config | `config/pilot_config.py` (seed 7319, 80/50/15/25, cutoffs 15m/1h/6h, B_ref=1024, losses, gate thresholds, LORO rotations) |
| Data | `data/weibo22_adapter.py`, `data/pilot_split.py`, `data/snapshot_bridge.py`, `data/structural_stats.py` |
| Intervention | `intervention/evidence_units.py`, `semantic_reference.py` (SRC), `intervention_generator.py` (I0–I5), `interaction_controls.py` (I3/I5 matching) |
| Readers | `readers/base_reader.py`, `sequence_scorer.py` (teacher-forced A/B), `qwen_reader.py`, `glm_reader.py`, `internlm_reader.py`, `mock_reader.py` (tests only) |
| Models | `models/bitte.py`, `text_baseline.py` (B0), `scalar_structure_baseline.py` (B1), `utility_heads.py` (shared + reader-residual + sign), `robust_selector.py` (S0–S6, 50% packing) |
| Training | `training/utility_dataset.py`, `train_utility.py` (B3 + row-level B0/B1), `checkpointing.py` |
| Evaluation | `evaluation/heterogeneity.py`, `structural_interaction.py`, `utility_prediction.py`, `unseen_reader.py`, `bootstrap.py` |
| Tests | `tests/` — the plan §35 inventory plus supporting unit tests |

The retired TC-DSCR components (1021D adjacency signature, Static Proxy,
MF-TSR/MS-TSR) are referenced only as *prohibited* items in prose; the verifier
strips comments/docstrings and asserts they never appear in CR-TSER code.

### Scripts (plan §30–§34)

| Script | Purpose |
|---|---|
| `scripts/cr_tser_p0_audit.py` | Weibo22 raw-field audit + normalized smoke, three-reader hash audit, A/B sanity (≥20 examples scored twice, 100% identity) |
| `scripts/cr_tser_build_manifests.py` | §31 manifests: `event_split.json`, `snapshot_manifest.jsonl`, `intervention_manifest.jsonl`, `hashes.json` |
| `scripts/cr_tser_generate_labels.py` | §32 utility-label cache (one row per dataset/event/cutoff/reader/intervention), resumable, hash-keyed |
| `scripts/cr_tser_train_predictors.py` | §33 steps 1–6: B0/B1/B3 per LORO rotation, seeds 7319/7320/7321 |
| `scripts/cr_tser_run_selection.py` | §22/§23: freeze arms, then score with the held-out reader |
| `scripts/cr_tser_run_pilot.py` | Gate evaluation + `CR_TSER_PILOT_REPORT.md` / `CR_TSER_PILOT_SUMMARY.json` |
| `scripts/cr_tser_verify_pilot.py` | §34 verifier, `--mode code` (static) and `--mode pilot` (artifacts) |
| `scripts/cr_tser_common.py` | Script-layer glue (dataset loading, snapshots, MiniLM, canonical tokenizer) |

## 2. P0 result — Weibo22 is not temporally usable

The audit was run against the locally available KPG release
(`data_raw/weibo22/extracted`, i.e. `Weibo_label_All.txt` +
`data.TD_RvNN.vol_5000.txt`):

```
timestamp_coverage        = 0.0
source_text_coverage      = 0.0
reply_text_coverage       = 0.0
verdict                   = WEIBO22_TEMPORAL_UNAVAILABLE
```

The released files carry only event id, binary label, topic, a parent-index
chain and vol_5000 vocabulary indices. There is **no** source/reply text and
**no** absolute timestamp of any kind. The plan forbids pseudo-time from row or
node order (§4.1), so the adapter does not invent one: `load_events` raises
`Weibo22TemporalUnavailable` and the P0 readiness file records `P0_FAIL`.

Consequences, stated plainly:

* the pilot cannot reach Gate P1–P4 on Weibo22, because no snapshot can be
  built and therefore no intervention can exist;
* a timestamp-bearing Weibo22 export (per-node text + timestamps + parent ids)
  is required before Checkpoint 3. `weibo22_adapter.load_events(raw, normalized_paths)`
  is the single entry point for such an export; nothing else has to change;
* PHEME plumbing is complete and can be exercised as a secondary check, but
  PHEME alone can never produce `FULL_GO` (§25).

Artifacts: `results/cr_tser/p0/{weibo22_audit.json,weibo22_audit.md,weibo22_smoke.json,reader_audit.json,label_scoring_sanity.json,p0_readiness.json,P0_READINESS.md}`.

## 3. Running the stages (server, after approval)

All paths come from `CRTSER_*` environment variables
(`CRTSER_WEIBO22_RAW`, `CRTSER_PHEME_RAW`, `CRTSER_SEMANTIC_MODEL`,
`CRTSER_QWEN_MODEL`, `CRTSER_GLM_MODEL`, `CRTSER_INTERLM_MODEL`,
`CRTSER_CANONICAL_TOKENIZER`, `CRTSER_OUT_ROOT`).

```bash
# Checkpoint 1 — data + reader readiness (no labels generated)
python scripts/cr_tser_p0_audit.py --readers --sanity

# Checkpoint 2 — tests only, never the full intervention dataset
python -m pytest project/cr_tser/tests -q
python scripts/cr_tser_verify_pilot.py --mode code     # issues = 0

# Checkpoint 3 — only after code approval
python scripts/cr_tser_build_manifests.py --dataset pheme
python scripts/cr_tser_generate_labels.py --dataset pheme
python scripts/cr_tser_train_predictors.py --dataset pheme
python scripts/cr_tser_run_selection.py  --dataset pheme
python scripts/cr_tser_run_pilot.py
python scripts/cr_tser_verify_pilot.py --mode pilot
```

`--smoke` (mock readers) exists for pipeline plumbing only and must never
produce formal results.

## 4. Decisions taken where the plan left a detail open

These are implementation choices, not protocol changes; each is isolated and
documented so the plan's owner can veto it cheaply.

1. **`log cutoff-time scalar`** (§14 `q_i` item 4) is implemented as
   `log1p(cutoff_minutes) / log1p(360)` so it stays in `[0, 1]`.
2. **SRC packing accounting** uses per-unit canonical tokens plus separator
   cost, then re-renders the final block and trims from the weakest pick until
   the *real* token count is within budget. The budget is therefore enforced
   against the rendered text, never against an estimate.
3. **I4 candidate rule** reads "at least two selected nodes in its current
   visible subtree" literally (including the root itself), then ranks candidate
   roots by selected-**descendant** count with earlier-timestamp tie-breaks.
4. **I3 matching** degrades in a fixed priority order
   (`exact_token_depth` → `token_only` → `structure_only`) and records which
   level it reached, so control quality is reported honestly.
5. **Sign convention** for §18 disagreement and for the tri-class label is the
   same frozen ±0.05 band, with correctness transitions overriding.
6. **B0/B1 share the §16 optimizer/early-stopping protocol** so the P3
   comparison is like-for-like; they are trained on rows pooled across the two
   training readers (their input has no reader identity).
7. **AUROC** uses the predictor's continuous class scores, not hard labels.

## 5. Local verification performed

* `python -m pytest project/cr_tser/tests -q` → **53 passed** (the plan §35
  inventory plus the review-round integration tests).
* `python scripts/cr_tser_verify_pilot.py --mode code` → **33 checks,
  issues = 0**.
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (pilot not executed, as instructed).
* `python scripts/cr_tser_p0_audit.py` → `P0_FAIL` (Weibo22 temporal
  unavailable; reader paths unset locally), written to `results/cr_tser/p0/`.
  **The P0 FAIL is the real state of the official public release and was not
  altered by these fixes.**
* `compileall` clean over `project/cr_tser` and `scripts/cr_tser_*.py`.

No formal pilot experiment was run: no formal utility label, no predictor
training and no unseen-reader evaluation.

---

# 6. Review fix round (code only — formal pilot NOT RUN)

Every finding from the review is listed with its fix location, the test that
pins it and the verifier check that now guards it. No scientific choice in
`CR_TSER_FEASIBILITY_PILOT_PLAN.md` was changed.

| # | Finding | Fix (location) | Test | Verifier check |
|---|---|---|---|---|
| 1 | Snapshot inherited the TC-DSCR `MAX_NODES=1021` cap | CR-TSER now owns `_assemble`/`build_causal_snapshot` with `MAX_NODES_CAP=None`; `assert_causal` rejects `cap_hit` (`project/cr_tser/data/snapshot_bridge.py`) | `test_snapshot_retains_more_than_1021_nodes`, `test_bitte_receives_all_snapshot_nodes` | `cr_tser_snapshot_uncapped`, `snapshot_uncapped:<dataset>` |
| 2 | Held-out reader labels could reach training | Entrances filter: `filter_rows_for_readers` + `UtilityDataset.__init__` calls `assert_groups_only_readers`; `run_rotation` filters the cache to the two training readers and records `held_out_rows_filtered` (`training/utility_dataset.py`, `scripts/cr_tser_train_predictors.py`) | `test_heldout_reader_not_in_training_batch`, `test_heldout_reader_not_in_early_stopping`, `test_batch_forward_never_touches_heldout_reader`, `test_rotation_builder_filters_three_reader_cache_at_the_entrance` | `loro_isolation_entrance`, `loro_isolation_wired`, `heldout_absent_from_predictions:<dataset>:<reader>` |
| 3 | Predictor/selector evidence identity not unified | Canonical `evidence_key = dataset\|event\|cutoff\|node` (`evidence_key`, `evidence_key_parts`) used by `utility_dataset`, `train_predictors`, `run_pilot`, `run_selection`; `predictions_for_snapshot` maps artifacts back to per-snapshot `{node_id: score}` | `test_predictor_selector_key_contract_end_to_end`, `test_p1_atomic_identity_includes_cutoff` | `evidence_key_defined`, `evidence_key_contract_consistent` |
| 4 | PHEME/Weibo22 artifacts could overwrite | All stages namespace by dataset: `manifests/<dataset>/`, `utility_labels/<dataset>/`, `predictor/<dataset>/`, `unseen_reader/<dataset>/`; the aggregator reads every dataset | `test_dataset_artifacts_do_not_overwrite` | `dataset_namespace:<script>` (5), `artifact_namespace_dataset_tagged`, `pheme_weibo22_artifacts_separate` |
| 5 | A/B scoring could double-add special tokens | `tokenize_prompt`/`tokenize_continuation` use `add_special_tokens=False`; `continuation_boundary` asserts the concatenation identity; P0 sanity records prompt token ids and each candidate's ids (`readers/sequence_scorer.py`, `scripts/cr_tser_p0_audit.py`) | `test_teacher_forced_tokenization_disables_special_tokens` | `teacher_forced_tokenization` |
| 6 | `prompt_hash` covered only evidence text | `prompt_hash`/`base_prompt_hash` hash the full chat-formatted prompt (`_chat_hash`); a cache hit is reused only when all of base context, intervened context, reader hash and prompt hash match, otherwise `CacheIdentityMismatch`; `model_weight_hash` now hashes shard head **and** tail plus config/index | `test_cache_fingerprint_mismatch_fails_closed` | `cache_fingerprint_fail_closed`, `prompt_hash_full_chat`, `cache_hashes_nonempty:<dataset>`, `reader_hashes_frozen:<dataset>` |
| 7 | `--force` could bypass manifest freezing | `assert_manifests_mutable` refuses unconditionally once any label cache exists; `--force` only permits a pre-freeze rebuild | `test_manifest_freeze_refuses_after_labels` | `manifest_immutable_after_labels` |
| 8 | Split could be drawn before viability filtering | `viable_event_ids` runs before `build_pilot_split`; the split manifest records `viable_event_count`/`viability_filtered` (`cr_tser_build_manifests.py`, `data/pilot_split.py`) | `test_viability_filter_precedes_split` | `viability_before_split`, `viability_before_split_recorded` |
| 9 | P1/P2/P3/P4 aggregation and gates | Unit-table keys include cutoff (no cross-cutoff collisions); P2 pairs within reader×snapshot then bootstraps over events; P3 requires the event-level paired bootstrap CI and pools **all** rotations; Weibo22 is primary, PHEME secondary only (`evaluation/structural_interaction.py`, `evaluation/utility_prediction.py`, `scripts/cr_tser_run_pilot.py`) | `test_p2_pairs_within_reader_and_snapshot`, `test_p3_gate_requires_bootstrap_ci`, `test_gate_p3_exact_thresholds`, `test_gate_logic_exact` | `p2_reader_snapshot_pairing`, `p3_bootstrap_gate`, `p3_all_rotations`, `primary_secondary_separated`, `pilot_summary_primary_is_weibo22` |
| 10 | PHEME-only B2/S6 legacy diagnostic missing | New `models/legacy_utility.py`: PHEME-only guard, inference-only (no grad, no training), reuses the frozen TC-DSCR Static Utility; S6 in the selection runner requires `--legacy` and refuses non-PHEME | `test_b2_s6_legacy_is_pheme_only_and_inference_only` | `b2_s6_pheme_only` |
| 11 | Code verifier too weak | 15 review checks added on top of the original inventory (33 checks total) | — | see the table above |

## 7. Review-round behaviour changes worth knowing

1. `UtilityDataset(groups, ...)` now **raises** if any row belongs to a reader
   outside `reader_keys`. Rotation builders must filter first; that is the
   point of the fix.
2. `gate_p3` **fails closed** when the event-level paired bootstrap is absent —
   a point estimate alone no longer satisfies P3.
3. `robust_score` / `rank_by_density` treat a missing prediction as `0.0`
   (neutral) instead of raising, because the §11 atomic cap can leave SRC units
   unlabeled; such units stay eligible for packing.
4. Manifest and utility-label paths are dataset-scoped. The old flat
   `utility_labels/<dataset>.jsonl` path is still read as a fallback so
   pre-existing local artifacts remain readable.

---

# 8. Final code-complete fix round (code only — formal pilot NOT RUN)

| # | Finding | Fix (location) | Test | Verifier check |
|---|---|---|---|---|
| 1 | `--normalized-events` existed only in the manifest builder | `weibo22_normalized` added to `PilotPaths`/`CRTSER_WEIBO22_NORMALIZED`; `cr_tser_common.load_dataset_events`/`dataset_registry` use it, so P0, manifest, labels, training and selection share one frozen source; `data/source_manifest.py` records `path`/`sha256`/`n_files` in `<manifests>/<dataset>/source.json` and `assert_frozen_source` fails every later stage on a changed source | `test_normalized_export_validation_ready_and_unavailable`, `test_raw_release_without_normalized_export_stays_unavailable`, `test_source_fingerprint_detects_change` | `weibo22_normalized_path_wired`, `p0_boundary_failure_blocks_pass` |
| 2 | P3 could derive `y_pred_sign` from ground-truth correctness | `run_pilot._pred_record` takes `predicted_sign`/`probs` from the model artifact only; `_sign_of` (the correctness-threshold helper) is deleted; B0/B1/B3 each contribute their own `predicted_sign`, three-class probs and continuous utility; `gate_p3` requires the event-level paired bootstrap CI | `test_p3_predicted_sign_comes_from_sign_head_not_correctness`, `test_p3_active_definition_includes_correctness_change`, `test_baseline_predictions_carry_own_sign_head` | `p3_predicted_sign_from_sign_head`, `p3_per_model_predictions` |
| 3 | B3 Spearman was copied onto baselines | `predict_baseline`/`_average_predictions` now return each model's own `utility`, `probs`, `predicted_sign`; `evaluate_utility` computes per-model Spearman/MAE/AUROC; bootstrap pairs identical `(event,key,reader)` rows | `test_baseline_predictions_carry_own_sign_head` | `p3_per_model_predictions`, `p3_bootstrap_gate` |
| 4 | The §11 top-20 label cap leaked into selector inference | `make_group` carries `infer_rows` (every `C_ref` candidate) beside the capped `unit_rows`; `infer_z` + `_reader_predictions`/`_shared_predictions` score all candidates; artifacts record `prediction_coverage`; `assert_full_coverage` makes S3/S4/S5/S6 fail closed on a missing candidate score (the silent `0.0` path is gone) | `test_label_cap_does_not_limit_inference_candidates`, `test_missing_candidate_prediction_fails_closed` | `inference_covers_all_candidates`, `selector_missing_prediction_fails_closed` |
| 5 | B2/S6 kept only a wrapper | `LegacyPHEMEUtility.per_unit_scores`/`score_items` run the real frozen path (TC-DSCR `build_light_item` → frozen encoder → `[h_source; h_i]` → `proxy.head`), `build_legacy_scorer` loads the frozen PHEME checkpoint via `load_frozen_components`, `run_selection` feeds the per-unit scores into S6 packing and refuses non-PHEME | `test_s6_consumes_legacy_scores`, `test_score_items_never_trains_and_is_pheme_only` | `s6_consumes_legacy_scores`, `b2_s6_pheme_only` |
| 6 | P0 recorded boundaries but did not fail on them | `cr_tser_p0_audit.evaluate_readiness` requires `identical_predictions` **and** `boundaries_ok` for every reader; `weibo22_temporal_check` prefers the normalized export and validates its fields | `test_p0_boundary_failure_blocks_pass` | `p0_boundary_failure_blocks_pass` |
| 7 | Cache fingerprint too weak | `reader_identity_hash` folds weight/tokenizer/template/dtype/model id; rows record `prompt_ids_hash` (actual tokenized prompt) and `reader_identity_hash`; `FINGERPRINT_FIELDS` covers all eight identities and any mismatch raises `CacheIdentityMismatch` | `test_cache_fingerprint_mismatch_fails_closed`, `test_cache_fails_closed_on_tokenizer_and_template_substitution` | `cache_identity_substitution_fails_closed`, `prompt_hash_full_chat` |
| 8 | S3a/S3b were reader-conditioned outputs of the two-reader model | `train_single_reader` trains one single-reader selector per reader (same architecture/protocol, three seeds averaged) and writes `predictor/<dataset>/single_<reader>/predictions.json` with `training_readers == [reader]`; `_load_single_predictions` refuses a multi-reader artifact; the two-reader B3 remains the S4/S5 source | `test_single_reader_artifact_guard` | `s3_single_reader_trained` |

## 9. Review-round behaviour changes (final round)

1. `--smoke` now redirects **all** writes to `<out_root>/smoke` (build manifests,
   labels, predictor, unseen reader), so a mock run can never freeze the formal
   cache or be mistaken for a formal result.
2. Every stage after manifest creation calls `assert_frozen_source`; swapping the
   Weibo22 export (or PHEME raw dir) mid-pipeline raises `SourceIdentityError`.
3. Formal S3/S4/S5/S6 raise `MissingPredictionError` instead of scoring an
   unlabeled candidate as `0.0`.
4. `predict_baseline` returns per-key `{utility, probs, predicted_sign,
   class_scores}` rather than a bare float, so B0/B1 are like-for-like with B3.
5. P0 exposes `weibo22_source_of_record` (`normalized_export` / `raw_release` /
   `none`) and `ab_boundaries_ok`; **P0 remains FAIL** because the only
   available Weibo22 source is the raw public KPG release, which has no
   per-node timestamps.

## 10. Verification (final round)

* `python -m pytest project/cr_tser/tests -q` → **67 passed**.
* `python scripts/cr_tser_verify_pilot.py --mode code` → **42 checks,
  issues = 0** (including the executed semantic checks).
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (pilot not executed).
* `python scripts/cr_tser_p0_audit.py` → `P0_FAIL`,
  `weibo22_source_of_record = raw_release`. The failure is the real state of
  the public release and was **not** altered by code.
* `compileall` clean over `project/cr_tser` and `scripts/cr_tser_*.py`.

No formal utility-label generation, predictor training, unseen-reader
evaluation or pilot run was performed.

---

# 11. Protocol closure fix round (code only — formal pilot NOT RUN)

| # | Finding | Fix (location) | Test | Verifier check |
|---|---|---|---|---|
| 1 | S6 used the Proxy logit difference instead of the frozen Static Utility | `LegacyPHEMEUtility` now keeps the frozen `StaticUtilitySelector` (required; `LegacyUnavailable` if absent) and computes `u_i = selector(h_i, event_repr, sem_i, sem_src, struct3_i)`; `score_items`/`per_unit_scores` drive S6; the Proxy is only the separate B2 classification diagnostic (`static_classification`/`b2_surface`, `diagnostic_only`, `participates_in_primary_gate=false`) | `test_legacy_s6_uses_static_utility_selector`, `test_legacy_selector_is_required`, `test_legacy_ranking_follows_selector_not_proxy` | `s6_uses_static_utility_selector` (executes a reversing Proxy and asserts it is never called) |
| 2 | P3 identity `(event,key,reader)` let a reader's second rotation overwrite the first | Observation key is `(event, rotation_id, evidence_key, reader)`; all three rotations contribute; the bootstrap resamples events with multiplicity preserved; B3 vs baseline pairs on identical observations | `test_p3_keeps_predictions_from_every_rotation` | `p3_keeps_all_rotations` |
| 3 | P3/P4 could pass with missing rotations | P3 returns `pass=false, reason="incomplete LORO predictor artifacts"` unless exactly three predictor artifacts exist; `gate_p4` checks completeness first (`rotation_completeness`) and `pheme_secondary` requires complete three-rotation evidence | `test_p3_missing_rotation_fails_closed`, `test_gate_p4_requires_exactly_three_distinct_rotations`, `test_pheme_secondary_requires_complete_rotations` | `p3_missing_rotation_fails_closed`, `p4_requires_three_rotations` |
| 4 | PHEME missing was treated as pass (fail-open) | `final_decision` requires `pheme["complete"] and pheme["pass"]`; missing/incomplete PHEME evidence can never yield `FULL_GO` (and forces `NO_GO`, not `PARTIAL_GO`) | `test_final_decision_requires_complete_pheme_evidence` | `missing_pheme_cannot_full_go` |
| 5 | §33 selection was a single un-auditable pass | `cr_tser_run_selection.py` splits into `freeze_subsets` (Stage A: training-reader predictors only, writes rotation-scoped frozen subset artifacts + `sha256`, never constructs a held-out reader) and `score_heldout` (Stage B: hash-verifies the frozen artifact first, fails closed on any change, then loads the held-out reader); `--mode freeze-subsets|score-heldout|both` | `test_freeze_stage_never_references_held_out_reader_builder`, `test_modified_frozen_subset_is_rejected`, `test_missing_frozen_subset_fails_closed` | `freeze_stage_excludes_heldout_reader`, `modified_frozen_subset_rejected` |
| 6 | Normalized validation was aggregate-ratio based and its parent denominator excluded missing parents | `validate_normalized_export` is node/reply-level: non-empty source/reply text, real source and per-node timestamps, resolvable parent ids, unique ids, valid labels, child-earlier-than-parent count; `parent_id` coverage divides by **every** reply and `original_order` is never consulted; the audit emits the full §30 field set | `test_missing_parent_blocks_ready`, `test_reply_text_coverage_counts_every_reply`, `test_unresolvable_parent_and_child_earlier_are_reported`, `test_valid_export_is_ready_with_full_30_fields`, `test_duplicate_node_ids_block_ready` | `normalized_parent_coverage_not_false_ready` |
| 7 | A CLI `--normalized-events` could fingerprint a different source than the one loaded | The override is removed: `CRTSER_WEIBO22_NORMALIZED` is the single effective source for loading, validation, `source_fingerprint`, `source.json` and the split; directory fingerprints traverse subdirectories in sorted order | `test_crtser_smoke_env_builds_smoke_root`, `test_source_fingerprint_detects_change` | `manifest_fingerprint_matches_source` |
| 8 | `paths_from_env` passed an unknown `smoke` field; `train_predictors` had a shadowed `run_rotation` | `CRTSER_SMOKE` now only rewrites `out_root` into the smoke namespace; the duplicate `run_rotation` definition is deleted | `test_crtser_smoke_env_builds_smoke_root` | `crtser_smoke_env_resolves` |

## 12. Final verification (protocol closure round)

* `python -m pytest project/cr_tser/tests -q` → **84 passed**.
* `python scripts/cr_tser_verify_pilot.py --mode code` → **52 checks,
  issues = 0** (including the executed protocol-closure semantics).
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (pilot not executed).
* `python scripts/cr_tser_p0_audit.py` → `P0_FAIL`,
  `weibo22_source_of_record = raw_release`. The public KPG release has no
  per-node timestamps, so `P0_PASS` was **not** fabricated.
* `compileall` clean over `project/cr_tser` and `scripts/cr_tser_*.py`.

---

# 13. Code freeze hotfix (code only — formal pilot NOT RUN)

Three integration/protocol defects, all of the "the unit test passes but the
formal aggregator contract does not" kind.

| # | Finding | Fix (location) | Test | Verifier check |
|---|---|---|---|---|
| 1 | `compute_gates` passed only `record["delta"]` into `gate_p4`/`pheme_secondary`, but Stage-B artifacts keep `held_out_reader` at the top level — so three complete rotations reached `rotation_completeness` as three identity-less rows and every P4/PHEME condition failed closed on valid evidence | The aggregator rebuilds the rotation contract, `{"held_out_reader": record["held_out_reader"], **record["delta"]}`, for both the Weibo22 primary and the PHEME secondary; `legacy_diagnostics` also records `participates_in_primary_gate=False` | `test_aggregator_preserves_three_rotation_identity`, `test_aggregator_two_rotations_fail_closed`, `test_aggregator_recognizes_complete_pheme_secondary`, `test_aggregator_incomplete_pheme_never_full_go` | `p4_rotation_identity_preserved` (asserts `distinct_held_out == {glm,internlm,qwen}`, not merely a count), `pheme_rotation_identity_complete`, `incomplete_rotations_fail_closed` |
| 2 | Stage A rewrote `unseen_reader/<dataset>/frozen/rotation_<held>.json` and its `.sha256` on every call, so a re-freeze with a different predictor silently changed what Stage B was scored on | `freeze_subsets` now calls `assert_unfrozen` **before any expensive work**; an existing frozen artifact raises `FrozenSubsetAlreadyExists`. There is deliberately no `--force`: a restart means explicitly removing the unused artifact set. Stage B still only reads/verifies; `--smoke` keeps its own namespace | `test_stage_a_is_write_once` (real first freeze, then a second freeze with a changed predictor, then byte-identity of the JSON and hash), `test_stage_a_refuses_when_a_single_rotation_exists` | `frozen_subset_write_once` (executes the double-freeze and asserts `refused and unchanged`) |
| 3 | B2 existed only as `b2_surface()`, so the PHEME legacy continuity diagnostic never reached an artifact or the report | The PHEME legacy execution stage (`--legacy` freeze) now calls `LegacyPHEMEUtility.b2_items` — the real frozen Proxy path, `classify_selected(proxy, h_source, mean(selected h_i))`, with `p_rumor = p[1]` aligned to TC-DSCR — and writes `unseen_reader/pheme/b2_legacy_diagnostic.json` (dataset, `B2_legacy_tcdscr`, `diagnostic_only`, `participates_in_primary_gate=false`, frozen fingerprint, frozen subset hashes, per-cutoff metrics, per-snapshot rows). `compute_gates` reads it into `legacy_diagnostics.b2_legacy_tcdscr` (rows summarised, never re-scored) and `write_report` prints it. S6 is untouched | `test_b2_surface_outputs_enter_pheme_artifact`, `test_pheme_legacy_freeze_writes_b2_artifact`, `test_b2_artifact_does_not_change_weibo22_gates`, `test_b2_is_never_part_of_p3_or_p4_comparison` | `b2_surface_enters_pheme_artifact`, `b2_never_changes_gates` (identical gate dicts with/without the artifact), `b2_excluded_from_p3_comparison`, `b2_excluded_from_p4_comparison` |

## 14. Final verification (code freeze hotfix)

* `python -m pytest project/cr_tser/tests -q` → **94 passed** (10 new
  integration tests, including the end-to-end synthetic aggregation:
  three Stage-B rotation artifacts → `compute_gates()` → P4 completeness
  `true`; three PHEME artifacts → PHEME completeness `true`).
* `python scripts/cr_tser_verify_pilot.py --mode code` → **60 checks,
  issues = 0** (the eight new checks execute the contracts rather than grep
  for them).
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (pilot not executed).
* `python scripts/cr_tser_p0_audit.py` → `P0_FAIL` /
  `WEIBO22_TEMPORAL_UNAVAILABLE`; no timestamp-bearing Weibo22 export is
  configured, so `P0_PASS` was **not** fabricated.
* `compileall` clean.

---

# 15. P0 prerequisite resolution and preflight (code only — formal pilot NOT RUN)

The frozen plan's P0 asks whether a strict temporal Weibo22 pilot can be run on
real data at all. This round resolves the data question and runs the readiness
preflight; it runs no pilot stage.

## 15.1 Weibo22 source resolution

The local `data_raw/weibo22` archives were verified **byte-for-byte** against
the official KPG release: each archive's git blob SHA-1 equals the blob hash
recorded from `github.com/kkkkk001/KPG` at `data/Weibo` during the preflight
(`Weibo_label_All.zip` → `9550ee43…`, `data.TD_RvNN.vol_5000.zip` →
`46f9af6a…`). Release tag `v1` (2024-06-28, "Release the dataset Weibo22")
carries no assets, `data/Weibo` was added by a single commit (`a9019754d8`,
2024-04-09), and the repository holds only these two archives for Weibo — i.e.
the complete public Weibo22 release.

Raw-field verdict (plan §4.1): event/source id, binary label, node id and
parent id are present; **source text, reply text, source timestamp and per-node
timestamp are absent**. Nodes carry `vol_5000` bag-of-words indices
(`index:frequency`), not text, and there is no timestamp column in either file.
The label file carries exactly `2087` rumor + `2087` non-rumor source events,
matching the CUHK dataset description, so the dataset identity is settled and
only the temporal fields are unpublished.

Rejected candidates, with provenance recorded in
`results/cr_tser/p0/weibo22_source_audit.json`: the CUHK RDM project page
(description of the dataset, no downloadable data package); Zenodo
"Weibo-Covid-19" (doi:10.5281/zenodo.13787781 — a different, keyword-filtered
vaccine-discourse corpus with no 2087+2087 rumor labels); and Ma-Weibo (plan
§4.3 forbids substituting it for Weibo22).

## 15.2 P0 verdict

`python scripts/cr_tser_p0_audit.py --weibo22-raw data_raw/weibo22/extracted --readers --sanity`
produced the readiness file from code, unedited:

```text
P0 = P0_FAIL
weibo22_temporal = WEIBO22_TEMPORAL_UNAVAILABLE
weibo22_source_of_record = raw_release
readers_ready = false; ab_sanity_ok = false
```

Two further prerequisites are also unmet on this machine and are recorded
rather than worked around: `CRTSER_PHEME_RAW` is unset, so the PHEME smoke is
`PHEME_RAW_MISSING`; and none of the three frozen readers is present locally
(no `Qwen/Qwen3-8B`, `zai-org/glm-4-9b-chat-hf` or
`internlm/internlm3-8b-instruct` weights, empty HF-cache matches), so the frozen
reader load/hash + A/B teacher-forced preflight cannot run and is reported as
`MODEL_PATH_MISSING`. No substitute model was used.

## 15.3 Evidence package

`scripts/cr_tser_p0_preflight.py` (new, read-only; no frozen code changed)
writes the package under `results/cr_tser/p0/`: `P0_REPORT.md`,
`weibo22_provenance.json`, `weibo22_source_audit.json`,
`weibo22_temporal_audit.json`, `weibo22_audit.{json,md}`,
`weibo22_smoke.json`, `pheme_adapter_smoke.json`, `environment_probe.json`,
`reader_identity_{qwen,glm,internlm}.json`,
`ab_scoring_{qwen,glm,internlm}.json`, alongside the frozen-script outputs
`reader_audit.json`, `label_scoring_sanity.json`, `p0_readiness.json` and
`P0_READINESS.md`.

## 15.4 Verification

* `python -m pytest project/cr_tser/tests -q` → **94 passed** (frozen code
  untouched).
* `python scripts/cr_tser_verify_pilot.py --mode code` → **60 checks,
  issues = 0**.
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (no manifests; the pilot has not run and must not).
* `compileall` clean.
* No manifest, utility label, predictor training, Stage-A freeze, held-out
  evaluation or P1–P4 run was performed in this round.

---

# 16. V2-M0 — dataset / orchestration protocol migration (experiments NOT RUN)

The V1 candidate primary dataset was rejected on evidence and the dataset
protocol was amended. Only dataset/orchestration code, tests and the verifier
changed; no experiment ran.

## 16.1 Decision recorded

```text
PRIMARY_DATASET   = Ma-Weibo
SECONDARY_DATASET = PHEME
WEIBO22           = REJECTED_PRIMARY_CANDIDATE (V1 evidence retained)
```

The amendment text lives in `docs/CR_TSER_DATASET_PROTOCOL_AMENDMENT_V2.md`;
`docs/CR_TSER_FEASIBILITY_PILOT_PLAN.md` keeps all of its original content and
only gained a non-destructive superseded notice at the top. The V1 P0 evidence
under `results/cr_tser/p0/weibo22_*` is untouched and now reads as
*candidate-dataset feasibility rejection*, not as a CR-TSER method failure.

## 16.2 Files changed

* `project/cr_tser/config/pilot_config.py` — `PRIMARY_DATASET`,
  `SECONDARY_DATASET`, `V2_DATASETS`, `REJECTED_PRIMARY_CANDIDATE`,
  `V2_MIN_VIABLE_EVENTS` (= 170), `V1_RESULTS_ROOT` / `V2_RESULTS_ROOT`;
  `PilotPaths` gained `maweibo_raw` / `maweibo_labels` and now defaults
  `out_root` to `results/cr_tser_v2`.
* `project/cr_tser/data/maweibo_bridge.py` — **new**. Reuses the audited
  `tcdscr.data.maweibo_adapter` (timestamp only from raw `t`, `original_text`
  with `text` fallback, raises on missing/multi root) and adds only the V2
  eligibility contract (`EMPTY_TEXT`, duplicate-id fail-closed), viability,
  cycle counting and the §9 P0-A audit.
* `project/cr_tser/data/source_manifest.py` — composite Ma-Weibo fingerprint
  (`raw_json` + `label_file` + `combined_source_sha256`) with the top-level
  aliases so the existing generic identity guard still fails closed.
* `project/cr_tser/data/snapshot_bridge.py` — added `valid_reply_parent_units`
  and `count_parent_cycles` (shared graph helpers); no existing behaviour
  changed.
* `project/cr_tser/intervention/evidence_units.py` — `build_evidence_units`
  skips nodes whose `status != "VALID"` (amendment §7). Unit structure, token
  accounting and rendering are unchanged.
* `project/cr_tser/data/pilot_split.py` — `viable_event_ids` now requires a
  `VALID` reply with a resolved, causally ordered parent inside one cutoff and
  rejects cyclic events (amendment §8).
* `scripts/cr_tser_common.py` — `load_dataset_events` / `dataset_registry` gained
  the `maweibo` branch through the bridge; new `default_out_root()` anchors the
  formal root to `results/cr_tser_v2`.
* `scripts/cr_tser_build_manifests.py`, `cr_tser_generate_labels.py`,
  `cr_tser_train_predictors.py`, `cr_tser_run_selection.py` — dataset choices
  are now `("maweibo", "pheme")` and the formal root is the V2 namespace.
* `scripts/cr_tser_run_pilot.py` — imports `PRIMARY_DATASET` /
  `SECONDARY_DATASET` from the config instead of hardcoding Weibo22.
* `scripts/cr_tser_p0_audit.py` — `--protocol v2` (default) implements P0-A
  Ma-Weibo integrity + viable ≥ 170, P0-B PHEME smoke, P0-C readers/A-B;
  `--protocol v1` keeps the historical Weibo22 preflight and always writes to
  the V1 namespace.
* `scripts/cr_tser_verify_pilot.py` — `--protocol v1|v2` (default v2) adds 17
  executed V2 checks without removing any V1 check.
* `docs/CR_TSER_FEASIBILITY_PILOT_PLAN.md` — superseded notice only.

## 16.3 Tests added

`project/cr_tser/tests/test_v2_dataset_protocol.py` (19 tests): dataset roles,
bridge→audited-adapter reuse, timestamp==raw `t`, `original_text` fallback,
`EXTERNAL_PARENT` / `EMPTY_TEXT` / `TEMPORAL_INVALID_NODE` handling, duplicate-id
and multi-root fail-closed, viability per cutoff, cycle rejection, viability
before split, composite fingerprint + replacement fail-closed, P0 blocked below
170 viable events / by PHEME smoke / by missing readers, Ma-Weibo B2-S6
forbidden, `common.load_dataset_events("maweibo")`, and the full §9 audit field
set. Existing aggregator/review tests were migrated from Weibo22 to Ma-Weibo as
the primary namespace.

## 16.4 Verifier checks added

`v2_primary_dataset_is_maweibo`, `weibo22_absent_from_v2_loops`,
`maweibo_bridge_reuses_audited_adapter`, `maweibo_timestamp_from_raw_t`,
`maweibo_fingerprint_is_composite`, `maweibo_source_replacement_fails_closed`,
`maweibo_p0a_fields_complete`, `maweibo_fixture_all_events_viable`,
`empty_text_not_an_evidence_unit`, `maweibo_viability_filter_runs`,
`viable_lt_170_blocks_p0`, `maweibo_b2_s6_forbidden`,
`v2_namespace_separate_from_v1`, `default_out_root_is_v2`,
`v2_p0_audit_runs_synthetic`, `maweibo_primary_path_has_no_legacy_artifacts`,
`v2_scientific_constants_unchanged`. They execute the contracts (synthetic
fixtures, real code paths) rather than grepping for strings.

## 16.5 Core research code untouched

No change was made to `models/bitte.py`, the utility head definitions, the
intervention utility equation, the structured-intervention definitions
(I2–I5), the reader scoring contract, the B0/B1/B3 architectures, the training
loss weights, the optimizer protocol, LORO, the robust-selector equation, the
bootstrap protocol, or any P1–P4 threshold. The amendment's frozen numbers are
re-asserted by `v2_scientific_constants_unchanged`
(seed 7319; 80/50/15/25; cutoffs 15/60/360; the three frozen readers;
0.05/0.10/0.02/0.02/0.01/−0.005/−0.005).

## 16.6 Historical Ma-Weibo firewall

Ma-Weibo gets no B2 and no S6: `legacy_arm_enabled()` is PHEME-only, and the
legacy TC-DSCR scorer stays behind the PHEME-only legacy gate in the selection
runner. The bridge never references a checkpoint, selector, proxy or historical
utility file. B2/S6 remain PHEME-only, diagnostic-only, and never enter a
Weibo22 or Ma-Weibo primary gate.

## 16.7 Verification (V2-M0)

* `python -m pytest project/cr_tser/tests -q` → **113 passed**.
* `python scripts/cr_tser_verify_pilot.py --mode code --protocol v2` →
  **77 checks, issues = 0**.
* `python scripts/cr_tser_verify_pilot.py --mode code --protocol v1` →
  **issues = 0** (historical V1 capability preserved).
* `python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2` →
  **issues = 0, pending = 1** (no manifests; V2 P0 has not run).
* `python -m compileall project/cr_tser scripts` → clean.
* Formal P0, reader loading, A/B real-reader sanity, manifests, utility labels,
  predictor training, Stage-A freeze, held-out evaluation and P1–P4 were **not**
  run.

## 16.8 Resulting state

```text
CR_TSER_V2_PROTOCOL_MIGRATION = CODE_READY
V2_P0 = NOT RUN
P1-P4 = NOT RUN
FORMAL PILOT = NOT RUN
```

---

# 17. V2-M0 closure hotfix (dataset/orchestration only, experiments NOT RUN)

Five issues found when the V2-M0 migration was reviewed. Only dataset /
orchestration / report code, tests and the verifier changed.

## 17.1 Eligibility filtering is dataset-aware again

`build_evidence_units()` had been changed to skip every `status != "VALID"`
node for **all** datasets, which silently altered the V1/PHEME evidence-unit
semantics. That unconditional filter is gone. The contract is now explicit and
dataset-scoped:

* `pilot_config.V2_STRICT_ELIGIBILITY_DATASETS = (PRIMARY_DATASET,)` is the
  single source of truth;
* `cr_tser_common.eligibility_for(dataset)` returns `"v2_strict"` for Ma-Weibo
  and `None` otherwise;
* `snapshot_artifacts()` stamps `snapshot["eligibility"]` from that contract, so
  `build_evidence_units()` reads **explicit snapshot metadata** — it never
  guesses a dataset from a string.

Ma-Weibo therefore enforces the V2 node contract while PHEME keeps the V1
behaviour (its historical missing/external-parent evidence is no longer
deleted). Regression:
`test_same_non_valid_reply_excluded_only_for_maweibo`,
`test_eligibility_for_is_explicit`, `test_pheme_keeps_v1_evidence_units`.

## 17.2 Strict Ma-Weibo Reply–Parent eligibility

A V2 textual Reply–Parent unit now requires **all** of: reply
`status == "VALID"`; parent present in the event; parent `status == "VALID"`;
non-empty reply text; non-empty parent text; `parent.timestamp <=
reply.timestamp`. A `VALID` child can no longer smuggle an `EMPTY_TEXT` /
`MISSING_PARENT` / `EXTERNAL_PARENT` parent into a unit. The same predicate is
shared by `valid_reply_parent_units()`, the Ma-Weibo viability calculation
(`viable_cutoffs` / `event_viable`), `pilot_split.viable_event_ids(...,
eligibility="v2_strict")` and the `"v2_strict"` branch of
`build_evidence_units()`. Tests:
`test_valid_child_with_empty_text_parent_has_no_unit`,
`test_valid_child_with_invalid_status_parent_has_no_unit`,
`test_valid_child_with_valid_textual_parent_has_unit`,
`test_invalid_parent_does_not_add_to_viable_pool`.

## 17.3 PHEME smoke is fail-closed

`status = "OK"` no longer depends on `future_leakage == []` alone. `pheme_smoke`
now verifies, on a real event, that the raw event loads, source text, reply
text, timestamps and parent relations are recoverable, all three 15m/1h/6h
snapshots build, and leakage is zero; otherwise it returns
`PHEME_SMOKE_FAIL` with an explicit `failures` list. A few unresolved parents
are normal in PHEME, so the contract requires at least one *resolved* relation
rather than one per reply. Tests: the six `test_pheme_smoke_*` cases.

## 17.4 V2 aggregator / report carries no V1 primary semantics

`compute_gates()` no longer reads `weibo22_reason`; the P0 detail is generated
from the V2 readiness fields (`maweibo_integrity`, `maweibo_viable`,
`pheme_smoke`, `readers_ready`, `ab_sanity_ok`) via `_p0_detail()`. The report
title is now “CR-TSER V2 Feasibility Pilot — Ma-Weibo primary”, the gate
section says “Ma-Weibo primary”, the B2/S6 copy says they never enter a
Ma-Weibo primary gate, and the module docstring states P1–P4 are
Ma-Weibo-primary with Weibo22 a rejected candidate. Test:
`test_v2_report_is_maweibo_primary` (asserts `primary = maweibo`,
`secondary = pheme`, no “Weibo22 primary”, and that a planted `weibo22_reason`
is never surfaced).

## 17.5 V1 namespace restored and frozen

`results/cr_tser/verifier/code_verify.json` was restored byte-for-byte to the
frozen baseline `b5cfa168245c46347e0f12833a4ef1017e93e94f`. The verifier now
writes **only** under `results/cr_tser_v2/verifier/` (a `--protocol v1` run
writes `v1_*.json` there), and pins the V1 artifacts
(`v1_historical_immutable:*`) plus the whole V1 verifier directory
(`v1_verifier_dir_immutable`) so a V2 run can never rewrite V1 history.

The Ma-Weibo audit ratios `parent_resolution_coverage`, `missing_parent_rate`
and `external_parent_rate` now use the **non-source reply count** as
denominator (plus new `reply_node_count` / `reply_status_counts` fields), so
the source node no longer dilutes them.

## 17.6 Verification (closure hotfix)

* `python -m pytest project/cr_tser/tests -q` → **128 passed** (15 new).
* `python scripts/cr_tser_verify_pilot.py --mode code --protocol v2` →
  **issues = 0**.
* `python scripts/cr_tser_verify_pilot.py --mode code --protocol v1` →
  **issues = 0** (writes `results/cr_tser_v2/verifier/v1_code_verify.json`).
* `python scripts/cr_tser_verify_pilot.py --mode pilot --protocol v2` →
  **issues = 0, pending = 1** (no manifests; V2 P0 not run).
* The V1 verifier directory hash is unchanged after a V2 verifier run
  (`efae6c5a…`), proving the historical namespace is read-only.
* `python -m compileall project/cr_tser scripts` → clean.
* No V2 P0, reader loading, A/B real-reader sanity, manifest, utility label,
  predictor training, Stage-A freeze, held-out evaluation or P1–P4 was run.

# 18. V2-P0 preflight execution (real run, formal pilot NOT RUN)

## 18.1 Baseline

Executed from the approved frozen baseline
`fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`, per
`docs/CR_TSER_V2_P0_EXECUTION_PLAN.md`. **No code was changed in this round.**

Pre-P0 health: `pytest project/cr_tser/tests -q` → 128 passed;
`verify_pilot.py --mode code --protocol v2` → issues = 0;
`compileall project/cr_tser scripts` → clean.

## 18.2 Verdict

```text
P0_FAIL
```

Sole blocker: the three frozen readers could not be resolved. This machine is
the **lightweight verification node** — the reader weights are server-side
resources (`resources/resource_manifest.md`), every `CRTSER_*_MODEL` variable
was unset, and the only GPU is a 4.00 GiB GTX 1050 Ti, which cannot host an
8B/9B bf16 reader in any case. The protocol forbids substituting a smaller /
quantized / API checkpoint, so P0-F and P0-G are a deployment-location result,
not a data or protocol finding. The heavyweight reader preflight must run on
the synchronized server.

## 18.3 Data-side checkpoints PASSED (real numbers)

Source of record — `CRTSER_MAWEIBO_RAW` =
`…/rumor_detection/data/dataset/Ma-WeiBo/Weibo`,
`CRTSER_MAWEIBO_LABELS` = `…/Ma-WeiBo/Weibo.txt` (original release layout):

| fingerprint | value |
|---|---|
| raw JSON files / bytes | 4664 / 3,995,877,128 |
| raw sha256 | `c6afd1c50a6cde8c27137d367d8e420b4a3afc43e017e3b19846e8b001ca1a27` |
| label sha256 | `032f10e2175fa461203bdba77ef2a492e0ebde1cdd24b9bcd2100e307bc6b8e4` |
| combined_source_sha256 | `b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d` |

Audit (recomputed, historical reference NOT assumed): 4664 raw / 4664 parsed /
0 invalid; labels 2351×0, 2313×1; source-text 1.0, reply-text
0.9999955274833517, timestamp 1.0, parent-resolution 0.9999918442343473;
duplicate_ids 0, cycles 0, multi-root 0; missing-parent 0, external-parent 30;
temporal-invalid 1; `events_with_ge1_valid_reply_parent_unit` 4663;
verdict `MAWEIBO_READY`.

Viability (V2 strict Reply–Parent rule): 15m 4296 / 1h 4486 / 6h 4591 →
`total_viable_events = 4591 >= 170`. Split untouched.

Snapshot integrity: 8 sampled viable events × 3 cutoffs — source present,
0 future leakage, no cap (`MAX_NODES_CAP=None`), parent visible only when
present, `parent.ts <= child.ts`, order = `(timestamp, original_order)`,
unreachable 0. One sample (`10031994215`, 941 nodes at 6h) exceeds the retired
1021 cap without truncation.

PHEME smoke: `status=OK` (6425 events; sample `552783238415265792`; source /
reply text, timestamps, ≥1 resolved parent relation, 15m/1h/6h snapshots,
zero leakage).

## 18.4 Evidence package

Under `results/cr_tser_v2/p0/`: `p0_readiness.json`, `P0_READINESS.md`,
`P0_EVIDENCE_PACKAGE.md`, `maweibo_audit.json`,
`maweibo_source_fingerprint.json`, `snapshot_integrity.json`,
`pheme_smoke.json`, `reader_audit.json`, `label_scoring_sanity.json`,
`environment.json`, `v1_namespace_baseline.json`. The verdict was produced by
the frozen entrypoint (`cr_tser_p0_audit.py --protocol v2 --readers --sanity`);
no artifact was hand-edited.

## 18.5 Verification and namespace protection

* `verify_pilot.py --mode code --protocol v2` → issues = 0.
* `verify_pilot.py --mode pilot --protocol v2` → issues = 0, pending = 1
  (`pilot_artifacts`, correct while P1–P4 have not run).
* `results/cr_tser/` (V1) directory SHA-256 identical before and after the
  whole round: `c55d4c9429483b314982b77816166258c197ef62a45d5e7aebe02ba976b79558`.
* All V2 output stayed under `results/cr_tser_v2/`.

## 18.6 Not run

formal manifest freeze, intervention generation, utility labels, predictor /
single-reader / shared-residual training, Stage-A freeze, held-out reader
evaluation, P1–P4, formal pilot report. No threshold, reader, cutoff, split,
eligibility rule or method was changed; no fallback was designed.

# 19. V2-P0 server completion (real run, formal pilot NOT RUN)

## 19.1 Dual-environment contract

```text
LOCAL  env = pytorch   (Windows, E:\anaconda\envs\pytorch, GTX 1050 Ti 4.00 GiB)
SERVER env = DGPA      (Linux, /data/jyz/envs/DGPA/bin/python, RTX 4090 23.52 GiB)
server root = /data/jyz/next/llm/ ; workspace = /data/jyz/next/llm/cr_tser_ws
```

Local (`pytorch`) ran only lightweight checks: git state at `b5f91e1`,
`pytest project/cr_tser/tests -q` → 128 passed,
`verify_pilot.py --mode code --protocol v2` → issues = 0, `compileall` clean.
No reader was loaded locally. Every reader load, identity hash and A/B score
ran on the server.

## 19.2 Server checkpoints

* **S1** — repository cloned to `/data/jyz/next/llm/cr_tser_ws`, HEAD
  `b5f91e1`; `git diff fa8ddbb..b5f91e1` = docs + `results/cr_tser_v2/p0/*`
  only, so `project/cr_tser` and `scripts` are unchanged from the frozen
  baseline.
* **S2** — reader acquisition. `huggingface.co` is unreachable from the
  server, but `hf-mirror.com` answers HTTP 200, so all three checkpoints were
  obtained **on the server**, under `/data/jyz/next/llm/model/`:
  `qwen3-8b` (`server_existing`, exposed at the llm root by a symlink to
  `/data/jyz/next/model/qwen3-8b`), `glm-4-9b-chat-hf` and
  `internlm3-8b-instruct` (`server_download`). No local non-C-drive transfer
  was used, so no local temporary weight path exists.
  Official integrity: weights verified as content sha256 against the mirror's
  `x-linked-etag`; non-LFS files as git blob sha1 against the HF API
  `blobId`. All three readers `ALL_MATCH`. GLM landed initially without
  `model-00003-of-00004.safetensors`; the exact shard was re-fetched and
  verified (`b5e6131e…`), completing the 4-shard set.
* **S3** — Ma-Weibo composite fingerprint `b982076d…` reproduced exactly on
  the server (raw 4664 files / 3,995,877,128 bytes, `c6afd1c5…`; labels
  `032f10e2…`). No source drift.
* **S4/S5** — see 19.3.
* **S6** — `python scripts/cr_tser_p0_audit.py --protocol v2 --readers
  --sanity` produced the verdict; nothing was hand-edited.
* **S7** — `--mode code` issues = 3, `--mode pilot` issues = 0 pending = 1.

## 19.3 Verdict and single blocker

```text
P0_FAIL
```

```text
qwen      loaded=true   boundaries_ok=true  identical_predictions=true  identity_rate=1.0
glm       loaded=true   boundaries_ok=true  identical_predictions=true  identity_rate=1.0
internlm  loaded=false  ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```

`internlm/internlm3-8b-instruct` cannot load under DGPA: its official
`modeling_internlm3.py` imports `LossKwargs`, which transformers **4.57.6
removed** (absent from both `transformers.utils` and
`transformers.utils.generic`), and this version also ships no built-in
`internlm3` model type — so the frozen reader's `trust_remote_code=True` path
fails. An official-checkpoint ↔ mandated-environment incompatibility, not a
data or protocol finding. No model was substituted, no threshold changed, no
patch of the official remote code applied.

Data side re-passed on the server: `MAWEIBO_READY`, labels 2351×0 / 2313×1,
viable events 15m 4296 / 1h 4486 / 6h **4591 >= 170**, snapshot integrity
(8 viable events × 3 cutoffs) source-present, 0 leakage, no cap,
`parent.ts <= child.ts`, exact `(timestamp, original_order)` ordering; PHEME
smoke `OK`.

## 19.4 V1 line-ending note (code verifier issues = 3)

On the server the three `v1_historical_immutable` /
`v1_verifier_dir_immutable` checks fail, but the V1 **content** is untouched:
the stored git blobs are identical on both hosts
(`code_verify.json` → `0d0e601e292481020b979ff8f5531c1920b59812`,
`pilot_verify.json` → `587d163f5c973d60c4875c7388130308fa57cb5f`). The local
checkout has `core.autocrlf=true` (8604 vs 8298 bytes = one `\r` per line) and
the frozen expectations were computed on that CRLF worktree. V1 is byte-stable
on the server across the whole round (`b1b348b8…` pre == post).

## 19.5 Not run

formal manifests, formal intervention generation, utility labels, predictor /
single-reader / shared-residual training, Stage-A freeze, held-out reader
evaluation, P1–P4, formal pilot report. No fallback was designed. The only
admissible remedy is an environment/research decision about the R3 runtime.

# 20. V2-P0 compatibility closure (P0_PASS, formal pilot NOT RUN)

Round code commit `808fd83`; baseline entering the round `99dab76`. Two
engineering problems, no science change.

## 20.1 InternLM3 ↔ Transformers compatibility

`internlm/internlm3-8b-instruct`'s official `modeling_internlm3.py` imports
`LossKwargs`, which transformers **4.57.6 removed** (absent from both
`transformers.utils` and `transformers.utils.generic`), and 4.57.6 ships no
built-in `internlm3`. Because the frozen `InternLMReader` uses
`trust_remote_code=True`, the R3 reader could not load.

Resolution — rollback-safe isolation, DGPA untouched:

* `dgpa_pip_freeze_before.txt` snapshots DGPA (transformers 4.57.6,
  tokenizers 0.22.2, torch 2.7.1+cu128) before anything changes;
* `transformers==4.53.3` plus `tokenizers>=0.21,<0.22` are installed into
  `/data/jyz/next/llm/.cr_tser_v2p0/tf453` and injected with `PYTHONPATH`;
* DGPA's own `site-packages` is never modified, so rollback is
  `unset PYTHONPATH`.

Result under 4.53.3 (`LossKwargs import: OK`): all three readers load
(`ALL_LOADED True`), and the frozen teacher-forced A/B sanity passes for all
three — qwen A=20/B=0, glm A=0/B=20, internlm A=17/B=3, each with
`boundaries_ok=true`, `identical_predictions=true`, `identity_rate=1.0`.
Qwen and GLM were re-scored as the minimal regression: the frozen scoring
contract is unaffected by the version change.

The official remote code was **not** modified, `LossKwargs` was **not**
patched, and no model was substituted.

## 20.2 Cross-platform V1 immutability

`_verify_v1_historical_immutable` pinned raw worktree bytes, so a Windows
checkout (`core.autocrlf=true`) and a Linux checkout could never both pass
even though the stored git blobs are identical; `_dir_sha256` also leaked
`os.sep` into the digest via `os.path.relpath`.

Identity is now the **canonical LF digest**: `_canonical_bytes` normalizes
CRLF→LF, `_canonical_sha256` digests a file, and `_dir_sha256` additionally
POSIX-normalizes relative paths. New pins:

```text
code_verify.json   134b32e2f00765ab221ee710a8def67ede91b009b97da95beecfac4463e22f0e
pilot_verify.json  c81000a97ed82c323c1953049d83fa864f9eba2b83640c6211af810076a061af
verifier dir       aba9f5169d63b39c7e423d886e01ec5e40547f43aa0f46a5c937bf73df4d90c2
```

These equal the digests the Linux worktree already produced, so the pins are
content-based and platform-stable; a real edit still changes them. New
`project/cr_tser/tests/test_v1_immutability_cross_platform.py` (11 tests)
covers CRLF/LF equivalence, content-change detection, directory digests, the
live pins, pass-on-either-line-ending, tamper-fails and missing-file-fails.

## 20.3 Final verification

```text
LOCAL/pytorch   pytest project/cr_tser/tests -q            -> 139 passed
LOCAL/pytorch   verify_pilot.py --mode code --protocol v2  -> issues = 0
LOCAL/pytorch   compileall project/cr_tser scripts         -> clean
SERVER/DGPA     verify_pilot.py --mode code --protocol v2  -> issues = 0
SERVER/DGPA     cr_tser_p0_audit.py --protocol v2 --readers --sanity
                                                          -> P0_PASS (exit 0)
SERVER/DGPA     verify_pilot.py --mode pilot --protocol v2 -> issues = 0, pending = 1
```

`P0_PASS` prerequisites: `MAWEIBO_READY`, viable 4591 ≥ 170, PHEME smoke OK,
all three readers loaded, all boundaries OK, all repeated scoring
deterministic.

One operational note: the server's first `git fetch` in this round failed with
`GnuTLS recv error (-54)` and the workspace silently stayed on `b5f91e1`,
which reproduced the old three CRLF failures. After retrying the fetch
(`b5f91e1..808fd83`) the chain was re-run in full; the reported results are
from that corrected run.

## 20.4 Not run

formal manifests, utility labels, predictor training, Stage A/B, held-out
reader evaluation, P1–P4, formal pilot. `P0_PASS` depends on the 4.53.3
`PYTHONPATH` override — every later stage that loads the readers must use it,
or DGPA must be moved to 4.53.3 by the research owner.

# 21. V2-P1A — blocked by a manifest-freeze defect (NO labels generated)

Baseline `c4773a3`. Round stopped at the Ma-Weibo formal manifest build.

Local checks passed (139 tests, code verifier `issues = 0`, `compileall`
clean), the server workspace was synced to `c4773a3`, the 4.53.3 overlay was
active, the Ma-Weibo composite fingerprint reproduced exactly
(`b982076d…`, no drift) and **PHEME manifests built successfully** (270
snapshots / 3412 interventions / 5669 viable events).

`python scripts/cr_tser_build_manifests.py --dataset maweibo` then failed:

```text
scripts/cr_tser_build_manifests.py", line 108, in <dictcomp>
    "source": {k: source[k] for k in ("kind", "path", "sha256",
KeyError: 'exists'
```

`source_fingerprint()`'s `maweibo_composite` branch supplies the top-level
aliases `path` / `sha256` / `bytes` / `n_files` but **not `exists`**, which
`build_manifests` requires; the non-composite branch delegates to
`fingerprint_path()` and therefore always has it. `source["exists"]` has
exactly one consumer in the repository, so only the Ma-Weibo (composite) path
is affected — P0 never consumed it, which is why the V2-M0/P0 rounds passed
while this stage had never actually run for Ma-Weibo.

Per plan §10 the round stopped, evidence was preserved
(`results/cr_tser_v2/p1a/`), and no patch was applied. A re-run after an
approved fix needs no `--force` (only `source.json` exists under
`manifests/maweibo/`), though PHEME would need `--force` to rebuild. No
utility label was generated, so nothing is frozen yet.

# 22. V2-P1A manifest hotfix — unified composite fingerprint contract

Baseline `c3c08dc`. Fix confined to `project/cr_tser/data/source_manifest.py`:
the `maweibo_composite` branch of `source_fingerprint()` now carries the same
top-level field set as every other source kind (`path`, `exists`, `sha256`,
`bytes`, `n_files`), with

```python
"exists": bool(raw.get("exists")) and bool(labels.get("exists")),
```

i.e. a real conjunction of the raw directory and the label file — never a
hard-coded `True`. The builder is untouched: no Ma-Weibo special case was
added at the consumer. `raw_json`, `label_file` and `combined_source_sha256`
are preserved, and the composite hash algorithm and the frozen Ma-Weibo
identity (`b982076d…`) are unchanged.

Note on the alias semantics: `fingerprint_path()` reports `exists` for *path
presence*, so a raw directory that exists but holds no files still reports
`True`; that is the pre-existing single-source semantics and was deliberately
not tightened here. Source replacement stays fail-closed regardless, because
`assert_same_source()` compares `kind` / `sha256` / `n_files`.

New regression tests (`project/cr_tser/tests/test_v2_p1a_manifest_hotfix.py`):

```text
composite fingerprint exposes the unified top-level field set
exists = true when raw + labels exist
exists = false when either half is missing
real Ma-Weibo manifest build writes source.json / event_split.json /
  snapshot_manifest.jsonl / intervention_manifest.jsonl / hashes.json
composite replacement still raises SourceIdentityError, including on a
  --force rebuild against the swapped source
```

No split seed/sizes, cutoff, eligibility, fingerprint semantics, reader
contract, utility definition or P1–P4 threshold was touched.

LOCAL/pytorch: `143 passed`, code verifier `issues = 0`, `compileall` clean.

# 23. V2-P1A execution — manifests frozen, utility labels partial (GLM blocked)

Baseline `c3c08dc`, hotfix `57895a3`. Full detail in
`results/cr_tser_v2/p1a/P1A_MANIFEST_UTILITY_LABEL_REPORT.md`.

Frozen manifests (server-authoritative hashes committed with the round):

```text
maweibo  270 snapshots / 3839 interventions / 12 zero-reply / 4591 viable
         max_snapshot_nodes 24192 (the retired MAX_NODES=1021 cap is not
         applied); source.json 80b07954…, event_split.json 24e18ed1…,
         snapshot_manifest.jsonl 611cb9c6…,
         intervention_manifest.jsonl f37c3ffc…, hashes.json fac953a5…
pheme    not rebuilt; byte-identical to the previous round
both     split 80/50/15/25 seed 7319, label-balanced, event-disjoint,
         cutoffs 15/60/360 only, no future-node leak, no cap hit
```

Ma-Weibo's composite identity reproduced exactly
(`b982076d…`, no drift). Utility labels were then generated one reader at a
time on SERVER/DGPA under the transformers 4.53.3 overlay: `maweibo/qwen`,
`maweibo/internlm`, `pheme/qwen` and `pheme/internlm` all completed and passed
the plan §8 audit (3202 / 3202 / 2879 / 2879 rows, zero duplicate keys, zero
missing fields, zero NaN/Inf, reader identity matching the frozen manifest). No
`CacheIdentityMismatch` occurred.

`maweibo/glm` and `pheme/glm` are **blocked by shared-GPU memory contention**,
not by a code defect: GLM-4-9B needs ≈19.3 GiB in bf16 while another user's six
python jobs hold ~3.3 GiB of the 24 GiB RTX 4090 and return immediately after
any short idle window, so the 674 MiB `logits.float()` peak cannot be
allocated. Two attempts OOM'd (23:08:33 after 121 rows, 01:03:30 after 0 rows);
tracebacks are preserved under `.cr_tser_v2p0/logs/` and
`results/cr_tser_v2/p1a/logs/`. The model, dtype, token budget, cutoffs and
protocol were not changed; the remaining GPU work waits for a sustained free
window and the retry sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(the allocator setting torch itself recommends), which changes no numeric
computation. The round stops for approval with those two pairs pending.

Verifiers at `57895a3` on the server: code `issues = 0, pending = 0`; pilot
`issues = 0, pending = 3` (the P1–P4 artifacts this round must not produce).
Predictor training, Stage A/B, held-out-reader scoring and P1–P4 were not run.




