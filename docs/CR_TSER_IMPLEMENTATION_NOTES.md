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

