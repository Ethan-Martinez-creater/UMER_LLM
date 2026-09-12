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

* `python -m pytest project/cr_tser/tests -q` → **39 passed** (includes all 23
  plan §35 test names).
* `python scripts/cr_tser_verify_pilot.py --mode code` → **issues = 0**.
* `python scripts/cr_tser_verify_pilot.py --mode pilot` → **issues = 0,
  pending = 1** (pilot not executed, as instructed).
* `python scripts/cr_tser_p0_audit.py` → `P0_FAIL` (Weibo22 temporal
  unavailable; reader paths unset locally), written to `results/cr_tser/p0/`.
* `compileall` clean over `project/cr_tser` and `scripts/cr_tser_*.py`.

No formal pilot experiment was run.
