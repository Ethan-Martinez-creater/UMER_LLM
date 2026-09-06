# TC-DSCR V2 Implementation Notes

Implementation record for Stage A (Code Complete) of
`TC_DSCR_V2_FORMAL_EXECUTION_PLAN.md`. Every section maps to the plan; the
final section lists deviations, which are engineering decisions on
under-specified points, not protocol changes.

## Git

- commit: recorded in the follow-up commit immediately after the Code
  Complete submission (a commit cannot contain its own hash)
- branch: `main`, repository `Ethan-Martinez-creater/UMER_LLM`

## Implemented Modules

All code lives in the isolated package `project/tcdscr/` (frozen UMER code in
`project/src/` is untouched) plus entry points `scripts/tcdscr_*.py`:

| Plan module | Files |
|---|---|
| Config (§4/6/9/11/17–21 constants) | `config/schema.py`, `config/pheme.yaml`, `config/maweibo.yaml` |
| M1 Dataset Adapters (§13) | `data/pheme_adapter.py`, `data/maweibo_adapter.py`, `data/normalized_schema.py` |
| M2 Causal Snapshot Builder (§14) | `data/snapshot_builder.py` (`build_snapshot`, `build_source_only`) |
| Text cleaning (§3.2/§7) | `data/text_cleaning.py` (`clean_tweet_pheme`, `clean_text_weibo`) |
| 384D semantic + cache (§7/§15) | `data/semantic_encoder.py` (`SemanticEncoder`, `SnapshotFeatureCache`) |
| 1021D signature (§11) | `data/adjacency_signature.py` |
| 3D summary + feature assembly (§9/§15) | `data/structural_features.py` |
| Splits (§5) | `data/temporal_split.py` |
| Manifests / cap statistics (§27) | `data/manifests.py` |
| M4 Causal Social Encoder (§16) | `models/causal_social_encoder.py` (`CausalSocialEncoder`, `load_umer_init`, `collate_snapshots`) |
| M5 Static Utility Selector (§17) | `models/selector.py` |
| Selector proxy + loss (§18) | `models/selector_proxy.py` |
| M6 Dynamic Evidence Memory (§19) | `models/evidence_memory.py` |
| Evidence unit (§20) | `context/evidence_unit.py` |
| Token budget (§21) | `context/token_budget.py` |
| Prompt templates (§22) | `context/prompts.py`, `context/packer.py` |
| M7 Qwen wrapper (§23) | `llm/qwen_wrapper.py` |
| Parser (§23) | `llm/parser.py` |
| LLM cache (§24) | `llm/cache.py` |
| Training loops (§36 smoke level) | `training/train_encoder.py`, `training/train_selector.py`, `training/sampler.py`, `training/checkpointing.py` |
| Baselines (§25/§26) | `evaluation/baselines.py` (8 arms incl. TC-DSCR itself) |
| Metrics (§42) | `evaluation/metrics.py`, `evaluation/temporal_metrics.py`, `evaluation/evidence_metrics.py` |
| Leakage scanner (§33/§29) | `evaluation/leakage_scanner.py` |
| Aggregation (§31) | `evaluation/aggregate.py` |
| Entry points | `scripts/tcdscr_build_cache.py`, `tcdscr_build_snapshots.py`, `tcdscr_train_encoder.py`, `tcdscr_train_selector.py`, `tcdscr_smoke.py`, `tcdscr_run_llm.py`, `tcdscr_summarize.py`, shared `scripts/tcdscr_common.py` |

## Reused UMER Modules

- `OriginalGraphBranch` — re-typed verbatim (384D semantic / 1024D struct
  inputs) as the TC-DSCR causal encoder backbone: `Linear(384→768)+LN+GELU`
  node projection, `StructureFeatureEncoder` (adj 1021→256, summary 3→256,
  concat→768), CLS token, one pre-norm Transformer layer (d_model 768, 8
  heads, FFN 1536), CLS+mean readout.
- The classifier shape `LayerNorm(768) → Dropout → Linear(768→2)` matches the
  UMER detector classifier so §16.1 can copy it.
- NOT reused (per §16): `NodeTokenSetEncoder`, tri-fusion `fusion_proj`,
  retrieval memory, DeBERTa four-view branch, anything 11D-dependent.
- Historical preprocessing functions `clean_tweet_pheme` / `clean_text_weibo`
  are re-included verbatim in `data/text_cleaning.py`.

## Dataset Adapters

Both adapters emit the §13 normalized schema (event_id, label, source_id,
source_timestamp, nodes with `status`) and never compute features.

- **PHEME** (`all-rnr-annotated-threads` layout): `created_at` parsed with
  `%a %b %d %H:%M:%S %z %Y`; label rumours=1 / non-rumours=0; parent from
  `in_reply_to_status_id_str`; `._` metadata files skipped. Parse failures
  raise (frozen audit: 100% timestamp coverage). Status flags: MISSING_PARENT
  (field absent/None), EXTERNAL_PARENT (target outside the event),
  TEMPORAL_INVALID_NODE (timestamp < source timestamp).
- **Ma-Weibo** (`<eid>.json` post lists + `Weibo.txt` labels): `t` is an
  absolute unix second; source = the single `parent: None` post (multi-root
  raises; frozen audit says 0 exist); text prefers `original_text` and falls
  back to `text` only when the field is absent (historical behavior verified
  by the D6 parity run, original_text coverage 99.9996%).

## Causal Snapshot Rules

- Inclusion: `status != TEMPORAL_INVALID_NODE` and
  `timestamp <= source_timestamp + cutoff` (§14.1).
- SOURCE_ONLY is a strict node-id match on the source (§4.1) — never emulated
  by a timestamp filter (unit-tested against a same-timestamp node).
- Edges require all four §14.2 conditions (child included, parent included,
  relation resolved, `parent_ts <= child_ts`); violating pairs keep the child
  with parent `[UNAVAILABLE]` (§10.2). No timestamps are ever modified.
- Node order: timestamp ascending, ties by adapter `original_order` (§11).
- SOURCE_ONLY leaves the dynamic memory empty, so the first dynamic snapshot
  scores novelty = 1 everywhere.

## Semantic Preprocessing

- PHEME: `clean_tweet_pheme(text)`; Ma-Weibo:
  `clean_text_weibo(original_text)` — the step the server pipeline lost to
  version drift (D6 finding) is restored explicitly (§3.2/§7).
- Encoder: frozen `paraphrase-multilingual-MiniLM-L12-v2` (384D,
  self-normalizing) for both datasets. No LLM embeddings, no per-split
  refitting anywhere.
- Snapshot-level disk cache keyed by dataset / event_id / cutoff /
  `tcdscr_v2.0` / semantic model hash (§15).

## Structural Features

- 1021D adjacency signature rebuilt from G_t edges only:
  `adj[parent, child] = 1` + diagonal self-loop, each row divided by its own
  out-degree (+1e-8), columns padded to 1021 (P4-verified historical
  construction; no full-graph masking).
- norm_degree: `max(outDeg−1, 0)`, normalized by the snapshot-internal max;
  all-zero degree ⇒ 0 (§9.1).
- norm_depth: BFS depth from the source / 19, never clipped; overflow counted
  per snapshot. Unreachable nodes keep the historical −1 sentinel (−1/19),
  audited as `unreachable_count` (see Known Issues).
- norm_time: `bin = clip(floor(elapsed/1800), 0, 479)`, norm = bin/480 (§9.3).

## Source Timestamp Handling

`elapsed_i = timestamp_i − source_timestamp` with `t0 = source_timestamp`
(§8). The historical `event_min_t` basis is used only where legacy parity
requires it — TC-DSCR snapshots, features, and evidence all derive from the
real source timestamp (unit-tested with an event containing a pre-source
node).

## 1021 Node Cap

Cap keeps the earliest 1021 nodes in the deterministic order; randomness,
degree heuristics, selector awareness, and label awareness are absent
(unit-tested, including invariance to raw node order). Per-dataset §27
statistics for the smoke sample are in
`results/tcdscr/code_smoke/cap_report.json` (0 cap hits at ≤32 nodes
sampled; the machinery is exercised by `test_1021_cap_deterministic` with
1,500 nodes). Full-dataset cap reports are produced by
`scripts/tcdscr_build_snapshots.py` when Stage B starts.

## Encoder Initialization

`load_umer_init` (§16.1) maps the server checkpoint's
`graph_model.graph_branch.*` / `graph_model.classifier.*` keys onto the
TC-DSCR encoder: node projection sliced `W_old[:, :384]` (395→384), plus
StructureFeatureEncoder, Transformer, CLS token, readout, and classifier
(shape-compatible). Node-token branch, fusion, retrieval are never copied.
Shape mismatches raise. Both Random Init and UMER Init are available for
PHEME and Ma-Weibo checkpoints (Stage B compares them).

## Selector

`z = [h_i; g_t; r_i; struct3]` = 1540D with `r_i = cos(e_i, e_source)`; frozen
scorer `LayerNorm(1540) → Linear(1540,256) → GELU → Dropout(0.1) →
Linear(256,1)`. The source is excluded from the candidate pool (defensive
`candidate_mask` ⇒ `-inf` ⇒ α = 0, unit-tested). No attention/GNN/Transformer/
LLM features are added.

## Dynamic Memory

Single-step rolling M_{t-1} = the previous snapshot's selected evidence
(node ids + semantic rows). novelty = 1 − max cos (normalize applied
defensively; MiniLM rows are already unit-norm), persistence = id membership,
`d = u + λ_n·novelty + λ_p·persist` with the frozen grids
λ_n ∈ {0, 0.25, 0.5, 1.0}, λ_p ∈ {0, 0.1, 0.25} (constructor validates).
Memory only ever contains past selections (unit-tested).

## Evidence Unit

Minimal Reply–Parent Pair per §20: `[Ei] / time = <s> / depth = <d> /
parent: <text| [UNAVAILABLE]> / reply: <text>`. Source is always presented
separately and never inside a pair. No full paths, summaries, stance or
relevance annotations.

## LLM Configuration

Frozen Qwen3-8B, greedy (`do_sample=False`, `num_beams=1`),
`max_new_tokens=8`, thinking disabled when the chat template supports
`enable_thinking=False`. Exact-match parsing to {RUMOR, NON_RUMOR}, else
INVALID_OUTPUT (no fuzzy matching; unit-tested). §24 cache key covers
model_id, model_revision, prompt_sha256, generation_config_sha256, dataset,
event_id, cutoff, selector_checkpoint_hash, budget; same key is never
inferred twice.

## Tests

53 tests, all passing on the server runtime
(`results/tcdscr/code_smoke/test_report.txt`):

- Data: `test_pheme_timestamp`, `test_maweibo_timestamp`,
  `test_maweibo_clean_text`, `test_source_only_exactly_one_node`
- Causality: node/edge monotonicity, no-future text/topology,
  source-timestamp-as-t0, temporal-invalid exclusion,
  child-before-parent edge removal
- Structure: snapshot-internal norm_degree, depth/19 (incl. overflow),
  30-min bins, snapshot-only adjacency signature, deterministic 1021 cap
- Split: fold integrity, no cross-fold events, all snapshots same fold,
  stratification balance, chronological 70/15/15 shapes
- Selector: output shape, source excluded from the candidate pool, frozen
  dims
- Memory: past-only novelty, persistence flag, dynamic score, frozen grids
- Context: pair rendering, [UNAVAILABLE], whole-pair budgeting, no-gold,
  no-future prompts
- LLM/cache: exact and invalid parsing, deterministic cache keys, no
  double inference
- Integration (§29): synthetic PHEME-style and Ma-Weibo-style events walk
  raw → SOURCE_ONLY/15m/1h → features → encoder → selector → memory →
  packer with finite tensors, correct dims, no leakage, deterministic
  selection and prompts.

## Smoke Run

Tiny smoke (§30) on the server (RTX 4090, python 3.11.13, torch 2.7.1,
transformers 4.57.6): 16 events per dataset, cutoffs SOURCE_ONLY / 15m / 1h,
1 encoder epoch, 1 selector epoch, exactly 10 LLM requests per dataset (20
total, budget 512), fixed memory λ_n=0.5 / λ_p=0.1 (no tuning). Outputs in
`results/tcdscr/code_smoke/`: `test_report.txt`, `smoke_manifest.json`,
`smoke_predictions.jsonl` (96 rows), `leakage_report.json` (0 failures),
`cap_report.json`, `prompt_examples.md`, `model_shapes.json`,
`known_issues.md`, `final_summary.json` (status PASS).

Gate results: tests pass; no future leakage (96/96 rows clean); LLM budget
respected; cap statistics recorded; invalid rate 0%. Per §30.1 the smoke
numbers (e.g. PHEME accuracy 0.30 on 10 scored rows, Ma-Weibo 1.00 with
10/10 rumor support) are pipeline checks only and carry no evidential value.

## Deviations from V2 Plan

No protocol deviations. Engineering decisions on points the plan leaves
open, all recorded in `results/tcdscr/code_smoke/known_issues.md`:

1. Unreachable nodes keep the historical depth = −1 sentinel (norm −1/19),
   audited per snapshot.
2. Prompt text is the raw social text; the frozen cleaning functions apply to
   the semantic pipeline only (plan does not specify prompt-side cleaning).
3. Dynamic memory is single-step rolling (M_{t-1} = previous selection, not a
   cumulative history).
4. `recent_budget` tie-break keeps snapshot order (plan fixes ties only for
   the structural budget).
5. Leakage-scan matching is substring-based over normalized text, exempting
   fragments shorter than 4 characters and text already covered by an
   included node's text (repost quotes), so template replies do not
   fabricate false alarms.

## Known Issues

- The cap statistics in the smoke run reflect the ≤32-event sample only;
  Stage B must regenerate full-dataset cap reports before formal training.
- `scripts/tcdscr_run_llm.py` uses `model_revision="local"` and a
  placeholder selector hash until Stage B freezes checkpoints.
- The smoke sample's class composition is skewed (Ma-Weibo: 10/10 rumor);
  this is a sampling artifact of a 16-event draw and is irrelevant to the
  Stage A gate.
- Server cache/outputs live under `/data/jyz/next/llm/tcdscr_cache/` and
  `/data/jyz/next/llm/results/tcdscr/code_smoke/`; the repository carries
  only the reports, never raw data or embeddings.
