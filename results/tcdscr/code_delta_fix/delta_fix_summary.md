# TC-DSCR V2 Code Complete Delta Fix Summary

## Overall Status
PASS

## Git
- commit: `2cd12b7` — "TC-DSCR V2 Code Complete delta fix: fold-aware
  protocol, source binding, recent/all-current baselines, full cap audit"
- final patch commit: `ac74242` — "TC-DSCR final patch: All-current token
  accounting aligned with Qwen chat template"

## Fix 1 — Five-Fold Protocol
- helper fixed: `event_folds_to_snapshot_folds(snapshot_event_ids,
  event_folds)` now returns **fold ids** and raises `ValueError` for an
  event without a fold assignment (`project/tcdscr/data/temporal_split.py`)
- train entry fold-aware: `scripts/tcdscr_train_encoder.py --fold 0..4
  --partition-seed 3090`; the split is computed over the full event
  registry, only train events are loaded for optimization, and
  `splits/split_manifest_<dataset>_fold<k>.json` records every §7.3 field
- selector entry fold-aware: `scripts/tcdscr_train_selector.py` with the
  same discipline (train-only loading, test completely frozen)
- cross-fold events: 0 (PHEME 6,425 events; Ma-Weibo 4,664 events)
- snapshot fold mismatches: 0 (checked per event × 7 cutoffs during the
  full-dataset audit)
- formal split: `build_primary_fold_split` — StratifiedKFold(n_splits=5,
  shuffle=True, random_state=3090) for the outer fold + stratified 10%
  inner validation draw; the three event sets are pairwise disjoint and
  cover all events (unit-tested)

## Fix 2 — Source Binding
- source lookup: the smoke/inference glue now reads the source claim from
  the snapshot itself (`snap["texts"][src_pos]` with
  `src_pos = snap["node_ids"].index(snap["source_id"])`); the cross-index
  pattern `event["nodes"][src_pos]` was found in exactly two places
  (`scripts/tcdscr_smoke.py`, `tests/test_integration_pipeline.py`) and both
  are fixed; a repository-wide search for `src_pos` / `source_pos` /
  `event["nodes"][...]` / `source_text` found no remaining offender
- adversarial raw-order test:
  `test_source_text_correct_when_raw_order_differs` +
  `test_prompt_source_bound_by_source_id` (raw order reply_A, source,
  reply_B with a different timestamp order) — both pass
- failures: 0 (`prompt_source_binding_failures = 0` over all 96 smoke rows)

## Fix 3 — Recent Baseline
- scoring direction: `rank_candidates("recent_budget")` now returns
  `+elapsed_seconds`, so descending score = most recent reply first
- ordering test: `test_recent_budget_prefers_latest` (10s/100s/1000s →
  C, B, A) plus the adversarial integration check that the top-ranked reply
  always carries the max visible elapsed — pass
- tie-break: unchanged, deterministic snapshot order (earlier original_order
  first)

## Fix 4 — All-current Context Limit
- context length: 40960, resolved from `model.config.max_position_embeddings`
  of the frozen Qwen3-8B via `resolve_context_length` (tokenizer placeholder
  sentinels are rejected; an unsolvable case raises
  `ContextLengthUnresolved` — never guessed)
- max_new_tokens: 8 (generation reserve enforced as
  `input_tokens + 8 <= context_length`)
- overflow failures: 0 (every smoke row's bounded all-current selection;
  largest observed all-current prompt 35,804 tokens on Ma-Weibo 24h-scale
  synthetic stress, still inside 40,960)
- partial-pair failures: 0 (units are added whole in snapshot order and the
  walk stops at the first pair that would not fit)
- per-row §22 fields recorded: `all_current_truncated`,
  `units_before/after_truncation`, `tokens_dropped`, `context_length`,
  `input_tokens`

## Fix 5 — Full Dataset Cap Audit

Full-dataset metadata audit (no training, no embeddings, no LLM calls):
PHEME 6,425 events, Ma-Weibo 4,664 events, 7 formal cutoffs each
(SOURCE_ONLY excluded), §27 class breakdown included. Raw reports:
`full_cap_report.json`, `fold_integrity_report.json`,
`maweibo_text_fallback_report.json`, `audit_summary.json`.

### PHEME
| cutoff | cap hit rate | P90 | P99 | max | total removed |
|---|---:|---:|---:|---:|---:|
| 5m | 0.0000 | 8 | 17 | 48 | 0 |
| 15m | 0.0000 | 15 | 24 | 98 | 0 |
| 30m | 0.0000 | 19 | 34 | 146 | 0 |
| 60m | 0.0000 | 21 | 46 | 207 | 0 |
| 180m | 0.0000 | 24 | 67 | 267 | 0 |
| 360m | 0.0000 | 27 | 80 | 337 | 0 |
| 1440m | 0.0000 | 29 | 90 | 340 | 0 |

### Ma-Weibo
| cutoff | cap hit rate | P90 | P99 | max | total removed |
|---|---:|---:|---:|---:|---:|
| 5m | 0.0000 | 68 | 220 | 781 | 0 |
| 15m | 0.0021 | 166 | 549 | 3,563 | 6,846 |
| 30m | 0.0077 | 270 | 885 | 16,665 | 39,453 |
| 60m | 0.0191 | 382 | 1,365 | 24,192 | 92,088 |
| 180m | 0.0530 | 667 | 2,983 | 24,192 | 321,306 |
| 360m | 0.0772 | 816 | 3,930 | 25,856 | 536,952 |
| 1440m | 0.1192 | 1,189 | 8,622 | 51,678 | 1,446,013 |

Class breakdown (cap may bite classes asymmetrically — reported, not acted
on, per §30): Ma-Weibo 24h rumor 0.1064 vs nonrumor 0.1319; PHEME both 0.
Worst-50 events per dataset are listed in `full_cap_report.json`.
The 1021 cap rule was NOT changed.

## KL Fix
- implemented: `l_fid = Σ p_ref (log p_ref − log q)` with
  `p_ref = softmax(p_full.detach())` — the logged value is now the true
  KL(stopgrad(p_full) ‖ p_sel) (gradient-equivalent to the previous form)
- numeric test: `test_fidelity_loss_matches_manual_kl` (value equality with
  the manual KL, zero for identical distributions, direction check) — pass

## All-current Chat-template Accounting: PASS

Final patch: All-current token accounting now matches real Qwen3-8B
inference exactly.

- Shared implementation: `format_chat_prompt(tokenizer, prompt)` /
  `count_chat_input_tokens(tokenizer, prompt)` in
  `project/tcdscr/llm/qwen_wrapper.py` — single user message,
  `add_generation_prompt=True`, `enable_thinking=False`; `QwenRumorLLM` and
  the All-current limit check call the same functions, so they cannot drift.
- Strict separation: `evidence_tokens` counts only the rendered kept
  Reply–Parent units; `input_tokens` is the full chat-formatted prompt
  (source + snapshot metadata + evidence + task). SOURCE_ONLY rows report
  `evidence_tokens = 0`; the largest smoke row is input 35,816 /
  evidence 35,461, both well inside 40,960.
- Truncation decisions run on the chat-formatted prompt: candidate units ->
  pack_context -> apply_chat_template -> tokenize ->
  `input_tokens + 8 <= context_length` (40960 resolved dynamically via
  `resolve_context_length`, never hard-coded).
- Report: `results/tcdscr/code_delta_fix/all_current_context_report.json` —
  model_context_length 40960, max_new_tokens 8, token_counting_mode
  "qwen_chat_template", overflow/partial-pair/chat-mismatch failures all 0;
  synthetic near-limit stress (120 whole pairs) stops at input_tokens
  40,868 (+8 reserve = 40,876 <= 40,960), evidence_tokens 40,686, 96/120
  units kept, truncated.
- Tests: `test_chat_token_count_matches_qwen_wrapper`,
  `test_all_current_limit_uses_chat_formatted_tokens`,
  `test_all_current_near_limit_never_overflows`,
  `test_all_current_evidence_tokens_exclude_source_and_instruction` — pass.
- Note: the Qwen3 template pre-fills an EMPTY `<think>

</think>` block
  when thinking is disabled; that is the expected disabled-thinking marker
  and the tests assert the block is empty.

## Ma-Weibo Text Fallback
- total nodes: 3,805,656
- fallback count: 0 (`original_text` used for every node; 17 blank-text
  nodes)
- fallback rate: 0.0 — the D6-era conclusion holds on the full dataset;
  the adapter keeps the branch and the audit makes it non-silent
  (`text_source_counts` per event, `maweibo_text_fallback_report.json`)

## Tests
- total: 83
- passed: 83
- failed: 0 (`test_report.txt`; includes the 13 delta-fix tests, the §37
  adversarial integration test, and the 4 chat-accounting tests)

## Tiny Smoke
- PHEME: 16 events, SOURCE_ONLY/15m/1h, 10 LLM requests — completed
- Ma-Weibo: 16 events, SOURCE_ONLY/15m/1h, 10 LLM requests — completed
- LLM requests: 20 total (budget 512, encoder/selector 1 epoch each)
- future leakage failures: 0 (96/96 rows clean)
- source binding failures: 0
- fold failures: 0
- context overflow failures: 0 (context_length 40960, reserve 8)
- recent baseline order failures: 0

## Blocking Issues
1. None blocking. Observation for the research approver (report only,
   §30): the 1021 cap bites only Ma-Weibo late cutoffs (up to 11.9% of
   events at 24h, P99 ≈ 8.6k nodes before cap, nonrumor events hit more
   often than rumor ones). Whether this asymmetry matters for Stage B
   design is a research decision, not a Code-Delta-Fix change.
2. Note: PHEME raw threads contain 772 duplicate reaction files (same
   id_str shipped under two file names). The adapter keeps the first file
   in deterministic name order and audits the count — matching the
   historical pipeline's dict-based de-duplication.

## Recommended Status
PASS
