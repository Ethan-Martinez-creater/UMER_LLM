# CR-TSER V2R1-P1A — formal manifest freeze, verified cache migration, Mistral labels

Approved baseline: `5295d5838be3f78daca53a34e9d6a9c661a45e29`
Plan: `docs/CR_TSER_V2R1_P1A_EXECUTION_PLAN.md`
Namespace: `results/cr_tser_v2r1/` (protocol `v2r1`)

This round froze the formal v2r1 manifests, migrated the existing
Qwen/InternLM labels fail-closed, verified that the ordinary generator reuses
every migrated row, generated the formal Mistral labels for both datasets, and
audited the final 3-reader caches. No predictor training, Stage A/B, held-out
evaluation, P1–P4 or final Pilot was run.

## 1. Source identity

| dataset | kind | exists | n_files | bytes | sha256 |
|---|---|---|---|---|---|
| maweibo | `maweibo_composite` | true | 4665 | 4060340423 | `b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d` |
| pheme | `pheme_raw` | true | 255945 | 322990857 | `88a27f230b577502674f9439ee1ca579d1b5e5a872ec93d5ed96a281f9816e6b` |

The Ma-Weibo fingerprint equals the value the plan requires, so no source drift
occurred.

## 2. Frozen v2r1 manifests

Built with `CRTSER_OUT_ROOT=results/cr_tser_v2r1`, no `--smoke`.

| file | canonical LF sha256 |
|---|---|
| `manifests/maweibo/source.json` | `80b07954ce3199c57cb25e7ca11b07115dd0e20a84787b97effc1cfed291b783` |
| `manifests/maweibo/event_split.json` | `24e18ed10954e8387c49d9a119e78934e2c8d33ccb582aa62e4f2cf201b8dde2` |
| `manifests/maweibo/hashes.json` | `f106af7a60b5cb76fd338a4bea53c56a9de79ee700c43a2525c848384d7d7045` |
| `manifests/maweibo/snapshot_manifest.jsonl` | `611cb9c6ca43afbaec8a0eba10d71c1fcc4dcb9af3ebbd7cd90a2660a2f434dc` |
| `manifests/maweibo/intervention_manifest.jsonl` | `f37c3ffcb07e8e0142a082326ec405417c8f99399a1b09fd907cfaba9b680782` |
| `manifests/pheme/source.json` | `1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363` |
| `manifests/pheme/event_split.json` | `f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385` |
| `manifests/pheme/hashes.json` | `a3c46403d9173bdcacc00604706ecd4586f8e80b5a97e5d8992f91a373fe9cd9` |
| `manifests/pheme/snapshot_manifest.jsonl` | `8c2ec0462a1fbf9ff681d237fb9e57df09afd9af5c7e402779c2defe8812315f` |
| `manifests/pheme/intervention_manifest.jsonl` | `a0c0ad7abb5c8132ecb9483143ad8c70bc6836568a183e32797a8af73201d2e3` |

Equivalence against the frozen V2 namespace:

| check | maweibo | pheme |
|---|---|---|
| `source.json` identical to V2 | yes | yes |
| `event_split.json` identical to V2 | yes | yes |
| `snapshot_manifest.jsonl` identical to V2 | yes | yes |
| `intervention_manifest.jsonl` identical to V2 | yes | yes |
| cutoffs | 15/60/360 | 15/60/360 |
| future-leakage rows | 0 | 0 |
| snapshots at the retired 1021 cap | 0 | 0 |
| `snapshot_cap` | `null` | `null` |
| split (`foundation/utility_train/utility_dev/utility_eval/unused`) | 80/50/15/25/4421 | 80/50/15/25/5499 |
| split seed | 7319 | 7319 |
| `event_split_sha256` | `d54dda6a33c027c18e3678d646ecf187e6348b831872fd94f2f64a36c0a37710` | same as V2 |
| viable events | 4591 | 5669 |
| snapshot rows / interventions | 270 / 3839 | 270 / 3412 |
| `glm` occurrences in the manifest blob | 0 | 0 |
| `hashes.json` readers | qwen/mistral/internlm | qwen/mistral/internlm |

`hashes.json` is the only file that differs from V2, and its diff is exactly
`readers` + `loro_rotations` — the reader-set change amendment R1 mandates.
Every other field (dataset, cutoffs, `event_split_sha256`, `source`,
`n_snapshots`, `n_interventions`, `n_zero_reply_snapshots`,
`max_snapshot_nodes`, `snapshot_cap`, `viable_events`) is identical to V2.

The ten hashes are pinned in `scripts/cr_tser_verify_pilot.py`
(`V2R1_FROZEN_MANIFEST_SHA256`), checked by `v2r1_manifest_pins`,
`v2r1_manifests_match_v2` and `v2r1_reader_contract_in_manifest`, and guarded
from the package side by `project/cr_tser/tests/test_v2r1_p1a_freeze.py`.

## 3. Cache migration (fail-closed)

Dry-run and apply produced identical selections:

| dataset | source rows | migrated | retired (GLM) | unknown readers | duplicate keys | readers migrated |
|---|---|---|---|---|---|---|
| maweibo | 6525 | **6404** | 121 | 0 | 0 | internlm, qwen |
| pheme | 5758 | **5758** | 0 | 0 | 0 | internlm, qwen |

* maweibo: qwen 3202 + internlm 3202 = 6404 (plan expected 6404)
* pheme: qwen 2879 + internlm 2879 = 5758 (plan expected 5758)

Target was empty before the apply; the apply used `open(..., "x")`, so an
existing cache could not be overwritten. The eight identity fields
(`base_context_hash`, `intervened_context_hash`, `reader_hash`,
`reader_identity_hash`, `tokenizer_hash`, `chat_template_hash`, `prompt_hash`,
`prompt_ids_hash`) are preserved row by row.

Migration fidelity (line-by-line comparison against the V2 originals, on the
final cache as well):

| dataset | checked | missing | changed | extra |
|---|---|---|---|---|
| maweibo | 6404 | 0 | 0 | 0 |
| pheme | 5758 | 0 | 0 | 0 |

Migrated cache files after the apply:

* `utility_labels/maweibo/labels.jsonl` — 6404 rows, 10616388 bytes, `a8fe1bb16cc6e33e9b3d8bea80beb43f7d15a10efc0ecdf900e3c326791e0729`
* `utility_labels/pheme/labels.jsonl` — 5758 rows, 9587921 bytes, `29f6e4a0a48c28de4f66c26e3cc39cc05d3a8fbeffbcea8431bda4c3cf8fa3b9`

## 4. Reuse verification (ordinary generator)

`scripts/cr_tser_generate_labels.py --dataset <ds> --reader <qwen|internlm>`
against the migrated v2r1 caches:

| run | rows written | rows reused | expected reuse | `CacheIdentityMismatch` |
|---|---|---|---|---|
| maweibo / qwen | **0** | 3202 | 3202 | none |
| maweibo / internlm | **0** | 3202 | 3202 | none |
| pheme / qwen | **0** | 2879 | 2879 | none |
| pheme / internlm | **0** | 2879 | 2879 | none |

The reuse pass left both cache files byte-identical (same sha256 as after the
apply), so the migrated rows were genuinely reused rather than rewritten.

## 5. Mistral labels

| dataset | rows written | reused | mismatches |
|---|---|---|---|
| maweibo | **3202** | 0 | none |
| pheme | **2879** | 0 | none |

Both equal the plan's expectation. No `--smoke`, no `--limit-snapshots`.

## 6. Final 3-reader cache audit

| dataset | rows | qwen | internlm | mistral | bytes | sha256 |
|---|---|---|---|---|---|---|
| maweibo | **9606** (3202 × 3) | 3202 | 3202 | 3202 | 16015182 | `6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3` |
| pheme | **8637** (2879 × 3) | 2879 | 2879 | 2879 | 14457787 | `773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15` |

Audit result for both datasets:

* reader set exactly `qwen/mistral/internlm`, no unknown reader
* GLM rows = **0**
* duplicate cache keys = **0**
* non-finite (`NaN`/`Inf`) numeric values = **0**
* missing fields = **0** (all 36 row fields present on every row)
* cutoffs only 15/60/360
* identity single-valued per reader (`model_id`, `dtype`, `reader_hash`,
  `reader_identity_hash`, `tokenizer_hash`, `chat_template_hash`)
* `prompt_hash`, `prompt_ids_hash`, `base_prompt_hash` and
  `reader_identity_hash` non-empty on every row
* no smoke contamination, and no cache written inside a smoke namespace

Per-reader `prompt_tokens` range (observation, nothing tuned from it):

| dataset | qwen | internlm | mistral |
|---|---|---|---|
| maweibo | 152–1289 | 165–1460 | 183–2204 |
| pheme | 133–1193 | 145–1416 | 145–1438 |

The Mistral ranges are the longest because Mistral's tokenizer splits the same
frozen text into more tokens; the evidence budget is accounted in canonical
Qwen3-8B tokens (`CANDIDATE_TOKENIZER`/`BUDGET_REF`), which is the frozen
contract, so this is a tokenizer-density property rather than a budget
violation.

## 7. Verifiers

| run | issues | pending |
|---|---|---|
| `--mode code --protocol v2r1` (server) | **0** | 0 |
| `--mode pilot --protocol v2r1` (server) | **0** | 3 |

The three pending pilot checks are the downstream artifacts this round must not
produce: `unseen_reader_artifacts:maweibo`, `unseen_reader_artifacts:pheme` and
`pilot_summary: CR_TSER_PILOT_SUMMARY.json missing`.

LOCAL/pytorch: `192 passed`, code verifier `issues = 0` for v1, v2 and v2r1,
`compileall` clean.

## 8. V2 historical immutability

`results/cr_tser_v2/manifests` (10 files) and
`results/cr_tser_v2/utility_labels` (2 files) were hashed before the round and
again at the end; both comparisons report `added: [], removed: [], changed: []`.
Nothing in the V2 namespace was rewritten, no GLM row was deleted or renamed,
and no V2 cache was overwritten.

## 9. Environment

| | |
|---|---|
| LOCAL | `pytorch`, Python 3.9.18, lightweight tests only |
| SERVER | `DGPA`, Python 3.11.13, torch `2.7.1+cu128`, transformers `4.53.3`, tokenizers `0.21.4` |
| overlay | `/data/jyz/next/llm/.cr_tser_v2p0/tf453` (DGPA site-packages untouched) |
| workspace | `/data/jyz/next/llm/cr_tser_ws` at `5295d58` |
| GPU | RTX 4090 24564 MiB; free at stage-D start 20030 MiB (other users held ~3.9 GiB) |

All server commands went through `next/.codex_tmp_ssh_run.cmd`; every transfer
went through `next/.codex_tmp_scp_run.cmd`. No host/port/user change, no
reconnection workaround.

## 10. Interruptions and resumes

None. Every stage ran start to finish in one pass, and no `--resume` path was
needed: the repository was already at the approved baseline, the target
namespace was empty, the migration gate and the reuse gate both passed on the
first attempt, and no `CacheIdentityMismatch`, `SourceIdentityError` or
`MigrationRefused` was raised.

## 11. Not run

`cr_tser_train_predictors.py`, Stage A/B, held-out evaluation, P1–P4 and the
final Pilot report were **not** run. The round stops after manifests +
migration + Mistral labels + cache audit + verifiers.

## 12. Evidence files

`results/cr_tser_v2r1/p1a_execution/` holds the raw server outputs:
`precheck.json`, `manifest_equivalence.json`, `migrate_<dataset>_{dryrun,apply}.json`,
`migration_gate_{dryrun,apply}.json`, `cache_audit_post_migration.json`,
`reuse_<dataset>_<reader>.json`, `reuse_check.json`, `mistral_<dataset>.json`,
`mistral_gate.json`, `cache_audit_final.json`, and the V2 tree hashes
(`v2_manifests_{pre,mid,post}.json`, `v2_labels_{pre,post}.json`).
