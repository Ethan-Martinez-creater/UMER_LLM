# CR-TSER V2-P1A — Manifest Freeze + Frozen-Reader Utility Labels

Baseline `c3c08dc` → hotfix `57895a3`. Round scope: build and freeze the
formal Ma-Weibo / PHEME manifests, then generate the frozen-reader utility
labels on SERVER/DGPA and audit the cache. No predictor training, no Stage A/B,
no held-out evaluation, no P1–P4 gate and no final pilot report was run.

## 1. Commits

```text
c3c08dc   V2-P1A blocked — Ma-Weibo manifest freeze hit KeyError 'exists'
57895a3   V2-P1A hotfix — composite source fingerprint exposes the unified
          top-level 'exists' (this round)
```

The hotfix is confined to `project/cr_tser/data/source_manifest.py` (the
`maweibo_composite` branch of `source_fingerprint()` now carries the same
top-level field set as every other source kind, with
`"exists": bool(raw["exists"]) and bool(labels["exists"])`) plus the new
regression suite `project/cr_tser/tests/test_v2_p1a_manifest_hotfix.py`.
Split seed/sizes, cutoffs, eligibility, fingerprint semantics, reader contract,
utility definition and P1–P4 thresholds are untouched.

## 2. LOCAL/pytorch checks (baseline and post-hotfix)

```text
python -m pytest project/cr_tser/tests -q                       143 passed
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2 issues = 0
python -m compileall project/cr_tser scripts                    clean
```

## 3. SERVER/DGPA workspace

```text
workspace   /data/jyz/next/llm/cr_tser_ws
HEAD        57895a3f4945b073b1a248d8b5632e520d463bf7
conda env   DGPA
overlay     PYTHONPATH=/data/jyz/next/llm/.cr_tser_v2p0/tf453
python      3.11.13
torch       2.7.1+cu128
transformers 4.53.3
tokenizers  0.21.4
GPU         NVIDIA GeForce RTX 4090, 24564 MiB (shared)
```

Wrappers used for every server interaction:

```text
next/.codex_tmp_ssh_run.cmd
next/.codex_tmp_scp_run.cmd
```

One operational note: the first server-side `git fetch origin main` of this
round timed out (`Failed to connect to github.com port 443`); the retry
succeeded (`c4773a3..57895a3`) and the workspace was checked out to the hotfix
commit. No host/port/user/connection method was changed.

## 4. Data and reader paths

```text
CRTSER_MAWEIBO_RAW          /data/jyz/next/llm/data/maweibo_raw
CRTSER_MAWEIBO_LABELS       /data/jyz/next/llm/data/maweibo_labels.txt
CRTSER_PHEME_RAW            /data/jyz/next/llm/data/pheme_raw
CRTSER_SEMANTIC_MODEL       /data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2
CRTSER_CANONICAL_TOKENIZER  /data/jyz/next/llm/model/qwen3-8b
CRTSER_QWEN_MODEL           /data/jyz/next/llm/model/qwen3-8b
CRTSER_GLM_MODEL            /data/jyz/next/llm/model/glm-4-9b-chat-hf
CRTSER_INTERNLM_MODEL       /data/jyz/next/llm/model/internlm3-8b-instruct
```

No model was downloaded this round; no checkpoint was substituted, quantized or
re-converted.

## 5. Ma-Weibo source fingerprint

```text
kind                      maweibo_composite
exists                    true            (new unified top-level alias)
raw_json.n_files          2542
combined_source_sha256    b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d
```

`MATCH_EXPECTED True` — the approved composite identity reproduced exactly, no
source drift, and the composite hash algorithm is unchanged.

## 6. Frozen manifests

Both datasets, `results/cr_tser_v2/manifests/<dataset>/`, audited before any
reader utility call. Ma-Weibo was rebuilt with the hotfix (no `--force` was
needed: only the failed run's `source.json` existed and no label cache was
present). PHEME was **not** rebuilt — its fingerprints are byte-identical to
the previous round's.

```text
Ma-Weibo split      foundation_train 80 / utility_train 50 / utility_dev 15 /
                    utility_eval 25, seed 7319, event-disjoint
                    label counts 40/40, 25/25, 8/7, 12/13
PHEME split         identical sizes, seed and label balance

maweibo  snapshots 270   interventions 3839   zero_reply 12   viable 4591
         max_snapshot_nodes 24192 (no 1021 cap applied; cap_hit all false)
pheme    snapshots 270   interventions 3412   zero_reply 9    viable 5669
         max_snapshot_nodes 79
cutoffs  [15, 60, 360] only, both datasets
```

File identities (server-authoritative):

| dataset | file | bytes | sha256 |
|---|---|---|---|
| maweibo | source.json | 752 | `80b07954ce3199c57cb25e7ca11b07115dd0e20a84787b97effc1cfed291b783` |
| maweibo | event_split.json | 101074 | `24e18ed10954e8387c49d9a119e78934e2c8d33ccb582aa62e4f2cf201b8dde2` |
| maweibo | snapshot_manifest.jsonl | 88494 | `611cb9c6ca43afbaec8a0eba10d71c1fcc4dcb9af3ebbd7cd90a2660a2f434dc` |
| maweibo | intervention_manifest.jsonl | 800357 | `f37c3ffcb07e8e0142a082326ec405417c8f99399a1b09fd907cfaba9b680782` |
| maweibo | hashes.json | 2133 | `fac953a5aaec6571f2ae90ebeb7a5f017c20673896ff2a65ae14991a2254154b` |
| pheme | source.json | 228 | `1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363` |
| pheme | event_split.json | 136816 | `f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385` |
| pheme | snapshot_manifest.jsonl | 87801 | `8c2ec0462a1fbf9ff681d237fb9e57df09afd9af5c7e402779c2defe8812315f` |
| pheme | intervention_manifest.jsonl | 724632 | `a0c0ad7abb5c8132ecb9483143ad8c70bc6836568a183e32797a8af73201d2e3` |
| pheme | hashes.json | 1588 | `4b865d5a811efe74a560e921bf888e6a7838ef3a20c854dbaa8b2bcf5b8b08f3` |

The manifest audit (`p1a_manifest_audit.json`, both datasets `all_checks_pass
= true`) covers: frozen source kind and `exists`, split sizes/seed/disjointness,
cutoffs limited to 15/60/360, no future-node leak, no cap hit and
`snapshot_cap = None`, snapshots only from labelled events, no dataset mixing,
unique snapshot keys, reader ids equal to the approved frozen readers and
non-empty reader weight/tokenizer hashes.

Largest Ma-Weibo snapshots (evidence that the retired TC-DSCR `MAX_NODES=1021`
cap is not applied):

```text
3918595647661796  cutoff 60   24192 nodes / 24191 replies
3918595647661796  cutoff 360  24192 nodes / 24191 replies
3918595647661796  cutoff 15    3563 nodes /  3562 replies
3484377063826489  cutoff 360   2143 nodes /  2142 replies
3912003082833585  cutoff 360   1289 nodes /  1288 replies
```

## 7. Utility labels

Cache: `results/cr_tser_v2/utility_labels/<dataset>/labels.jsonl` (server-only
heavyweight artifact, deliberately not committed — see §10). One reader at a
time, no `--smoke`, no `--limit-snapshots`. Every row carries the §32 identity
fields (`reader_hash`, `reader_identity_hash`, `tokenizer_hash`,
`chat_template_hash`, `prompt_hash`, `prompt_ids_hash`,
`base_context_hash`, `intervened_context_hash`).

Per dataset/reader, audited (plan §8) — `label_audit/audit_<dataset>_<reader>.json`:

| dataset | reader | rows | snapshots | base rows | audit | row-level file sha256 (at audit time) |
|---|---|---|---|---|---|---|
| maweibo | qwen | 3202 | 258 | 258 | PASS | `8bdfbe400c6f8845b9b6739f4ca3e897011bcb50b2b0ef69c29af6433351b2a7` |
| maweibo | internlm | 3202 | 258 | 258 | PASS | `8bfecf4d503dc200131e06be0a1ac228734bc61a6eb450d77d7445b6095256f4` |
| maweibo | glm | 121 | — | — | BLOCKED (§9) | — |
| pheme | qwen | 2879 | 261 | 261 | PASS | `0b450035183416d10d39bd9f5deba4f6e7ac8c58144af91cf7a932d3cd3170c4` |
| pheme | internlm | 2879 | 261 | 261 | PASS | `29f6e4a0a48c28de4f66c26e3cc39cc05d3a8fbeffbcea8431bda4c3cf8fa3b9` |
| pheme | glm | 0 | — | — | BLOCKED (§9) | — |

Intervention-type coverage (both readers of a dataset agree exactly, because
the type set comes from the frozen manifest):

```text
maweibo  I0_base 258  I1_atomic 2549  I2_parent_child 101
         I3_matched_nonadjacent 100  I4_subtree 101  I5_matched_disconnected 93
pheme    I0_base 261  I1_atomic 2107  I2_parent_child 130
         I3_matched_nonadjacent 127  I4_subtree 131  I5_matched_disconnected 123
```

Sign and gold distributions (no thresholds were tuned or reinterpreted):

```text
maweibo/qwen      NEUTRAL 2686 HARMFUL 261 HELPFUL 255   gold 0:1714 1:1488
maweibo/internlm  NEUTRAL 2609 HARMFUL 246 HELPFUL 347   gold 0:1714 1:1488
pheme/qwen        NEUTRAL 2348 HARMFUL 245 HELPFUL 286   gold 0:1319 1:1560
pheme/internlm    NEUTRAL 2052 HARMFUL 475 HELPFUL 352   gold 0:1319 1:1560
```

Every audit checks: rows present, zero duplicate cache keys, zero missing
required fields, zero NaN/Inf, cutoffs limited to 15/60/360, dataset/reader
field consistency, reader weight/tokenizer hash equal to the frozen manifest,
model id equal to the approved reader, a single reader identity per pair, one
`I0` base row per snapshot, recorded dtype, binary gold, and no smoke
namespace. All checks pass for the four completed pairs. No
`CacheIdentityMismatch` was raised at any point.

Whole-cache file identities at the time of writing (they grow as the GLM pairs
complete):

```text
maweibo/labels.jsonl  6525 rows  10816212 bytes
  8bfecf4d503dc200131e06be0a1ac228734bc61a6eb450d77d7445b6095256f4
pheme/labels.jsonl    5758 rows   9587921 bytes
  29f6e4a0a48c28de4f66c26e3cc39cc05d3a8fbeffbcea8431bda4c3cf8fa3b9
```

## 8. Verifier (SERVER/DGPA, HEAD 57895a3)

```text
cr_tser_verify_pilot.py --mode code  --protocol v2
    issues = 0, pending = 0
cr_tser_verify_pilot.py --mode pilot --protocol v2
    issues = 0, pending = 3
      [pending] unseen_reader_artifacts:maweibo
      [pending] unseen_reader_artifacts:pheme
      [pending] pilot_summary
```

The three pending items are the downstream P1–P4 artifacts this round is
forbidden to produce; nothing was fabricated for them.

## 9. Interruption — GLM-4-9B blocked by shared-GPU memory contention

`maweibo/glm` and `pheme/glm` could not complete. This is a resource-contention
failure on the shared GPU, **not** a code or protocol defect, and no mitigation
that would change the science was applied.

```text
first attempt            2026-09-15 23:08:33  OOM after 121 rows   (p1a_labels_master.log)
second attempt           2026-09-16 01:03:30  OOM after 0 rows     (p1a_resume_master.log)
                         torch.OutOfMemoryError: Tried to allocate 674.00 MiB;
                         GPU 0 has 542.62 MiB free; this process uses 19.30 GiB
```

Root cause: GLM-4-9B needs ≈19.3 GiB in bf16 (151k vocabulary, the 674 MiB peak
is `logits.float()` in `sequence_logprob`), while a *different user's* six
python jobs (`/home/shx/anaconda3/envs/shx/bin/python`, ~3.3 GiB total) hold the
rest of the 24 GiB card and return immediately after any short idle window. The
8B readers fit alongside that load; the 9B reader does not.

Evidence preserved on the server and in this repository:

```text
.cr_tser_v2p0/logs/oom_maweibo_glm.log          (first attempt, full traceback)
.cr_tser_v2p0/logs/oom_attempt1_maweibo_glm.log (second attempt, kept by the retry script)
results/cr_tser_v2/p1a/logs/labels_maweibo_glm.log
results/cr_tser_v2/p1a/logs/oom_maweibo_glm.log
```

Nothing about the model, dtype, token budget, cutoffs, reader contract or
protocol was changed to work around it. The only measures taken are operational:
the remaining GPU work now waits for a *sustained* free window (≥23500 MiB for
four consecutive 30 s samples) before starting GLM, and the retry sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the allocator setting torch
itself recommends in the OOM message, which reclaims the ~350 MiB of
reserved-but-unallocated segments and changes no numeric computation. The
generator is resumable by design (plan §7), so the 121 cached `maweibo/glm` rows
are reused, not discarded.

`maweibo/glm` and `pheme/glm` therefore remain **pending**, and this round stops
here for approval rather than reporting a partial stage as complete.

## 10. Artifact retention

Committed to Git (this round): the frozen manifests (both datasets, ~1.9 MB
total), the label-cache audit JSONs, the manifest audit, the OOM evidence logs
and this report.

Left at their authoritative server location as heavyweight artifacts, with
paths and SHA256 recorded above and in `label_audit/*.json`:

```text
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/utility_labels/maweibo/labels.jsonl
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/utility_labels/pheme/labels.jsonl
```

No Git LFS, compression scheme or `.gitignore` change was introduced.

Note on line endings: the repository has `core.autocrlf=true` and no
`.gitattributes`, so a Windows checkout materialises the committed manifests
with CRLF while the stored blob (and the server working tree) is LF. Every
SHA256 recorded here is the server/LF identity.

## 11. Not run

```text
cr_tser_train_predictors.py        NOT RUN
formal Stage-A subset freeze       NOT RUN
held-out-reader scoring            NOT RUN
P1 / P2 / P3 / P4 final gates      NOT RUN
cr_tser_run_pilot.py final report  NOT RUN
```
