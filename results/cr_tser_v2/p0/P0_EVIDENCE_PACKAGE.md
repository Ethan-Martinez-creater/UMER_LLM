# CR-TSER V2-P0 — Preflight Evidence Package

Executed from frozen baseline `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`
("CR-TSER V2-M0 closure hotfix …"). No code change was made in this round.

## 1. Verdict

```text
P0_FAIL
```

**Single blocker**: the three frozen readers could not be resolved on this
machine. Every `CRTSER_*_MODEL` path was unset, so the frozen P0 entrypoint
resolved `model_path` to the empty string for Qwen3-8B, GLM-4-9B-Chat and
InternLM3-8B-Instruct, and no local copy of any of them exists.

Failure reasons (execution plan §14):

```text
QWEN_MODEL_MISSING
GLM_MODEL_MISSING
INTERNLM_MODEL_MISSING
```

The data side of P0 **passed in full** — see §4–§7.

Production of this verdict is frozen-code-only:
`python scripts/cr_tser_p0_audit.py --protocol v2 --readers --sanity`
wrote `p0_readiness.json`. No artifact was hand-edited.

## 2. Execution context

| Item | Value |
|---|---|
| git commit | `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f` |
| protocol baseline | `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f` (match) |
| working tree | clean for tracked files |
| python | 3.11.15 (`E:\miniconda3\envs\bettafish\python.exe`) |
| torch | 2.5.1+cu118 |
| transformers | 5.14.1 |
| CUDA | 11.8, available |
| GPU | NVIDIA GeForce GTX 1050 Ti — **4.00 GiB total, 3.28 GiB free**, cc 6.1 |
| deployment role | lightweight verification machine; heavyweight resources (reader weights, GPU) run on the synchronized server |

Baseline health before P0 (execution plan §4):

```text
pytest project/cr_tser/tests -q      -> 128 passed
verify_pilot.py --mode code --protocol v2 -> issues = 0
compileall project/cr_tser scripts   -> clean
```

## 3. Ma-Weibo source of record (P0-A)

```text
CRTSER_MAWEIBO_RAW    = E:\Graduate_work_folder\rumor_detection\data\dataset\Ma-WeiBo\Weibo
CRTSER_MAWEIBO_LABELS = E:\Graduate_work_folder\rumor_detection\data\dataset\Ma-WeiBo\Weibo.txt
```

This is the original Ma-Weibo release layout (raw per-event JSON directory +
`Weibo.txt` label file), matching the authoritative source recorded in
`resources/resource_manifest.md`. No tensors, processed graph cache, UMER
cache, TC-DSCR predictions or synthetic time were used.

Composite fingerprint (`maweibo_source_fingerprint.json`):

| Field | Value |
|---|---|
| raw JSON files | 4664 |
| raw bytes | 3,995,877,128 |
| raw sha256 | `c6afd1c50a6cde8c27137d367d8e420b4a3afc43e017e3b19846e8b001ca1a27` |
| label file bytes | 64,463,295 |
| label sha256 | `032f10e2175fa461203bdba77ef2a492e0ebde1cdd24b9bcd2100e307bc6b8e4` |
| **combined_source_sha256** | `b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d` |

Timestamps come only from raw `post["t"]`; `original_order` is a
deterministic tie-break and never enters the temporal inclusion rule
(verified per-event in `snapshot_integrity.json`).

## 4. Ma-Weibo integrity audit (P0-B)

From `maweibo_audit.json` (frozen `audit_maweibo`, amendment §9 fields):

```text
raw_event_count                     4664
raw_json_files                      4664
parsed_event_count                  4664
invalid_event_count                 0
label_distribution                  {"0": 2351, "1": 2313}
source_text_coverage                1.0
reply_text_coverage                 0.9999955274833517
source_timestamp_coverage           1.0
timestamp_coverage                  1.0
parent_resolution_coverage          0.9999918442343473
reply_node_count                    3800992
duplicate_ids                       0
cycle_count                         0
multi_root_event_count              0
missing_parent_count                0
missing_parent_rate                 0.0
external_parent_count               30
external_parent_rate                7.892676438150883e-06
temporal_invalid_node_count         1
empty_text_count                    17
node_status_counts                  {VALID: 3805608, EXTERNAL_PARENT: 30,
                                     EMPTY_TEXT: 17, TEMPORAL_INVALID_NODE: 1}
events_with_ge1_valid_reply_parent_unit  4663
verdict                             MAWEIBO_READY
```

The historical TC-DSCR reference numbers were **not** assumed; they were
recomputed from the current source of record and independently reproduced
(text coverage ≈ 0.9999955, parent resolution ≈ 0.9999918, 0 cycles,
0 duplicate IDs, 0 multi-root events).

Reply–parent ratios use the non-source reply count (3,800,992) as the
denominator; the source node never dilutes them.

## 5. Ma-Weibo viability (P0-C)

V2 viability rule (amendment §8): valid binary label, exactly one
source/root, valid source timestamp, non-empty source text, unique node IDs,
no valid-node cycle, and ≥1 legal Reply–Parent Evidence Unit in at least one
of 15m / 1h / 6h. A legal unit requires
`reply.status == VALID ∧ reply text non-empty ∧ parent exists ∧
parent.status == VALID ∧ parent text non-empty ∧
parent.timestamp <= reply.timestamp`.

```text
events_viable_15m        4296
events_viable_1h         4486
events_viable_6h         4591
total_viable_events      4591   (required >= 170)
```

**4591 >= 170 — satisfied.** The frozen 80/50/15/25 split was not touched,
and no formal event split was created.

## 6. Causal snapshot integrity (P0-D)

`snapshot_integrity.json` samples the first 8 streaming events satisfying the
V2 viability rule and probes all three cutoffs (24 snapshots). Every probe:

```text
source_present                    true
future_leak_count                 0
cap_hit                           false        (MAX_NODES_CAP = None)
num_nodes_before_cap == after     true
parent_visibility_ok              true
parent_time_ok                    true   (parent.ts <= child.ts)
order == timestamp, original_order true
unreachable_count                 0
```

Largest sampled event: `10031994215` (1062 nodes total, 941 nodes at 6h) —
above the retired 1021 cap, confirming the cap can no longer truncate a
snapshot. Zero-reply cutoffs are recorded, not event-deleting; no sample
cutoff was zero-reply.

## 7. PHEME smoke (P0-E)

`pheme_smoke.json` — `status = OK`, raw dir
`E:/Graduate_work_folder/rumor_detection/data/dataset/PHEME_extension/all-rnr-annotated-threads`:

```text
n_events                  6425
sample_event_id           552783238415265792
source_text_available     true
reply_text_available      true
timestamps_available      true
parent_relation_available true
snapshots_built           15m:1 node, 1h:5 nodes, 6h:10 nodes
future_leakage            []
failures                  []
```

The smoke is fail-closed: it demands a loadable event with recoverable
source/reply text, timestamps, at least one resolved parent relation, all
three snapshots and zero leakage — not a parent for every reply.

## 8. Frozen readers (P0-F) — the blocker

`reader_audit.json`: all three readers resolved to `model_path = ""`.

| reader key | expected model id | model_path_exists | loaded | weight/tokenizer/template hash |
|---|---|---|---|---|
| qwen | `Qwen/Qwen3-8B` | false | false | empty |
| glm | `zai-org/glm-4-9b-chat-hf` | false | false | empty |
| internlm | `internlm/internlm3-8b-instruct` | false | false | empty |

Reader model environment at run time:

```text
CRTSER_QWEN_MODEL       (unset)
CRTSER_GLM_MODEL        (unset)
CRTSER_INTERLM_MODEL    (unset)
```

Corroborating local search: a filesystem scan of `C:\`, `D:\` and `E:\` to
depth 4 matched no directory name containing `qwen3-8b`, `glm-4-9b` or
`internlm3-8b`; the HuggingFace hub cache
(`C:\Users\PC\.cache\huggingface\hub`) contains only bert-base-chinese,
all-MiniLM-L6-v2, sentiment-roberta-large-english and difraud — no
instruction-tuned 8B/9B reader.

Hardware corroboration: the only GPU is a 4.00 GiB GTX 1050 Ti
(3.28 GiB free). An 8B/9B reader in bf16 needs roughly 16–18 GiB of weights
alone, before activations and KV cache — so this machine cannot host a frozen
reader even if the weights were materialized locally. The three readers are
documented server-side resources (`resources/resource_manifest.md`).

The protocol forbids substituting a smaller, quantized, API or otherwise
different checkpoint, so the frozen reader set cannot be satisfied here.

Deployment note: this machine is the **lightweight verification node**, by
design. Data-side checkpoints (P0-A … P0-E) and all code-level checks are
reproducible locally and passed here; the heavyweight reader checkpoints
(P0-F / P0-G) belong on the synchronized server, where
`resources/resource_manifest.md` places the reader weights and where a
GPU with enough VRAM exists. The local `P0_FAIL` is therefore a
deployment-location result, not a data or protocol finding.

## 9. Teacher-forced A/B sanity (P0-G)

Not executable: `label_scoring_sanity.json` records
`status = "MODEL_PATH_MISSING"` for all three readers, with
`loaded=false`, `boundaries_ok` absent, `identical_predictions=false`.

No `generate()`, free-form generation, generated confidence, self-reported
probability or answer parsing was used anywhere; the frozen scoring contract
remains `teacher_forced_logprob_sum`.

Resulting criteria:

```text
boundaries_ok        false   (not evaluable)
identical_predictions false  (not evaluable)
identity_rate        n/a
```

## 10. P0 PASS criteria vs actual

| Criterion (amendment §10 / plan §13) | Actual |
|---|---|
| Ma-Weibo source integrity PASS | ✅ `MAWEIBO_READY` |
| Ma-Weibo viable events >= 170 | ✅ 4591 |
| PHEME smoke = OK | ✅ OK |
| Qwen loaded | ❌ model path missing |
| GLM loaded | ❌ model path missing |
| InternLM loaded | ❌ model path missing |
| all boundaries_ok | ❌ not evaluable |
| all repeated scoring deterministic | ❌ not evaluable |

→ `P0_FAIL` (a reader prerequisite is unmet).

## 11. What was deliberately NOT done

```text
formal manifest freeze          NOT RUN
formal intervention generation  NOT RUN
formal utility labels           NOT RUN
predictor / single-reader /
shared-residual training        NOT RUN
Stage-A subset freeze           NOT RUN
held-out reader evaluation      NOT RUN
P1 / P2 / P3 / P4               NOT RUN
formal pilot report             NOT RUN
```

No threshold, reader, cutoff, split size, eligibility rule or method was
changed. No historical TC-DSCR artifact was used and B2/S6 were not brought
into Ma-Weibo. No fallback was designed in response to this failure.

## 12. Historical namespace protection

```text
results/cr_tser/   (V1)  read-only throughout
results/cr_tser_v2/ (V2) all P0 outputs
```

V1 `results/cr_tser` directory SHA-256 before the V2 verifier run:
`c55d4c9429483b314982b77816166258c197ef62a45d5e7aebe02ba976b79558`
(re-checked after the verifier run — see `v1_namespace_baseline.json` and the
verifier's `v1_historical_*` checks).

## 13. Allowed next step

Per the execution plan §14/§20 this round stops at a documented
`P0_FAIL`. The only admissible remedy is a **research/environment decision**,
not a code or protocol change:

1. provision the three frozen readers on a host with sufficient VRAM
   (server workspace `/data/jyz/next/llm/model/` already holds at least the
   Qwen3-8B copy), and
2. re-run the same frozen entrypoint
   (`cr_tser_p0_audit.py --protocol v2 --readers --sanity`) there.
3. carry this package's data-side evidence (P0-A … P0-E, identical source
   fingerprint `b982076d…`) across so the server run re-verifies the same
   Ma-Weibo source of record rather than a fresh one.

Until that happens, no V2 manifest, label, training or evaluation stage may
start.
