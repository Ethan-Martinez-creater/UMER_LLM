# CR-TSER V2-P1A — Blocked by a formal-manifest-freeze defect

Round: **V2-P1A (formal manifest freeze + frozen-reader utility labels)**
Baseline: `c4773a3ec605e40d6aa26cd2244a41ccb51b998a`
Status: **STOPPED — genuine code defect, not patched**

> **Follow-up (V2-P1A manifest hotfix, commit `57895a3`).** The defect was
> approved for repair and fixed in the unified fingerprint contract
> (`project/cr_tser/data/source_manifest.py`): the `maweibo_composite` branch
> now exposes the same top-level field set as every other source kind, with
> `"exists"` a real conjunction of the raw directory and the label file.
> Ma-Weibo manifests were then built successfully (270 snapshots / 3839
> interventions) and utility labels were generated; see
> `P1A_MANIFEST_UTILITY_LABEL_REPORT.md`. The text below is the original
> stop-time record and is kept unchanged.

Per plan §10 ("If a genuine runtime/code defect blocks this frozen stage:
STOP, preserve evidence, report the defect, do not patch and continue
automatically"), the round stopped at the Ma-Weibo manifest build. No source
file was modified, and no utility label was generated.

## 1. What completed before the stop

| step | result |
|---|---|
| LOCAL/pytorch baseline (pytorch env) | 139 tests passed, code verifier `issues = 0`, `compileall` clean |
| server workspace synced | `/data/jyz/next/llm/cr_tser_ws` @ `c4773a3` |
| compatibility overlay | `PYTHONPATH=/data/jyz/next/llm/.cr_tser_v2p0/tf453` → transformers 4.53.3, tokenizers 0.21.4 (DGPA unmodified) |
| Ma-Weibo source identity | **reproduced exactly**, no drift (see §5) |
| PHEME manifest build | **succeeded** (`--dataset pheme`) |
| Ma-Weibo manifest build | **FAILED** — `KeyError: 'exists'` |
| utility labels | not started |

## 2. The defect

Command (server, DGPA + 4.53.3 overlay):

```bash
python scripts/cr_tser_build_manifests.py --dataset maweibo
```

Error:

```text
  File "/data/jyz/next/llm/cr_tser_ws/scripts/cr_tser_build_manifests.py", line 108, in <dictcomp>
    "source": {k: source[k] for k in ("kind", "path", "sha256",
                  ~~~~~~^^^
KeyError: 'exists'
maweibo_rc=1
```

Root cause — `project/cr_tser/data/source_manifest.py`, `source_fingerprint()`,
`maweibo_composite` branch (lines 101–119):

```python
record = {
    "dataset": dataset,
    "kind": kind,
    "raw_json": raw,
    "label_file": labels,
    "combined_source_sha256": combined,
    # top-level aliases so the generic identity guard below also
    # covers the composite source
    "path": raw_dir,
    "sha256": combined,
    "bytes": raw.get("bytes", 0) + labels.get("bytes", 0),
    "n_files": raw.get("n_files", 0) + labels.get("n_files", 0),
}
```

The comment claims the top-level aliases make the composite source satisfy
the generic identity consumers, but the alias set is incomplete: `path`,
`sha256`, `bytes` and `n_files` are provided, **`exists` is not**.

The non-composite branch (line 120–122) avoids the problem by delegating to
`fingerprint_path()`, which always returns `path`, `exists`, `sha256`,
`bytes`, `n_files`.

The consumer — `scripts/cr_tser_build_manifests.py` lines 108–109:

```python
"source": {k: source[k] for k in ("kind", "path", "sha256",
                                  "n_files", "exists")},
```

## 3. Blast radius

`source["exists"]` has exactly one consumer in the repository
(`cr_tser_build_manifests.py:109`; verified by grep across `scripts/` and
`project/`). Therefore:

* **Ma-Weibo (V2 primary, `maweibo_composite`) always fails** this stage;
* PHEME is unaffected (its fingerprint flows through `fingerprint_path`);
* `assert_same_source` is unaffected — it only compares `kind`, `sha256`,
  `n_files`, all of which the composite record does provide;
* `cr_tser_p0_audit.py` is unaffected: P0 does not consume
  `source["exists"]`, which is why the V2-P0 approval round passed.

Why this was never seen before: the V1 candidate path used Weibo22
(non-composite), and the V2-M0 migration only ever exercised the PHEME
manifest path. The Ma-Weibo formal manifest build had never actually been
executed until this round.

Observed side effect on the server — `manifests/maweibo/source.json` **was**
written (the `write_frozen_source` call at line 101 precedes the failing
dict comprehension) and it also lacks a top-level `exists`:

```json
{
 "dataset": "maweibo",
 "kind": "maweibo_composite",
 "raw_json": {"path": "...", "exists": true, "sha256": "c6afd1c5…", "bytes": 3995877128, "n_files": 4664},
 "label_file": {"path": "...", "exists": true, "sha256": "032f10e2…", "bytes": 64463295, "n_files": 1},
 "combined_source_sha256": "b982076d…",
 "path": "/data/jyz/next/llm/data/maweibo_raw",
 "sha256": "b982076d…",
 "bytes": 4060340423,
 "n_files": 4665
}
```

No `event_split.json`, `snapshot_manifest.jsonl`,
`intervention_manifest.jsonl` or `hashes.json` was produced for Ma-Weibo.

## 4. Suggested fix (NOT applied)

Add the missing alias in the composite branch of `source_fingerprint()`, e.g.

```python
"exists": bool(raw.get("exists")) and bool(labels.get("exists")),
```

A round-trip regression test that runs `build_manifests` over a synthetic
composite source and asserts every key in
`("kind", "path", "sha256", "n_files", "exists")` exists would prevent
recurrence. Both changes are scientific-protocol-neutral and belong in a
separately approved fix round.

A re-run after the fix does **not** need `--force`: `manifests/maweibo/`
holds only `source.json`, so `assert_manifests_mutable` finds no
`event_split.json` and treats the build as a clean start. The existing
`source.json` also passes `assert_same_source` (kind/sha256/n_files all match).

## 5. Source identity (confirmed, no drift)

```text
CRTSER_MAWEIBO_RAW    = /data/jyz/next/llm/data/maweibo_raw
CRTSER_MAWEIBO_LABELS = /data/jyz/next/llm/data/maweibo_labels.txt
raw_json     : 4664 files, 3,995,877,128 bytes, sha256 c6afd1c50a6cde8c27137d367d8e420b4a3afc43e017e3b19846e8b001ca1a27
label_file   : 64,463,295 bytes, sha256 032f10e2175fa461203bdba77ef2a492e0ebde1cdd24b9bcd2100e307bc6b8e4
combined_source_sha256 = b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d
MATCH_EXPECTED True
```

Environment actually used: DGPA + `PYTHONPATH` overlay
(`/data/jyz/next/llm/.cr_tser_v2p0/tf453`), python 3.11.13, torch
2.7.1+cu128, transformers 4.53.3, tokenizers 0.21.4, RTX 4090.
Semantic encoder: `/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2`
(384d, the audited historical TC-DSCR encoder). Canonical tokenizer:
`/data/jyz/next/llm/model/qwen3-8b` (`CANONICAL_TOKENIZER_ID = "Qwen/Qwen3-8B"`).

## 6. Server artifacts left in place (not committed)

PHEME manifests were built successfully and are left at their authoritative
server location. They are **not frozen**: no utility label row exists yet, so
per plan §31 they remain rebuildable with `--force`.

```text
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/pheme/event_split.json
  f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/pheme/hashes.json
  4b865d5a811efe74a560e921bf888e6a7838ef3a20c854dbaa8b2bcf5b8b08f3
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/pheme/snapshot_manifest.jsonl
  8c2ec0462a1fbf9ff681d237fb9e57df09afd9af5c7e402779c2defe8812315f
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/pheme/intervention_manifest.jsonl
  a0c0ad7abb5c8132ecb9483143ad8c70bc6836568a183e32797a8af73201d2e3
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/pheme/source.json
  1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363
/data/jyz/next/llm/cr_tser_ws/results/cr_tser_v2/manifests/maweibo/source.json
  1bb858b110b692f3719fea05ca4dda59d4d7c5cd6c6dd48caf073bf4254ac62c
```

PHEME build statistics reported by the frozen builder: 270 snapshots,
3412 interventions, 9 zero-reply snapshots, 5669 viable events,
max snapshot nodes 79.

## 7. Not run

Utility-label generation (all six dataset/reader combinations), label-cache
audit, `cr_tser_verify_pilot.py --mode code|pilot --protocol v2` on the server
for this stage, predictor training, Stage A/B, held-out evaluation, P1–P4,
final Pilot report.

## 8. Evidence in this directory

```text
server_manifest_build.log     full server run log (fingerprint check, both builds, traceback)
maweibo_fingerprint.json      composite fingerprint reproduced on the server
P1A_DEFECT_REPORT.md          this report
```
