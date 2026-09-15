# CR-TSER V2-P0 Compatibility Closure — Execution Report

Round: **V2-P0 Compatibility Closure**
Code commit for this round: `808fd83e61f71b0935a5a54543859519e6d0473d`
Baseline entering the round: `99dab765266aa8cee18baa66e1d2237c83eead4c`
Frozen scientific baseline: `fa8ddbbc6f2b4b86f963a8913a22d2f88b8bd82f`

This round resolved exactly two engineering problems and changed no science:

1. the InternLM3 ↔ Transformers compatibility failure in the server `DGPA`
   environment;
2. the Windows-CRLF ↔ Linux-LF instability of the V1 immutable verifier pins.

**Result: `P0_PASS`.**

## 1. Dual-environment execution contract

```text
LOCAL  env  = pytorch   (lightweight verification)
SERVER env  = DGPA      (GPU / large-model / formal runs)
server root = /data/jyz/next/llm/
```

Server commands were issued only through `next/.codex_tmp_ssh_run.cmd`; file
transfer only through `next/.codex_tmp_scp_run.cmd`.

## 2. Task A — InternLM3 compatibility closure

### 2.1 Root cause (confirmed earlier on the server)

```text
internlm/internlm3-8b-instruct  modeling_internlm3.py line 40:
    from transformers.utils import LossKwargs
DGPA transformers 4.57.6:
    transformers.utils.LossKwargs            -> absent
    transformers.utils.generic.LossKwargs    -> absent
    built-in internlm3 model type            -> absent
```

The frozen `InternLMReader` uses `trust_remote_code=True`, so the official
`auto_map` implementation is executed and cannot import. The official
`modeling_internlm3.py` was **not** modified and `LossKwargs` was **not**
patched.

### 2.2 Rollback-safe isolation (DGPA itself untouched)

DGPA was snapshotted first (`dgpa_pip_freeze_before.txt`, shipped in the
evidence package):

```text
python 3.11.13 / torch 2.7.1+cu128 / transformers 4.57.6
tokenizers 0.22.2 / accelerate 1.9.0 / safetensors 0.7.0 / numpy 1.26.4
```

The candidate version was then installed into an **isolated target
directory** and injected with `PYTHONPATH`, leaving `DGPA`'s site-packages
byte-untouched:

```text
/data/jyz/next/llm/.cr_tser_v2p0/tf453
transformers 4.53.3
+ tokenizers 0.21.4   (4.53.3 requires tokenizers>=0.21,<0.22)
```

Raw `pypi.org` is unreachable from the server; the Tsinghua mirror answers
HTTP 200 and served both wheels.

Rollback is `unset PYTHONPATH` (or delete the `tf453` directory); nothing in
DGPA was modified, so no other project on that machine is affected.

### 2.3 Version confirmation under `PYTHONPATH` override

```text
transformers 4.53.3   (/data/jyz/next/llm/.cr_tser_v2p0/tf453/transformers/__init__.py)
tokenizers   0.21.4
torch        2.7.1+cu128   (DGPA, unchanged)
LossKwargs import: OK
```

### 2.4 Frozen reader load audit (all three)

```text
qwen      loaded=True
glm       loaded=True
internlm  loaded=True        <-- previously ImportError
ALL_LOADED True
```

### 2.5 Frozen teacher-forced A/B sanity (all three, 20 prompts × 2)

Executed with the frozen `label_scoring_sanity` implementation, so the
scoring contract is unchanged (`teacher_forced_logprob_sum`,
`generated_text_used=false`, `generated_confidence_used=false`).

| reader | loaded | boundaries_ok | identical_predictions | identity_rate | A/B distribution |
|---|---|---|---|---|---|
| qwen | true | true | true | 1.0 | A=20, B=0 |
| glm | true | true | true | 1.0 | A=0, B=20 |
| internlm | true | true | true | 1.0 | A=17, B=3 |

```text
ALL_SANITY_OK True
```

Qwen and GLM were re-scored under 4.53.3 as the minimal regression required
by the plan: both still satisfy the original frozen scoring contract, with
finite log-probabilities, no NaN/Inf and no empty continuations. The full
Ma-Weibo / PHEME data audit was deliberately not repeated as part of this
check (it was re-run afterwards as part of the authoritative P0 entrypoint).

## 3. Task B — cross-platform immutable verifier

### 3.1 The defect

`_verify_v1_historical_immutable` pinned **raw worktree bytes**. A Windows
checkout with `core.autocrlf=true` and a Linux checkout carry the same git
blob but different bytes (8604 vs 8298 for `code_verify.json`), so the pins
could never be satisfied on both hosts. `_dir_sha256` additionally used
`os.path.relpath` directly, leaking the platform path separator into the
digest.

### 3.2 The fix

Identity is now the **canonical LF digest**:

* `_canonical_bytes(data)` normalizes CRLF → LF;
* `_canonical_sha256(path)` digests a file's canonical content;
* `_dir_sha256(path)` LF-normalizes content *and* POSIX-normalizes relative
  paths (`os.sep` → `/`).

New pins (canonical digests):

```text
results/cr_tser/verifier/code_verify.json   134b32e2f00765ab221ee710a8def67ede91b009b97da95beecfac4463e22f0e
results/cr_tser/verifier/pilot_verify.json  c81000a97ed82c323c1953049d83fa864f9eba2b83640c6211af810076a061af
results/cr_tser/verifier (dir)              aba9f5169d63b39c7e423d886e01ec5e40547f43aa0f46a5c937bf73df4d90c2
```

Those three values are exactly the digests the Linux LF worktree already had,
and they are also what the CRLF Windows worktree now canonicalizes to —
verified on both hosts. The V1 content is unchanged; only the *identity
function* changed.

### 3.3 Immutability is not relaxed

The digest is computed over file content, so a real edit still changes it.
`test_v1_immutability_fails_on_tampered_artifact` rewrites `"issues": 0` to
`"issues": 7` and asserts both the per-file pin and the directory pin flip to
`fail`.

### 3.4 Regression tests

New file `project/cr_tser/tests/test_v1_immutability_cross_platform.py`
(11 tests):

```text
canonical bytes normalize CRLF; bare CR is left alone
canonical sha256 is line-ending independent
canonical sha256 detects a real content change
directory digest ignores line endings / detects content changes
the live pins equal the canonical digests of the real files
the check passes on an LF checkout and on a CRLF checkout (parametrized)
the check fails on a tampered artifact
the check fails when an artifact is missing
```

## 4. Final verification on SERVER / DGPA

Environment actually used for the authoritative run:

```text
CONDA_DEFAULT_ENV = DGPA
PYTHONPATH        = /data/jyz/next/llm/.cr_tser_v2p0/tf453
python            = 3.11.13
torch             = 2.7.1+cu128
transformers      = 4.53.3
tokenizers        = 0.21.4
GPU               = RTX 4090, 23.52 GiB
repository        = /data/jyz/next/llm/cr_tser_ws @ 808fd83
```

| step | command | result |
|---|---|---|
| code verifier | `cr_tser_verify_pilot.py --mode code --protocol v2` | **issues = 0, pending = 0** |
| authoritative P0 | `cr_tser_p0_audit.py --protocol v2 --readers --sanity` | **P0_PASS** (exit 0) |
| pilot verifier | `cr_tser_verify_pilot.py --mode pilot --protocol v2` | **issues = 0, pending = 1** |

`p0_readiness.json`:

```json
{
 "protocol": "v2",
 "P0": "P0_PASS",
 "maweibo_source_integrity": true,
 "maweibo_source_verdict": "MAWEIBO_READY",
 "maweibo_viable_events": 4591,
 "maweibo_viable_required": 170,
 "maweibo_viable_ok": true,
 "pheme_smoke": "OK",
 "pheme_smoke_ok": true,
 "readers_ready": true,
 "readers_checked": true,
 "ab_sanity_ok": true,
 "ab_boundaries_ok": true,
 "next_step": "COMMIT PUSH STOP — research approval required before expensive labels (amendment V2 §26)"
}
```

Reader identities recorded by the frozen audit (`reader_audit.json`,
`transformers_version = 4.53.3` for all three):

| reader | weight_hash | tokenizer_hash | loaded |
|---|---|---|---|
| qwen | `345a676964219bec…` | `26a5805b76938647…` | true |
| glm | `50b8b4f31d1ce5d7…` | `263851b177e6d7a0…` | true |
| internlm | `3ac547bf153173e2…` | `c0ed5fe5c9f2d713…` | true |

The pilot verifier `pending = 1` is the expected `pilot_artifacts` entry (no
manifests; P1–P4 not run). No downstream artifact was fabricated.

An earlier attempt in this round ran the chain before the server had fetched
the new commit — the server's first `git fetch` failed with
`GnuTLS recv error (-54)`, so it silently stayed on `b5f91e1` and reproduced
the old three CRLF failures. The fetch was retried successfully
(`b5f91e1..808fd83`), the workspace was checked out to `808fd83`, and the
whole chain was re-run; the results reported above are from that corrected
run.

## 5. LOCAL / pytorch checks

```text
python -m pytest project/cr_tser/tests -q                 -> 139 passed
python scripts/cr_tser_verify_pilot.py --mode code --protocol v2
                                                          -> issues = 0
python -m compileall project/cr_tser scripts              -> clean
```

The local machine loaded none of the three large readers.

## 6. Files changed in this round

```text
scripts/cr_tser_verify_pilot.py                              (canonical pins + helpers)
project/cr_tser/tests/test_v1_immutability_cross_platform.py (new, 11 tests)
docs/CR_TSER_IMPLEMENTATION_NOTES.md                         (section 20)
results/cr_tser_v2/p0/*                                      (P0_PASS evidence)
results/cr_tser_v2/verifier/*                                (issues = 0)
```

No file under `project/cr_tser` other than the new test file was touched, and
no scientific threshold, reader definition, cutoff, split or P1–P4 rule was
changed.

## 7. Explicitly NOT run

```text
formal manifest freeze        NOT RUN
formal utility labels         NOT RUN
predictor training            NOT RUN
Stage A / Stage B             NOT RUN
held-out reader evaluation    NOT RUN
P1 / P2 / P3 / P4             NOT RUN
formal Pilot                  NOT RUN
```

## 8. Environment follow-up note

`P0_PASS` was obtained with transformers 4.53.3 injected through `PYTHONPATH`
while DGPA itself stayed at 4.57.6. Every later CR-TSER stage that loads the
three readers must use the same override (the run scripts and the evidence
`environment.json` record it), or the R3 reader will fail again under the
unmodified DGPA default. Making 4.53.3 the DGPA default — if desired — is an
environment decision for the research owner, outside this round.
