# BCR-Utility M1-E — Conditional-GO Attribution Audit Report

```text
protocol            = bcr_v1
approved M1 commit  = fef08cd581fab3b2208cc5b55c49cccb5e9915d4
M1-E code commit    = 1c243ac0ad788407d108fb01adbeada5cdb90ecb
STATUS              = INFRASTRUCTURE_PAUSE
frozen M1 verdict   = M1_CONDITIONAL_GO (unchanged, re-verified)
```

## 0. Status and scope

M1-E is a **post-hoc diagnostic attribution audit**. It defines no gate,
modifies no frozen M1 artifact, calls no reader and generates no label.

| task | content | status |
|---|---|---|
| A | frozen-evidence pinning | **DONE** (LOCAL) |
| B | feature-group ablations D0/D2/D3/D5 | **BLOCKED** (SERVER) |
| C | fingerprint increment D4 vs D5 | **BLOCKED** (SERVER) |
| D | dataset-shift audit | **DONE** (LOCAL) |
| E | reader/cutoff/class concentration | **DONE** (LOCAL) |
| F | B3 deployment-contract audit | **DONE** (LOCAL) |

Tasks B/C require SERVER training and were stopped by the connection
contract: the approved wrappers returned

```text
scp: Connection closed
scp: Connection closed
Connection closed by 60.215.128.50 port 32985
Connection closed by 60.215.128.50 port 32985
```

No host/port/user was changed, no alternative SSH/SCP was used, no retry
loop was entered, and no server work was moved to LOCAL. The raw error is
preserved in `INFRASTRUCTURE_PAUSE.json`. Tasks B/C resume unchanged once
the tunnel is restored.

## 1. Task A — frozen M1 evidence (fail-closed)

```text
commit verified   = fef08cd581fab3b2208cc5b55c49cccb5e9915d4
artifacts pinned  = 36 files (git blob + raw SHA256 + bytes each)
tracked changes under results/bcr_utility_v1/m1/ = 0
M1 verdict        = M1_CONDITIONAL_GO  (zero_touch_passed=False,
                                        light_touch_passed=True)
```

Every M1 artifact hashes to its committed blob; `M1_VERDICT.json`,
probe responses, fingerprints, E0/E1/E2/E3, both evaluation files, both
prediction files and the historical caches are byte-identical to the
approved commit. Artifact: `m1_evidence_pins.json`
(sha256 `5aff65110076cae53d5f1245168150515ce0ee148b0b0f4ad4d3d30574fcb558`).

## 2. Task D — Ma-Weibo vs PHEME E3 shift

Per reader x cutoff x feature: mean / std / median / IQR on both datasets and
the fixed 1-D Wasserstein distance (`dataset_shift.json`). Mean levels at
cutoff 15m (Ma-Weibo → PHEME) with W1:

| reader | feature | Ma-Weibo | PHEME | W1 |
|---|---|---|---|---|
| qwen | source_only_margin | **−6.17** | **+9.15** | **15.73** |
| qwen | conditional_evidence_nll | 2.45 | 3.07 | 0.67 |
| qwen | nll_gap | 1.99 | 1.57 | 0.53 |
| qwen | source_nll | 4.15 | 4.72 | 0.76 |
| mistral | source_only_margin | 9.61 | 7.41 | 2.30 |
| mistral | conditional_evidence_nll | 1.62 | 2.29 | 0.69 |
| mistral | nll_gap | 1.71 | 1.30 | 0.45 |
| internlm | source_only_margin | −0.41 | +0.17 | 0.82 |
| internlm | conditional_evidence_nll | 2.02 | 2.56 | 0.56 |
| internlm | nll_gap | 1.82 | 1.41 | 0.43 |

The same direction holds at 60m and 360m (see the artifact). Two systematic
shifts stand out:

1. **source-only A/B margin flips sign for qwen** (−6.17 → +9.15, W1 15.7)
   and moves substantially for mistral; only internlm's margin stays small;
2. **evidence-familiarity features move coherently**: the conditional
   evidence NLL rises on PHEME while the NLL gap shrinks, for all three
   readers (W1 0.43–0.69 on a NLL scale of ≈1–4), i.e. PHEME's evidence is
   systematically less predictable given its source.

Association with the frozen target (per reader, per split, in
`dataset_shift.json`) is **weak and reader-dependent**: single-feature
Spearman with continuous utility stays within |ρ| ≤ 0.20 on every split, and
HELPFUL AUROC of `source_nll` ranges from 0.70 (qwen) to 0.39 (mistral) on
Ma-Weibo utility_train. Association only — no causal claim is made.

**Reading:** PHEME's B5 failure coincides with a clear, systematic E3
domain shift (especially the qwen source-margin sign flip and the
conditional-NLL / NLL-gap movement). This is an association observed on the
same frozen artifacts, not a demonstrated cause.

## 3. Task E — reader / cutoff / class concentration (existing predictions)

Ma-Weibo B5−B0 (aggregate +0.0616), decomposed:

| level | value |
|---|---|
| readers | qwen +0.1247, mistral +0.0469, internlm +0.0131 |
| cutoffs | 15m +0.0692, 60m +0.0511, 360m +0.0601 |
| classes | HELPFUL +0.0467, NEUTRAL +0.0348, HARMFUL +0.1031 |

```text
largest reader contribution = qwen     (0.68 of the positive reader mass)
largest cutoff contribution = 15m      (0.38 of the positive cutoff mass)
largest class contribution  = HARMFUL  (0.56 of the positive class mass)
```

So the Ma-Weibo gain is **not** a single-cutoff artifact — all three cutoffs
move by a similar amount. It is, however, reader-skewed (qwen carries most
of it) and class-skewed (HARMFUL dominates, consistent with B5's higher
HARMFUL recall). For contrast, the failed ZERO-TOUCH B4−B0 (−0.0038) is
dragged down by mistral (−0.0978) and by the 360m cutoff (−0.0331), while
PHEME's LIGHT-TOUCH failure is negative in all three cutoffs, all three
readers except mistral, and concentrated in NEUTRAL (−0.2872).

## 4. Task F — B3 deployment contract

```text
Ma-Weibo : same-evidence cross-reader transfer = TRUE
           prediction == nearest training reader's label at the same key
           (611/611 rows in every rotation)
           nearest reader chosen from fingerprint distance only = TRUE
           held-out utility labels used for selection = FALSE
PHEME    : identical contract (695/695 rows per rotation)
```

**Conclusion (verbatim, also in `b3_contract.json`):** *B3 is a
same-evidence cross-reader transfer baseline: it predicts a held-out
reader's utility for an evidence key by copying the real utility label of
the fingerprint-nearest training reader at that same key. It therefore
requires the evidence to already carry a utility label from at least one
existing reader and is not equivalent to a new-event deployment scenario
that calls no additional utility oracle.*

B3's ≈0.396 Macro-F1 is reported as-is; nothing was changed to lower it.

## 5. Task B/C — pending (blocked, not executed)

The frozen variant definitions are implemented and tested
(`attribution/feature_ablation.py`, `scripts/bcr_run_attribution.py`):
D0 = Z, D1 = Z+F (= frozen B4, reused), D2 = Z+F+S, D3 = Z+F+C,
D4 = Z+F+S+C (= frozen B5, reused), D5 = Z+S+C (E3 present, no
fingerprint), all through the identical LORO / splits / 8-config grid /
3 seeds / 10000-iteration seed-7319 event bootstrap. **No D0–D5 result is
reported here because the SERVER training could not run.**

## 6. Task G — research summary

```text
Established
  reader-specific atomic evidence utility exists (frozen M0/M1 evidence)

Failed
  the zero-touch BCR mechanism (B4 = evidence + behavioral fingerprint)
  on Ma-Weibo: mean Δ(B4-B0) = -0.0038, CI covers 0, worst -0.0978

Promising (single dataset, mechanism not yet attributed)
  light-touch reader-forward compatibility (B5) on Ma-Weibo:
  mean Δ(B5-B0) = +0.0616, CI [+0.0068, +0.1084], 3/3 positive,
  worst +0.0131; gain is cutoff-balanced but reader-skewed (qwen 0.68)
  and class-skewed (HARMFUL 0.56)

Unresolved
  cross-dataset robustness: PHEME LIGHT-TOUCH = -0.1063 (all cutoffs
  negative) and coincides with a systematic E3 domain shift, most visibly
  the qwen source-margin sign flip and the conditional-NLL / NLL-gap move
  behavioral-fingerprint incremental contribution (Task C, blocked)
  source-state vs evidence-familiarity contribution (Task B, blocked)
  B3's same-evidence cross-reader transfer requires an existing-reader
  utility oracle label, so it does not answer the new-event deployment
  question
```

Explicitly **not** claimed anywhere in this report: that BCR generalises,
that unseen-reader transfer is solved, or that LIGHT-TOUCH holds across
datasets. None of those is supported by the current evidence.

## 7. Immutability and boundary confirmations

```text
frozen M1 artifacts / verdict : unchanged (36/36 blob-pinned, 0 tracked changes)
new reader inference          : none (no reader was loaded in M1-E)
new utility labels            : none
new gate defined              : none (all M1-E output is post-hoc diagnostic)
reader panel expansion        : none; Phi/Gemma/Qwen3-0.6B not deployed
M2                            : not entered
```

Artifacts under `results/bcr_utility_v1/m1e/`: `m1_evidence_pins.json`,
`dataset_shift.json`, `concentration.json`, `b3_contract.json`,
`M1E_VERDICT.json`, `INFRASTRUCTURE_PAUSE.json`, this report.
