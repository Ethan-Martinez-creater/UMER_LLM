# BCR-Utility M1-E — Conditional-GO Attribution Audit Report

```text
protocol            = bcr_v1
approved M1 commit  = fef08cd581fab3b2208cc5b55c49cccb5e9915d4
M1-E code commits   = 1c243ac, 2184e63, 41a9b7e
STATUS              = COMPLETED (all six tasks)
frozen M1 verdict   = M1_CONDITIONAL_GO (unchanged, re-verified)
```

M1-E is a **post-hoc diagnostic attribution audit**. It defines no gate,
modifies no frozen M1 artifact, calls no reader and generates no label. The
one infrastructure interruption (server connection closed during the first
Task B/C attempt) is preserved in `INFRASTRUCTURE_PAUSE.json`; the tasks
were resumed unchanged after the tunnel was restored.

## 1. Task A — frozen M1 evidence (fail-closed)

```text
commit verified   = fef08cd581fab3b2208cc5b55c49cccb5e9915d4
artifacts pinned  = 36 files (git blob + raw SHA256 + bytes each)
tracked changes under results/bcr_utility_v1/m1/ = 0
M1 verdict        = M1_CONDITIONAL_GO (zero_touch_passed=False,
                                       light_touch_passed=True)
```

Artifact: `m1_evidence_pins.json`.

## 2. Task B — feature-group ablations (frozen protocol)

Groups: **Z** = E0+E1+E2, **F** = behavioral fingerprint, **S** = source-state
E3, **C** = evidence-familiarity E3. Every variant ran the identical
three-reader LORO, `utility_train`/`dev`/`eval`, 8-config grid, 3 seeds, and
the 10000-iteration seed-7319 event bootstrap. D1 = frozen B4 and
D4 = frozen B5 are reused verbatim (their re-exported predictions reproduce
the frozen M1 numbers exactly: −0.0038 and +0.0616).

Per-reader Macro-F1 and the paired bootstrap against the frozen **B0** gate
baseline (the M1 gate's own comparison):

| variant | Ma-Weibo qwen / mistral / internlm | Ma-Weibo Δ vs B0 (95% CI) | PHEME Δ vs B0 (95% CI) |
|---|---|---|---|
| D0 = Z | 0.1948 / 0.1588 / 0.2138 | **−0.0471** [−0.0793, −0.0183] | +0.0136 [−0.0136, +0.0439] |
| D1 = Z+F (=B4) | 0.2400 / 0.1649 / 0.2924 | −0.0038 [−0.0360, +0.0255] | +0.0312 [+0.0033, +0.0622] |
| D2 = Z+F+S | 0.2736 / 0.2856 / 0.2940 | +0.0482 [+0.0164, +0.0788] | −0.0718 [−0.1041, −0.0392] |
| D3 = Z+F+C | 0.2585 / 0.2818 / 0.2241 | +0.0185 [−0.0188, +0.0496] | −0.0704 [−0.1062, −0.0324] |
| D4 = Z+F+S+C (=B5) | 0.3535 / 0.3095 / 0.2303 | +0.0616 [+0.0068, +0.1084] | −0.1063 [−0.1367, −0.0759] |
| D5 = Z+S+C (no F) | 0.2943 / 0.3035 / 0.3239 | **+0.0710** [+0.0448, +0.0974] | +0.0277 [−0.0128, +0.0595] |

Relative to the frozen D0 baseline (`attribution/<dataset>.json`,
`comparisons_vs_D0`):

```text
Ma-Weibo   D1 +0.0433 [+0.0130,+0.0755] 3/3   D2 +0.0953 [+0.0537,+0.1417] 3/3
           D3 +0.0657 [+0.0376,+0.0940] 3/3   D4 +0.1087 [+0.0604,+0.1538] 3/3
           D5 +0.1181 [+0.0768,+0.1639] 3/3   (worst reader +0.0995)
PHEME      D1 +0.0176 [-0.0115,+0.0462] 2/3   D2 -0.0854 [-0.1265,-0.0462] 1/3
           D3 -0.0840 [-0.1157,-0.0548] 1/3   D4 -0.1199 [-0.1635,-0.0782] 0/3
           D5 +0.0141 [-0.0335,+0.0522] 1/3
```

Three findings follow directly:

1. **E2 alone is negative on Ma-Weibo**: D0 (Z, no fingerprint, no E3) is
   −0.0471 against B0, 0/3 readers positive. The ZERO-TOUCH evidence group
   does not by itself beat the E0+E1 baseline.
2. **The E3 groups carry the gain**: with F present, adding S (D2) is worth
   +0.0953 and adding C (D3) +0.0657 on Ma-Weibo — S is the stronger group,
   and S+C (D4/D5) is stronger than either alone.
3. **The gain is dataset-conditional**: on PHEME every F-containing variant
   with E3 is clearly negative (D2 −0.0854, D3 −0.0840, D4 −0.1199), while
   the same features without F (D5) return to neutral (+0.0141).

## 3. Task C — fingerprint incremental contribution (D4 vs D5)

| dataset | mean Δ(D4−D5) | 95% CI | positive readers | per reader |
|---|---|---|---|---|
| Ma-Weibo | **−0.0094** | [−0.0675, +0.0435] | 2/3 | qwen +0.0592, mistral +0.0060, internlm −0.0935 |
| PHEME | **−0.1340** | [−0.1582, −0.1000] | 0/3 | qwen −0.0982, mistral −0.1672, internlm −0.1367 |

Per-reader secondary metrics (`comparisons_vs_B0.json`):

| dataset | reader | D4 MacroF1 / HARMFUL-AUPRC / Spearman | D5 MacroF1 / HARMFUL-AUPRC / Spearman |
|---|---|---|---|
| Ma-Weibo | qwen | 0.3535 / 0.1067 / −0.0601 | 0.2943 / **0.1442** / **+0.1079** |
| Ma-Weibo | mistral | 0.3095 / 0.0812 / +0.0375 | 0.3035 / 0.0799 / +0.0375 |
| Ma-Weibo | internlm | 0.2303 / 0.0689 / +0.0946 | **0.3239** / **0.1434** / −0.0709 |
| PHEME | qwen | 0.0819 / 0.0987 / +0.1378 | **0.1801** / **0.1094** / +0.0314 |
| PHEME | mistral | 0.2066 / 0.1360 / −0.1214 | **0.3738** / **0.1734** / +0.0047 |
| PHEME | internlm | 0.1344 / 0.2571 / −0.0257 | **0.2711** / **0.2801** / +0.0532 |

**Answer to the Task C question — does the behavioral fingerprint still add
increment once E3 exists? No.** On Ma-Weibo the increment is −0.0094 with a
CI covering zero (and it makes the worst reader worse: D4 worst +0.0131 vs
D5 worst +0.0409). On PHEME it is −0.1340, significantly negative, and D5
beats D4 on Macro-F1 for all three readers and on HARMFUL AUPRC for all
three. The frozen B5 is untouched by this finding.

## 4. Task D — Ma-Weibo vs PHEME E3 shift

Per reader x cutoff x feature mean/std/median/IQR on both datasets plus the
fixed 1-D Wasserstein distance (`dataset_shift.json`). Mean levels at 15m
(Ma-Weibo → PHEME) with W1:

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

The same direction holds at 60m and 360m. Two systematic shifts:

1. **the source-only A/B margin flips sign for qwen** (−6.17 → +9.15,
   W1 15.7) and moves substantially for mistral;
2. **the evidence-familiarity features move coherently**: the conditional
   evidence NLL rises on PHEME while the NLL gap shrinks, for all three
   readers (W1 0.43–0.69 on a NLL scale of ≈1–4).

Association with the frozen target is **weak and reader-dependent**:
single-feature Spearman with continuous utility stays within |ρ| ≤ 0.20 on
every split, and HELPUL AUROC of `source_nll` ranges from 0.70 (qwen) to
0.39 (mistral) on Ma-Weibo utility_train. Association only — no causal
claim. PHEME's LIGHT-TOUCH failure and its fingerprint-driven damage
(§2–§3) coincide with this shift; the audit reports the coincidence.

## 5. Task E — reader / cutoff / class concentration

Ma-Weibo B5−B0 (+0.0616) decomposed from the existing predictions:

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

The gain is **not** a single-cutoff artifact (all three cutoffs move
similarly) but is reader-skewed (qwen) and class-skewed (HARMFUL). For
contrast: the failed B4−B0 (−0.0038) is dragged down by mistral (−0.0978)
and the 360m cutoff (−0.0331); PHEME's B5 failure is negative in all three
cutoffs and concentrated in NEUTRAL (−0.2872).

## 6. Task F — B3 deployment contract

```text
Ma-Weibo : same-evidence cross-reader transfer = TRUE
           prediction == the fingerprint-nearest training reader's real
           label at the same key (611/611 rows, every rotation)
           nearest reader chosen from fingerprint distance only = TRUE
           held-out utility labels used for selection = FALSE
PHEME    : identical contract (695/695 rows per rotation)
```

**Conclusion:** *B3 is a same-evidence cross-reader transfer baseline: it
predicts a held-out reader's utility for an evidence key by copying the real
utility label of the fingerprint-nearest training reader at that same key.
It therefore requires the evidence to already carry a utility label from at
least one existing reader and is not equivalent to a new-event deployment
scenario that calls no additional utility oracle.*

B3's ≈0.396 Macro-F1 is reported as-is; nothing was changed to lower it.

## 7. Task G — research summary

```text
Established
  reader-specific atomic evidence utility exists (frozen M0/M1 evidence)
  B3 is a same-evidence cross-reader transfer baseline that consumes an
  existing-reader utility label

Failed
  the zero-touch BCR mechanism (B4 = Z + fingerprint) on Ma-Weibo:
  Δ(B4-B0) = -0.0038, CI covers 0, worst -0.0978; D0 shows the
  reader-agnostic Z group alone is -0.0471 against B0

Promising (single dataset, attribution now known)
  the LIGHT-TOUCH Ma-Weibo gain (+0.0616 vs B0) is attributable to the E3
  source-state and evidence-familiarity features: D5 (E3 without the
  fingerprint) reaches +0.0710 with 3/3 positive readers and the best worst
  reader (+0.0409)

Unresolved / negative
  behavioral-fingerprint incremental contribution: NONE with E3 present
  (Ma-Weibo -0.0094, CI covers 0; PHEME -0.1340, significantly negative)
  cross-dataset robustness: PHEME LIGHT-TOUCH = -0.1063 (all cutoffs
  negative) and coincides with a systematic E3 domain shift, most visibly
  the qwen source-margin sign flip; the same E3 groups are negative on
  PHEME whenever the fingerprint is present
  source-state vs evidence-familiarity: S is the stronger group on
  Ma-Weibo (D2 +0.0953 vs D3 +0.0657 relative to D0) but both are negative
  on PHEME with F present, so their cross-dataset behaviour is not settled
```

Explicitly **not** claimed: that BCR generalises, that unseen-reader
transfer is solved, or that LIGHT-TOUCH holds across datasets. None is
supported by the current evidence. The frozen `M1_CONDITIONAL_GO` verdict
stands as recorded; M1-E only explains what that result was made of.

## 8. Immutability and boundary confirmations

```text
frozen M1 artifacts / verdict : unchanged (36/36 blob-pinned, 0 tracked changes)
new reader inference          : none (no reader was loaded in M1-E)
new utility labels            : none
new gate defined              : none (all M1-E output is post-hoc diagnostic)
reader panel expansion        : none; Phi/Gemma/Qwen3-0.6B not deployed
M2                            : not entered
```

Artifacts under `results/bcr_utility_v1/m1e/`: `m1_evidence_pins.json`,
`attribution/{maweibo,pheme}.json`, `attribution/predictions_<dataset>_<variant>.jsonl`
(12 files), `attribution.log`, `dataset_shift.json`, `concentration.json`,
`b3_contract.json`, `comparisons_vs_B0.json`, `M1E_VERDICT.json`,
`INFRASTRUCTURE_PAUSE.json`, this report.
