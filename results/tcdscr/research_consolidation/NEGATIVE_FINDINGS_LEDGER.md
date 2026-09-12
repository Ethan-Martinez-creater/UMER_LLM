# TC-DSCR Negative Findings Ledger

A negative finding is a hypothesis that a designed experiment falsified. None of these entries may be deleted or softened to make the paper story look better; they are part of the contribution (see `FINAL_CONTRIBUTION_MATRIX.md`, Contribution E).

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

---

## N01 — Point-wise novelty + persistence dynamic reranking

**Hypothesized.** A temporal dynamic bonus (novelty of new replies + persistence of retained evidence) added to the static utility score would make early detection follow the evolving propagation structure and improve Macro-F1.

**Falsified by.** E3 held-out test (30 frozen runs, 94,104 + 79,173 rows). Delta Macro-F1 was +0.0001037 (PHEME) and +0.0000120 (Ma-Weibo); both paired bootstrap CIs cross zero. The E3 failure diagnosis then measured *why*: the bonus std was only 1.9-3.9% of the utility std, and selection changes were invisible to the proxy (prediction change given selection change 0.0071 / 0.0043).

**What was learned.** Ranking change is not selection change, and selection change is not prediction change. A point-wise bonus can move the ranking while leaving the budgeted top-k set, and therefore the decision, untouched. Scale calibration relative to the base utility matters as much as the bonus design.

**How it influenced the next design.** The next stage stopped scoring individual candidates and instead refined the *set* (MF-TSR), which is the only way to change what the reader actually sees.

## N02 — Teacher-KL fidelity as a downstream utility surrogate

**Hypothesized.** If a refined evidence subset reproduces the full-graph teacher distribution (low teacher-KL distortion), the downstream classification should be preserved or improved.

**Falsified by.** Dynamic V2 / MF-TSR validation. The refiner did reduce distortion and change the set substantially (set-change rate 0.561-0.976), yet downstream Macro-F1 moved by +0.00293 on PHEME and -0.00286 on Ma-Weibo against a -0.002 non-inferiority floor: gate FAIL on both datasets.

**What was learned.** Distribution matching against a teacher is not the same objective as preserving the decision-relevant evidence. A fidelity proxy can be optimized while the discriminative content is not preserved.

**How it influenced the next design.** V3-A replaced fidelity-to-teacher with an explicit decision-preservation condition (C1) plus a margin-retention condition (C2), which is a strictly weaker and more testable requirement.

## N03 — Proxy decision preservation as a universal LLM sufficiency guarantee

**Hypothesized.** If the compressed context keeps the frozen proxy's decision and a fraction of its signed margin, then the context is "sufficient" and any capable reader should reach the same decision.

**Falsified by.** The V3-A / V3-B pair. V3-A reported ΔMacro-F1 = 0 and prediction change rate 0.0 on both datasets — but this is structural (condition C1 plus static fallback), so it measures nothing about readers. V3-B then evaluated the same contexts with an independent frozen Qwen3-8B reader: PHEME +0.01974 (WEAK_TRANSFER, CI crosses zero) and Ma-Weibo -0.00933 (TRANSFER_FAIL), overall PARTIAL.

**What was learned.** Proxy sufficiency does not automatically transfer. The correct claim is "does not automatically transfer", never "never transfers": one dataset was mildly positive and the other mildly negative, and neither is statistically significant on its own.

**How it influenced the next design.** It produced the final V3 recommendation `MS_TSR_COMPRESSION_ONLY`: keep MS-TSR as a context-efficiency component, and do not present it as a universal sufficiency selector.

## N04 — Simple compression severity as a sufficient reader-risk predictor

**Hypothesized.** Reader degradation under compression can be predicted from how aggressively the context was compressed (token reduction / evidence-count reduction).

**Falsified by.** The V3-B failure diagnosis. `OVER_COMPRESSION` was not triggered under the predefined rule (Ma-Weibo CW vs CC+WC mean reduction 0.8909 vs 0.8513, a 0.04 gap against a 0.05 threshold, with MS units 1.00 vs 1.10 against a 0.5 threshold). More importantly, no proxy-side signal in the whole diagnostic cleared its fixed threshold in a way that predicted the Ma-Weibo failure: Static-utility→citation AUC was 0.6158 (Ma-Weibo) vs 0.4370 (PHEME), MS-removal-vs-citation gap was +0.0005 (Ma-Weibo) and -0.0676 (PHEME), and proxy-margin retention vs the correctness transition had Spearman -0.0929 (Ma-Weibo) and +0.0751 (PHEME).

**What was learned.** Compression magnitude alone does not explain the transfer failure. The diagnosis flagged `PROXY_READER_MARGIN_MISMATCH` on exactly this pattern: mean proxy margin retention 1.2986 (the compressed set retained, in fact exceeded, the static signed margin) while the reader delta was -0.00933. Mean retention was above 1 on both datasets (PHEME 1.21646, Ma-Weibo 1.27774), so the proxy's margin is not a proxy for reader difficulty.

**How it influenced the next design.** Any future reader-risk rule must be calibrated fold-locally before use; no global compression threshold is implemented. The candidate signals in the diagnosis report are explicitly *not* promoted to rules.

## N05 — Qwen citation preservation as a sufficient safety rule

**Hypothesized.** If the compressed context keeps the reader's evidence citations valid and non-fabricated, grounding is preserved and the compressed arm is safe to use.

**Falsified by.** The V3-B reader pilot. Grounding was in fact intact on both datasets (valid citation rate 1.0, unsupported citation rate 0.0, no-citation rate ≤ 0.177), and yet Ma-Weibo still FAILED the non-inferiority gate. Citation validity therefore cannot certify reader sufficiency.

**What was learned.** Evidence-ID validity is a necessary hygiene property, not a sufficiency criterion. It also cannot be used to claim hallucination was eliminated: the check only verifies that a cited E# exists in the supplied block, not that the cited evidence supports the stated reason.

**How it influenced the next design.** The final diagnosis recommendation does not include any citation-based safety rule, and the prohibited-claims list (`PROHIBITED_CLAIMS.md`) forbids the "hallucination eliminated" phrasing.
