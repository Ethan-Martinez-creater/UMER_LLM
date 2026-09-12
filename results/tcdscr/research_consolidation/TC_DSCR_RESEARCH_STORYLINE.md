# TC-DSCR Research Storyline

This is a paper-level logic chain, not a paper draft. It abstracts the stage history (E1 / E2 / E3 / Dynamic V1 / V2 / V3-A / V3-B) into a narrative that stands on its own.

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

---

## The chain

**Problem.** Early rumor detection on social media must decide from partial propagation. The full reply graph at the end of observation is not available at detection time, and it is long: mean social context reaches 472.8 evidence tokens on PHEME and 855.6 on Ma-Weibo at the primary cutoffs, with 6.1 and 9.8 evidence units respectively. Feeding everything into a reader is expensive and noisy.

**Why full social context is problematic.** Two reasons, both measured. First, most of the context is redundant: a subset that preserves the frozen decision removes 57.9% (PHEME) and 82.7% (Ma-Weibo) of social tokens. Second, the reader does not use most of what it is given and the part it uses is not the part a utility score would rank highest (Static-utility -> reader-citation AUC 0.437 on PHEME, 0.616 on Ma-Weibo).

**Temporal causality requirement.** Therefore the evaluation must be causal: each snapshot is built at the real cutoff, every event is kept at every cutoff including no-reply snapshots, and no future node may enter an early snapshot. This is enforced by construction and verified by a leakage checker.

**Static evidence utility.** The first usable signal is a utility score over candidate evidence, learned on the causal propagation representation. It beats random and semantic-similarity selection on both datasets (+0.00218 and +0.02095 Macro-F1 over the best simple baseline), though with clearly different strength.

**Attempt 1 — Point-wise Temporal Re-ranking.** *Hypothesis:* add a per-candidate temporal bonus (novelty of new replies, persistence of retained evidence) so the ranking follows the evolving structure. *Result:* rejected. The bonus std is 1.9-3.9% of the utility std; rankings move (Spearman 0.988-0.998) but the budgeted selection barely changes, and when the selection does change the frozen proxy changes its decision only 0.43-0.71% of the time. Net effect on Macro-F1 is within ±0.0001 with CIs crossing zero. *Lesson:* ranking change is not decision change; a point-wise score cannot control what the reader sees.

**Attempt 2 — Set-level Fidelity Refinement.** *Hypothesis:* if the point-wise view fails, refine the *set* so that the compressed context reproduces the full-graph teacher distribution. *Result:* the context does change substantially (set-change rate 0.56-0.98), and distortion falls, but downstream Macro-F1 does not improve (+0.00293 PHEME, -0.00286 Ma-Weibo; gate FAIL). *Lesson:* fidelity to a teacher is not the same objective as preserving decision-relevant evidence. This attempt also produced the redundancy discovery that motivates the next step.

**Minimal-Sufficient Context Refinement.** *Hypothesis:* drop the teacher surrogate and require the two properties directly — the compressed set must keep the frozen decision (condition C1) and retain a fraction of its signed margin (condition C2). *Result:* at the proxy level this passes with large compression (ΔMacro-F1 = 0, prediction change 0, mean token reduction 57.9% / 82.7%). *Important caveat:* ΔMacro-F1 = 0 is a structural consequence of C1 plus static fallback, not a measured gain; the proxy is also insensitive to evidence substitution. So the proxy result establishes compression, not reader-level quality.

**Independent LLM reader transfer.** *Question:* does the proxy-defined minimal-sufficient context transfer to a completely different frozen reader? *Design:* the same paired contexts are fed to a frozen Qwen3-8B with an identical frozen prompt and greedy decoding, 300 paired event-cutoffs per dataset, under a pre-registered gate (ΔMacro-F1 ≥ -0.005, social token reduction ≥ 30%, unsupported citation rate ≤ 1%). *Result:* PHEME +0.01974 with CI crossing zero (WEAK_TRANSFER), Ma-Weibo -0.00933 (TRANSFER_FAIL), overall PARTIAL.

**Dataset-dependent boundary.** *Question:* why did one dataset transfer and the other not? *Design:* a post-hoc, no-new-inference diagnosis over paired outcome groups (CC/CW/WC/WW), compression severity, evidence-count thresholds, reader evidence loss, proxy margin retention, citation alignment, and cutoff / pressure / label strata. *Result:* root causes `PROXY_READER_MARGIN_MISMATCH` and `DATASET_SPECIFIC_CONTEXT_NEED`. The proxy retained or exceeded its margin (mean retention 1.2986) while the reader degraded, and no proxy-side signal cleared its fixed threshold. The reader need is dataset-dependent: Ma-Weibo's static context supplies 9.83 units and the reader cites 7.65 of them, while PHEME supplies 6.05 and cites 4.66, yet MS-TSR compresses both to about one unit.

**Final conclusion.** Proxy-defined sufficiency does not automatically transfer to an LLM reader. MS-TSR is retained as a **context-compression component**, not as a universal sufficiency selector. The final recommendation is `MS_TSR_COMPRESSION_ONLY`.

**Current result.** Reader transfer is validation-supported but dataset-dependent (PHEME +0.01974 with a CI crossing zero; Ma-Weibo -0.00933).

**Remaining final evidence.** A fold-local held-out reader comparison with token-matched compression baselines (Gap B + Gap C, one combined reader run), an all-event held-out closure of the Dynamic V1 comparison (Gap F), and a held-out Static vs Random vs Semantic comparison (Gap A).

The reader contribution is **not** complete: Contribution D is currently `VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING`.

**Held-out evidence note (scope audit).** The E3 held-out run dropped every event-cutoff with no candidate (173,277 rows kept of 199,602 expected), so its absolute Macro-F1 values are `CONDITIONAL_HELD_OUT` / `DEPRECATED_ABSOLUTE`. Dynamic V1 remains a supported historical negative finding based on its negligible candidate-conditioned held-out effect, corrected bootstrap intervals crossing zero, and consistent all-event validation diagnostics; however, because Macro-F1 is nonlinear, the final all-event held-out effect is not inferred from identical no-candidate predictions and must be established by Gap F. Details: `E3_NO_CANDIDATE_SCOPE_AUDIT.md`.

---

## Stage-to-paper mapping

The paper must **not** be structured as "V1 failed, V2 failed, V3 partially worked". Use the abstract names.

| Internal stage | Paper name | Paper role |
|---|---|---|
| Dynamic V1 | Point-wise Temporal Re-ranking | ablation / design evolution |
| Dynamic V2 (MF-TSR) | Set-level Fidelity Refinement | ablation / negative finding |
| Dynamic V3-A (MS-TSR) | Minimal-Sufficient Context Refinement | method component (compression) |
| Dynamic V3-B | Proxy-to-Reader Transfer Analysis | analysis / reader-transfer section |
| E1 / E2 / E3 | Causal encoder + Static Evidence Utility | method / main results |
| Protocol corrections + failure diagnoses | Evaluation Protocol + Analysis | setup + discussion |

## Narrative invariants

- Every failure is reported with its measured mechanism, never as an unexplained negative.
- The proxy result and the reader result are always reported together; neither is allowed to stand alone.
- The reader pilot is always labeled a validation pilot.
- The final method claim is compression, not sufficiency.
