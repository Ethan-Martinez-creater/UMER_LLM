# TC-DSCR Next Experiment Gap Analysis

This file does **not** design a new algorithm and does not authorise any run. It answers:

> If a complete paper were written from the current state, what is still missing from the evidence chain?

No experiment in this document has been executed. Nothing here may be started without explicit approval. **Gap A / Gap B / Gap C execution = NOT APPROVED. V3-C = NOT APPROVED. Full E4 = NOT APPROVED.**

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`. Consolidation finalization commit: `47aa201`.

Current reader contribution status: **VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING**.
Current E3 held-out status: **CONDITIONAL_HELD_OUT** (scope audit `CASE_B_ZERO_CANDIDATE_SKIPPED`).

---

## MUST_HAVE

### Gap A — Static Utility held-out comparison

**Question.** Is the final held-out evidence for the Static Utility Selector complete?

**Answer: no — MUST_HAVE.** Corrected E2 readiness is validation-only and uses the all-event canonical metric; the E3 held-out run compared static only against Dynamic V1 (and is itself candidate-conditioned). There is no held-out, fold-local, paired comparison of Static vs Random vs Semantic selection.

**Minimum future experiment (recorded only; not executed).**
Arms: Static, Random, Semantic.
Requirements: frozen checkpoints, fold-local, all-event protocol, held-out test split, paired statistics, no retraining.

### Gap B — Fold-local frozen-LLM reader evaluation

**Question.** If the reader-transfer / context-efficiency contribution stays in the paper, is a strict fold-local reader evaluation missing?

**Answer: yes — MUST_HAVE**, because Contribution D is currently only validation-pilot supported.

**Protocol requirement (recorded only; not executed).**

```
for each outer fold k:
  development / calibration uses only fold-k train + validation
  fold-k test stays completely untouched
```

The current pilot merges all five outer-fold validation pools (285/285 PHEME and 275/275 Ma-Weibo sampled events also appear in another fold's test split), so it must not be used to design a global rule. Gap B forbids that pattern: any reader-side decision must be calibrated inside the outer fold.

### Gap C — Simple context-compression comparators under the reader

**Question.** Are simple, accepted compression baselines needed?

**Answer: yes — MUST_HAVE.** The current reader comparison pits MS-TSR against the *full* static selection, which cannot separate "compression helps" from "this particular subset helps".

**Frozen comparator family (requirement only; implementation details are not decided here).**

```
Static Full Context
Static Utility Token-Matched  (or Static Top-k Token-Matched)
Random Token-Matched
MS-TSR
```

"Token-Matched" means the baseline's social-token budget is matched to the MS-TSR context rather than fixing an arbitrary k. This is what makes the compression benefit separable from the subset-selection benefit.

### Gap F — All-event held-out re-evaluation of the Dynamic V1 comparison

**Question.** Is the held-out evidence all-event?

**Answer: no — MUST_HAVE.** The scope audit found the E3 held-out run dropped every no-candidate event-cutoff (173,277 of 199,602 expected rows). The absolute values are therefore `CONDITIONAL_HELD_OUT` / `DEPRECATED_ABSOLUTE`.

**Minimum future experiment (recorded only; not executed).** Re-evaluate the frozen E3 fold configurations on the held-out split under the all-event protocol and report the paired comparison. No retraining, no re-search. Validation all-event diagnostics are **not** a substitute.

## SHOULD_HAVE

### Gap D — Formal token-cost / latency comparison

**Answer: SHOULD_HAVE.** Token counts already exist for both arms. Missing: prompt tokens, generated tokens, wall-clock latency and cost per 1,000 decisions on the same hardware. Cheap to add alongside any future reader run.

## OPTIONAL

### Gap E — Hallucination robustness experiment

**Answer: OPTIONAL, outside the current main line.** The pilot supports one narrow observation (no fabricated evidence-ID references). A real hallucination study needs a different task design (distractor evidence, adversarial context, faithfulness annotation) and would not change the current conclusions.

---

## Recommended option

### Option A1 — Consolidation with Final Evidence Closure

**Positioning (frozen):**

```
no new selector
no Dynamic V4
no new hallucination research topic
```

Only close the evidence the paper needs:

```
Gap A
Gap B + Gap C  (executed as ONE combined reader experiment)
optional Gap D
plus Gap F (all-event held-out closure for the Dynamic V1 comparison)
```

**Why A1 and not the earlier Option A.** Option A planned only Gap A + Gap C, yet the contribution matrix keeps Contribution D (proxy-to-LLM reader transfer) as a paper contribution. That is inconsistent: a reader contribution cannot be closed without Gap B. A1 makes the recommendation consistent with the contribution matrix — if Contribution D stays, Gap B is mandatory.

**Why Gap B and Gap C should be one experiment.**

```
same frozen Qwen model
same outer-fold test samples
same prompt
same source text
same inference configuration
```

Arms in that single run: `Static Full`, `Static Token-Matched`, `Random Token-Matched`, `MS-TSR`. Running them separately would double the Qwen cost for no methodological gain.

### Option B — LLM robustness extension (not recommended now)

Center the work on distractor robustness and add Gap E. Requires a new reader experiment design, new annotations and a new gate; no guaranteed positive result.

### Option C — Reader-aware future work (explicitly a different paper)

Treat the proxy-reader mismatch as a new topic. The diagnosis already isolates the phenomenon with fixed thresholds and no stable signal, so it is well-posed as follow-up work rather than a component of this paper.

---

## Recommendation

**Recommended Next Research Option: A1 — Consolidation with Final Evidence Closure.**

**Innovation.** No new mechanism is needed. The novelty sits in the causal evaluation protocol, the redundancy/compression measurement, and the proxy-to-reader transfer analysis.

**Evidence already available.** Static utility validation (all-event), the Dynamic V1 negative finding (relative verdict intact), MF-TSR and MS-TSR validation, the 600-pair reader pilot, and a closed failure diagnosis.

**Remaining workload.** Bounded and frozen-checkpoint-only: Gap A, the combined Gap B+C reader run, Gap F, optionally Gap D. No training, no new selector, no new dataset.

**Risk.** Option A1's residual risk is that the reader result stays mixed and that the all-event held-out comparison does not change the Dynamic V1 verdict. Both risks are already priced into the current framing ("does not automatically transfer"; Dynamic V1 REJECTED).

**Continuity with the current UMER codebase.** Reuses frozen artifacts plus the existing evaluation, bootstrap and manifest machinery. No change to MS-TSR, the Static Selector, the Proxy or the prompt.

**Why not simply "do more".** Two refinement attempts have already been tried and rejected with measured causes. The minimum work that makes the existing claims reviewable is to close Gaps A, B+C and F and stop.
