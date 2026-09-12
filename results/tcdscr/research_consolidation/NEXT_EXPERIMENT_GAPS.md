# TC-DSCR Next Experiment Gap Analysis

This file does **not** design a new algorithm and does not authorise any run. It answers one question:

> If a complete paper were written from the current state, what is still missing from the evidence chain?

No experiment in this document has been executed. Nothing here may be started without explicit approval (V3-C held-out test and Full E4 remain **NOT APPROVED**).

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

---

## MUST_HAVE

### Gap A — Static Utility held-out evidence

**Question.** Is the final held-out evidence for the Static Utility Selector complete?

**Answer: not yet — MUST_HAVE, and it is the cheapest gap to close.** The corrected E2 readiness is validation-only, and the E3 held-out test used the static selector only as the *comparison arm* for Dynamic V1. There is no held-out, fold-local, paired comparison of static vs random selection and static vs semantic selection. Without it, Contribution B (the static utility model) cannot be reported as a held-out result at all; it can only be reported as a validation-stage finding.

**Minimum needed.** One frozen-configuration held-out evaluation reporting paired static / random / semantic Macro-F1 with the same bootstrap procedure already implemented. No new training.

### Gap B — A strict fold-local or held-out reader evaluation

**Question.** If LLM grounding / context efficiency is a core contribution, is a strict fold-local / held-out reader evaluation missing?

**Answer: yes, if the claim stays in the paper — MUST_HAVE for that claim.** The V3-B pilot is a validation pilot and additionally mixes outer-fold pools (285/285 PHEME and 275/275 Ma-Weibo sampled events also appear in another fold's test split). A reader evaluation that is calibrated and reported inside the outer fold, with the fold's test untouched, is the minimum required to make any reader-transfer statement that a reviewer will accept as evidence.

**If the reader section is demoted to an analysis/limitation discussion**, this gap stays SHOULD_HAVE rather than MUST_HAVE — but then the paper must not claim compression is reader-safe.

### Gap C — A simple, accepted context-compression baseline under the reader

**Question.** Is a simple, well-known compression baseline needed (Static top-k, random compression, utility-only compression)?

**Answer: yes — MUST_HAVE.** The current reader comparison is MS-TSR vs the *full* static selection, which conflates "compression works" with "MS-TSR's specific subset works". Without at least Static top-k and random compression under the identical reader and prompt, the reader result cannot attribute anything to MS-TSR as a method. Only the gap is recorded here; **do not execute it in this round.**

## SHOULD_HAVE

### Gap D — Formal token-cost / latency comparison

**Answer: SHOULD_HAVE.** Token counts already exist for both arms (social and total input tokens, per-sample). What is missing is a formal cost table: prompt tokens, generated tokens, wall-clock latency, and cost per 1,000 decisions, measured on the same hardware. This is cheap to add alongside any future reader run and is a standard reviewer request for an efficiency-flavoured contribution.

## OPTIONAL

### Gap E — Hallucination robustness experiment

**Answer: OPTIONAL, and outside the current main line.** The pilot supports exactly one narrow observation (no fabricated evidence-ID references; unsupported citation rate 0.0). A real hallucination study would need a different task design — distractor evidence, adversarial context, or faithfulness annotation of the stated reason — and would not change the current conclusions. It should not be added to this paper's critical path.

---

## Candidate next research options

### Option A — Consolidation (收束型)

**Goal.** Design no new selector. Close only the evidence the paper needs: Gap A (held-out static baseline comparison), Gap C (simple compression baselines under the reader), and Optionally Gap D.

**Why it fits.** Contribution A (protocol) and Contribution B (static utility) are already supported; Contribution C (compression) is supported at the proxy level; Contribution E (negative findings) is well evidenced. The missing pieces are baselines and an honest reader section, not new mechanisms.

### Option B — LLM robustness extension

**Goal.** Make context redundancy and distractor robustness the core, and add a grounding / hallucination experiment (Gap E) around a frozen reader.

**Why it might fit.** It would turn the current negative reader finding into a positive research question. It requires a new reader experiment design, new annotations, and a new gate.

### Option C — Reader-aware future work

**Goal.** Treat the proxy-reader mismatch as a new research topic (reader-aware sufficiency, reader calibration), explicitly outside this paper.

**Why it might fit later.** The diagnosis already isolates the phenomenon (`PROXY_READER_MARGIN_MISMATCH`, `DATASET_SPECIFIC_CONTEXT_NEED`) with fixed thresholds and no stable signal. That is a well-posed opening for a follow-up paper, not a component of this one.

---

## Recommendation

**Recommended Next Research Option: A — Consolidation.**

**Innovation.** The paper's novelty does not come from an additional mechanism. It comes from the causal evaluation protocol (Contribution A), the redundancy/compression measurement (Contribution C), and the proxy-to-reader transfer analysis (Contribution D) — all of which already exist. Option B would trade a solid, novel framing for a new experiment family that the current evidence does not yet support; Option C is by construction a different paper.

**Evidence already available.** Static utility validation (both datasets), the full held-out Dynamic V1 verdict, MF-TSR and MS-TSR validation, the 600-pair reader pilot, and the closed failure diagnosis. Only two evidence cells are empty: a held-out static-vs-baseline comparison, and simple compression baselines under the reader.

**Remaining workload.** Small and bounded: frozen-checkpoint evaluation only (Gap A) plus a reader evaluation over already-defined contexts (Gap C). No training, no new selector, no new dataset. That is the minimum work that makes the existing claims reviewable.

**Risk.** Option A's main risk is that the reader section stays modest — PHEME not significant, Ma-Weibo crossing the boundary. That risk is already priced into the current storyline: the finding is "does not automatically transfer", and it is defensible. Options B and C carry the larger risk of an unbounded new experiment programme with no guaranteed positive result.

**Continuity with the current UMER codebase.** Option A reuses frozen artifacts and the existing evaluation, bootstrap and manifest machinery end to end. It requires no change to MS-TSR, the Static Selector, the Proxy, or the prompt.

**Why not "do more work" by default.** Additional mechanisms have already been tried twice and rejected with measured causes. The cheapest path to a complete, credible evidence chain is to fill the two missing baseline cells and stop, not to add a third refinement attempt.
