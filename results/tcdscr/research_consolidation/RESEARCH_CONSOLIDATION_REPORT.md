# TC-DSCR Research Consolidation

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`
Created: 2026-09-12 (UTC)

This report consolidates the completed TC-DSCR research into an auditable evidence system. It does not run anything, it does not change any method, and it does not hide any failed stage.

## 1. Frozen Research State

| Item | Value |
|---|---|
| Freeze commit | `c96ccbd` |
| E1 Random-init Causal Encoder | VALID |
| Corrected E2 Static Utility Selector | VALID (dataset-dependent) |
| Dynamic V1 (novelty + persistence) | NOT SUPPORTED |
| Dynamic V2 (MF-TSR teacher fidelity) | NOT SUPPORTED AS FINAL METHOD |
| Dynamic V3-A (MS-TSR proxy) | PASS AT PROXY / COMPRESSION LEVEL |
| Dynamic V3-B (frozen Qwen3-8B reader) | PARTIAL — PHEME WEAK_TRANSFER, Ma-Weibo TRANSFER_FAIL |
| V3-B failure diagnosis | CLOSED |
| Final V3 recommendation | MS_TSR_COMPRESSION_ONLY |
| V3-C held-out test | **NOT APPROVED** |
| Full E4 | **NOT APPROVED** |
| New training / new Qwen inference / held-out test this round | all false |

## 2. What Has Been Successfully Demonstrated

- A temporally causal evaluation protocol: real-cutoff snapshots, all events retained at every cutoff, no future leakage (Contribution A).
- A Static Utility Selector that beats random and semantic-similarity selection on the causal representation, with a dataset-dependent effect (Contribution B).
- Large, stable social-context redundancy: a compressed subset preserves the frozen proxy decision while removing 57.9% (PHEME) and 82.7% (Ma-Weibo) of social tokens (Contribution C).
- A controlled proxy-to-reader transfer measurement with an independent frozen reader under a pre-registered gate (Contribution D).
- A coherent chain of negative and boundary findings with measured mechanisms rather than unexplained failures (Contribution E).

## 3. What Has Been Rejected

- Dynamic V1 point-wise novelty/persistence reranking: delta Macro-F1 +0.0001037 / +0.0000120 with both bootstrap CIs crossing zero (held-out test).
- MF-TSR teacher-fidelity set refinement as a final method: gate FAIL on both datasets (+0.00293 / -0.00286).
- Proxy-defined sufficiency as a universal LLM-reader sufficiency guarantee: the reader failed the frozen gate on Ma-Weibo.
- Compression severity as a reader-risk predictor, and citation validity as a sufficient safety rule (both falsified by the V3-B diagnosis).

## 4. Method Contributions

1. **Temporally Causal Social-Context Evaluation Protocol** (Contribution A) — fully supported.
2. **Static Evidence Utility / Refinement** (Contribution B) — supported, dataset-dependent effect stated explicitly.
3. **Minimal-Sufficient Context Compression (MS-TSR)** (Contribution C) — supported **only as a compression component**. It must not be presented as a universal sufficiency selector.

## 5. Finding Contributions

Point-wise dynamic scoring failure; teacher-fidelity / downstream-utility mismatch; proxy-reader mismatch; dataset-specific context requirement; social-context redundancy; reader confidence change under compression; narrow-scope citation-validity observation. These are analysis / negative / design-implication contributions, not method contributions.

## 6. Canonical Experimental Evidence

Full tables with split, dataset, method, metric, value and status are in `CANONICAL_RESULTS_TABLE.md`. Highlights:

- E1 (TRAIN): UMER init vs random init Macro-F1 deltas +0.00632 to +0.01477 (PHEME), +0.00586 to +0.01285 (Ma-Weibo).
- Corrected E2 (VALIDATION): static beats the best simple baseline by +0.02095 (Ma-Weibo, readiness pass) and +0.00218 (PHEME, readiness fail).
- Dynamic V1 (HELD_OUT_TEST): delta Macro-F1 +0.0001037 / +0.0000120, corrected CIs crossing zero.
- MF-TSR (VALIDATION): +0.00293 / -0.00286, gate FAIL.
- MS-TSR V3-A (VALIDATION): 57.9% / 82.7% mean token reduction, prediction change 0.
- V3-B reader (**VALIDATION PILOT**): +0.01974 (PHEME, CI crossing zero) / -0.00933 (Ma-Weibo), overall PARTIAL.

## 7. Negative Findings

See `NEGATIVE_FINDINGS_LEDGER.md`: N01 point-wise dynamic reranking; N02 teacher-KL fidelity as a downstream surrogate; N03 proxy decision preservation as a universal LLM sufficiency guarantee; N04 compression severity as a reader-risk predictor; N05 citation preservation as a sufficient safety rule.

## 8. Protocol Corrections

See `PROTOCOL_CORRECTIONS.md`: P01 no-candidate snapshot correction (CLOSED); P02 bootstrap multiplicity bug (CLOSED, no conclusion changed); P03 structural position-vs-ID degree bug (CLOSED, diagnosis-only, no root cause or recommendation changed); P04 cross-fold reader-development contamination risk (OPEN BY DESIGN; any future rule must be fold-local).

## 9. Claims We Can Make

- C01 Temporal causal evaluation is required.
- C02 No-candidate snapshots must remain in early evaluation.
- C03 Static social evidence refinement is useful (per dataset, with stated effect sizes).
- C06 Set-wise refinement can substantially change evidence context.
- C07 Teacher fidelity is not discriminative sufficiency.
- C08 Social context contains substantial redundancy.
- C09 Proxy-defined sufficiency does not automatically transfer to an LLM reader.
- C10 Reader context need is dataset-dependent.
- C12 Fabricated evidence-ID references were not observed (narrow scope).
- Diagnostics C05 (scale + budget masking) and C11 (confidence drop, ECE worse).

## 10. Claims We Cannot Make

See `PROHIBITED_CLAIMS.md`. In short: no "dynamic memory significantly improves detection"; no "MF-TSR improves classification by matching the full graph"; no "MS-TSR finds universally minimal sufficient evidence"; no "proxy confidence reliably predicts reader sufficiency"; no "compression eliminates hallucination"; no "PHEME +1.97pt is significant"; no "Ma-Weibo proves compression is harmful".

## 11. Current Paper Story

See `TC_DSCR_RESEARCH_STORYLINE.md`. The chain is: problem -> why full context is problematic -> temporal causality requirement -> static evidence utility -> point-wise attempt fails -> set-level fidelity attempt fails (redundancy discovered) -> minimal-sufficient compression at the proxy -> independent LLM reader transfer -> dataset-dependent boundary -> final conclusion. Internal stage names (V1/V2/V3) must be replaced by "Point-wise Temporal Re-ranking", "Set-level Fidelity Refinement" and "Minimal-Sufficient Context Refinement".

## 12. Remaining Evidence Gaps

See `NEXT_EXPERIMENT_GAPS.md`.

- **MUST_HAVE Gap A** — held-out, fold-local paired comparison of the Static Utility Selector against random and semantic baselines (none exists; the held-out test only compared static vs dynamic).
- **MUST_HAVE Gap B** — a strict fold-local reader evaluation if the reader section is to carry any claim (the current pilot mixes outer-fold pools and is validation-only).
- **MUST_HAVE Gap C** — at least Static top-k and random compression baselines under the same frozen reader and prompt.
- **SHOULD_HAVE Gap D** — formal token-cost / latency comparison.
- **OPTIONAL Gap E** — hallucination robustness experiment; outside the current critical path.

## 13. Candidate Next Research Options

### Option A — Consolidation

No new selector. Close Gap A and Gap C, optionally Gap D. Report the existing contributions honestly, with the reader section labeled a validation pilot.

### Option B — LLM robustness extension

Center the work on context redundancy and distractor robustness and add a grounding / hallucination experiment (Gap E) around a frozen reader.

### Option C — Reader-aware future work

Treat the proxy-reader mismatch as a new research topic (reader-aware sufficiency / reader calibration), explicitly outside this paper.

## 14. Recommended Next Option

**Recommended Next Research Option: A — Consolidation.**

## 15. Why

**Innovation.** The novelty already exists in the causal evaluation protocol, the redundancy/compression measurement and the proxy-to-reader transfer analysis. Adding a mechanism would not add a new idea.

**Evidence already available.** Static utility validation, the full held-out Dynamic V1 verdict, MF-TSR and MS-TSR validation, the 600-pair reader pilot, and a closed failure diagnosis. Only two cells are empty: a held-out static-vs-baseline comparison and simple compression baselines under the reader.

**Remaining workload.** Small and bounded: frozen-checkpoint evaluation (Gap A) plus a reader evaluation over already-defined contexts (Gap C). No training, no new selector, no new dataset.

**Risk.** Option A's residual risk is that the reader result stays mixed; that is already priced into the "does not automatically transfer" framing and remains defensible. Options B and C open unbounded new experiment programmes with no guaranteed positive outcome.

**Continuity with the current UMER codebase.** Option A reuses the frozen artifacts, the evaluation, bootstrap and manifest machinery as-is. No change to MS-TSR, the Static Selector, the Proxy or the prompt.

**Why not simply "do more".** Two refinement attempts have already been tried and rejected with measured causes. The minimum work that makes the existing claims reviewable is to fill the two baseline cells and stop.

## 16. STOP Decision

STOP. This round performed read / audit / aggregate / summarize / document only. No training, no Qwen inference, no held-out test, no selector change, no gate retuning, no deletion of failed experiments. V3-C and full E4 remain **NOT APPROVED** and require new explicit approval.

## Verifier
issues = 0 (stages=11, claims={'SUPPORTED': 9, 'PARTIAL': 0, 'REJECTED': 1, 'DIAGNOSTIC_ONLY': 2})
