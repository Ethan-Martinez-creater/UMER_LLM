# TC-DSCR Research Consolidation

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`
Created: 2026-09-12 (UTC)

This report consolidates the completed TC-DSCR research into an auditable evidence system. It does not run anything, it does not change any method, and it does not hide any failed stage.

## 1. Frozen Research State

| Item | Value |
|---|---|
| Freeze commit | `c96ccbd` |
| Consolidation finalization commit | `47aa201` |
| E1 Random-init Causal Encoder | VALID — reported split **VALIDATION** (training split TRAIN) |
| Corrected E2 Static Utility Selector | VALID (dataset-dependent) — canonical metric is **ALL-EVENT**; historical candidate-conditioned readiness retained for audit |
| Dynamic V1 (novelty + persistence) | NOT SUPPORTED — held-out run is **CONDITIONAL_HELD_OUT** (scope audit CASE_B); relative verdict retained |
| Dynamic V2 (MF-TSR teacher fidelity) | NOT SUPPORTED AS FINAL METHOD |
| Dynamic V3-A (MS-TSR proxy) | PASS AT PROXY / COMPRESSION LEVEL |
| Dynamic V3-B (frozen Qwen3-8B reader) | PARTIAL — PHEME WEAK_TRANSFER, Ma-Weibo TRANSFER_FAIL; `VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING` |
| V3-B failure diagnosis | CLOSED |
| Final V3 recommendation | MS_TSR_COMPRESSION_ONLY |
| Recommended next option | **A1** — Gap A + Gap B + Gap C (Gap B and Gap C combined into one reader experiment) |
| V3-C held-out test / Gap A/B/C execution | **NOT APPROVED** |
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

- E1 (**VALIDATION** fold readout, training split TRAIN): UMER init vs random init Macro-F1 deltas +0.00632 to +0.01477 (PHEME), +0.00586 to +0.01285 (Ma-Weibo).
- Corrected E2 (VALIDATION, all-event canonical): static beats the best simple baseline by +0.02030 (Ma-Weibo) and +0.00186 (PHEME). Historical candidate-conditioned readiness (Ma-Weibo pass, PHEME fail) is retained for audit only.
- Dynamic V1 (**CONDITIONAL_HELD_OUT**): delta Macro-F1 +0.0001037 / +0.0000120, corrected CIs crossing zero; absolute values are DEPRECATED_ABSOLUTE because the run skipped no-candidate snapshots.
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

- **MUST_HAVE Gap A** — held-out, fold-local paired comparison of the Static Utility Selector against Random and Semantic selection (none exists).
- **MUST_HAVE Gap B** — a strict fold-local frozen-LLM reader evaluation (required as long as Contribution D stays in the paper; the current pilot mixes outer-fold pools and is validation-only).
- **MUST_HAVE Gap C** — token-matched compression comparators under the same frozen reader and prompt (Static Full, Static Token-Matched, Random Token-Matched, MS-TSR). Gap B and Gap C should be executed as **one combined reader experiment**.
- **MUST_HAVE Gap F** — an all-event held-out re-evaluation of the Dynamic V1 comparison (the current held-out run is candidate-conditioned).
- **SHOULD_HAVE Gap D** — formal token-cost / latency comparison.
- **OPTIONAL Gap E** — hallucination robustness experiment; outside the current critical path.

## 13. Candidate Next Research Options

### Option A1 — Consolidation with Final Evidence Closure

No new selector, no Dynamic V4, no new hallucination topic. Close Gap A, Gap B + Gap C (one combined reader experiment), Gap F, optionally Gap D. This is the recommended option.

### Option B — LLM robustness extension

Center the work on context redundancy and distractor robustness and add a grounding / hallucination experiment (Gap E) around a frozen reader.

### Option C — Reader-aware future work

Treat the proxy-reader mismatch as a new research topic (reader-aware sufficiency / reader calibration), explicitly outside this paper.

## 14. Recommended Next Option

**Recommended Next Research Option: A1 — Consolidation with Final Evidence Closure.**

## 15. Why

**Innovation.** The novelty already exists in the causal evaluation protocol, the redundancy/compression measurement and the proxy-to-reader transfer analysis. Adding a mechanism would not add a new idea.

**Why A1 rather than the earlier Option A.** Option A planned only Gap A + Gap C, but Contribution D (proxy-to-LLM reader transfer) remains a paper contribution. A reader contribution cannot be closed without Gap B, so A1 makes the recommendation consistent with the contribution matrix.

**Evidence already available.** Static utility validation (all-event), the Dynamic V1 negative finding (relative verdict intact), MF-TSR and MS-TSR validation, the 600-pair reader pilot, and a closed failure diagnosis. The empty cells are: a held-out static-vs-baseline comparison, a fold-local reader evaluation with token-matched baselines, and an all-event held-out closure.

**Remaining workload.** Bounded and frozen-checkpoint-only: Gap A, the combined Gap B+C reader run, Gap F, optionally Gap D. No training, no new selector, no new dataset. Running Gap B and Gap C together avoids duplicating the Qwen cost.

**Risk.** Option A1's residual risk is that the reader result stays mixed and that the all-event held-out closure does not change the Dynamic V1 verdict. Both are already priced into the current framing ("does not automatically transfer"; Dynamic V1 REJECTED). Options B and C open unbounded new experiment programmes with no guaranteed positive outcome.

**Continuity with the current UMER codebase.** Option A1 reuses the frozen artifacts, the evaluation, bootstrap and manifest machinery as-is. No change to MS-TSR, the Static Selector, the Proxy or the prompt.

**Why not simply "do more".** Two refinement attempts have already been tried and rejected with measured causes. The minimum work that makes the existing claims reviewable is to close Gaps A, B+C and F and stop.

## Consolidation Finalization Audit

E1 split corrected:
YES — reported split VALIDATION (training split TRAIN); the earlier TRAIN labeling was wrong.

E2 canonical migrated to all-event:
YES — PHEME 0.85669 / 0.85482 / 0.85382 (Static / Random / Semantic, delta +0.00186); Ma-Weibo 0.93387 / 0.91357 / 0.90893 (delta +0.02030). Values read from `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json`.

Historical E2 readiness preserved:
YES — candidate-conditioned, retained as HISTORICAL_ONLY / audit only (PHEME FAIL, Ma-Weibo PASS, overall PARTIAL).

E3 no-candidate scope audited:
YES — runner behavior: the held-out runner dropped event-cutoffs with `candidate_count == 0`; expected rows 199,602, actual rows 173,277, rows with `n_candidates == 0` = 0; missing-row rate (PHEME 0.18630, Ma-Weibo 0.05693) matches the validation no-candidate rate (0.18774 / 0.05071). Artifacts: `E3_NO_CANDIDATE_SCOPE_AUDIT.md`, `e3_no_candidate_scope_audit.json`.

E3 held-out canonical status:
CONDITIONAL_HELD_OUT / DEPRECATED_ABSOLUTE — the relative negative finding (Dynamic V1 not supported) is retained; no re-run was performed; validation all-event diagnostics are not a held-out substitute.

Reader contribution status:
VALIDATION-PILOT / FINAL-EVIDENCE-PENDING

Recommended next option:
A1

Required next gaps:
A + B + C  (Gap B and Gap C combined into one reader experiment; plus Gap F for the all-event held-out closure)

New experiment executed:
NO

## 16. STOP Decision

STOP. This round was the consolidation finalization: read / audit / trace / relabel / correct documentation / update verifier and tests only. No training, no inference, no Qwen, no held-out re-run, no Gap A/B/C implementation, no selector or MS-TSR change. V3-C, full E4, and Gap A/B/C execution remain **NOT APPROVED** and require new explicit approval.

## Verifier
issues = 0 (stages=11, claims={'SUPPORTED': 9, 'PARTIAL': 0, 'REJECTED': 1, 'DIAGNOSTIC_ONLY': 2})
