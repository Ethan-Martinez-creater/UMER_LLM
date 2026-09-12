# TC-DSCR Research Evidence Ledger

- Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`. Consolidation finalization commit: `47aa201`.
- Machine version: `research_evidence_ledger.json` (same content, authoritative for the verifier).
- Evidence strength: **STRONG** = clean held-out test (all-event) + paired statistics; **MODERATE** = validation across folds/seeds, independent validation reader pilot, or a held-out run with an audited protocol gap; **WEAK** = small subgroup, post-hoc, or descriptive diagnostic.
- C01 is a protocol claim: its STRONG rating refers to protocol enforcement, not to a predictive effect.

## Summary

| Claim | Statement | Status | Strength | Split | protocol_status | In paper |
|---|---|---|---|---|---|---|
| C01 | Temporal-causal cutoff evaluation is required | SUPPORTED | STRONG (protocol) | VALIDATION | ALL_EVENT | yes |
| C02 | No-candidate snapshots must stay in early evaluation | SUPPORTED | MODERATE | VALIDATION | ALL_EVENT | yes |
| C03 | Static utility refinement is useful (all-event) | SUPPORTED | MODERATE | VALIDATION | ALL_EVENT | yes |
| C04 | Point-wise novelty/persistence reranking is not sufficient | REJECTED | MODERATE | HELD_OUT_TEST (conditional) | CANDIDATE_CONDITIONED | yes (negative) |
| C05 | Dynamic bonus failed via score scale + budget masking | DIAGNOSTIC_ONLY | MODERATE | HELD_OUT_TEST (conditional) | CANDIDATE_CONDITIONED | yes (diagnostic) |
| C06 | Set-wise refinement can substantially change the context | SUPPORTED | MODERATE | VALIDATION | ALL_EVENT | yes |
| C07 | Teacher fidelity is not discriminative sufficiency | SUPPORTED | MODERATE | VALIDATION | ALL_EVENT | yes |
| C08 | Social context contains substantial redundancy | SUPPORTED | MODERATE | VALIDATION | ALL_EVENT | yes |
| C09 | Proxy sufficiency does not automatically transfer to an LLM reader | SUPPORTED | MODERATE | VALIDATION | VALIDATION_PILOT | yes |
| C10 | Reader context need is dataset-dependent | SUPPORTED | MODERATE | VALIDATION | VALIDATION_PILOT | yes |
| C11 | Reduced context lowers reader confidence | DIAGNOSTIC_ONLY | WEAK | VALIDATION | VALIDATION_PILOT | yes (diagnostic) |
| C12 | No fabricated evidence-ID citation was observed | SUPPORTED | WEAK | VALIDATION | VALIDATION_PILOT | yes (narrow scope) |

Status counts: SUPPORTED 9, PARTIAL 0, REJECTED 1, DIAGNOSTIC 2.

Finalization notes: E1's reported split is VALIDATION (training split TRAIN). The canonical E2 metric is all-event. The E3 held-out run is `CONDITIONAL_HELD_OUT` (scope audit CASE_B), so C04/C05 are MODERATE rather than held-out-STRONG. Contribution D is `VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING` (Gap B).

---

## C01 — Temporal causal evaluation is necessary

**Claim.** Social propagation context must be built at the real detection cutoff; a future node must never enter an early snapshot.

**Support.** Every formal stage (E1, corrected E2, E3 validation, E3 held-out, all Dynamic stages) constructs per-cutoff snapshots and passes the future-leakage verifier. Cutoffs are fixed at (5, 15, 30, 60, 180, 360) minutes.

**Metrics.** Leakage verifier = 0 findings across all stages.

**Strength.** STRONG **for protocol enforcement**, not for predictive effect. **Limitations.** This is a design requirement rather than an empirical effect. The E3 held-out run that used it skipped no-candidate snapshots, so at the held-out level the protocol is only partially evidenced.

## C02 — No-candidate snapshots must remain in early evaluation

**Claim.** Events with no reply yet at an early cutoff must not be dropped from the metric, otherwise evaluation becomes conditional on the presence of social evidence.

**Support.** V2 protocol correction re-ran corrected-E2 and E3-V1 with all events at every real cutoff.

**Metrics.** 5-minute no-candidate rate: PHEME 0.35525, Ma-Weibo 0.12674.

**Strength.** MODERATE. **Limitations.** The correction changes denominators rather than showing a large effect; pre-correction absolute values are DEPRECATED, not wrong. The E3 held-out run predates the correction and skipped no-candidate snapshots (scope audit CASE_B), so the correction is evidenced on validation metrics; the all-event held-out cell is Gap F.

## C03 — Static social evidence refinement is useful

**Claim.** A causal-propagation-based Static Utility Selector outperforms random and semantic selection under the corrected all-event protocol, with a strong dataset-dependent effect.

**Support.** Corrected E2 + V2 protocol correction, validation only. Values read from `results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json`.

| Dataset | Static | Random | Semantic | Delta vs best | historical readiness (candidate-conditioned) |
|---|---|---|---|---|---|
| PHEME | 0.85669 | 0.85482 | 0.85382 | +0.00186 | fail |
| Ma-Weibo | 0.93387 | 0.91357 | 0.90893 | +0.02030 | pass |

**Strength.** MODERATE. **Limitations.** Strongly dataset-dependent; PHEME stays below the original 0.005 readiness threshold (that historical verdict used the candidate-conditioned metric and is retained for audit only). No held-out fold-local comparison against random/semantic exists yet (Gap A).

## C04 — Point-wise novelty/persistence dynamic reranking is not sufficient

**Claim.** The Dynamic V1 point-wise temporal bonus does not improve rumor detection over the static selector.

**Metrics.** PHEME: static 0.84643, dynamic 0.84653, delta +0.0001037, corrected CI [-0.0000969, +0.0003214]. Ma-Weibo: static 0.92678, dynamic 0.92679, delta +0.0000120, corrected CI [-0.0001399, +0.0001614].

**Strength.** MODERATE — the held-out run is `CONDITIONAL_HELD_OUT` (candidate-conditioned), so it is not a clean all-event held-out result. **Status.** REJECTED as a method, retained as a supported historical negative finding based on its negligible candidate-conditioned held-out effect, corrected bootstrap intervals crossing zero, and consistent all-event validation diagnostics. **Limitations.** Because Macro-F1 is nonlinear, the final all-event held-out effect is not inferred from identical no-candidate predictions and must be established by Gap F.

## C05 — Dynamic bonus suffered from scale and budget-boundary problems

**Claim.** The failure is explained by (a) a bonus scale far below the utility scale and (b) selection changes the frozen proxy does not see.

**Metrics.** PHEME: u std 6.8016, bonus std 0.2666, ratio 0.0392, exact-match 0.9052, Jaccard 0.9691, ranking-changed-but-selection-same 0.1990, prediction change given selection change 0.0071, Spearman 0.9882. Ma-Weibo: u std 11.6088, bonus std 0.2206, ratio 0.0190, exact-match 0.8852, Jaccard 0.9753, ranking-changed-but-selection-same 0.4865, prediction change given selection change 0.0043, Spearman 0.9984.

**Strength.** MODERATE (post-hoc). **Status.** DIAGNOSTIC_ONLY. The proxy-insensitivity component is a property of the frozen proxy and must not be generalized to real readers. The underlying rows are candidate-conditioned, which does not change the direction of the diagnostic.

## C06 — Set-wise refinement can substantially change evidence context

**Claim.** MF-TSR set refinement can change the evidence set a lot while preserving the proxy decision.

**Metrics.** Set-change rate by stage — PHEME 0.5610 / 0.7506 / 0.8139; Ma-Weibo 0.8300 / 0.9266 / 0.9755. Tokens PHEME 472.77 -> 178.69; Ma-Weibo 855.55 -> 516.33.

**Strength.** MODERATE. **Limitations.** Context-change claim only; not a classification improvement (Ma-Weibo -0.00286, overall gate FAIL).

## C07 — Teacher fidelity is not equivalent to discriminative sufficiency

**Claim.** Reducing the distortion to the full-graph teacher distribution does not reliably improve downstream classification.

**Metrics.** PHEME +0.00293 with 62.2% token reduction; Ma-Weibo -0.00286 with 39.6% token reduction; gate FAIL on both.

**Strength.** MODERATE. **Limitations.** Two datasets, one fidelity objective, one refiner design; measured against a proxy that is itself only weakly sensitive to set replacement.

## C08 — Social context contains substantial redundancy

**Claim.** A minimal-sufficient subset can preserve the proxy decision while removing most social tokens.

**Metrics.** PHEME: mean token reduction 0.57859, median 0.8058, mean static units 6.0655, dual-view agreement 0.94730, compression attempt rate 0.77103, static fallback 0.28189. Ma-Weibo: 0.82738, median 0.9453, static units 9.8333, agreement 0.97323, attempt 0.92579, fallback 0.07818. Prediction change rate 0.0 on both.

**Strength.** MODERATE. **Limitations.** The preserved decision is the frozen proxy's decision by construction (C1 plus static fallback), so this is proxy-decision redundancy, not proof of reader sufficiency. PHEME compression is bimodal (p25 = 0).

## C09 — Proxy-defined sufficiency does not guarantee LLM-reader sufficiency

**Claim.** Proxy-defined sufficiency does not automatically transfer to an independent frozen LLM reader.

**Metrics.** V3-A proxy delta Macro-F1 = 0 on both datasets with prediction change rate 0.0 (structural). V3-B reader: PHEME +0.01974 (CI [-0.02192, +0.06144], WEAK_TRANSFER), Ma-Weibo -0.00933 (CI [-0.04010, +0.02129], TRANSFER_FAIL), overall PARTIAL.

**Strength.** MODERATE. Status: `VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING`. **Limitations.** One frozen reader, one prompt version, one validation reader pilot of 300 paired samples per dataset. Required wording: "does not automatically transfer", never "never transfers". Ma-Weibo's degradation is inside its own CI. No fold-local held-out reader evaluation exists (Gap B).

## C10 — Context need is dataset-dependent

**Claim.** The amount of social context a reader needs is dataset-dependent.

**Metrics.** PHEME: static 6.05 units cited 4.66, MS 1.18333 units cited 1.13, reduction 0.68700, reader delta +0.01974. Ma-Weibo: static 9.83333 units cited 7.65333, MS 1.10333 units cited 1.05667, reduction 0.85322, reader delta -0.00933.

**Strength.** MODERATE. **Limitations.** Validation reader pilot only; the diagnosis found no stable proxy-side predictor, so this claim is descriptive and licenses no per-dataset rule.

## C11 — Reduced context can lower reader confidence

**Claim.** Reducing the evidence context lowers the reader's stated confidence.

**Metrics.** PHEME delta -0.05867, ECE 0.17767 -> 0.19367. Ma-Weibo delta -0.05667, ECE 0.19000 -> 0.24067. 10 bins.

**Strength.** WEAK. **Status.** DIAGNOSTIC_ONLY. Must not be read as improved hallucination behaviour: ECE worsened on both datasets, and on Ma-Weibo the MS arm is more confident on wrong answers than on correct ones.

## C12 — Evidence-ID citation hallucination was not observed in the pilot

**Claim.** Fabricated evidence-ID references were not observed: every cited E# existed in the supplied evidence block.

**Metrics.** PHEME static / MS valid citation rate 1.0 / 1.0, unsupported 0.0 / 0.0, cited totals 1398 / 339. Ma-Weibo the same, cited totals 2296 / 317.

**Strength.** WEAK. **Limitations.** Narrow scope: verifies only that a cited E# exists in the supplied block; it does not verify that the cited evidence supports the stated reason and covers no other hallucination form. Required wording: "fabricated evidence-ID references were not observed", never "LLM hallucination was eliminated".
