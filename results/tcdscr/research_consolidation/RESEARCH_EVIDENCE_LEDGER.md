# TC-DSCR Research Evidence Ledger

- Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`
- Read the machine version at `research_evidence_ledger.json` (same content, authoritative for the verifier).
- Evidence strength: **STRONG** = held-out test + clean protocol + paired statistics; **MODERATE** = validation across folds/seeds or an independent reader pilot; **WEAK** = small subgroup, post-hoc, or descriptive diagnostic.

## Summary

| Claim | Statement | Status | Strength | Split | In paper |
|---|---|---|---|---|---|
| C01 | Temporal-causal cutoff evaluation is required | SUPPORTED | STRONG | VALIDATION | yes |
| C02 | No-candidate snapshots must stay in early evaluation | SUPPORTED | MODERATE | VALIDATION | yes |
| C03 | Static utility refinement is useful | SUPPORTED | MODERATE | VALIDATION | yes |
| C04 | Point-wise novelty/persistence reranking is not sufficient | REJECTED | STRONG | HELD_OUT_TEST | yes (negative) |
| C05 | Dynamic bonus failed via score scale + budget masking | DIAGNOSTIC_ONLY | MODERATE | HELD_OUT_TEST | yes (diagnostic) |
| C06 | Set-wise refinement can substantially change the context | SUPPORTED | MODERATE | VALIDATION | yes |
| C07 | Teacher fidelity is not discriminative sufficiency | SUPPORTED | MODERATE | VALIDATION | yes |
| C08 | Social context contains substantial redundancy | SUPPORTED | MODERATE | VALIDATION | yes |
| C09 | Proxy sufficiency does not automatically transfer to an LLM reader | SUPPORTED | MODERATE | VALIDATION | yes |
| C10 | Reader context need is dataset-dependent | SUPPORTED | MODERATE | VALIDATION | yes |
| C11 | Reduced context lowers reader confidence | DIAGNOSTIC_ONLY | WEAK | VALIDATION | yes (diagnostic) |
| C12 | No fabricated evidence-ID citation was observed | SUPPORTED | WEAK | VALIDATION | yes (narrow scope) |

Status counts: SUPPORTED 9, PARTIAL 0, REJECTED 1, DIAGNOSTIC 2.

---

## C01 — Temporal causal evaluation is necessary

**Claim.** Social propagation context must be built at the real detection cutoff; a future node must never enter an early snapshot.

**Support.** Every formal stage (E1, corrected E2, E3 validation, E3 held-out test, all Dynamic stages) constructs per-cutoff snapshots and passes the future-leakage verifier. Cutoffs are fixed at (5, 15, 30, 60, 180, 360) minutes.

**Metrics.** Leakage verifier = 0 findings across all stages; the E3 held-out test contains 94,104 (PHEME) and 79,173 (Ma-Weibo) event-cutoff rows.

**Strength.** STRONG. **Limitations.** This is a protocol requirement rather than an empirical effect; its support is that it was enforced everywhere and never violated.

## C02 — No-candidate snapshots must remain in early evaluation

**Claim.** Events with no reply yet at an early cutoff must not be dropped from the metric, otherwise evaluation becomes conditional on the presence of social evidence.

**Support.** V2 protocol correction re-ran corrected-E2 and E3-V1 with all events at every real cutoff.

**Metrics.** 5-minute no-candidate rate: PHEME 0.35525, Ma-Weibo 0.12674. Metric shift (all-event minus conditioned) for corrected E2 on PHEME: static +0.00097, random +0.00130, semantic +0.00150.

**Strength.** MODERATE. **Limitations.** The correction changes denominators rather than showing a large effect; pre-correction absolute values are DEPRECATED, not wrong.

## C03 — Static social evidence refinement is useful

**Claim.** A Static Utility Selector built on the causal propagation representation selects more useful social evidence than random or semantic-similarity selection.

**Support.** Corrected E2 readiness, 30 runs (5 folds x 3 seeds), validation only.

**Metrics.**

| Dataset | Static | Random | Semantic | Static - best baseline | Readiness |
|---|---|---|---|---|---|
| PHEME | 0.85591 | 0.85372 | 0.85254 | +0.00218 | fail (threshold 0.005) |
| Ma-Weibo | 0.93503 | 0.91408 | 0.90908 | +0.02095 | pass |

**Strength.** MODERATE. **Limitations.** The effect strength differs materially by dataset and must be reported separately; PHEME does not clear the pre-registered readiness threshold. The all-event correction moves PHEME static to 0.85669.

## C04 — Point-wise novelty/persistence dynamic reranking is not sufficient

**Claim.** The Dynamic V1 point-wise temporal bonus does not improve rumor detection over the static selector.

**Support.** E3 held-out test (30 frozen runs) plus the E3 failure diagnosis.

**Metrics.** PHEME: static 0.84643, dynamic 0.84653, delta +0.0001037, paired bootstrap CI [-0.0000502, +0.0002557] (multiplicity-corrected: [-0.0000969, +0.0003214]), flip-rate delta +0.000687, 3 positive / 1 negative folds. Ma-Weibo: static 0.92678, dynamic 0.92679, delta +0.0000120, CI [-0.0000999, +0.0001204] (corrected [-0.0001399, +0.0001614]), 3 positive / 2 negative folds.

**Strength.** STRONG. **Status.** REJECTED as a method (both CIs cross zero), while retained as a supported negative finding: the bonus moves rankings but not outcomes.

## C05 — Dynamic bonus suffered from scale and budget-boundary problems

**Claim.** The Dynamic V1 failure is explained by (a) a bonus scale far below the utility scale and (b) selection changes that the frozen proxy does not see.

**Metrics.** PHEME: u std 6.8016, bonus std 0.2666, ratio 0.0392, mean|bonus|/mean|u| 0.0535, exact-match 0.9052, Jaccard 0.9691, ranking-changed-but-selection-same 0.1990, prediction change given selection change 0.0071, Spearman 0.9882. Ma-Weibo: u std 11.6088, bonus std 0.2206, ratio 0.0190, mean|bonus|/mean|u| 0.0224, exact-match 0.8852, Jaccard 0.9753, ranking-changed-but-selection-same 0.4865, prediction change given selection change 0.0043, Spearman 0.9984.

**Strength.** MODERATE (post-hoc). **Status.** DIAGNOSTIC_ONLY. The proxy-insensitivity component is a property of the frozen proxy and must not be generalized to real readers.

## C06 — Set-wise refinement can substantially change evidence context

**Claim.** MF-TSR set refinement can change the evidence set a lot while preserving the proxy decision.

**Metrics.** Set-change rate by stage — PHEME 0.5610 / 0.7506 / 0.8139 (very early / early / mid); Ma-Weibo 0.8300 / 0.9266 / 0.9755. Tokens PHEME 472.77 -> 178.69; Ma-Weibo 855.55 -> 516.33.

**Strength.** MODERATE. **Limitations.** This is a context-change claim only. It must not be read as a classification improvement: Ma-Weibo degraded (delta -0.00286) and the overall gate FAILED.

## C07 — Teacher fidelity is not equivalent to discriminative sufficiency

**Claim.** Reducing the distortion to the full-graph teacher distribution does not reliably improve downstream classification.

**Metrics.** PHEME delta +0.00293 with 62.2% token reduction; Ma-Weibo delta -0.00286 with 39.6% token reduction; gate FAIL on both. The refiner was driven by the teacher-KL fidelity objective.

**Strength.** MODERATE. **Limitations.** Two datasets, one fidelity objective, one refiner design; a methodological finding rather than a theorem. The measurement uses the frozen proxy, which is itself only weakly sensitive to set replacement.

## C08 — Social context contains substantial redundancy

**Claim.** A minimal-sufficient subset can preserve the proxy decision while removing most social tokens.

**Metrics.** PHEME: mean token reduction 0.57859, median 0.8058, mean static units 6.0655, dual-view agreement 0.94730, compression attempt rate 0.77103, static fallback 0.28189. Ma-Weibo: mean token reduction 0.82738, median 0.9453, mean static units 9.8333, dual-view agreement 0.97323, compression attempt rate 0.92579, static fallback 0.07818. Prediction change rate 0.0 on both.

**Strength.** MODERATE. **Limitations.** The preserved decision is the frozen proxy's decision by construction (condition C1 plus static fallback), so this is proxy-decision redundancy, not proof of reader sufficiency. PHEME compression is bimodal (p25 = 0), so the mean overstates the typical case.

## C09 — Proxy-defined sufficiency does not guarantee LLM-reader sufficiency

**Claim.** Proxy-defined sufficiency does not automatically transfer to an independent frozen LLM reader.

**Support.** V3-A defined the minimal-sufficient context with the proxy; V3-B evaluated the same context with a completely independent frozen Qwen3-8B reader.

**Metrics.** V3-A proxy: delta Macro-F1 = 0 on both datasets with prediction change rate 0.0 (structural, see C08 limitations). V3-B reader: PHEME +0.01974 (CI [-0.02192, +0.06144], label WEAK_TRANSFER), Ma-Weibo -0.00933 (CI [-0.04010, +0.02129], label TRANSFER_FAIL), overall PARTIAL.

**Strength.** MODERATE. **Limitations.** One frozen reader, one prompt version, one validation-only pilot of 300 paired samples per dataset. The required wording is "does not automatically transfer" — never "never transfers". The Ma-Weibo degradation is inside its own confidence interval.

## C10 — Context need is dataset-dependent

**Claim.** The amount of social context a reader needs is dataset-dependent.

**Metrics.** PHEME: static 6.05 units cited 4.66, MS 1.18333 units cited 1.13, reduction 0.68700, reader delta +0.01974. Ma-Weibo: static 9.83333 units cited 7.65333, MS 1.10333 units cited 1.05667, reduction 0.85322, reader delta -0.00933. MS-TSR compresses both to about one unit, yet the outcomes differ.

**Strength.** MODERATE. **Limitations.** Validation reader pilot only. The failure diagnosis found no stable proxy-side signal predicting which sample will degrade, so this claim is descriptive and does not license a per-dataset rule.

## C11 — Reduced context can lower reader confidence

**Claim.** Reducing the evidence context lowers the reader's stated confidence.

**Metrics.** PHEME confidence delta -0.05867, ECE 0.17767 -> 0.19367. Ma-Weibo confidence delta -0.05667, ECE 0.19000 -> 0.24067. 10 bins.

**Strength.** WEAK. **Status.** DIAGNOSTIC_ONLY. The drop must not be interpreted as improved hallucination behaviour: ECE worsened on both datasets, and on Ma-Weibo the MS arm is more confident on wrong answers than on correct ones.

## C12 — Evidence-ID citation hallucination was not observed in the pilot

**Claim.** Fabricated evidence-ID references were not observed: every cited E# existed in the supplied evidence block.

**Metrics.** PHEME static / MS valid citation rate 1.0 / 1.0, unsupported rate 0.0 / 0.0, cited totals 1398 / 339. Ma-Weibo the same, cited totals 2296 / 317.

**Strength.** WEAK. **Limitations.** Narrow scope: this verifies only that a cited E# exists in the supplied block. It does not verify that the cited evidence supports the stated reason, and it does not cover any other hallucination form. Required wording: "fabricated evidence-ID references were not observed" — never "LLM hallucination was eliminated".
