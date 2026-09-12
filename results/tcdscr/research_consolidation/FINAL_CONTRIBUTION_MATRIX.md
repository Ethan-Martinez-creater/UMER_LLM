# TC-DSCR Final Contribution Matrix

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

## Contribution overview

| Contribution | Research novelty | Experimental evidence | Evidence strength | Paper role | Remaining risk |
|---|---|---|---|---|---|
| **A** Temporally Causal Social-Context Evaluation Protocol | Real-cutoff snapshot construction with all-event retention and a leakage verifier | E1/E2/E3 + all Dynamic stages; leakage verifier 0 findings; no-candidate audit on both datasets | STRONG (protocol) | Method / Experimental Setup | Low. The protocol is enforced and auditable; the only risk is a venue treating it as engineering hygiene rather than a contribution. |
| **B** Causal Social Evidence Utility Modeling (Static Utility Selector) | Utility learned on a causal propagation representation rather than on text similarity | Corrected E2 readiness, 30 runs; static beats random and semantic on both datasets | MODERATE | Method / Main Results | Medium. Effect is dataset-dependent: +2.10 pt on Ma-Weibo, +0.22 pt on PHEME (below the 0.005 readiness threshold). |
| **C** Evidence-Set Redundancy / Compression Analysis (MS-TSR) | Minimal-sufficient subset search with an explicit decision-preservation condition and margin-retention condition | V3-A proxy validation: 57.9% / 82.7% mean social-token reduction with 0 proxy prediction change | MODERATE | Method component + Analysis | Medium. ΔMacro-F1 = 0 is structural, so the proxy evidence cannot support a quality claim for the method; only compression and set metrics are meaningful. |
| **D** Proxy-to-LLM Reader Transfer Analysis | Same frozen contexts evaluated by an independent frozen LLM reader under a pre-registered gate | V3-B pilot: 600 paired event-cutoffs, 1,200 arm evaluations, deterministic decoding, 0 context overflow, 0 determinism warnings | **MODERATE / PENDING FINAL EVIDENCE** | Analysis / Reader Transfer | Medium-High. Validation-only pilot, one reader, one prompt; PHEME CI crosses zero; no fold-local held-out reader evaluation exists (Gap B). Must never be presented as held-out evidence. |
| **E** Negative / Boundary Findings for Dynamic Evidence Refinement | A coherent chain of falsified hypotheses with measured mechanisms (scale, budget masking, teacher-fidelity mismatch, proxy-reader mismatch) | E3 test + E3 diagnosis + V2 MF-TSR + V3-B diagnosis | MODERATE (the E3 negative finding survives the held-out scope gap; the held-out run is candidate-conditioned) | Analysis / Discussion / Ablation | Low-Medium. Valuable only if presented as design implications rather than as a record of failed attempts. |

## Reader contribution status

`VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING`

Contribution D stays in the matrix because the pilot is the only independent-reader measurement the project has, but it is not fully established. Its final closure requires Gap B (a strict fold-local frozen-LLM reader evaluation) combined with Gap C (token-matched compression baselines), as defined in `NEXT_EXPERIMENT_GAPS.md`. Until then, every reader-result statement must be labeled a validation pilot.

## Method contributions (what can stand as method)

- **Causal snapshot protocol** — per-cutoff construction, all events retained, no future leakage, no-candidate snapshots preserved. Contribution A. This is fully supported.
- **Static evidence utility / refinement** — the selector plus the budgeted top-k evidence packer. Contribution B. Supported on validation with the all-event canonical metric (Static - best simple baseline: PHEME +0.00186, Ma-Weibo +0.02030), with the dataset-dependence caveat stated explicitly.
- **MS-TSR compression mechanism** — the set-refinement machinery that finds a low-cost subset preserving the frozen proxy decision. Contribution C. This may be presented **only as a context-compression component**, never as a universal sufficiency selector.

## Finding contributions (what can stand as analysis)

- Point-wise dynamic scoring failure (Dynamic V1) — N01, C04, C05.
- Teacher-fidelity / downstream-utility mismatch (MF-TSR) — N02, C07.
- Proxy-reader mismatch (V3-A vs V3-B) — N03, C09.
- Dataset-specific context requirement — C10.
- Social-context redundancy — C08.
- Reader confidence change under compression — C11.
- Narrow-scope citation validity observation — C12.

These can be written up as analysis contributions, negative results and design implications. They are not method contributions and must not be placed in a "our method improves X" claim.

## What must not be claimed as a contribution

- MS-TSR as a universal minimal-sufficient evidence selector (see `PROHIBITED_CLAIMS.md` §3).
- Proxy decision preservation as a guarantee of LLM-reader sufficiency (§3, §4).
- Any hallucination-elimination claim (§5).
- Any statistically significant PHEME reader gain (§6).
- Any general harmfulness claim from the Ma-Weibo result (§7).
