# TC-DSCR Protocol Corrections

Each correction records what went wrong, what it affected, what it did not affect, and its final status. The purpose is that later writing and reproduction never re-use the old convention.

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

---

## P01 — No-candidate snapshot correction

**Original issue.** Early-cutoff metrics were computed only over snapshots with `candidate_count > 0`. Events that had no reply yet at that cutoff were silently dropped, so evaluation was conditioned on the presence of social evidence.

**Impact.** The metric denominator was wrong for early detection, which is exactly the regime TC-DSCR targets. The distortion is dataset-dependent: the 5-minute no-candidate rate is 0.35525 on PHEME but only 0.12674 on Ma-Weibo, so PHEME was affected roughly three times as much.

**Fix.** Every event is now evaluated at every real cutoff; a no-candidate snapshot keeps its empty candidate array and contributes its (source-only) prediction. Recorded in `results/tcdscr/dynamic_v2_protocol/protocol_manifest.json` with `test_split_read: false` and `no_retraining: true`.

**Affected.**

- **E3 validation/readiness metrics (candidate-conditioned):** corrected E2 PHEME static 0.85572 -> 0.85669 (all-event), random 0.85352 -> 0.85482, semantic 0.85232 -> 0.85382; Ma-Weibo static 0.93506 -> 0.93387, random 0.91394 -> 0.91357, semantic 0.90904 -> 0.90893. These old values are DEPRECATED (D02).
- **E3 held-out absolute values:** the held-out run predates this correction and skipped no-candidate snapshots (scope audit `CASE_B_ZERO_CANDIDATE_SKIPPED`: 173,277 rows kept of 199,602 expected; missing-row rate 0.18630 / 0.05693 vs validation no-candidate rate 0.18774 / 0.05071). Its absolute Macro-F1 values are `CONDITIONAL_HELD_OUT` / `DEPRECATED_ABSOLUTE` (D06). No re-run was performed.

**Not affected.** E1 (all-event by construction), the V3-A and V3-B protocols (both all-event), and every V3-B reader metric. Dynamic V1 remains a supported historical negative finding based on its negligible candidate-conditioned held-out effect, corrected bootstrap intervals crossing zero, and consistent all-event validation diagnostics. However, because Macro-F1 is nonlinear, the final all-event held-out effect is not inferred from identical no-candidate predictions and must be established by Gap F. E3 held-out **relative** verdict status: historical negative finding, final all-event confirmation pending Gap F.

**Final status.** CLOSED. All-event evaluation is mandatory from this point on; no-candidate coverage is now a reported statistic at every cutoff.

---

## P02 — Bootstrap multiplicity bug

**Original issue.** The paired bootstrap resampled event indices with replacement, but the resampled list was then converted into a unique-event dictionary. Duplicated events were collapsed, so the resample lost the multiplicity that makes bootstrap work, and the resulting confidence intervals were too narrow.

**Impact.** The E3 held-out confidence intervals for delta Macro-F1 and delta flip rate were invalid. The point estimates were unaffected.

**Fix.** `results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json` re-implements the resample so the sampled event list keeps every occurrence (10,000 iterations, seed 3090, paired). The corrected function is the one used in the V3-B reader summary as well.

**Affected.** The bootstrap fields of `results/tcdscr/formal_e3_test/e3_test_summary.json` (D03).

**Not affected.** The classification point estimates, the flip-rate point estimates, the WEAK_POSITIVE status and the REVIEW_DYNAMIC_DESIGN recommendation: the corrected CIs still cross zero on both datasets (PHEME delta Macro-F1 [-0.0000969, +0.0003214]; Ma-Weibo [-0.0001399, +0.0001614]).

**Final status.** CLOSED with no change to any research conclusion.

---

## P03 — Structural position-vs-ID degree bug

**Original issue.** In the V3-B failure diagnosis, `snapshot["edge_index"]` stores edge *position* indices (`edges.append([i, pos[parent_id]])`), but the degree accumulator was keyed by position while node lookups used node IDs. Every lookup missed, so node degree collapsed to ~0 and `is_leaf` (computed from undirected degree) became ~1.0 for all nodes.

**Impact.** Degenerate degree / leaf / child-count structural statistics in `structural_role_analysis.json`, `removed_evidence_feature_analysis.json`, `reader_evidence_use_analysis.json` and `node_level_features.jsonl`. The structural conclusions were uninformative, not inverted.

**Fix.** Commit `c96ccbd` adds `structural_counts(node_ids, edge_index)`, which converts edge positions to node IDs before aggregating, adds an explicit `child_count`, and defines `is_leaf` as `child_count == 0`. Consistency assertions (`sum(child_count) == n_edges`, `sum(degree) == 2 * n_edges`) are stored with the artifact and re-checked by the verifier. Five regression tests use a synthetic snapshot with string node IDs.

**Affected.** Only the structural fields listed above. The affected artifacts were regenerated by deterministic feature extraction; no Qwen inference was added.

**Not affected.** Frozen Qwen generations, all V3-B detection metrics, compression metrics, citation mapping, reader-evidence-loss analysis, proxy-margin analysis, label asymmetry, cutoff analysis, the cross-fold audit, the root causes, and the recommendation. Frozen artifact hashes were re-verified unchanged.

**Final status.** CLOSED. After the fix, no stable structural reader-risk signal emerged; the largest difference (Ma-Weibo retained vs removed source-child rate 0.699 vs 0.852) is recorded as a candidate observation only.

---

## P04 — Cross-fold reader-development contamination risk

**Original issue.** The V3-B pilot samples 300 event-cutoffs per dataset from the union of the five outer-fold validation pools. Because the temporal split assigns an event to validation in one fold and to test in another, an event used in the pilot may be an outer test event in a different fold.

**Impact.** Quantified in `results/tcdscr/v3b_failure_diagnosis/cross_fold_development_audit.json`: 285/285 sampled PHEME events and 275/275 sampled Ma-Weibo events occur in another fold's test split (rate 1.0000); 47 and 53 events respectively appear in more than one validation fold.

**Fix / mitigation.** No re-sampling was performed (the pilot is frozen). The consequence is stated as a protocol limitation: the aggregate pilot must not be used to design a global heuristic while claiming the original 5-fold test was untouched.

**Affected.** Any future attempt to derive a reader-calibration rule from the aggregated 600-sample pilot.

**Not affected.** The pilot's own measured reader metrics, which remain valid as a validation-only transfer measurement, and the recommendation `MS_TSR_COMPRESSION_ONLY`, which does not introduce any rule.

**Final status.** OPEN BY DESIGN. If a fold-local reader calibration is ever introduced, it must be calibrated inside the outer fold only, leaving that fold's test untouched.
