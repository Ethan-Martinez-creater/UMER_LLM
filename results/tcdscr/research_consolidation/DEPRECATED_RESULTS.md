# TC-DSCR Deprecated Results

Artifacts and numbers on this page must **not** be cited as canonical results. They are retained in the repository for auditability only. Each entry names its replacement artifact and replacement commit.

```json
{
  "deprecated_entries": [
    {
      "entry_id": "D01",
      "scope": "results/tcdscr/formal_e2/ (whole directory: E2_READINESS_REPORT.md, readiness/, pheme/, maweibo/)",
      "reason": "INVALID: the proxy was trained with [z_sel ; g_t] but evaluated with [z_sel ; h_source], violating the frozen proxy contract proxy input = [h_source ; z_sel].",
      "impact": "Every readiness number in that directory is unusable for scientific claims.",
      "replacement": "results/tcdscr/formal_e2_corrected/",
      "replacement_artifact": "results/tcdscr/formal_e2_corrected/readiness/e2_summary.json",
      "replacement_commit": "beaac68",
      "status": "DEPRECATED_WHOLE_DIRECTORY"
    },
    {
      "entry_id": "D02",
      "scope": "candidate-conditioned E2 and E3 validation/readiness metrics (metric computed only over snapshots with candidate_count > 0); the E3 held-out run is NOT in this entry",
      "reason": "Metric conditioned on the presence of social evidence; no-candidate events were silently dropped, which violates the all-event protocol.",
      "impact": "Absolute values shifted after re-evaluation; pre-correction numbers are not comparable with post-correction numbers. No-candidate rates at 5m: PHEME 0.35525, Ma-Weibo 0.12674.",
      "deprecated_numbers": ["0.8557205505338684", "0.853524186296833", "0.8523170709420275", "0.9350558488145194", "0.9139445843843265", "0.9090430147579556"],
      "replacement": "results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json",
      "replacement_artifact": "results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json",
      "replacement_commit": "a0a0d31",
      "status": "DEPRECATED_NUMBERS",
      "note": "The historical readiness decision (PHEME FAIL, Ma-Weibo PASS, overall PARTIAL) is retained for audit only; the canonical paper metric is the all-event value."
    },
    {
      "entry_id": "D03",
      "scope": "bootstrap confidence intervals stored in results/tcdscr/formal_e3_test/e3_test_summary.json (fields datasets.*.bootstrap.*)",
      "reason": "Multiplicity bug: the with-replacement resample was collapsed into a unique-event dict, so duplicated events were dropped and the CIs were invalid.",
      "impact": "The published E3 CIs were too narrow. The corrected CIs still cross zero, so the WEAK_POSITIVE / REVIEW_DYNAMIC_DESIGN verdict is unchanged.",
      "replacement": "results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json",
      "replacement_artifact": "results/tcdscr/e3_failure_diagnosis/bootstrap_fixed.json",
      "replacement_commit": "1d97fd1",
      "status": "DEPRECATED_FIELD",
      "note": "The macro_f1 point estimates in formal_e3_test/e3_test_summary.json are themselves separately classified as CONDITIONAL_HELD_OUT (see D06); only the bootstrap fields are deprecated by this entry."
    },
    {
      "entry_id": "D04",
      "scope": "structural degree / leaf / node-role fields in results/tcdscr/v3b_failure_diagnosis/structural_role_analysis.json, removed_evidence_feature_analysis.json, reader_evidence_use_analysis.json and node_level_features.jsonl",
      "reason": "Structural bug: snapshot['edge_index'] stores edge position indices (not node IDs), but the degree counter was keyed by position while queries used node IDs, so degree collapsed to ~0 and every node appeared to be a leaf.",
      "impact": "Pre-fix mean degree was ~0 and leaf rate ~1.0, which is degenerate. Detection metrics, compression metrics, citation mapping, reader-evidence-loss, proxy-margin, label-asymmetry, cutoff, cross-fold and the final recommendation were NOT affected.",
      "replacement": "results/tcdscr/v3b_failure_diagnosis/structural_role_analysis.json",
      "replacement_note": "the same paths regenerated with edge positions converted to node IDs and is_leaf defined by child_count == 0",
      "replacement_artifact": "results/tcdscr/v3b_failure_diagnosis/structural_role_analysis.json",
      "replacement_commit": "c96ccbd",
      "status": "DEPRECATED_FIELD"
    },
    {
      "entry_id": "D05",
      "scope": "the first E1 submission's fold-parity / epochs_run reporting",
      "reason": "UMER_INIT_FOLD_PARITY_FAIL found in the finalization audit; epochs_run and cap-aware diagnostics needed correction.",
      "impact": "E1 was not retrained; only the audit/report layer changed. E1 encoder weights and the E1 per-cutoff metrics are unaffected.",
      "replacement": "results/tcdscr/formal_e1_finalization/",
      "replacement_artifact": "results/tcdscr/formal_e1_finalization/E1_FINALIZATION_REPORT.md",
      "replacement_commit": "26029af",
      "status": "DEPRECATED_REPORT_LAYER"
    },
    {
      "entry_id": "D06",
      "scope": "absolute Macro-F1 values of the E3 held-out run (results/tcdscr/formal_e3_test/e3_test_summary.json classification values)",
      "reason": "Scope audit CASE_B_ZERO_CANDIDATE_SKIPPED: the held-out runner dropped every event-cutoff with candidate_count == 0 (173,277 of 199,602 expected rows; zero retained rows have n_candidates == 0). The missing-row rate matches the validation no-candidate rate (PHEME 0.18630 vs 0.18774; Ma-Weibo 0.05693 vs 0.05071).",
      "impact": "The absolute held-out Macro-F1 values are not all-event and cannot be the paper's all-event held-out numbers.",
      "retained_use": "The relative negative finding is retained as a historical negative finding: Dynamic V1 is not supported (both corrected CIs cross zero). On a no-candidate snapshot both arms predict source-only, so restoring the dropped rows would move both arms toward the same value and cannot turn the negative delta positive.",
      "replacement": "results/tcdscr/research_consolidation/NEXT_EXPERIMENT_GAPS.md",
      "replacement_artifact": "results/tcdscr/research_consolidation/E3_NO_CANDIDATE_SCOPE_AUDIT.md",
      "replacement_commit": "47aa201",
      "status": "DEPRECATED_ABSOLUTE",
      "conditional_status": "CONDITIONAL_HELD_OUT",
      "note": "No re-run was performed. Validation all-event diagnostics are not a substitute for the held-out result; the all-event held-out cell is a recorded MUST_HAVE gap (Gap F)."
    }
  ],
  "deprecated_artifact_roots": [
    "results/tcdscr/formal_e2/"
  ],
  "deprecated_numbers": [
    "0.8557205505338684",
    "0.853524186296833",
    "0.8523170709420275",
    "0.9350558488145194",
    "0.9139445843843265",
    "0.9090430147579556"
  ]
}
```

## Reading guide

- **D01** is a whole-directory deprecation. Nothing under `results/tcdscr/formal_e2/` may be cited.
- **D02** is a denominator/protocol deprecation for E2 and E3 validation/readiness metrics. It explicitly does not cover the held-out run.
- **D03** is a field-level deprecation (bootstrap CIs) inside an otherwise historically retained artifact.
- **D04** is a field-level deprecation fixed in the V3-B finalization commit.
- **D05** is a report-layer deprecation; the underlying E1 numbers stand.
- **D06** is the held-out scope deprecation: absolute values are `CONDITIONAL_HELD_OUT` / `DEPRECATED_ABSOLUTE`; the relative verdict is retained.

## Not deprecated

- The E3 held-out **relative** verdict (Dynamic V1 REJECTED) as a historical negative finding.
- The corrected E2 all-event static/random/semantic values (these are the canonical E2 numbers).
- The V3-A proxy summary (compression, dual-view agreement, fallback rates), with the structural ΔMF1 = 0 caveat.
- All V3-B reader-pilot detection, compression and citation metrics, and the diagnosis recommendation MS_TSR_COMPRESSION_ONLY.
