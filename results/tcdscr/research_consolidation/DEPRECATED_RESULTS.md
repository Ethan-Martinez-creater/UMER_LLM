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
      "scope": "candidate-conditioned E2 / E3-V1 metrics (metrics computed only over snapshots with candidate_count > 0)",
      "reason": "Metric conditioned on the presence of social evidence; no-candidate events were silently dropped, which violates the all-event protocol.",
      "impact": "Absolute values shifted after re-evaluation; pre-correction numbers are not comparable with post-correction numbers. No-candidate rates: PHEME 0.35525 at 5m, Ma-Weibo 0.12674 at 5m.",
      "deprecated_numbers": ["0.8557205505338684", "0.853524186296833", "0.8523170709420275"],
      "replacement": "results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json and results/tcdscr/dynamic_v2_protocol/e3_v1_all_event_diagnostic.json",
      "replacement_artifact": "results/tcdscr/dynamic_v2_protocol/e2_corrected_all_event_metrics.json",
      "replacement_commit": "a0a0d31",
      "status": "DEPRECATED_NUMBERS"
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
      "note": "The macro_f1 point estimates in formal_e3_test/e3_test_summary.json remain canonical; only its bootstrap fields are deprecated."
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
    }
  ],
  "deprecated_artifact_roots": [
    "results/tcdscr/formal_e2/"
  ],
  "deprecated_numbers": [
    "0.8557205505338684",
    "0.853524186296833",
    "0.8523170709420275"
  ]
}
```

## Reading guide

- **D01** is a whole-directory deprecation. Nothing under `results/tcdscr/formal_e2/` may be cited.
- **D02** is a denominator/protocol deprecation. Use the all-event artifacts instead.
- **D03** is a field-level deprecation inside an otherwise canonical artifact.
- **D04** is a field-level deprecation fixed in the V3-B finalization commit.
- **D05** is a report-layer deprecation; the underlying E1 numbers stand.

## Not deprecated

- The E3 held-out macro-F1 point estimates and the REJECTED verdict for Dynamic V1.
- The corrected E2 static/random/semantic means on Ma-Weibo and PHEME (post-correction values).
- The V3-A proxy summary (compression, dual-view agreement, fallback rates), with the structural ΔMF1 = 0 caveat.
- All V3-B reader-pilot detection, compression and citation metrics, and the diagnosis recommendation MS_TSR_COMPRESSION_ONLY.
