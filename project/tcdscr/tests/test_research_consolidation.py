"""Research-consolidation tests (protocol §25).

Ten checks over the consolidation artifacts.  They are pure file/JSON reads
plus the verifier's own helpers, so no GPU, dataset or Qwen model is needed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_verify_research_consolidation as V  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _consolidation_root():
    local = REPO / "results" / "tcdscr" / "research_consolidation"
    if local.exists():
        return local
    server = Path("/data/jyz/next/llm/results/tcdscr/research_consolidation")
    if server.exists():
        return server
    return None


ROOT = _consolidation_root()
pytestmark = pytest.mark.skipif(ROOT is None,
                                reason="consolidation artifacts not present")


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _json(name):
    return json.loads(_read(name))


# 1 --------------------------------------------------------------------------
def test_evidence_ledger_claims_have_sources():
    ledger = _json("research_evidence_ledger.json")
    registry = _json("experiment_registry.json")
    stage_ids = {st["stage_id"] for st in registry["stages"]}
    claims = ledger["claims"]
    assert ledger["claim_count"] == len(claims)
    for claim in claims:
        assert claim.get("supporting_stage"), claim["claim_id"]
        assert set(claim["supporting_stage"]) <= stage_ids, claim["claim_id"]
        assert claim.get("artifacts"), claim["claim_id"]
        assert claim.get("metrics"), claim["claim_id"]
        for art in claim["artifacts"]:
            assert (REPO / art.rstrip("/")).exists(), art


# 2 --------------------------------------------------------------------------
def test_canonical_result_has_split_type():
    text = _read("CANONICAL_RESULTS_TABLE.md")
    sections = V._sections(text)
    splits_seen = set()
    for heading, _body in sections:
        for split in ("TRAIN", "VALIDATION", "HELD_OUT_TEST", "DIAGNOSTIC"):
            if split in heading:
                splits_seen.add(split)
    assert {"TRAIN", "VALIDATION", "HELD_OUT_TEST"} <= splits_seen
    # every table body must carry per-row status markers
    assert text.count("| CANONICAL") >= 10


# 3 --------------------------------------------------------------------------
def test_validation_not_marked_test():
    text = _read("CANONICAL_RESULTS_TABLE.md")
    for heading, body in V._sections(text):
        if "HELD_OUT_TEST" in heading:
            assert "reader_transfer_summary.json" not in body, heading
    pilot = [b for h, b in V._sections(text) if "VALIDATION PILOT" in h]
    assert pilot, "no VALIDATION PILOT section"
    assert any("reader_transfer_summary.json" in b for b in pilot)


# 4 --------------------------------------------------------------------------
def test_deprecated_results_not_canonical():
    dep_text = _read("DEPRECATED_RESULTS.md")
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    block = re.search(r"```json\s*(\{.*?\})\s*```", dep_text, re.S)
    assert block, "deprecated ledger has no json block"
    dep = json.loads(block.group(1))
    for root in dep["deprecated_artifact_roots"]:
        assert root not in canonical, root
    for number in dep["deprecated_numbers"]:
        assert number not in canonical, number


# 5 --------------------------------------------------------------------------
def test_prohibited_claims_present():
    text = _read("PROHIBITED_CLAIMS.md")
    for pid in V.PROHIBITED_CLAIM_IDS:
        assert f"### {pid} " in text, pid
    meta = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
                      .group(1))
    assert meta["n_prohibited"] == len(V.PROHIBITED_CLAIM_IDS)
    assert "never transfers" in text  # required phrasing rule present


# 6 --------------------------------------------------------------------------
def test_ms_tsr_status_compression_only():
    manifest = _json("research_freeze_manifest.json")
    report = _read("RESEARCH_CONSOLIDATION_REPORT.md")
    diag = json.loads((REPO / V.DIAG_DIR / "diagnosis_summary.json")
                      .read_text(encoding="utf-8"))
    assert manifest["final_v3_recommendation"] == "MS_TSR_COMPRESSION_ONLY"
    assert diag["recommendation"] == "MS_TSR_COMPRESSION_ONLY"
    assert "MS_TSR_COMPRESSION_ONLY" in report
    matrix = _read("FINAL_CONTRIBUTION_MATRIX.md")
    assert "never as a universal sufficiency selector" in matrix


# 7 --------------------------------------------------------------------------
def test_v3b_marked_validation_pilot():
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    report = _read("RESEARCH_CONSOLIDATION_REPORT.md")
    ledger = _read("RESEARCH_EVIDENCE_LEDGER.md")
    assert "VALIDATION PILOT" in canonical
    assert "validation pilot" in report.lower()
    low = ledger.lower()
    assert "validation pilot" in low or "validation reader pilot" in low


# 8 --------------------------------------------------------------------------
def test_no_new_training_artifact():
    bad = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name.lower().endswith(
                V.CHECKPOINT_SUFFIXES):
            bad.append(path.name)
    assert bad == []
    manifest = _json("research_freeze_manifest.json")
    assert manifest["new_training_allowed"] is False
    assert manifest["new_qwen_inference_allowed"] is False
    assert manifest["held_out_test_allowed"] is False


# 9 --------------------------------------------------------------------------
def test_no_new_qwen_generation():
    frozen = json.loads((REPO / V.DIAG_DIR / "frozen_artifacts.json")
                        .read_text(encoding="utf-8"))
    reader = REPO / V.READER_DIR
    assert V.sha256_file(str(reader / "sampling_manifest.json")) == \
        frozen["sampling_manifest_sha256"]
    for ds in V.DATASETS:
        assert V.sha256_file(str(reader / "parsed" / f"{ds}.jsonl")) == \
            frozen["parsed_sha256"][ds]
        assert V.sha256_file(
            str(reader / "raw_generations" / f"{ds}.jsonl")) == \
            frozen["raw_generations_sha256"][ds]
        assert V.sha256_file(str(reader / "prompts" / f"{ds}.jsonl")) == \
            frozen["prompts_sha256"][ds]


# 10 -------------------------------------------------------------------------
def test_consolidation_required_files():
    for name in V.REQUIRED:
        assert (ROOT / name).exists(), name
    result = json.loads((ROOT / "research_consolidation_verify.json")
                        .read_text(encoding="utf-8"))
    assert result["n_issues"] == 0, result["issues"]
    assert result["recommendation"] == "MS_TSR_COMPRESSION_ONLY"
    assert result["v3c_approved"] is False
    assert result["full_e4_approved"] is False


# --- consolidation finalization (protocol section 27) -----------------------

# 11 -------------------------------------------------------------------------
def test_e1_reported_split_is_validation():
    registry = _json("experiment_registry.json")
    e1 = [st for st in registry["stages"] if st["stage_id"] == "E1"][0]
    assert e1["split_used"] == "VALIDATION"
    assert e1["training_split"] == "TRAIN"
    assert e1["reported_evaluation_split"] == "VALIDATION"
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    for heading, body in V._sections(canonical):
        if "E1" in heading and "Causal Encoder" in heading:
            assert "split: VALIDATION" in heading or \
                "split: VALIDATION" in body
            # "training split: TRAIN" is correct and must not be read as a
            # TRAIN evidence label.
            assert not re.search(r"(?<!training )split: TRAIN", heading)
            assert not re.search(r"(?<!training )split: TRAIN", body)


# 12 -------------------------------------------------------------------------
def test_e2_canonical_values_are_all_event():
    source = json.loads((REPO / "results" / "tcdscr" / "dynamic_v2_protocol"
                         / "e2_corrected_all_event_metrics.json")
                        .read_text(encoding="utf-8"))
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    ledger = _json("research_evidence_ledger.json")
    c03 = [c for c in ledger["claims"] if c["claim_id"] == "C03"][0]
    assert c03["protocol_status"] == "ALL_EVENT"
    for ds in ("pheme", "maweibo"):
        static = source["datasets"][ds]["mean_over_runs_new"]["static"]
        assert f"{static:.5f}" in canonical, ds
        assert abs(c03["metrics"][ds]["static"] - static) < 1e-9, ds


# 13 -------------------------------------------------------------------------
def test_candidate_conditioned_e2_not_canonical():
    dep = json.loads(re.search(
        r"```json\s*(\{.*?\})\s*```", _read("DEPRECATED_RESULTS.md"), re.S)
        .group(1))
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    for number in dep["deprecated_numbers"]:
        assert number not in canonical, number
    for line in canonical.splitlines():
        if "CANDIDATE_CONDITIONED" in line:
            assert "HISTORICAL_ONLY" in line or \
                "CONDITIONAL_HELD_OUT" in line or \
                "DEPRECATED_ABSOLUTE" in line, line


# 14 -------------------------------------------------------------------------
def test_e3_no_candidate_scope_audit_present():
    audit = _json("e3_no_candidate_scope_audit.json")
    md = _read("E3_NO_CANDIDATE_SCOPE_AUDIT.md")
    assert audit["scope_verdict"] in ("CASE_A_ALL_EVENTS_RETAINED",
                                     "CASE_B_ZERO_CANDIDATE_SKIPPED")
    assert audit["totals"]["expected_rows"] > 0
    assert audit["totals"]["actual_rows"] > 0
    assert audit["cross_check_consistent"] is True
    assert isinstance(audit["runner"]["current_source_keeps_zero_candidate"],
                      bool)
    if audit["scope_verdict"] == "CASE_B_ZERO_CANDIDATE_SKIPPED":
        assert audit["runner"]["artifacts_observed_skip"] is True
    assert audit["scope_verdict"] in md
    recomputed = V.e3_scope_audit()
    assert recomputed["scope_verdict"] == audit["scope_verdict"]
    assert recomputed["totals"]["actual_rows"] == audit["totals"]["actual_rows"]


# 15 -------------------------------------------------------------------------
def test_e3_canonical_status_matches_scope_audit():
    audit = _json("e3_no_candidate_scope_audit.json")
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    assert audit["held_out_status"] in canonical
    if audit["scope_verdict"] == "CASE_B_ZERO_CANDIDATE_SKIPPED":
        assert "CONDITIONAL_HELD_OUT" in canonical
        assert "DEPRECATED_ABSOLUTE" in _read("DEPRECATED_RESULTS.md")
        gaps = _read("NEXT_EXPERIMENT_GAPS.md")
        assert "Gap F" in gaps
    else:
        assert "HELD_OUT" in canonical


# 16 -------------------------------------------------------------------------
def test_contribution_d_requires_gap_b():
    matrix = _read("FINAL_CONTRIBUTION_MATRIX.md")
    gaps = _read("NEXT_EXPERIMENT_GAPS.md")
    report = _read("RESEARCH_CONSOLIDATION_REPORT.md")
    assert "Proxy-to-LLM Reader Transfer Analysis" in matrix
    assert "VALIDATION-PILOT SUPPORTED; FINAL HELD-OUT EVIDENCE PENDING" in \
        matrix
    assert "Gap B" in gaps
    assert "fold-local" in gaps.lower()
    assert "A1" in report


# 17 -------------------------------------------------------------------------
def test_option_a1_contains_gap_a_b_c():
    gaps = _read("NEXT_EXPERIMENT_GAPS.md")
    report = _read("RESEARCH_CONSOLIDATION_REPORT.md")
    for gap in ("Gap A", "Gap B", "Gap C"):
        assert gap in gaps, gap
        assert gap in report, gap
    assert "A1" in gaps and "A1" in report
    assert "combined" in gaps.lower()
    assert "NOT APPROVED" in gaps


# 18 -------------------------------------------------------------------------
def test_deprecated_numeric_values_not_canonical():
    dep = json.loads(re.search(
        r"```json\s*(\{.*?\})\s*```", _read("DEPRECATED_RESULTS.md"), re.S)
        .group(1))
    canonical = _read("CANONICAL_RESULTS_TABLE.md")
    assert dep["deprecated_numbers"], "no deprecated numbers declared"
    for root in dep["deprecated_artifact_roots"]:
        assert root not in canonical, root
    for number in dep["deprecated_numbers"]:
        assert number not in canonical, number
    # the deprecated E2 values must not be reachable as canonical rows
    for line in canonical.splitlines():
        if "formal_e2_corrected/readiness/e2_summary.json" in line:
            assert "HISTORICAL_ONLY" in line, line

