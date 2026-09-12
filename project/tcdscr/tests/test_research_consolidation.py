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
