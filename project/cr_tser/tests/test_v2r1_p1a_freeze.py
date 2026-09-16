"""V2R1-P1A — the formal v2r1 manifests are frozen.

The P1A round freezes the v2r1 manifests and makes them immutable from the
first formal label row onward. These tests pin them from the *package* side, so
a manifest edit is caught by the ordinary test run and not only by the verifier:

* every pinned file matches its canonical LF digest;
* the four files the V2 namespace also has are byte-equal to it — which is what
  makes reusing the migrated Qwen/InternLM labels legitimate;
* ``hashes.json`` carries the R1 reader contract (qwen/mistral/internlm, the R1
  LORO rotations, no snapshot cap, cutoffs 15/60/360).

No model, no GPU and no label cache is needed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

V2 = "results/cr_tser_v2"
V2R1 = "results/cr_tser_v2r1"
SHARED_FILES = ("source.json", "event_split.json", "snapshot_manifest.jsonl",
                "intervention_manifest.jsonl")


def _canonical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _pins():
    import cr_tser_verify_pilot as verifier
    return verifier.V2R1_FROZEN_MANIFEST_SHA256


def test_v2r1_manifest_files_match_their_pins():
    pins = _pins()
    assert len(pins) == 10
    drifted = {}
    for rel, expected in pins.items():
        path = REPO / rel
        if not path.exists():
            drifted[rel] = "missing"
        elif _canonical(path) != expected:
            drifted[rel] = "changed"
    assert drifted == {}, f"frozen v2r1 manifests drifted: {drifted}"


def test_v2r1_matches_v2_for_every_shared_manifest_file():
    same = {}
    for dataset in ("maweibo", "pheme"):
        for name in SHARED_FILES:
            old = REPO / V2 / "manifests" / dataset / name
            new = REPO / V2R1 / "manifests" / dataset / name
            assert old.exists(), f"{old} missing from the frozen V2 namespace"
            assert new.exists(), f"{new} missing from the v2r1 namespace"
            same[f"{dataset}/{name}"] = _canonical(old) == _canonical(new)
    assert all(same.values()), f"v2r1 diverged from V2: {same}"


def test_v2r1_hashes_json_carries_the_r1_reader_contract():
    from ..config.pilot_config import (CUTOFFS_MIN, LORO_ROTATIONS,
                                       READER_KEYS)
    for dataset in ("maweibo", "pheme"):
        frozen = json.loads((REPO / V2R1 / "manifests" / dataset /
                             "hashes.json").read_text(encoding="utf-8"))
        assert list(frozen["readers"]) == list(READER_KEYS)
        assert "glm" not in frozen["readers"]
        assert frozen["cutoffs"] == list(CUTOFFS_MIN) == [15, 60, 360]
        assert frozen["snapshot_cap"] is None
        assert tuple(tuple(r) for r in frozen["loro_rotations"]) == \
            tuple(LORO_ROTATIONS)
        for reader in READER_KEYS:
            record = frozen["readers"][reader]
            assert record.get("weight_hash"), f"{dataset}/{reader} weights"
            assert record.get("tokenizer_hash"), f"{dataset}/{reader} tokenizer"


def test_v2r1_hashes_json_differs_from_v2_only_in_the_reader_contract():
    """The reader set and LORO are exactly what amendment R1 changed."""
    for dataset in ("maweibo", "pheme"):
        old = json.loads((REPO / V2 / "manifests" / dataset /
                          "hashes.json").read_text(encoding="utf-8"))
        new = json.loads((REPO / V2R1 / "manifests" / dataset /
                          "hashes.json").read_text(encoding="utf-8"))
        differing = sorted(k for k in set(old) | set(new)
                           if old.get(k) != new.get(k))
        assert differing == ["loro_rotations", "readers"], \
            f"{dataset}: unexpected hashes.json drift {differing}"
        assert sorted(old["readers"]) == ["glm", "internlm", "qwen"]


def test_v2r1_split_is_the_frozen_80_50_15_25_seed_7319_split():
    from ..config.pilot_config import PARTITION_SEED, SPLIT_SIZES
    for dataset in ("maweibo", "pheme"):
        split = json.loads((REPO / V2R1 / "manifests" / dataset /
                            "event_split.json").read_text(encoding="utf-8"))
        assert split["split_seed"] == PARTITION_SEED == 7319
        assert split["sizes"] == dict(SPLIT_SIZES)
        assert {k: len(split[k]) for k in SPLIT_SIZES} == dict(SPLIT_SIZES)
        assert split["source"]["exists"] is True
        assert split["source"]["n_files"] > 0
        segments = [set(split[k]) for k in SPLIT_SIZES]
        for i, left in enumerate(segments):
            for right in segments[i + 1:]:
                assert not left & right, f"{dataset}: split segments overlap"


def test_v2r1_source_is_the_approved_maweibo_fingerprint():
    split = json.loads((REPO / V2R1 / "manifests" / "maweibo" /
                        "event_split.json").read_text(encoding="utf-8"))
    assert split["source"]["sha256"] == \
        "b982076df8f8ea8ce1eb167538801eecd5304a11bd6aa8bb23ec361e7df8c90d"


def test_the_p1a_verifier_checks_are_wired_into_the_v2r1_code_pass():
    import cr_tser_verify_pilot as verifier

    report = verifier.Report()
    verifier.verify_code(report, "v2r1")
    status = {c["check"]: c["status"] for c in report.checks}
    for check in ("v2r1_manifest_pins", "v2r1_manifests_match_v2",
                  "v2r1_reader_contract_in_manifest"):
        assert check in status, f"{check} is not part of the v2r1 code pass"
        assert status[check] == "pass", f"{check}: {status[check]}"

    # the v2 pass must not gain v2r1-only manifest checks
    v2_report = verifier.Report()
    verifier.verify_code(v2_report, "v2")
    v2_status = {c["check"] for c in v2_report.checks}
    assert "v2r1_manifest_pins" not in v2_status
