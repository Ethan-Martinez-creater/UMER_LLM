"""M1-F report rendering tests (pure string assembly over the payloads)."""
from __future__ import annotations

from pathlib import Path

from ..config import protocol as P
from ..m1f import report
from ..m1f import task_validity as tv
from .test_verifier_m1f import (_active_payload, _baseline_payload,
                                _robustness_payload, _verdict_payload,
                                _within_payload)

REPO_DIR = Path(__file__).resolve().parents[3]


def _report_data(datasets=P.DATASETS):
    active = _active_payload()
    return {
        "commit": "b" * 40,
        "environment": "TEST",
        "task_validity": tv.task_validity(str(REPO_DIR)),
        "baselines": {ds: _baseline_payload(ds) for ds in datasets},
        "active": active,
        "within": _within_payload(),
        "robustness": _robustness_payload(),
        "pheme": {"diagnostic_only": True, "decides_gate": False,
                  "comparator": {"selected": P.M1F_S2},
                  "primary_comparison": active["primary"], "note": "n/a"},
        "verdict": _verdict_payload(),
        "pins": {"n_groups": 10, "n_artifacts": 43},
    }


def test_build_report_renders_every_section():
    md = report.build_report(_report_data())
    for marker in ("Strong-Baseline", "Task validity", "Strong baselines",
                   "Primary comparison", "Active HELPFUL/HARMFUL",
                   "Within-snapshot", "Fixed threshold", "PHEME diagnostic",
                   "M1-F decision", "Immutability"):
        assert marker in md, marker
    assert "M1F_CLOSE_BCR_UTILITY_METHOD" in md
    assert "M2 entered                  = False" in md


def test_build_report_lists_the_ladder_and_the_comparator():
    md = report.build_report(_report_data())
    for variant in P.M1F_VARIANTS:
        assert variant in md
    assert P.M1F_S2 in md


def test_build_report_handles_a_single_dataset_run():
    md = report.build_report(_report_data(datasets=(P.PRIMARY_DATASET,)))
    section = md.split("## 2. Strong baselines")[1].split("## 3.")[0]
    assert "### maweibo" in section
    assert "### pheme" not in section
