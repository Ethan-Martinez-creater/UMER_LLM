"""Smoke aggregation (plan §31): fold the per-run artifacts into one
summary with an overall PASS / PARTIAL / FAIL status for Code Complete."""
from __future__ import annotations


def aggregate_smoke(dataset_reports: dict, checks: dict) -> dict:
    """``dataset_reports``: {dataset: metrics-dict}; ``checks``: named gates.

    Each gate is {"pass": bool, "detail": str}. Overall status is FAIL if any
    hard gate fails, PARTIAL for soft failures, PASS otherwise.
    """
    hard_gates = ("tests_pass", "no_future_leakage", "cap_deterministic",
                  "snapshots_finite", "prompt_deterministic")
    status = "PASS"
    for name, gate in checks.items():
        if gate.get("pass"):
            continue
        status = "FAIL" if name in hard_gates else "PARTIAL"
    return {"status": status, "datasets": dataset_reports, "checks": checks}
