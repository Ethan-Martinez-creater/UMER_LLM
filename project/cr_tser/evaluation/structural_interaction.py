"""Structured social interaction (plan §12) and the P2 gate (plan §25).

    Interaction_r(A)  = u_r(A) - sum_{e_i in A} u_r({e_i})
    Delta_edge        = E|I(parent-child)| - E|I(matched-nonadjacent)|
    Delta_subtree     = E|I(subtree)|      - E|I(matched-disconnected)|

Interactions are magnitudes; the bootstrap is event-level and includes only
events that carry a valid structured **and** matched control group (plan §24).
"""
from __future__ import annotations

from collections import defaultdict

from ..config.pilot_config import P2_EDGE_DELTA_MIN
from .bootstrap import mean, paired_event_bootstrap

TYPE_EDGE = "I2"
TYPE_EDGE_CONTROL = "I3"
TYPE_SUBTREE = "I4"
TYPE_SUBTREE_CONTROL = "I5"


def interaction(utility_group: float, member_utilities) -> float:
    """``u_r(A) - sum u_r({e_i})`` (plan §12)."""
    return float(utility_group) - sum(float(m) for m in member_utilities)


def build_event_payloads(records) -> dict:
    """``{event: {pc, na, sub, disc}}`` of interaction magnitudes (plan §12)."""
    by = defaultdict(lambda: {"pc": [], "na": [], "sub": [], "disc": []})
    for r in records:
        value = abs(interaction(r["utility"], r.get("members") or []))
        slot = {TYPE_EDGE: "pc", TYPE_EDGE_CONTROL: "na",
                TYPE_SUBTREE: "sub", TYPE_SUBTREE_CONTROL: "disc"}.get(r["type"])
        if slot:
            by[r["event"]][slot].append(value)
    return dict(by)


def _flatten(payloads, key):
    out = []
    for p in payloads:
        out.extend(p[key])
    return out


def delta_statistic(key_a, key_b):
    def statistic(payloads):
        a, b = _flatten(payloads, key_a), _flatten(payloads, key_b)
        if not a or not b:
            return float("nan")
        return mean(a) - mean(b)
    return statistic


def structural_interaction_report(records, iterations=None, seed=None) -> dict:
    """Edge and subtree interaction gaps with event bootstrap (plan §12)."""
    payloads = build_event_payloads(records)
    edge_events = {e: p for e, p in payloads.items()
                   if p["pc"] and p["na"]}
    subtree_events = {e: p for e, p in payloads.items()
                      if p["sub"] and p["disc"]}
    out = {}
    if edge_events:
        boot = paired_event_bootstrap(edge_events,
                                      delta_statistic("pc", "na"),
                                      iterations=iterations, seed=seed)
        out["edge"] = {"mean_parent_child": mean(_flatten(
            list(edge_events.values()), "pc")),
            "mean_matched_nonadjacent": mean(_flatten(
                list(edge_events.values()), "na")),
            "delta": boot["observed"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "n_events": boot["n_events"]}
    else:
        out["edge"] = {"delta": float("nan"), "n_events": 0,
                       "reason": "no event has both I2 and I3"}
    if subtree_events:
        boot = paired_event_bootstrap(subtree_events,
                                      delta_statistic("sub", "disc"),
                                      iterations=iterations, seed=seed)
        out["subtree"] = {"mean_subtree": mean(_flatten(
            list(subtree_events.values()), "sub")),
            "mean_matched_disconnected": mean(_flatten(
                list(subtree_events.values()), "disc")),
            "delta": boot["observed"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "n_events": boot["n_events"]}
    else:
        out["subtree"] = {"delta": float("nan"), "n_events": 0,
                          "reason": "no event has both I4 and I5"}
    return out


def gate_p2(report) -> dict:
    """P2: edge delta ≥ 0.02 with event-bootstrap CI lower bound > 0 (§25)."""
    edge = report["edge"]
    delta = edge.get("delta", float("nan"))
    low = edge.get("ci_low", float("nan"))
    passed = (delta == delta and low == low and delta >= P2_EDGE_DELTA_MIN
              and low > 0.0)
    return {
        "gate": "P2_structured_interaction",
        "edge_delta": delta, "edge_ci_low": low,
        "edge_threshold": P2_EDGE_DELTA_MIN,
        "pass": bool(passed),
        "subtree_confirmatory": report.get("subtree", {}),
    }
