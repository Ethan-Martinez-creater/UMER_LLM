"""Structured social interaction (plan §12) and the P2 gate (plan §25).

    Interaction_r(A)  = u_r(A) - sum_{e_i in A} u_r({e_i})
    Delta_edge        = E|I(parent-child)| - E|I(matched-nonadjacent)|
    Delta_subtree     = E|I(subtree)|      - E|I(matched-disconnected)|

The review fixes are structural, not cosmetic:

* matching stays **reader-specific and snapshot-specific** — an I2 for reader
  ``r`` at ``(event, cutoff)`` is paired only with the I3 for the same reader
  and the same snapshot;
* the event-level bootstrap (plan §24) resamples events, and every paired
  snapshot of a sampled event moves with it, preserving multiplicity.
"""
from __future__ import annotations

from collections import defaultdict

from ..config.pilot_config import P2_EDGE_DELTA_MIN
from .bootstrap import mean, paired_event_bootstrap

TYPE_EDGE = "I2"
TYPE_EDGE_CONTROL = "I3"
TYPE_SUBTREE = "I4"
TYPE_SUBTREE_CONTROL = "I5"

SLOTS = {TYPE_EDGE: "pc", TYPE_EDGE_CONTROL: "na",
         TYPE_SUBTREE: "sub", TYPE_SUBTREE_CONTROL: "disc"}


def interaction(utility_group: float, member_utilities) -> float:
    """``u_r(A) - sum u_r({e_i})`` (plan §12)."""
    return float(utility_group) - sum(float(m) for m in member_utilities)


def build_event_payloads(records) -> dict:
    """``{event: {slot: [(reader, snapshot_key, magnitude)]}}`` (plan §12).

    Each record must expose ``event``, ``cutoff``, ``reader``, ``type``,
    ``utility`` and ``members`` so the pairing can be reader- and
    snapshot-specific.
    """
    by = defaultdict(lambda: {"pc": [], "na": [], "sub": [], "disc": []})
    for r in records:
        slot = SLOTS.get(r["type"])
        if slot is None:
            continue
        magnitude = abs(interaction(r["utility"], r.get("members") or []))
        snapshot = f"{r['event']}|{int(r['cutoff'])}"
        by[r["event"]][slot].append((r["reader"], snapshot, magnitude))
    return dict(by)


def _pair_deltas(payloads, key_a, key_b):
    """Per-(reader, snapshot) ``|I_a| - |I_b|`` for jointly present pairs."""
    a_index = {(reader, snap): value for reader, snap, value in payloads[key_a]}
    b_index = {(reader, snap): value for reader, snap, value in payloads[key_b]}
    shared = sorted(set(a_index) & set(b_index))
    return [a_index[k] - b_index[k] for k in shared]


def _delta_statistic(key_a, key_b):
    def statistic(payloads):
        deltas = []
        for payload in payloads:
            deltas.extend(_pair_deltas(payload, key_a, key_b))
        return mean(deltas)
    return statistic


def structural_interaction_report(records, iterations=None, seed=None) -> dict:
    """Edge and subtree interaction gaps with event-level bootstrap (plan §12)."""
    payloads = build_event_payloads(records)
    edge_events = {e: p for e, p in payloads.items()
                   if p["pc"] and p["na"]}
    subtree_events = {e: p for e, p in payloads.items()
                      if p["sub"] and p["disc"]}
    out = {"matching_unit": "reader_x_snapshot",
           "bootstrap_unit": "event"}
    if edge_events:
        boot = paired_event_bootstrap(edge_events,
                                      _delta_statistic("pc", "na"),
                                      iterations=iterations, seed=seed)
        deltas = []
        for payload in edge_events.values():
            deltas.extend(_pair_deltas(payload, "pc", "na"))
        out["edge"] = {
            "n_matched_pairs": len(deltas),
            "mean_delta": mean(deltas),
            "delta": boot["observed"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "n_events": boot["n_events"]}
    else:
        out["edge"] = {"delta": float("nan"), "n_events": 0,
                       "reason": "no event has both I2 and I3"}
    if subtree_events:
        boot = paired_event_bootstrap(subtree_events,
                                      _delta_statistic("sub", "disc"),
                                      iterations=iterations, seed=seed)
        deltas = []
        for payload in subtree_events.values():
            deltas.extend(_pair_deltas(payload, "sub", "disc"))
        out["subtree"] = {
            "n_matched_pairs": len(deltas), "mean_delta": mean(deltas),
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
        "n_matched_pairs": edge.get("n_matched_pairs", 0),
        "matching_unit": report.get("matching_unit"),
        "bootstrap_unit": report.get("bootstrap_unit"),
        "pass": bool(passed),
        "subtree_confirmatory": report.get("subtree", {}),
    }
