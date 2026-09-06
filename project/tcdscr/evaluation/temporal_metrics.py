"""Dynamic/temporal metrics (plan §42): prediction flip rate, evidence
turnover, persistent/new ratio.

Inputs are per-event ordered traces: each event yields a list of snapshot
steps ``[{cutoff, prediction, selected_ids}]`` in causal order.
"""
from __future__ import annotations


def flip_rate(traces) -> dict:
    """Prediction flip rate over consecutive snapshot transitions."""
    transitions = flips = 0
    for trace in traces:
        preds = [s["prediction"] for s in trace
                 if s["prediction"] is not None]
        for a, b in zip(preds, preds[1:]):
            transitions += 1
            flips += 1 if a != b else 0
    return {"transitions": transitions, "flips": flips,
            "flip_rate": flips / max(transitions, 1)}


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    if not sa or not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def evidence_turnover(traces) -> dict:
    """Mean 1-Jaccard between consecutive selected evidence sets."""
    distances = []
    for trace in traces:
        sets = [s["selected_ids"] for s in trace]
        for a, b in zip(sets, sets[1:]):
            d = _jaccard(a, b)
            if d is not None:
                distances.append(1.0 - d)
    return {"transitions": len(distances),
            "mean_turnover": sum(distances) / max(len(distances), 1)}


def persistent_new_ratio(traces) -> dict:
    """Share of persistent evidence across consecutive selections."""
    persistent = new = 0
    for trace in traces:
        sets = [set(s["selected_ids"]) for s in trace]
        for prev, cur in zip(sets, sets[1:]):
            if not prev and not cur:
                continue
            persistent += len(prev & cur)
            new += len(cur - prev)
    total = persistent + new
    return {"persistent": persistent, "new": new,
            "persistent_ratio": persistent / max(total, 1)}
