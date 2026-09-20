"""M1 paired event-level bootstrap for model deltas (M1 plan §14).

The primary comparison ``B4 - B0`` is resampled over **events** (never over
independent atomic rows), 10000 iterations, seed 7319, with the exact RNG
sequence the frozen CR-TSER bootstrap uses. B4 and B0 predictions are scored
on the *same* resample, so the pairing is preserved.

Per-reader intervals and the three-reader aggregate share one draw sequence:
every held-out reader has the same number of utility_eval events, so
``bootstrap_draws`` with the frozen seed yields identical draws per reader
and the aggregate replicate is the per-iteration mean of the reader deltas.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from .reader_geometry import bootstrap_draws
from .utility_metrics import SIGN_TO_INDEX


class BootstrapRefused(RuntimeError):
    """Raised when a bootstrap input violates the event-level contract."""


def _weighted_macro_f1(gold, pred, weights) -> float:
    n_classes = len(P.SIGN_CLASSES)
    f1s = []
    for c in range(n_classes):
        tp = fp = fn = 0.0
        for g, p, w in zip(gold, pred, weights):
            if p == c and g == c:
                tp += w
            elif p == c and g != c:
                fp += w
            elif p != c and g == c:
                fn += w
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def _argmax_rows(probs) -> list:
    return [max(range(len(P.SIGN_CLASSES)), key=lambda c: row[c])
            for row in probs]


def _percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _prepare(rows, probs_a, probs_b):
    if len(rows) != len(probs_a) or len(rows) != len(probs_b):
        raise BootstrapRefused("rows/predictions length mismatch")
    if not rows:
        raise BootstrapRefused("empty eval fold")
    events = sorted({str(r["event_id"]) for r in rows})
    pos = {e: i for i, e in enumerate(events)}
    row_event = [pos[str(r["event_id"])] for r in rows]
    gold = [SIGN_TO_INDEX[r["sign"]] for r in rows]
    return events, row_event, gold, _argmax_rows(probs_a), _argmax_rows(probs_b)


def _delta_on_draw(row_event, gold, pa, pb, mult) -> float:
    weights = [mult[e] for e in row_event]
    return (_weighted_macro_f1(gold, pa, weights)
            - _weighted_macro_f1(gold, pb, weights))


def paired_event_delta(rows, probs_a, probs_b,
                       iterations=P.BOOTSTRAP_ITERATIONS,
                       seed=P.BOOTSTRAP_SEED) -> dict:
    """``Macro-F1(A) - Macro-F1(B0)`` bootstrapped over events."""
    events, row_event, gold, pa, pb = _prepare(rows, probs_a, probs_b)
    n = len(events)
    ones = [1] * n
    observed = _delta_on_draw(row_event, gold, pa, pb, ones)
    reps = []
    for draw in bootstrap_draws(n, iterations, seed):
        mult = [0] * n
        for e in draw:
            mult[e] += 1
        reps.append(_delta_on_draw(row_event, gold, pa, pb, mult))
    ordered = sorted(reps)
    return {
        "observed": observed,
        "ci_low": _percentile(ordered, P.GATE_CI_ALPHA / 2),
        "ci_high": _percentile(ordered, 1 - P.GATE_CI_ALPHA / 2),
        "iterations": iterations,
        "seed": seed,
        "n_events": n,
        "unit": P.BOOTSTRAP_UNIT,
        "reps": reps,
    }


def aggregate_delta(per_reader, iterations=P.BOOTSTRAP_ITERATIONS,
                    seed=P.BOOTSTRAP_SEED) -> dict:
    """Mean of the per-reader deltas with a shared draw sequence.

    ``per_reader``: ``[(rows, probs_a, probs_b), ...]`` in a fixed reader
    order. Every reader is resampled with the same frozen draw sequence
    (same seed, same event count), and each aggregate replicate is the mean
    of that iteration's per-reader deltas.
    """
    if not per_reader:
        raise BootstrapRefused("no readers to aggregate")
    prepared = [_prepare(rows, pa, pb) for rows, pa, pb in per_reader]
    counts = {len(events) for events, _re, _g, _pa, _pb in prepared}
    if len(counts) != 1:
        raise BootstrapRefused(
            f"readers have different eval event counts {sorted(counts)}; "
            "the shared-sequence aggregate requires equality")
    n = counts.pop()
    observed = sum(_delta_on_draw(re, g, pa, pb, [1] * n)
                   for _ev, re, g, pa, pb in prepared) / len(prepared)
    reps = []
    for draw in bootstrap_draws(n, iterations, seed):
        mult = [0] * n
        for e in draw:
            mult[e] += 1
        reps.append(sum(_delta_on_draw(re, g, pa, pb, mult)
                        for _ev, re, g, pa, pb in prepared) / len(prepared))
    ordered = sorted(reps)
    return {
        "observed": observed,
        "ci_low": _percentile(ordered, P.GATE_CI_ALPHA / 2),
        "ci_high": _percentile(ordered, 1 - P.GATE_CI_ALPHA / 2),
        "iterations": iterations,
        "seed": seed,
        "n_events_per_reader": n,
        "n_readers": len(prepared),
        "unit": P.BOOTSTRAP_UNIT,
        "reps": reps,
    }
