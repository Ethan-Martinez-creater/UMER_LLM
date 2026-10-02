"""M1-F signed-utility metrics (pure functions; import-safe without torch).

Covers the metric surface the M1-F plan fixes in advance:

* Task C — overall signed utility: Macro-F1, per-class F1, balanced accuracy,
  continuous-utility Spearman/MAE, HARMFUL/HELPFUL AUPRC;
* Task D — active HELPFUL/HARMFUL audit on gold-active rows only, where a
  NEUTRAL prediction on an active row is an error and is never dropped;
* Task E — within-snapshot evidence discrimination (per-snapshot Spearman,
  pairwise order concordance, snapshot-centred pooled Spearman);
* Task F — fixed-threshold diagnostics at ±0.03/±0.05/±0.07 plus activity,
  conditional sign and near-boundary frequency.

Statistics are always event-level: every bootstrap resamples whole events
(``(event, cutoff)`` clusters), never independent atomic rows.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from ..evaluation import utility_metrics as um

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.evaluation.bootstrap import spearman_rank
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser.evaluation.bootstrap read-only") from exc


class SignedMetricRefused(RuntimeError):
    """Raised when a metric input violates the M1-F contract."""


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def variance(values) -> float:
    values = [float(v) for v in values]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def mean(values) -> float:
    values = [float(v) for v in values]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def percentile(sorted_values, q: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def spearman(a, b) -> float:
    """Spearman rho that returns ``nan`` instead of raising on degenerate input."""
    a, b = [float(v) for v in a], [float(v) for v in b]
    if len(a) != len(b):
        raise SignedMetricRefused("spearman length mismatch")
    if len(a) < 2 or variance(a) <= 0.0 or variance(b) <= 0.0:
        return float("nan")
    value = spearman_rank(a, b)
    return float(value)


# --------------------------------------------------------------------------
# Task C — overall signed-utility metrics
# --------------------------------------------------------------------------
def balanced_accuracy(gold_signs, pred_signs, classes=P.SIGN_CLASSES) -> float:
    """Macro-averaged recall over the classes present in ``gold_signs``."""
    gold_signs, pred_signs = list(gold_signs), list(pred_signs)
    if len(gold_signs) != len(pred_signs):
        raise SignedMetricRefused("gold/pred length mismatch")
    recalls = []
    for c in classes:
        n = sum(1 for g in gold_signs if g == c)
        if n == 0:
            continue
        hits = sum(1 for g, p in zip(gold_signs, pred_signs)
                   if g == c and p == c)
        recalls.append(hits / n)
    if not recalls:
        return float("nan")
    return sum(recalls) / len(recalls)


def per_class_f1(gold_signs, pred_signs, classes=P.SIGN_CLASSES) -> dict:
    return {c: um.class_f1(gold_signs, pred_signs, c) for c in classes}


def auprc_for(sign_probs, gold_signs, target: str) -> float:
    idx = um.SIGN_TO_INDEX[target]
    return um.auprc([p[idx] for p in sign_probs],
                    [s == target for s in gold_signs])


def overall_metrics(rows, prediction: dict) -> dict:
    """Task C metric set of one prediction on one set of rows."""
    if len(rows) != len(prediction["sign_probs"]):
        raise SignedMetricRefused("rows/prediction length mismatch")
    gold = [r["sign"] for r in rows]
    preds = um.predicted_signs(prediction)
    utility = [float(r["utility"]) for r in rows]
    return {
        "n": len(rows),
        "macro_f1": um.macro_f1(gold, preds),
        "per_class_f1": per_class_f1(gold, preds),
        "balanced_accuracy": balanced_accuracy(gold, preds),
        "utility_spearman": spearman(utility, prediction["utility"]),
        "mae": um.mean_absolute_error(utility, prediction["utility"]),
        "harmful_auprc": auprc_for(prediction["sign_probs"], gold, "HARMFUL"),
        "helpful_auprc": auprc_for(prediction["sign_probs"], gold, "HELPFUL"),
    }


# --------------------------------------------------------------------------
# Task D — active (signed) audit
# --------------------------------------------------------------------------
def active_mask(rows, classes=P.M1F_ACTIVE_CLASSES) -> list:
    return [r["sign"] in classes for r in rows]


def take_prediction(prediction: dict, indices) -> dict:
    """Sub-select a prediction by row indices (order preserved)."""
    indices = list(indices)
    return {"utility": [prediction["utility"][i] for i in indices],
            "sign_probs": [prediction["sign_probs"][i] for i in indices]}


def subset_prediction(prediction: dict, mask) -> dict:
    return take_prediction(prediction, [i for i, keep in enumerate(mask)
                                        if keep])


def active_metrics(rows, prediction: dict,
                   classes=P.M1F_ACTIVE_CLASSES) -> dict:
    """Task D: metrics on gold-active rows, NEUTRAL predictions kept as errors."""
    mask = active_mask(rows, classes)
    n_active = sum(mask)
    base = {"n_rows": len(rows), "n_active": n_active,
            "active_prevalence": (n_active / len(rows)) if rows else
            float("nan")}
    if n_active == 0:
        return {**base, "defined": False}
    active_rows = [r for r, keep in zip(rows, mask) if keep]
    active_pred = subset_prediction(prediction, mask)
    gold = [r["sign"] for r in active_rows]
    preds = um.predicted_signs(active_pred)
    neutral = sum(1 for p in preds if p == "NEUTRAL")
    return {
        **base,
        "defined": True,
        "helpful_vs_harmful_macro_f1": um.macro_f1(gold, preds, classes),
        "balanced_accuracy": balanced_accuracy(gold, preds, classes),
        "per_class_f1": per_class_f1(gold, preds, classes),
        "harmful_auprc": auprc_for(active_pred["sign_probs"], gold, "HARMFUL"),
        "helpful_auprc": auprc_for(active_pred["sign_probs"], gold, "HELPFUL"),
        "neutral_prediction_rate": neutral / len(preds),
        "n_neutral_predictions": neutral,
        "n_gold_helpful": sum(1 for g in gold if g == "HELPFUL"),
        "n_gold_harmful": sum(1 for g in gold if g == "HARMFUL"),
    }


def conditional_sign_matrix(rows, prediction: dict,
                            classes=P.SIGN_CLASSES) -> dict:
    """``P(predicted class | gold class)`` — the Task F conditional-sign view."""
    gold = [r["sign"] for r in rows]
    preds = um.predicted_signs(prediction)
    matrix = {}
    for g in classes:
        idx = [i for i, s in enumerate(gold) if s == g]
        if not idx:
            matrix[g] = {"n": 0}
            continue
        row = {p: sum(1 for i in idx if preds[i] == p) / len(idx)
               for p in classes}
        matrix[g] = {"n": len(idx), **{f"p_pred_{k}": v
                                       for k, v in row.items()}}
    return matrix


# --------------------------------------------------------------------------
# Task F — fixed-threshold diagnostics
# --------------------------------------------------------------------------
def threshold_signs(utilities, threshold: float) -> list:
    """Map continuous predicted utility to signs at a **fixed** threshold."""
    out = []
    for value in utilities:
        value = float(value)
        if value >= threshold:
            out.append("HELPFUL")
        elif value <= -threshold:
            out.append("HARMFUL")
        else:
            out.append("NEUTRAL")
    return out


def activity_rate(signs) -> float:
    """Fraction of rows whose sign is active (HELPFUL or HARMFUL)."""
    signs = list(signs)
    if not signs:
        return float("nan")
    return sum(1 for s in signs if s in P.M1F_ACTIVE_CLASSES) / len(signs)


def near_boundary_rate(utilities, threshold: float = P.M1F_OFFICIAL_THRESHOLD,
                       band: float = P.M1F_NEAR_BOUNDARY_BAND) -> float:
    """Fraction of ``|u|`` strictly inside ``(threshold-band, threshold+band)``."""
    utilities = [abs(float(u)) for u in utilities]
    if not utilities:
        return float("nan")
    lo, hi = threshold - band, threshold + band
    return sum(1 for v in utilities if lo < v < hi) / len(utilities)


def threshold_diagnostics(rows, prediction: dict,
                          thresholds=P.M1F_THRESHOLDS) -> dict:
    """Diagnostic-only view at each fixed threshold; no threshold is selected."""
    gold = [r["sign"] for r in rows]
    out = {}
    for threshold in thresholds:
        preds = threshold_signs(prediction["utility"], threshold)
        out[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "macro_f1": um.macro_f1(gold, preds),
            "balanced_accuracy": balanced_accuracy(gold, preds),
            "per_class_f1": per_class_f1(gold, preds),
            "predicted_activity_rate": activity_rate(preds),
        }
    return out


# --------------------------------------------------------------------------
# Task E — within-snapshot evidence discrimination
# --------------------------------------------------------------------------
def snapshot_key(row) -> tuple:
    """The frozen snapshot identity: source-state features are constant here."""
    return (str(row["event_id"]), int(row["cutoff"]), str(row["reader"]))


def group_snapshots(rows) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(snapshot_key(row), []).append(row)
    return groups


def pairwise_concordance(gold, pred) -> tuple:
    """``(concordant, comparable)`` evidence pairs; gold ties are not comparable."""
    gold, pred = [float(v) for v in gold], [float(v) for v in pred]
    if len(gold) != len(pred):
        raise SignedMetricRefused("concordance length mismatch")
    concordant = comparable = 0
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            if gold[i] == gold[j]:
                continue
            comparable += 1
            if (gold[i] - gold[j]) * (pred[i] - pred[j]) > 0:
                concordant += 1
    return concordant, comparable


def within_snapshot_stats(rows, min_rows=P.M1F_WITHIN_MIN_ROWS,
                          variance_eps=P.M1F_WITHIN_VARIANCE_EPS,
                          pairwise=False) -> dict:
    """Per-snapshot Spearman/concordance plus the snapshot-centred pool.

    ``rows`` must carry ``utility`` (gold) and ``utility_pred`` (predicted).
    Snapshots with fewer than ``min_rows`` evidence rows or a zero-variance
    gold utility are excluded and counted, never silently dropped.
    """
    groups = group_snapshots(rows)
    skipped = {"too_few_rows": 0, "zero_variance": 0}
    per_snapshot = []
    centred_gold, centred_pred = [], []
    raw_gold, raw_pred = [], []
    concordant = comparable = 0
    rhos = []
    for key in sorted(groups):
        group = groups[key]
        gold = [float(r["utility"]) for r in group]
        pred = [float(r["utility_pred"]) for r in group]
        if len(group) < min_rows:
            skipped["too_few_rows"] += 1
            continue
        if variance(gold) <= variance_eps:
            skipped["zero_variance"] += 1
            continue
        rho = spearman(gold, pred)
        rhos.append(rho)
        entry = {"event_id": key[0], "cutoff": key[1], "reader": key[2],
                 "n": len(group), "gold_variance": variance(gold),
                 "spearman": rho}
        if pairwise:
            c, t = pairwise_concordance(gold, pred)
            entry["concordant"] = c
            entry["comparable"] = t
            entry["concordance"] = (c / t) if t else float("nan")
            concordant += c
            comparable += t
        per_snapshot.append(entry)
        mg, mp = mean(gold), mean(pred)
        centred_gold += [v - mg for v in gold]
        centred_pred += [v - mp for v in pred]
        raw_gold += gold
        raw_pred += pred

    finite_rhos = [r for r in rhos if math.isfinite(r)]
    out = {
        "n_snapshots_total": len(groups),
        "n_snapshots_used": len(per_snapshot),
        "n_rows_used": len(centred_gold),
        "skipped": skipped,
        "mean_within_snapshot_spearman": mean(finite_rhos),
        "n_finite_snapshot_spearman": len(finite_rhos),
        "raw_pooled_spearman": spearman(raw_gold, raw_pred),
        "centered_pooled_spearman": spearman(centred_gold, centred_pred),
        "centered_rows": len(centred_gold),
    }
    if pairwise:
        out["pairwise_concordance"] = (concordant / comparable) \
            if comparable else float("nan")
        out["pairwise_concordant"] = concordant
        out["pairwise_comparable"] = comparable
    out["per_snapshot"] = per_snapshot
    return out


def with_prediction_rows(rows, prediction: dict) -> list:
    """Attach the predicted utility to each row (copy; input untouched)."""
    if len(rows) != len(prediction["utility"]):
        raise SignedMetricRefused("rows/prediction length mismatch")
    return [{**row, "utility_pred": float(u)}
            for row, u in zip(rows, prediction["utility"])]


def event_clusters(rows) -> tuple:
    """``(events, {event: [row indices]})`` — the resampling unit is the event."""
    events = sorted({str(r["event_id"]) for r in rows})
    index = {e: [] for e in events}
    for i, row in enumerate(rows):
        index[str(row["event_id"])].append(i)
    return events, index


def _event_resample(rows, index, events, draw):
    take = []
    for event_idx in draw:
        take += index[events[event_idx]]
    return take


def clustered_event_bootstrap(rows, pred_a: dict, pred_b: dict,
                              metric_fn, iterations=P.BOOTSTRAP_ITERATIONS,
                              seed=P.BOOTSTRAP_SEED) -> dict:
    """Event-clustered paired bootstrap of ``metric_fn(A) - metric_fn(B)``.

    Whole events are resampled with the frozen draw sequence; every replicate
    scores A and B on the *same* resample, so the pairing is preserved.
    Non-finite replicates are counted and excluded from the interval.
    """
    from ..evaluation.reader_geometry import bootstrap_draws

    events, index = event_clusters(rows)
    n = len(events)
    if n == 0:
        raise SignedMetricRefused("no events to bootstrap")
    observed = float(metric_fn(rows, pred_a)) - float(metric_fn(rows, pred_b))
    reps, n_non_finite = [], 0
    for draw in bootstrap_draws(n, iterations, seed):
        take = _event_resample(rows, index, events, draw)
        sample = [rows[i] for i in take]
        value = float(metric_fn(sample, take_prediction(pred_a, take))) \
            - float(metric_fn(sample, take_prediction(pred_b, take)))
        if not math.isfinite(value):
            n_non_finite += 1
            continue
        reps.append(value)
    ordered = sorted(reps)
    return {
        "observed": observed,
        "ci_low": percentile(ordered, P.M1F_CI_ALPHA / 2),
        "ci_high": percentile(ordered, 1 - P.M1F_CI_ALPHA / 2),
        "iterations": iterations,
        "seed": seed,
        "n_events": n,
        "n_finite_replicates": len(reps),
        "n_non_finite_replicates": n_non_finite,
        "unit": P.BOOTSTRAP_UNIT,
        "reps": reps,
    }


def clustered_event_ci(rows, pred: dict, metric_fn,
                       iterations=P.BOOTSTRAP_ITERATIONS,
                       seed=P.BOOTSTRAP_SEED) -> dict:
    """Event-clustered bootstrap interval of a single prediction's metric."""
    from ..evaluation.reader_geometry import bootstrap_draws

    events, index = event_clusters(rows)
    n = len(events)
    if n == 0:
        raise SignedMetricRefused("no events to bootstrap")
    observed = float(metric_fn(rows, pred))
    reps, n_non_finite = [], 0
    for draw in bootstrap_draws(n, iterations, seed):
        take = _event_resample(rows, index, events, draw)
        sample = [rows[i] for i in take]
        value = float(metric_fn(sample, take_prediction(pred, take)))
        if not math.isfinite(value):
            n_non_finite += 1
            continue
        reps.append(value)
    ordered = sorted(reps)
    return {
        "observed": observed,
        "ci_low": percentile(ordered, P.M1F_CI_ALPHA / 2),
        "ci_high": percentile(ordered, 1 - P.M1F_CI_ALPHA / 2),
        "iterations": iterations,
        "seed": seed,
        "n_events": n,
        "n_finite_replicates": len(reps),
        "n_non_finite_replicates": n_non_finite,
        "unit": P.BOOTSTRAP_UNIT,
        "reps": reps,
    }
