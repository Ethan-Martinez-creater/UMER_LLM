"""M1-E Task D — Ma-Weibo vs PHEME E3 distribution and association audit.

Reports, per reader and per cutoff, the frozen E3 features' location and
spread (mean / std / median / IQR) on both datasets, the fixed distribution
distance (1-D Wasserstein / earth-mover distance), and the association of
each E3 feature with the frozen continuous utility and the HELPFUL / HARMFUL
sign, separately for utility_train / utility_dev / utility_eval.

The utility_eval association is a POST-HOC DIAGNOSTIC: it is reported only
and is never used to retrain, re-select or re-tune anything. Everything here
is association, never causation.
"""
from __future__ import annotations

import bisect
import math

from ..config import protocol as P

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.evaluation.bootstrap import auroc, spearman_rank
    from cr_tser.intervention.evidence_units import evidence_key_parts
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser bootstrap/evidence units read-only") from exc


class ShiftAuditRefused(RuntimeError):
    """Raised when the distribution audit cannot run as specified."""


# --------------------------------------------------------------------------
# statistics primitives
# --------------------------------------------------------------------------
def summarise(values) -> dict:
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "iqr": None, "min": None, "max": None}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0

    def quantile(q):
        if n == 1:
            return values[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    q1, q3 = quantile(0.25), quantile(0.75)
    return {"n": n, "mean": mean, "std": var ** 0.5,
            "median": quantile(0.5), "iqr": q3 - q1,
            "q1": q1, "q3": q3, "min": values[0], "max": values[-1]}


def _cdf(sorted_values, x) -> float:
    return bisect.bisect_right(sorted_values, x) / len(sorted_values)


def wasserstein_1d(a, b) -> float:
    """Exact 1-D Wasserstein (earth-mover) distance of two samples.

    Integrates ``|F_a(x) - F_b(x)|`` over the merged value grid, which is the
    exact W1 of the two empirical distributions.
    """
    a = sorted(float(v) for v in a)
    b = sorted(float(v) for v in b)
    if not a or not b:
        raise ShiftAuditRefused("empty sample for Wasserstein distance")
    grid = sorted(set(a) | set(b))
    total = 0.0
    for lo, hi in zip(grid, grid[1:]):
        total += abs(_cdf(a, lo) - _cdf(b, lo)) * (hi - lo)
    return total


def auprc(scores, positives) -> float:
    from ..evaluation.utility_metrics import auprc as _auprc
    return _auprc(scores, positives)


# --------------------------------------------------------------------------
# E3 tables
# --------------------------------------------------------------------------
def e3_by_reader_cutoff(e3_rows) -> dict:
    """``{reader: {cutoff: {feature: [values]}}}``."""
    out = {}
    for row in e3_rows:
        parts = evidence_key_parts(row["key"])
        bucket = out.setdefault(row["reader"], {}).setdefault(
            parts["cutoff"], {name: [] for name in P.E3_FEATURE_NAMES})
        for name in P.E3_FEATURE_NAMES:
            bucket[name].append(float(row["e3"][name]))
    return out


def distribution_report(e3_by_dataset: dict) -> dict:
    """Per dataset x reader x cutoff x feature descriptive statistics."""
    report = {}
    for dataset, by_reader in e3_by_dataset.items():
        report[dataset] = {}
        for reader, by_cutoff in by_reader.items():
            report[dataset][reader] = {}
            for cutoff, features in by_cutoff.items():
                report[dataset][reader][str(cutoff)] = {
                    name: summarise(values)
                    for name, values in features.items()}
    return report


def shift_report(e3_by_dataset: dict, datasets=None) -> dict:
    """Fixed Wasserstein distance between the two datasets per cell."""
    datasets = tuple(datasets or P.DATASETS)
    if len(datasets) != 2:
        raise ShiftAuditRefused("the shift audit compares exactly two datasets")
    left, right = datasets
    if left not in e3_by_dataset or right not in e3_by_dataset:
        raise ShiftAuditRefused(f"missing dataset in {sorted(e3_by_dataset)}")
    out = {}
    for reader in P.READER_KEYS:
        out[reader] = {}
        for cutoff in P.CUTOFFS_MIN:
            cell = {}
            for name in P.E3_FEATURE_NAMES:
                a = (e3_by_dataset[left].get(reader, {})
                     .get(cutoff, {}).get(name))
                b = (e3_by_dataset[right].get(reader, {})
                     .get(cutoff, {}).get(name))
                if not a or not b:
                    raise ShiftAuditRefused(
                        f"{reader}/{cutoff}/{name}: empty sample")
                sa, sb = summarise(a), summarise(b)
                cell[name] = {
                    "wasserstein": wasserstein_1d(a, b),
                    "mean_left": sa["mean"], "mean_right": sb["mean"],
                    "mean_shift": sa["mean"] - sb["mean"],
                    "std_left": sa["std"], "std_right": sb["std"],
                    "median_left": sa["median"], "median_right": sb["median"],
                    "iqr_left": sa["iqr"], "iqr_right": sb["iqr"],
                }
            out[reader][str(cutoff)] = cell
    return {"datasets": [left, right], "metric": "wasserstein_1d",
            "per_reader_cutoff": out}


# --------------------------------------------------------------------------
# association with the frozen utility / sign
# --------------------------------------------------------------------------
def association_report(e3_rows, atomic_entries, split) -> dict:
    """Feature <-> utility/sign association per reader and per frozen split."""
    utility_by_key = {e["key"]: e for e in atomic_entries}
    e3_by_key = {}
    for row in e3_rows:
        e3_by_key.setdefault(row["key"], {})[row["reader"]] = row["e3"]
    folds = {name: set(str(x) for x in split[name])
             for name in ("utility_train", "utility_dev", "utility_eval")}
    out = {}
    for reader in P.READER_KEYS:
        out[reader] = {}
        for fold_name, events in folds.items():
            utilities, signs = [], []
            features = {name: [] for name in P.E3_FEATURE_NAMES}
            for key, entry in utility_by_key.items():
                if str(entry["event_id"]) not in events:
                    continue
                values = e3_by_key.get(key, {}).get(reader)
                if values is None:
                    continue
                utilities.append(float(entry["utility"][reader]))
                signs.append(entry["sign"][reader])
                for name in P.E3_FEATURE_NAMES:
                    features[name].append(float(values[name]))
            cell = {"n": len(utilities), "post_hoc": fold_name ==
                    "utility_eval"}
            for name in P.E3_FEATURE_NAMES:
                scores = features[name]
                cell[name] = {
                    "spearman_with_utility":
                        spearman_rank(scores, utilities)
                        if len(scores) > 1 else float("nan"),
                    "auroc_helpful": auroc(
                        scores, [s == "HELPFUL" for s in signs]),
                    "auroc_harmful": auroc(
                        scores, [s == "HARMFUL" for s in signs]),
                }
            out[reader][fold_name] = cell
    return out
