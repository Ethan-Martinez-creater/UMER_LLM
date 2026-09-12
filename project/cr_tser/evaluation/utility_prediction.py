"""Utility prediction metrics (plan §17, §25 P3).

Reported for B0 / B1 / B2 (PHEME) / B3::

    Macro-F1, per-class F1, AUROC helpful-vs-rest, AUROC harmful-vs-rest,
    Spearman (continuous utility), MAE, active-intervention sign accuracy

The P3 gate compares B3 with the strongest of B0/B1 on Macro-F1 and requires
the B3 continuous-utility Spearman not to be worse than that baseline's.
"""
from __future__ import annotations

from ..config.pilot_config import P3_MACRO_F1_DELTA_MIN, SIGN_CLASSES
from .bootstrap import (accuracy_from_counts, auroc, class_f1_from_counts,
                        macro_f1_from_counts, mean, spearman_rank)

POSITIVE = "HELPFUL"
NEGATIVE = "HARMFUL"


def sign_accuracy(y_true, y_pred, active_flags=None) -> float:
    """Sign accuracy, restricted to active interventions when flags given."""
    correct = total = 0
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if active_flags is not None and not active_flags[i]:
            continue
        total += 1
        if t == p:
            correct += 1
    return (correct / total) if total else float("nan")


def evaluate_utility(y_true_sign, y_pred_sign, y_true_cont, y_pred_cont,
                     helpful_scores=None, harmful_scores=None,
                     active_flags=None) -> dict:
    """The full plan §17 report surface for one predictor.

    ``helpful_scores`` / ``harmful_scores`` are the predictor's continuous
    class scores (e.g. softmax over the sign head); hard labels are not used
    as AUROC scores because they collapse the ranking.
    """
    counts = {}
    for t, p in zip(y_true_sign, y_pred_sign):
        counts[(t, p)] = counts.get((t, p), 0) + 1
    if helpful_scores is None:
        helpful_scores = [1.0 if p == POSITIVE else 0.0 for p in y_pred_sign]
    if harmful_scores is None:
        harmful_scores = [-1.0 if p == NEGATIVE else 0.0 for p in y_pred_sign]
    return {
        "macro_f1": macro_f1_from_counts(counts, labels=SIGN_CLASSES),
        "per_class_f1": {c: class_f1_from_counts(counts, c)
                         for c in SIGN_CLASSES},
        "auroc_helpful_vs_rest": auroc(
            list(helpful_scores), [t == POSITIVE for t in y_true_sign]),
        "auroc_harmful_vs_rest": auroc(
            list(harmful_scores), [t == NEGATIVE for t in y_true_sign]),
        "spearman": spearman_rank(y_true_cont, y_pred_cont),
        "mae": mean([abs(a - b) for a, b in zip(y_true_cont, y_pred_cont)]),
        "sign_accuracy": sign_accuracy(y_true_sign, y_pred_sign, active_flags),
        "n": len(y_true_sign),
    }


def gate_p3(b3: dict, baselines: dict) -> dict:
    """P3: B3 Macro-F1 ≥ strongest B0/B1 + 0.02 and Spearman not worse (§25)."""
    candidates = {name: m for name, m in baselines.items() if m is not None}
    if not candidates:
        return {"gate": "P3_structural_utility_increment", "pass": False,
                "reason": "no baseline available"}
    best_name, best = max(candidates.items(), key=lambda kv: kv[1]["macro_f1"])
    delta = b3["macro_f1"] - best["macro_f1"]
    spearman_ok = not (b3["spearman"] == b3["spearman"]
                       and best["spearman"] == best["spearman"]) or \
        b3["spearman"] >= best["spearman"]
    passed = delta >= P3_MACRO_F1_DELTA_MIN and spearman_ok
    return {
        "gate": "P3_structural_utility_increment",
        "strongest_baseline": best_name,
        "b3_macro_f1": b3["macro_f1"],
        "baseline_macro_f1": best["macro_f1"],
        "macro_f1_delta": delta,
        "threshold": P3_MACRO_F1_DELTA_MIN,
        "b3_spearman": b3["spearman"],
        "baseline_spearman": best["spearman"],
        "spearman_not_worse": bool(spearman_ok),
        "pass": bool(passed),
    }


def macro_f1_delta_bootstrap(per_event_b3, per_event_base, iterations=None,
                             seed=None):
    """Event-level paired bootstrap of the P3 Macro-F1 difference."""
    from .bootstrap import paired_diff_bootstrap

    def statistic(pairs):
        counts_b3, counts_base = {}, {}
        for b3_counts, base_counts in pairs:
            for key, n in b3_counts.items():
                counts_b3[key] = counts_b3.get(key, 0) + n
            for key, n in base_counts.items():
                counts_base[key] = counts_base.get(key, 0) + n
        return (macro_f1_from_counts(counts_b3, labels=SIGN_CLASSES)
                - macro_f1_from_counts(counts_base, labels=SIGN_CLASSES))

    return paired_diff_bootstrap(per_event_b3, per_event_base, statistic,
                                 iterations=iterations, seed=seed)
