"""Classification metrics (plan §42): Accuracy, Macro-F1, Weighted-F1,
Rumor-F1. Implemented without external deps so tests run anywhere.

INVALID_OUTPUT predictions (None) are excluded from metrics and reported
separately as the invalid rate.
"""
from __future__ import annotations


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def classification_metrics(gold, pred) -> dict:
    """``gold``: iterable of 0/1. ``pred``: iterable of 0/1/None."""
    pairs = [(g, p) for g, p in zip(gold, pred) if p is not None]
    invalid = len(gold) - len(pairs)
    n = max(len(pairs), 1)
    acc = sum(1 for g, p in pairs if g == p) / n

    tp1 = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp1 = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn1 = sum(1 for g, p in pairs if g == 1 and p == 0)
    tp0 = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp0 = sum(1 for g, p in pairs if g == 1 and p == 0)
    fn0 = sum(1 for g, p in pairs if g == 0 and p == 1)

    _, _, f1_rumor = _prf(tp1, fp1, fn1)
    _, _, f1_nonrumor = _prf(tp0, fp0, fn0)
    macro_f1 = (f1_rumor + f1_nonrumor) / 2
    support1 = tp1 + fn1
    support0 = tp0 + fn0
    weighted_f1 = (f1_rumor * support1 + f1_nonrumor * support0) / max(
        support1 + support0, 1)
    return {
        "n": len(gold),
        "n_scored": len(pairs),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rumor_f1": f1_rumor,
        "invalid_rate": invalid / max(len(gold), 1),
        "support_rumor": support1,
        "support_nonrumor": support0,
    }
