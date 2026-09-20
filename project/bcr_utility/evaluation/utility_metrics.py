"""M1 evaluation metrics (M1 plan §13).

Primary: HELPFUL / NEUTRAL / HARMFUL Macro-F1.
Secondary: utility Spearman, MAE, HARMFUL AUPRC, HARMFUL F1, HELPFUL AUPRC.

Pure functions over plain rows so LOCAL tests need no torch.
"""
from __future__ import annotations

from ..config import protocol as P

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.evaluation.bootstrap import spearman_rank
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser.evaluation.bootstrap read-only") from exc

SIGN_TO_INDEX = {name: i for i, name in enumerate(P.SIGN_CLASSES)}
INDEX_TO_SIGN = {i: name for name, i in SIGN_TO_INDEX.items()}


class MetricRefused(RuntimeError):
    """Raised when a metric input violates the evaluation contract."""


def macro_f1(gold_signs, pred_signs, classes=P.SIGN_CLASSES) -> float:
    gold_signs, pred_signs = list(gold_signs), list(pred_signs)
    if len(gold_signs) != len(pred_signs):
        raise MetricRefused("gold/pred length mismatch")
    if not gold_signs:
        return float("nan")
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gold_signs, pred_signs)
                 if g == c and p == c)
        fp = sum(1 for g, p in zip(gold_signs, pred_signs)
                 if g != c and p == c)
        fn = sum(1 for g, p in zip(gold_signs, pred_signs)
                 if g == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def class_f1(gold_signs, pred_signs, target: str) -> float:
    gold_signs, pred_signs = list(gold_signs), list(pred_signs)
    tp = sum(1 for g, p in zip(gold_signs, pred_signs)
             if g == target and p == target)
    fp = sum(1 for g, p in zip(gold_signs, pred_signs)
             if g != target and p == target)
    fn = sum(1 for g, p in zip(gold_signs, pred_signs)
             if g == target and p != target)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def auprc(scores, positives) -> float:
    """Average precision of ``scores`` against binary ``positives``."""
    scores, positives = list(scores), list(positives)
    if len(scores) != len(positives):
        raise MetricRefused("scores/positives length mismatch")
    n_pos = sum(1 for p in positives if p)
    if not scores or n_pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for i in order:
        if positives[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def mean_absolute_error(gold, pred) -> float:
    gold, pred = list(gold), list(pred)
    if len(gold) != len(pred):
        raise MetricRefused("gold/pred length mismatch")
    if not gold:
        return float("nan")
    return sum(abs(float(g) - float(p)) for g, p in zip(gold, pred)) / len(gold)


def predicted_signs(pred: dict) -> list:
    return [INDEX_TO_SIGN[max(range(len(P.SIGN_CLASSES)),
                              key=lambda c: probs[c])]
            for probs in pred["sign_probs"]]


def evaluate_predictions(gold_rows, pred: dict) -> dict:
    """The full M1 metric set of one prediction on one eval fold."""
    gold_signs = [r["sign"] for r in gold_rows]
    gold_utils = [float(r["utility"]) for r in gold_rows]
    pred_signs = predicted_signs(pred)
    harmful_idx = SIGN_TO_INDEX["HARMFUL"]
    helpful_idx = SIGN_TO_INDEX["HELPFUL"]
    return {
        "n": len(gold_rows),
        "macro_f1": macro_f1(gold_signs, pred_signs),
        "utility_spearman": spearman_rank(gold_utils, pred["utility"]),
        "mae": mean_absolute_error(gold_utils, pred["utility"]),
        "harmful_auprc": auprc([p[harmful_idx] for p in pred["sign_probs"]],
                               [s == "HARMFUL" for s in gold_signs]),
        "harmful_f1": class_f1(gold_signs, pred_signs, "HARMFUL"),
        "helpful_auprc": auprc([p[helpful_idx] for p in pred["sign_probs"]],
                               [s == "HELPFUL" for s in gold_signs]),
    }


def disagreement_mask(eval_rows, training_signs_by_key) -> list:
    """Mark eval rows whose *training-reader* signs disagree (M1 plan §13).

    ``training_signs_by_key`` maps ``key -> {training_reader: sign}``; a row
    is in the disagreement subset when the training readers' signs at that
    key are not all equal. Held-out labels are never consulted here.
    """
    mask = []
    for row in eval_rows:
        signs = (training_signs_by_key.get(row["key"]) or {})
        values = set(signs.values())
        mask.append(len(values) > 1 if values else False)
    return mask


def subset(rows, mask) -> list:
    return [r for r, keep in zip(rows, mask) if keep]


def subset_prediction(pred: dict, mask) -> dict:
    return {"utility": [u for u, keep in zip(pred["utility"], mask) if keep],
            "sign_probs": [p for p, keep in zip(pred["sign_probs"], mask)
                           if keep]}
