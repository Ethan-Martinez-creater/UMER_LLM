"""Event-level paired bootstrap and shared metric primitives (plan §24).

All intervals are event-level paired bootstraps with 10000 iterations and seed
7319. Resampling is done over **events**, and everything belonging to a sampled
event (every cutoff, every intervention) moves together; multiplicity is
preserved by repeating the event's payload once per draw.
"""
from __future__ import annotations

import math
import random

from ..config.pilot_config import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED


def macro_f1_from_counts(counts, labels=(0, 1)) -> float:
    """Macro-F1 from ``{(gold, pred): n}`` (safe for missing classes)."""
    f1s = []
    for label in labels:
        tp = counts.get((label, label), 0)
        fp = sum(counts.get((g, label), 0) for g in labels if g != label)
        fn = sum(counts.get((label, p), 0) for p in labels if p != label)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def class_f1_from_counts(counts, label) -> float:
    tp = counts.get((label, label), 0)
    fp = sum(n for (g, p), n in counts.items() if g != label and p == label)
    fn = sum(n for (g, p), n in counts.items() if g == label and p != label)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def accuracy_from_counts(counts) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    correct = sum(n for (g, p), n in counts.items() if g == p)
    return correct / total


def auroc(scores, positive: bool) -> float:
    """Rank-based AUROC of ``scores`` against binary ``positive`` flags."""
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, p in zip(ranks, positive) if p)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rank(xs, ys) -> float:
    """Spearman rank correlation; ``nan`` when undefined (n<2 or constant)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _rankdata(list(xs)), _rankdata(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def median(values) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci(samples, alpha: float = 0.05):
    """Two-sided percentile CI from a list of bootstrap replicates."""
    ordered = sorted(samples)
    return percentile(ordered, alpha / 2), percentile(ordered, 1 - alpha / 2)


def paired_event_bootstrap(per_event_values, statistic, iterations=None,
                           seed=None):
    """Bootstrap a statistic over events, preserving multiplicity (plan §24).

    ``per_event_values`` maps event_id -> payload; ``statistic`` receives the
    resampled list of payloads (with repeats) and returns a float.
    """
    iterations = iterations or BOOTSTRAP_ITERATIONS
    seed = BOOTSTRAP_SEED if seed is None else seed
    event_ids = sorted(per_event_values)
    payloads = [per_event_values[e] for e in event_ids]
    rng = random.Random(seed)
    observed = statistic(payloads)
    reps = []
    n = len(event_ids)
    for _ in range(iterations):
        draw = [payloads[rng.randrange(n)] for _ in range(n)]
        reps.append(statistic(draw))
    low, high = bootstrap_ci(reps)
    return {"observed": observed, "ci_low": low, "ci_high": high,
            "iterations": iterations, "seed": seed,
            "n_events": n, "samples": reps}


def paired_diff_bootstrap(per_event_a, per_event_b, statistic, iterations=None,
                          seed=None):
    """Paired difference ``A - B`` bootstrapped over shared events (plan §24)."""
    shared = sorted(set(per_event_a) & set(per_event_b))
    if not shared:
        return {"observed": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": 0, "seed": seed,
                "n_events": 0, "samples": []}
    iterations = iterations or BOOTSTRAP_ITERATIONS
    seed = BOOTSTRAP_SEED if seed is None else seed
    payloads = [(per_event_a[e], per_event_b[e]) for e in shared]
    rng = random.Random(seed)
    observed = statistic(payloads)
    reps = []
    for _ in range(iterations):
        draw = [payloads[rng.randrange(len(shared))]
                for _ in range(len(shared))]
        reps.append(statistic(draw))
    low, high = bootstrap_ci(reps)
    return {"observed": observed, "ci_low": low, "ci_high": high,
            "iterations": iterations, "seed": seed,
            "n_events": len(shared), "samples": reps}
