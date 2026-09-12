"""Held-out reader evaluation and the P4 gate (plan §23, §25).

After checkpoints and evidence subsets are frozen, the held-out reader scores
every selection arm. The primary comparison is::

    Delta_r = MF1(S5) - MF1(best simple baseline among S1/S2)

P4 passes when the mean delta is ≥ +0.01, at least 2/3 rotations are positive,
no rotation is worse than -0.005, and S5 obeys the same maximum token target.
"""
from __future__ import annotations

from ..config.pilot_config import (P4_MEAN_DELTA_MIN,
                                   P4_ROTATIONS_POSITIVE_REQUIRED,
                                   P4_WORST_ROTATION_MIN, PRIMARY_ARM,
                                   SIMPLE_BASELINE_ARMS)
from .bootstrap import (accuracy_from_counts, class_f1_from_counts,
                        macro_f1_from_counts, mean, median)


def reader_metrics(golds, preds) -> dict:
    """Accuracy / Macro-F1 / per-class F1 / token stats (plan §23)."""
    counts = {}
    for g, p in zip(golds, preds):
        counts[(int(g), int(p))] = counts.get((int(g), int(p)), 0) + 1
    return {
        "accuracy": accuracy_from_counts(counts),
        "macro_f1": macro_f1_from_counts(counts, labels=(0, 1)),
        "rumor_f1": class_f1_from_counts(counts, 1),
        "non_rumor_f1": class_f1_from_counts(counts, 0),
        "n": len(golds),
    }


def selection_metrics(golds, preds, social_tokens) -> dict:
    """Reader metrics plus mean/median social tokens for one arm (§23)."""
    out = reader_metrics(golds, preds)
    out["mean_social_tokens"] = mean(social_tokens)
    out["median_social_tokens"] = median(social_tokens)
    return out


def best_simple_baseline(arm_metrics, simple_arms=SIMPLE_BASELINE_ARMS):
    """Strongest S1/S2 by Macro-F1 (plan §23 P4 primary comparison)."""
    available = {name: arm_metrics[name] for name in simple_arms
                 if name in arm_metrics}
    if not available:
        raise ValueError("no simple baseline arm available")
    name = max(available, key=lambda k: available[k]["macro_f1"])
    return name, available[name]


def rotation_delta(arm_metrics, token_target_ok: bool = True) -> dict:
    """One LORO rotation's Δ and its arm table."""
    name, baseline = best_simple_baseline(arm_metrics)
    primary = arm_metrics[PRIMARY_ARM]
    return {
        "primary_arm": PRIMARY_ARM,
        "primary_macro_f1": primary["macro_f1"],
        "best_simple_arm": name,
        "best_simple_macro_f1": baseline["macro_f1"],
        "delta": primary["macro_f1"] - baseline["macro_f1"],
        "primary_tokens": primary.get("mean_social_tokens"),
        "baseline_tokens": baseline.get("mean_social_tokens"),
        "token_target_ok": bool(token_target_ok),
    }


def gate_p4(rotations) -> dict:
    """Apply the four plan §25 P4 conditions across all rotations."""
    deltas = [r["delta"] for r in rotations]
    positive = sum(1 for d in deltas if d > 0)
    worst = min(deltas) if deltas else float("nan")
    mean_delta = mean(deltas)
    targets_ok = all(r.get("token_target_ok", True) for r in rotations)
    passed = (mean_delta == mean_delta and mean_delta >= P4_MEAN_DELTA_MIN
              and positive >= P4_ROTATIONS_POSITIVE_REQUIRED
              and worst >= P4_WORST_ROTATION_MIN and targets_ok)
    return {
        "gate": "P4_unseen_reader_transfer",
        "rotations": len(rotations),
        "deltas": deltas,
        "mean_delta": mean_delta,
        "mean_threshold": P4_MEAN_DELTA_MIN,
        "positive_rotations": positive,
        "positive_required": P4_ROTATIONS_POSITIVE_REQUIRED,
        "worst_delta": worst,
        "worst_threshold": P4_WORST_ROTATION_MIN,
        "token_targets_ok": bool(targets_ok),
        "pass": bool(passed),
    }


def pheme_secondary(rotations) -> dict:
    """PHEME secondary condition: mean Δ ≥ -0.005, never enough for FULL_GO."""
    deltas = [r["delta"] for r in rotations]
    value = mean(deltas)
    from ..config.pilot_config import PHEME_MEAN_DELTA_MIN
    return {
        "condition": "pheme_secondary",
        "mean_delta": value,
        "threshold": PHEME_MEAN_DELTA_MIN,
        "pass": bool(value == value and value >= PHEME_MEAN_DELTA_MIN),
        "note": "PHEME alone cannot produce FULL_GO (plan §25)",
    }


def final_decision(gates, pheme=None) -> dict:
    """Combine gates into the plan §26 recommendation (no auto redesign)."""
    required = ["P0", "P1", "P2", "P3", "P4"]
    values = {k: bool(gates.get(k, {}).get("pass")) for k in required}
    pheme_pass = True if pheme is None else bool(pheme.get("pass"))
    if all(values.values()) and pheme_pass:
        return {"recommendation": "START_FULL_CR_TSER_METHOD_DEVELOPMENT",
                "decision": "FULL_GO", "gates": values,
                "pheme_secondary": pheme_pass}
    core_failures = [k for k, ok in values.items() if not ok]
    if len(core_failures) <= 1 and values.get("P0"):
        return {"recommendation": "STOP_FOR_RESEARCH_REVIEW",
                "decision": "PARTIAL_GO", "gates": values,
                "failed_gates": core_failures,
                "pheme_secondary": pheme_pass}
    return {"recommendation": "STOP_CR_TSER", "decision": "NO_GO",
            "gates": values, "failed_gates": core_failures,
            "pheme_secondary": pheme_pass}
