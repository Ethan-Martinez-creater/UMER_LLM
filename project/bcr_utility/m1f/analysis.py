"""M1-F Tasks C–G — signed-utility analysis, evidence discrimination, diagnostics.

Task C  overall signed-utility metrics and the primary paired comparison;
Task D  active HELPFUL/HARMFUL audit (NEUTRAL predictions on active rows are
        errors and stay in the data);
Task E  within-snapshot evidence discrimination (source-state features are
        constant inside ``(event, cutoff, reader)``, so this is the only test
        that can show evidence-level signed discrimination);
Task F  fixed ±0.03/±0.05/±0.07 diagnostics plus activity, conditional sign and
        near-boundary frequency — no threshold is ever selected;
Task G  the same view on PHEME, flagged ``diagnostic_only``.

Every prediction is first aligned to the **same row order** as the reference
variant, and the gold sign/utility is re-checked during alignment, so a paired
bootstrap can never compare mismatched rows.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from ..evaluation import bootstrap as boot
from ..evaluation import utility_metrics as um
from . import metrics as M


class AnalysisRefused(RuntimeError):
    """Raised when the M1-F analysis contract is violated."""


# --------------------------------------------------------------------------
# alignment and pooling
# --------------------------------------------------------------------------
def align_rotations(results: dict, order_variant: str = P.M1F_S0) -> dict:
    """``{held: {variant: {"rows", "prediction"}}}`` on one shared row order."""
    if order_variant not in results:
        raise AnalysisRefused(f"reference variant {order_variant} missing")
    aligned = {}
    for held, base_rot in results[order_variant]["rotations"].items():
        base_rows = base_rot["eval_rows"]
        base_keys = {r["key"] for r in base_rows}
        entry = {}
        for variant, result in results.items():
            rot = result["rotations"].get(held)
            if rot is None:
                raise AnalysisRefused(f"{variant}/{held}: rotation missing")
            rows = rot["eval_rows"]
            if {r["key"] for r in rows} != base_keys:
                raise AnalysisRefused(f"{variant}/{held}: evidence key set "
                                      "differs from the reference variant")
            index = {r["key"]: i for i, r in enumerate(rows)}
            take = [index[r["key"]] for r in base_rows]
            for pos, base_row in zip(take, base_rows):
                other = rows[pos]
                if other["sign"] != base_row["sign"] or \
                        abs(float(other["utility"])
                            - float(base_row["utility"])) > 1e-12:
                    raise AnalysisRefused(
                        f"{variant}/{held}/{base_row['key']}: gold drift "
                        "against the reference variant")
            entry[variant] = {"rows": base_rows,
                              "prediction": M.take_prediction(
                                  rot["prediction"], take)}
        if set(entry) != set(P.M1F_VARIANTS):
            raise AnalysisRefused(f"{held}: variants {sorted(entry)}")
        aligned[held] = entry
    return aligned


def pool(aligned: dict, variant: str) -> tuple:
    """Concatenate one variant's three rotations into one evaluation set."""
    rows, utility, probs = [], [], []
    for held in sorted(aligned):
        entry = aligned[held][variant]
        rows += entry["rows"]
        utility += [float(u) for u in entry["prediction"]["utility"]]
        probs += entry["prediction"]["sign_probs"]
    return rows, {"utility": utility, "sign_probs": probs}


def rotation_pairs(aligned: dict, variant: str) -> list:
    return [(aligned[held][variant]["rows"],
             aligned[held][variant]["prediction"]["sign_probs"])
            for held in sorted(aligned)]


# --------------------------------------------------------------------------
# metric callables (event-clustered bootstrap)
# --------------------------------------------------------------------------
def _gold(rows):
    return [r["sign"] for r in rows]


def macro_f1_metric(rows, pred) -> float:
    return um.macro_f1(_gold(rows), um.predicted_signs(pred))


def balanced_accuracy_metric(rows, pred) -> float:
    return M.balanced_accuracy(_gold(rows), um.predicted_signs(pred))


def active_macro_f1_metric(rows, pred) -> float:
    return M.active_metrics(rows, pred).get("helpful_vs_harmful_macro_f1",
                                            float("nan"))


def harmful_auprc_metric(rows, pred) -> float:
    return M.auprc_for(pred["sign_probs"], _gold(rows), "HARMFUL")


def helpful_auprc_metric(rows, pred) -> float:
    return M.auprc_for(pred["sign_probs"], _gold(rows), "HELPFUL")


def utility_spearman_metric(rows, pred) -> float:
    return M.spearman([r["utility"] for r in rows], pred["utility"])


def centered_spearman_metric(rows, pred) -> float:
    stats = M.within_snapshot_stats(M.with_prediction_rows(rows, pred),
                                    pairwise=False)
    return stats["centered_pooled_spearman"]


def centered_spearman_or_zero_metric(rows, pred) -> float:
    """Centered Spearman with an explicit zero base.

    ``S2`` sees only source-state features, which are constant inside
    ``(event, cutoff, reader)``, so its snapshot-centred prediction has no
    variance at all and the centred Spearman is undefined. That is the
    *defined* semantics of the S2 reference — no evidence-level discrimination
    — so an undefined value is scored as 0 and the fact is recorded next to the
    comparison instead of silently propagating a NaN.
    """
    value = centered_spearman_metric(rows, pred)
    return 0.0 if not math.isfinite(value) else float(value)


def class_f1_metric(target: str):
    def metric(rows, pred) -> float:
        return um.class_f1(_gold(rows), um.predicted_signs(pred), target)
    return metric


METRIC_FUNCTIONS = {
    "macro_f1": macro_f1_metric,
    "balanced_accuracy": balanced_accuracy_metric,
    "active_macro_f1": active_macro_f1_metric,
    "harmful_auprc": harmful_auprc_metric,
    "helpful_auprc": helpful_auprc_metric,
    "utility_spearman": utility_spearman_metric,
    "centered_spearman": centered_spearman_metric,
    "centered_spearman_zero_base": centered_spearman_or_zero_metric,
    "harmful_f1": class_f1_metric("HARMFUL"),
    "helpful_f1": class_f1_metric("HELPFUL"),
    "neutral_f1": class_f1_metric("NEUTRAL"),
}


def delta_ci(rows, pred_a, pred_b, metric_name,
             iterations=P.BOOTSTRAP_ITERATIONS,
             seed=P.BOOTSTRAP_SEED) -> dict:
    """Event-clustered paired bootstrap of one metric difference."""
    metric_fn = METRIC_FUNCTIONS[metric_name]
    out = M.clustered_event_bootstrap(rows, pred_a, pred_b, metric_fn,
                                      iterations=iterations, seed=seed)
    out.pop("reps", None)
    out["metric"] = metric_name
    return out


# --------------------------------------------------------------------------
# Task C / D / F — one variant's full metric bundle
# --------------------------------------------------------------------------
def variant_bundle(aligned: dict, variant: str) -> dict:
    rows, pred = pool(aligned, variant)
    per_reader = {}
    for held in sorted(aligned):
        entry = aligned[held][variant]
        per_reader[held] = {
            "n": len(entry["rows"]),
            "macro_f1": um.macro_f1(_gold(entry["rows"]),
                                    um.predicted_signs(entry["prediction"])),
        }
    return {
        "variant": variant,
        "overall": M.overall_metrics(rows, pred),
        "active": M.active_metrics(rows, pred),
        "per_reader_macro_f1": per_reader,
        "threshold_diagnostics": M.threshold_diagnostics(rows, pred),
        "conditional_sign": M.conditional_sign_matrix(rows, pred),
        "activity_rate_gold": M.activity_rate(_gold(rows)),
        "activity_rate_predicted": M.activity_rate(um.predicted_signs(pred)),
        "near_boundary_predicted": M.near_boundary_rate(pred["utility"]),
        "near_boundary_gold": M.near_boundary_rate([r["utility"] for r in rows]),
    }


# --------------------------------------------------------------------------
# Task C — the primary paired comparison
# --------------------------------------------------------------------------
def primary_comparison(aligned: dict, primary: str, comparator: str,
                       iterations=P.BOOTSTRAP_ITERATIONS,
                       seed=P.BOOTSTRAP_SEED) -> dict:
    """``primary - comparator``: frozen per-reader Macro-F1 bootstrap first."""
    per_reader = {}
    payloads = []
    for held in sorted(aligned):
        p = aligned[held][primary]
        c = aligned[held][comparator]
        record = boot.paired_event_delta(p["rows"],
                                        p["prediction"]["sign_probs"],
                                        c["prediction"]["sign_probs"],
                                        iterations=iterations, seed=seed)
        record.pop("reps", None)
        per_reader[held] = record
        payloads.append((p["rows"], p["prediction"]["sign_probs"],
                         c["prediction"]["sign_probs"]))
    aggregate = boot.aggregate_delta(payloads, iterations=iterations, seed=seed)
    aggregate.pop("reps", None)
    deltas = {held: rec["observed"] for held, rec in per_reader.items()}
    rows, pred_primary = pool(aligned, primary)
    _rows, pred_comparator = pool(aligned, comparator)
    secondary = {}
    for name in ("balanced_accuracy", "active_macro_f1", "harmful_auprc",
                 "helpful_auprc", "utility_spearman", "centered_spearman",
                 "harmful_f1", "helpful_f1", "neutral_f1"):
        secondary[name] = delta_ci(rows, pred_primary, pred_comparator, name,
                                   iterations=iterations, seed=seed)
    return {
        "primary": primary,
        "comparator": comparator,
        "per_reader_macro_f1_delta": per_reader,
        "aggregate": {
            "mean_delta_macro_f1": aggregate["observed"],
            "ci_low": aggregate["ci_low"],
            "ci_high": aggregate["ci_high"],
            "iterations": aggregate["iterations"],
            "seed": aggregate["seed"],
            "n_events_per_reader": aggregate["n_events_per_reader"],
            "positive_readers": sum(1 for d in deltas.values() if d > 0),
            "worst_reader_delta": min(deltas.values()),
        },
        "secondary_metric_deltas": secondary,
    }


# --------------------------------------------------------------------------
# Task E — within-snapshot evidence discrimination
# --------------------------------------------------------------------------
def within_snapshot_variant(aligned: dict, variant: str,
                            iterations=P.BOOTSTRAP_ITERATIONS,
                            seed=P.BOOTSTRAP_SEED) -> dict:
    rows, pred = pool(aligned, variant)
    stats = M.within_snapshot_stats(M.with_prediction_rows(rows, pred),
                                    pairwise=True)
    ci = M.clustered_event_ci(rows, pred, centered_spearman_metric,
                              iterations=iterations, seed=seed)
    ci.pop("reps", None)
    out = dict(stats)
    out["variant"] = variant
    out["centered_spearman_ci"] = {
        "observed": stats["centered_pooled_spearman"],
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "iterations": ci["iterations"], "seed": ci["seed"],
        "n_events": ci["n_events"],
    }
    return out


def within_snapshot_comparison(aligned: dict, primary: str, reference: str,
                               iterations=P.BOOTSTRAP_ITERATIONS,
                               seed=P.BOOTSTRAP_SEED) -> dict:
    """Centered-Spearman increment of ``primary`` over the S2 reference.

    The reference's centred Spearman is undefined by construction (constant
    inside a snapshot) and is therefore scored as an explicit zero base; the
    raw values are reported alongside so the substitution is auditable.
    """
    rows, pred_primary = pool(aligned, primary)
    _rows, pred_reference = pool(aligned, reference)
    primary_value = centered_spearman_metric(rows, pred_primary)
    reference_value = centered_spearman_metric(rows, pred_reference)
    out = delta_ci(rows, pred_primary, pred_reference,
                   "centered_spearman_zero_base",
                   iterations=iterations, seed=seed)
    out["primary"] = primary
    out["reference"] = reference
    out["primary_centered_spearman"] = primary_value
    out["reference_centered_spearman"] = reference_value
    out["reference_is_undefined_treated_as_zero"] = \
        not math.isfinite(reference_value)
    out["base_rule"] = "the S2 reference has no within-snapshot variance, so " \
                       "its centred Spearman is scored as an explicit 0"
    return out


def within_snapshot_audit(aligned: dict, dataset: str,
                          iterations=P.BOOTSTRAP_ITERATIONS,
                          seed=P.BOOTSTRAP_SEED) -> dict:
    variants = {}
    for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
        variants[variant] = within_snapshot_variant(aligned, variant,
                                                    iterations, seed)
    comparisons = {}
    for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
        if variant == P.M1F_S2:
            continue
        comparisons[variant] = within_snapshot_comparison(
            aligned, variant, P.M1F_S2, iterations, seed)
    best = max(comparisons, key=lambda v: comparisons[v]["observed"]) \
        if comparisons else None
    return {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f_within_snapshot",
        "dataset": dataset,
        "scope": "evidence_level_discrimination",
        "min_rows": P.M1F_WITHIN_MIN_ROWS,
        "variance_eps": P.M1F_WITHIN_VARIANCE_EPS,
        "variants": variants,
        "comparisons_vs_S2": comparisons,
        "largest_centered_increment": best,
        "reference_variant": P.M1F_S2,
    }


# --------------------------------------------------------------------------
# M1-F decision (plan §12)
# --------------------------------------------------------------------------
def gain_only_neutral(comparison: dict) -> dict:
    """Is the Macro-F1 gain explainable purely by predicting NEUTRAL?"""
    secondary = comparison["secondary_metric_deltas"]
    active = secondary["active_macro_f1"]["observed"]
    harmful = secondary["harmful_f1"]["observed"]
    helpful = secondary["helpful_f1"]["observed"]
    only_neutral = (not math.isfinite(active) or active <= 0.0) and \
        (not math.isfinite(harmful) or harmful <= 0.0) and \
        (not math.isfinite(helpful) or helpful <= 0.0)
    return {
        "gain_only_neutral": bool(only_neutral),
        "active_macro_f1_delta": active,
        "harmful_f1_delta": harmful,
        "helpful_f1_delta": helpful,
        "rule": "gain counts as NEUTRAL-only when the active Macro-F1 "
                "difference and both active-class F1 differences are all "
                "non-positive",
    }


def decide(comparison: dict, within: dict, validity_blocked: bool,
           active_improves: bool, harmful_auprc_not_decreased: bool) -> dict:
    """The seven pre-registered conditions of plan §12, all on Ma-Weibo."""
    agg = comparison["aggregate"]
    neutral = gain_only_neutral(comparison)
    within_delta = within["comparisons_vs_S2"].get(P.M1F_S4) or \
        within["comparisons_vs_S2"][P.M1F_S3]
    checks = {
        "mean_delta_gte_0.02": agg["mean_delta_macro_f1"]
        >= P.M1F_MEAN_DELTA_MIN,
        "ci_low_gt_0": agg["ci_low"] > 0.0,
        "positive_readers_gte_2": agg["positive_readers"]
        >= P.M1F_POSITIVE_READERS_MIN,
        "active_macro_f1_improves": bool(active_improves),
        "harmful_auprc_not_decreased": bool(harmful_auprc_not_decreased),
        "within_snapshot_centered_spearman_improves_over_S2":
            within_delta["observed"] > 0.0,
        "gain_not_only_neutral": not neutral["gain_only_neutral"],
    }
    if validity_blocked:
        verdict = "M1F_VALIDITY_BLOCKED"
    elif all(checks.values()):
        verdict = "M1F_CONTINUE_TO_CONFIRMATION_REVIEW"
    else:
        verdict = "M1F_CLOSE_BCR_UTILITY_METHOD"
    return {
        "verdict": verdict,
        "checks": checks,
        "conditions": {
            "mean_delta_min": P.M1F_MEAN_DELTA_MIN,
            "positive_readers_min": P.M1F_POSITIVE_READERS_MIN,
            "ci_alpha": P.M1F_CI_ALPHA,
        },
        "observed": {
            "mean_delta_macro_f1": agg["mean_delta_macro_f1"],
            "ci_low": agg["ci_low"], "ci_high": agg["ci_high"],
            "positive_readers": agg["positive_readers"],
            "worst_reader_delta": agg["worst_reader_delta"],
            "active_macro_f1_delta": neutral["active_macro_f1_delta"],
            "harmful_auprc_delta":
                comparison["secondary_metric_deltas"]["harmful_auprc"]
                ["observed"],
            "within_snapshot_centered_spearman_delta":
                within_delta["observed"],
            "within_snapshot_ci": [within_delta["ci_low"],
                                   within_delta["ci_high"]],
        },
        "gain_only_neutral_detail": neutral,
        "authorizes_m2": False,
        "note": "A CONTINUE outcome permits only a new research-review round "
                "for fresh-event / new-reader confirmation; it never "
                "authorizes M2.",
    }
