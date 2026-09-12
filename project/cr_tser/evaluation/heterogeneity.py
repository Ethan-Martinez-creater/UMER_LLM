"""Reader heterogeneity (plan §18) and the P1 gate (plan §25).

An atomic intervention is **active** for a reader when ``abs(u_r) >= 0.05`` or
the reader's correctness changes. Disagreement is measured only among
interventions active for both readers of a pair:

    Disagree(a,b) = P[ sign(u_a) != sign(u_b) ].

The sign convention uses the frozen ±0.05 neutral band, so it is the same
three-way sign the tri-class label uses (plan §10.1).
"""
from __future__ import annotations

from ..config.pilot_config import (P1_MEAN_DISAGREEMENT_MIN,
                                   P1_PAIR_DISAGREEMENT_MIN, P1_PAIRS_REQUIRED,
                                   UTILITY_THRESHOLD)
from .bootstrap import mean, spearman_rank


def sign_of(utility: float, correct_before=None, correct_after=None) -> int:
    """-1 / 0 / +1 with correctness transitions overriding (plan §10.1, §18)."""
    if correct_before is not None and correct_after is not None:
        if correct_before and not correct_after:
            return 1
        if not correct_before and correct_after:
            return -1
    if utility >= UTILITY_THRESHOLD:
        return 1
    if utility <= -UTILITY_THRESHOLD:
        return -1
    return 0


def is_active(utility: float, correct_before=None, correct_after=None) -> bool:
    """Active for a reader if ``|u| >= 0.05`` or correctness changes (§18)."""
    if correct_before is not None and correct_after is not None \
            and correct_before != correct_after:
        return True
    return abs(float(utility)) >= UTILITY_THRESHOLD


def _pair_rows(unit_table, a, b):
    """Jointly active rows for readers ``a`` and ``b``."""
    rows = []
    for key, entry in unit_table.items():
        ra, rb = entry.get(a), entry.get(b)
        if ra is None or rb is None:
            continue
        if not (is_active(ra["utility"], ra.get("correct_before"),
                          ra.get("correct_after"))
                and is_active(rb["utility"], rb.get("correct_before"),
                              rb.get("correct_after"))):
            continue
        rows.append((key, ra, rb))
    return rows


def pair_disagreement(unit_table, a, b) -> dict:
    """Sign disagreement, utility Spearman and the 3x3 contingency (plan §18)."""
    rows = _pair_rows(unit_table, a, b)
    if not rows:
        return {"reader_a": a, "reader_b": b, "n_active": 0,
                "disagreement": float("nan"),
                "sign_contingency": {}, "utility_spearman": float("nan")}
    disagree = 0
    contingency = {}
    for _key, ra, rb in rows:
        sa = sign_of(ra["utility"], ra.get("correct_before"),
                     ra.get("correct_after"))
        sb = sign_of(rb["utility"], rb.get("correct_before"),
                     rb.get("correct_after"))
        contingency[f"{sa}:{sb}"] = contingency.get(f"{sa}:{sb}", 0) + 1
        if sa != sb:
            disagree += 1
    ua = [ra["utility"] for _k, ra, _rb in rows]
    ub = [rb["utility"] for _k, _ra, rb in rows]
    return {
        "reader_a": a, "reader_b": b, "n_active": len(rows),
        "disagreement": disagree / len(rows),
        "sign_contingency": contingency,
        "utility_spearman": spearman_rank(ua, ub),
    }


def heterogeneity_report(unit_table, reader_keys) -> dict:
    """All reader pairs plus the macro mean (plan §18)."""
    pairs = []
    for i in range(len(reader_keys)):
        for j in range(i + 1, len(reader_keys)):
            pairs.append(pair_disagreement(unit_table, reader_keys[i],
                                           reader_keys[j]))
    finite = [p["disagreement"] for p in pairs
              if p["disagreement"] == p["disagreement"]]
    return {"pairs": pairs, "macro_mean_disagreement": mean(finite)}


def gate_p1(report) -> dict:
    """P1: mean disagreement ≥ 0.10 and ≥ 2/3 pairs ≥ 0.05 (plan §25)."""
    pairs = report["pairs"]
    n_ok = sum(1 for p in pairs
               if p["disagreement"] == p["disagreement"]
               and p["disagreement"] >= P1_PAIR_DISAGREEMENT_MIN)
    macro = report["macro_mean_disagreement"]
    passed = (macro == macro and macro >= P1_MEAN_DISAGREEMENT_MIN
              and n_ok >= P1_PAIRS_REQUIRED)
    return {
        "gate": "P1_reader_heterogeneity",
        "mean_disagreement": macro,
        "mean_threshold": P1_MEAN_DISAGREEMENT_MIN,
        "pairs_above_pair_threshold": n_ok,
        "pairs_required": P1_PAIRS_REQUIRED,
        "pair_threshold": P1_PAIR_DISAGREEMENT_MIN,
        "pass": bool(passed),
    }
