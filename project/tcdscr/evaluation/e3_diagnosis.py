"""E3 failure-diagnosis statistics (post-hoc; no model runs).

Pure functions used by the E3 failure-diagnosis pipeline:

- paired bootstrap with **multiplicity preserved**: with-replacement event
  resampling keeps every occurrence (never folded into a dict keyed by
  event_id), so repeated events contribute repeatedly to each iteration;
- selection-change metrics (exact match / Jaccard / replacements);
- prediction-transition categories (wrong->correct / correct->wrong);
- Spearman rank correlation (no scipy dependency);
- budget-masking A/B/C classification;
- selection-pressure bins (fixed thresholds).

None of these functions touch models, checkpoints or predictions files.
"""
from __future__ import annotations

import random
import statistics as st
from collections import Counter

PRIMARY_CUTOFFS = (5, 15, 30, 60, 180, 360)


def classification_metrics(pairs):
    """Identical to scripts/tcdscr_run_e2.classification_metrics (pairs list
    of (gold, pred)); keeps the E3/E3-B metric definition bit-for-bit."""
    n = max(len(pairs), 1)
    acc = sum(1 for g, p in pairs if g == p) / n
    tp1 = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp1 = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn1 = sum(1 for g, p in pairs if g == 1 and p == 0)
    tp0 = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp0 = sum(1 for g, p in pairs if g == 1 and p == 0)
    fn0 = sum(1 for g, p in pairs if g == 0 and p == 1)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * re / (pr + re) if pr + re else 0.0

    s1, s0 = tp1 + fn1, tp0 + fn0
    return {"n": len(pairs), "accuracy": acc,
            "macro_f1": (f1(tp1, fp1, fn1) + f1(tp0, fp0, fn0)) / 2,
            "weighted_f1": (f1(tp1, fp1, fn1) * s1
                            + f1(tp0, fp0, fn0) * s0) / max(s1 + s0, 1),
            "rumor_f1": f1(tp1, fp1, fn1)}


def mean_primary_macro_f1(metrics_by_cutoff):
    """Mean Macro-F1 over the six primary cutoffs (E3 test order §9)."""
    vals = [metrics_by_cutoff[str(c)]["macro_f1"]
            for c in PRIMARY_CUTOFFS
            if str(c) in metrics_by_cutoff]
    return sum(vals) / len(vals) if vals else 0.0

# ---------------------------------------------------------------------------
# Paired bootstrap (multiplicity-preserving)
# ---------------------------------------------------------------------------


def resample(keys, rng, n):
    """With-replacement draw of ``n`` keys; duplicates are kept."""
    return [keys[rng.randrange(len(keys))] for _ in range(n)]


def pooled_primary_from_events(events, arm):
    """Per-cutoff pooled pairs over an event list (multiplicity included)."""
    by_cut = {}
    for u in events:
        for c, triples in u["pairs"].items():
            bucket = by_cut.setdefault(c, [])
            for g, s, d in triples:
                bucket.append((g, s if arm == "static" else d))
    return mean_primary_macro_f1(
        {c: classification_metrics(by_cut[c]) for c in by_cut})


def flip_stats_from_events(events, arm):
    """(flips, transitions) over the event list (multiplicity included)."""
    flips = transitions = 0
    for u in events:
        seqs = u["static_seq"] if arm == "static" else u["dynamic_seq"]
        for seq in seqs:
            for i in range(1, len(seq)):
                transitions += 1
                if seq[i - 1] != seq[i]:
                    flips += 1
    return flips, transitions


def _delta_of_events(events):
    dmf = (pooled_primary_from_events(events, "dynamic")
           - pooled_primary_from_events(events, "static"))
    fs, ts = flip_stats_from_events(events, "static")
    fd, td = flip_stats_from_events(events, "dynamic")
    dfl = (fd / td if td else 0.0) - (fs / ts if ts else 0.0)
    return dmf, dfl


def paired_bootstrap(units, n_iter=10000, seed=3090):
    """Event-level paired bootstrap with preserved multiplicity.

    ``units`` maps event_id -> event unit (pairs + per-seed prediction
    sequences). Each iteration draws n events WITH replacement and keeps
    every occurrence — repeated events contribute their predictions
    repeatedly (they are never merged into a unique-key dict).

    Returns point estimates and 95% percentile CIs for Delta Macro-F1 and
    Delta FlipRate (Dynamic - Static).
    """
    keys = sorted(units)
    n = len(keys)
    if n == 0:
        raise ValueError("no events to bootstrap")
    point_mf1, point_flip = _delta_of_events([units[k] for k in keys])
    rng = random.Random(seed)
    deltas_mf1, deltas_flip = [], []
    for _ in range(n_iter):
        # list comprehension keeps multiplicity; each occurrence contributes
        events = [units[keys[i]]
                  for i in (rng.randrange(n) for _ in range(n))]
        dmf, dfl = _delta_of_events(events)
        deltas_mf1.append(dmf)
        deltas_flip.append(dfl)

    def ci(vals):
        vals = sorted(vals)
        return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

    lo_m, hi_m = ci(deltas_mf1)
    lo_f, hi_f = ci(deltas_flip)
    return {
        "n_events": n,
        "n_iterations": n_iter,
        "seed": seed,
        "multiplicity": "preserved",
        "delta_macro_f1": {"point": point_mf1, "ci_low": lo_m,
                           "ci_high": hi_m},
        "delta_flip_rate": {"point": point_flip, "ci_low": lo_f,
                            "ci_high": hi_f},
    }


# ---------------------------------------------------------------------------
# Selection change (Q1)
# ---------------------------------------------------------------------------


def selection_change_metrics(rows):
    """rows: per (event, cutoff) rows with static/dynamic selected ids.

    Returns dict with exact-match rate, Jaccard stats and replacement
    counts over the given rows.
    """
    jacs, exact, removals, additions, symdiff, any_repl = [], 0, 0, 0, 0, 0
    for r in rows:
        ss, ds = set(r["static_selected_node_ids"]), \
            set(r["dynamic_selected_node_ids"])
        j = (len(ss & ds) / len(ss | ds)) if (ss | ds) else 1.0
        jacs.append(j)
        if ss == ds:
            exact += 1
        else:
            any_repl += 1
        removals += len(ss - ds)
        additions += len(ds - ss)
        symdiff += len(ss ^ ds)
    n = len(rows)
    jacs_sorted = sorted(jacs)

    def pct(p):
        return jacs_sorted[min(int(p * len(jacs_sorted)),
                               len(jacs_sorted) - 1)]

    return {
        "n": n,
        "exact_match_count": exact,
        "exact_match_rate": exact / n if n else 0.0,
        "jaccard_mean": st.mean(jacs) if jacs else 0.0,
        "jaccard_median": pct(0.5) if jacs else 0.0,
        "jaccard_p25": pct(0.25) if jacs else 0.0,
        "jaccard_p75": pct(0.75) if jacs else 0.0,
        "jaccard_p90": pct(0.90) if jacs else 0.0,
        "removed_from_static": removals,
        "added_by_dynamic": additions,
        "symmetric_difference_size": symdiff,
        "mean_replacements_per_event": symdiff / n if n else 0.0,
        "events_with_any_replacement": any_repl,
        "replacement_rate": any_repl / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Selection -> prediction effect (Q2)
# ---------------------------------------------------------------------------


def prediction_transition_categories(rows):
    """Over rows where the selection changed: prediction transitions.

    Returns wrong_to_correct / correct_to_wrong / both_wrong / both_correct
    plus the net correction gain (wrong_to_correct - correct_to_wrong).
    """
    w2c = c2w = bw = bc = 0
    for r in rows:
        if set(r["static_selected_node_ids"]) == \
                set(r["dynamic_selected_node_ids"]):
            continue
        gold, sp, dp = r["gold"], r["static_prediction"], \
            r["dynamic_prediction"]
        static_ok = sp == gold
        dynamic_ok = dp == gold
        if not static_ok and dynamic_ok:
            w2c += 1
        elif static_ok and not dynamic_ok:
            c2w += 1
        elif not static_ok and not dynamic_ok:
            bw += 1
        else:
            bc += 1
    return {"wrong_to_correct": w2c, "correct_to_wrong": c2w,
            "both_wrong": bw, "both_correct": bc,
            "net_correction_gain": w2c - c2w}


def selection_change_prediction_effect(rows):
    """Q2 summary: prediction-change rate given selection change."""
    changed = [r for r in rows
               if set(r["static_selected_node_ids"]) !=
               set(r["dynamic_selected_node_ids"])]
    pred_changed = sum(1 for r in changed
                       if r["static_prediction"] != r["dynamic_prediction"])
    n = len(changed)
    return {
        "n_selection_changed": n,
        "prediction_same": n - pred_changed,
        "prediction_changed": pred_changed,
        "prediction_change_rate_given_selection_change":
            pred_changed / n if n else 0.0,
        "transitions": prediction_transition_categories(changed),
    }


# ---------------------------------------------------------------------------
# Ranking (Q5) / budget masking (Q6)
# ---------------------------------------------------------------------------


def spearman_rank_correlation(a, b):
    """Standard Spearman rho between two score lists (no scipy needed)."""
    n = len(a)
    if n < 2:
        return 1.0

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def topk_set_changed(ranked_ids_a, ranked_ids_b, k):
    """Whether the top-k id sets differ (k capped at list length)."""
    k = min(k, len(ranked_ids_a))
    return set(ranked_ids_a[:k]) != set(ranked_ids_b[:k])


def budget_masking_classification(rows, ranking_changed_fn):
    """Q6 A/B/C per row via a ranking_changed_fn(row) -> bool."""
    a = b = c = 0
    for r in rows:
        ranking_changed = ranking_changed_fn(r)
        selection_changed = set(r["static_selected_node_ids"]) != \
            set(r["dynamic_selected_node_ids"])
        if not ranking_changed and not selection_changed:
            a += 1
        elif ranking_changed and not selection_changed:
            b += 1
        elif ranking_changed and selection_changed:
            c += 1
        else:  # selection changed without ranking change is impossible
            # for score-ordered selection; count defensively as C
            c += 1
    n = a + b + c
    return {"A_unchanged": a, "B_ranking_changed_selection_same": b,
            "C_ranking_changed_selection_changed": c,
            "ranking_changed_but_selection_same_rate":
                b / n if n else 0.0,
            "selection_change_rate": (b + c) / n if n else 0.0}


# ---------------------------------------------------------------------------
# Selection pressure (Q10)
# ---------------------------------------------------------------------------


def pressure_bin(saturation):
    """Fixed bins: Low >= 0.8, Medium 0.4-0.8, High < 0.4."""
    if saturation >= 0.8:
        return "low"
    if saturation >= 0.4:
        return "medium"
    return "high"


def selection_pressure_summary(rows):
    """Group rows into fixed pressure bins; per-bin stats dict."""
    groups = {"low": [], "medium": [], "high": []}
    for r in rows:
        den = r.get("n_candidates") or 1
        sat = len(r["dynamic_selected_node_ids"]) / den
        groups[pressure_bin(sat)].append(r)
    out = {}
    for name, g in groups.items():
        out[name] = {"n": len(g)}
        if g:
            out[name]["static_macro_f1"] = classification_metrics(
                [(r["gold"], r["static_prediction"]) for r in g])
            out[name]["dynamic_macro_f1"] = classification_metrics(
                [(r["gold"], r["dynamic_prediction"]) for r in g])
            sel_changed = sum(1 for r in g
                              if set(r["static_selected_node_ids"]) !=
                              set(r["dynamic_selected_node_ids"]))
            pred_changed = sum(1 for r in g
                               if r["static_prediction"] !=
                               r["dynamic_prediction"])
            out[name]["selection_change_rate"] = sel_changed / len(g)
            out[name]["prediction_change_rate"] = pred_changed / len(g)
    return out


def duplicate_counts(keys, rng, n):
    """Exposed for tests: Counter of a with-replacement draw."""
    return Counter(resample(keys, rng, n))
