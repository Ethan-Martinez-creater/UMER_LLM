#!/usr/bin/env python
"""Final Evidence Closure statistics: Gap A + Gap F.

Paired event-level bootstrap, iterations 10000, seed 4096, fold-stratified,
multiplicity preserved (resampled events keep every occurrence).  Reads only
the frozen prediction artifacts produced by the Gap A / Gap F runners.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tcdscr_run_e2 import PRIMARY_CUTOFFS

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
ARM_NAMES = ("static", "random", "semantic")
N_ITER = 10000
BOOT_SEED = 4096
PRACTICAL_THRESHOLD = 0.005
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence"


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)


def macro_f1_from_counts(c):
    """c: [..., 4] = tp(class 1), fp, fn, tn."""
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]

    def f1(a, b, d):
        pr = np.divide(a, a + b, out=np.zeros_like(a, dtype=float),
                       where=(a + b) > 0)
        re = np.divide(a, a + d, out=np.zeros_like(a, dtype=float),
                       where=(a + d) > 0)
        return np.divide(2 * pr * re, pr + re, out=np.zeros_like(pr),
                         where=(pr + re) > 0)
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2.0


def counts_of(pairs_g, pairs_p):
    g = np.asarray(pairs_g, dtype=np.int64)
    p = np.asarray(pairs_p, dtype=np.int64)
    return np.array([int(((g == 1) & (p == 1)).sum()),
                     int(((g == 0) & (p == 1)).sum()),
                     int(((g == 1) & (p == 0)).sum()),
                     int(((g == 0) & (p == 0)).sum())], dtype=np.int64)


# ---------------------------------------------------------------- Gap A -----

def load_gap_a(out_root, dataset):
    """{fold: [(event_id, counts_by_cutoff[6,4])]} for each arm."""
    root = os.path.join(out_root, "gap_a", "runs", dataset)
    arms = {a: {} for a in ARM_NAMES}
    for fold in FOLDS:
        by_arm = {a: {} for a in ARM_NAMES}  # event -> [6][4]
        for seed in SEEDS:
            path = os.path.join(root, f"fold{fold}_seed{seed}",
                                "predictions.jsonl")
            if not os.path.exists(path):
                continue
            for r in read_jsonl(path):
                ci = PRIMARY_CUTOFFS.index(int(r["cutoff"]))
                key = r["event_id"]
                cell = by_arm[r["method"]].setdefault(
                    key, np.zeros((len(PRIMARY_CUTOFFS), 4), dtype=np.int64))
                cell[ci] += counts_of([r["gold"]], [r["pred"]])
        for a in ARM_NAMES:
            arms[a][fold] = by_arm[a]
    return arms


def stacked_fold(counts_by_fold):
    """fold -> (event_keys, array[n_events, 6, 4])."""
    out = {}
    for fold, d in counts_by_fold.items():
        keys = sorted(d.keys())
        if not keys:
            continue
        arr = np.stack([d[k] for k in keys])
        out[fold] = (keys, arr)
    return out


def stratified_bootstrap(stacks, n_iter=N_ITER, seed=BOOT_SEED):
    """stacks: {fold: (keys, arr)}. Returns (point, samples) mean-primary MF1.

    Resamples events with replacement inside each fold, keeping the fold
    size; multiplicity is preserved because a sampled event is applied as
    many times as it was drawn.
    """
    folds = sorted(stacks.keys())
    point = mean_primary_of(np.concatenate([stacks[f][1] for f in folds],
                                           axis=0))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        parts = []
        for f in folds:
            _keys, arr = stacks[f]
            n = arr.shape[0]
            idx = rng.integers(0, n, size=n)
            parts.append(arr[idx].sum(axis=0))
        total = np.sum(parts, axis=0)  # [6, 4]
        samples[i] = macro_f1_from_counts(total).mean()
    return point, samples


def mean_primary_of(arr):
    total = arr.sum(axis=0)
    return float(macro_f1_from_counts(total).mean())


def ci95(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def gap_a(args):
    summary = {"stage": "final_evidence_gap_a",
               "protocol": "ALL_EVENT held-out (test split)",
               "budget": 1024, "bootstrap": {
                   "unit": "event", "iterations": N_ITER, "seed": BOOT_SEED,
                   "stratified": "within outer fold",
                   "multiplicity": "preserved"},
               "datasets": {}}
    for dataset in DATASETS:
        arms = load_gap_a(args.out_root, dataset)
        stacks = {a: stacked_fold(arms[a]) for a in ARM_NAMES}
        points = {a: mean_primary_of(
            np.concatenate([stacks[a][f][1] for f in sorted(stacks[a])],
                           axis=0)) for a in ARM_NAMES}
        best = max(("random", "semantic"), key=lambda a: points[a])
        pair = {}
        for a in ARM_NAMES:
            point, samples = stratified_bootstrap(stacks[a])
            pair[a] = {"point": point, "samples": samples}
        entry = {"static": points["static"],
                 "random": points["random"],
                 "semantic": points["semantic"],
                 "best_simple_baseline": best,
                 "delta_static_minus_best": points["static"] - points[best],
                 "comparisons": {}}
        all_arr = {a: np.concatenate([stacks[a][f][1]
                                      for f in sorted(stacks[a])], axis=0)
                   for a in ARM_NAMES}
        entry["per_cutoff"] = {
            str(c): {a: float(macro_f1_from_counts(
                all_arr[a][:, ci, :].sum(axis=0)))
                for a in ARM_NAMES}
            for ci, c in enumerate(PRIMARY_CUTOFFS)}
        entry["per_fold"] = {
            str(f): {a: mean_primary_of(stacks[a][f][1]) for a in ARM_NAMES}
            for f in sorted(stacks["static"])}
        for a in ("random", "semantic"):
            delta = pair["static"]["samples"] - pair[a]["samples"]
            lo, hi = ci95(delta)
            point = points["static"] - points[a]
            entry["comparisons"][f"static_minus_{a}"] = {
                "point": point, "ci_low": lo, "ci_high": hi,
                "excludes_zero": bool(lo > 0 or hi < 0)}
        delta = pair["static"]["samples"] - pair[best]["samples"]
        lo, hi = ci95(delta)
        point = points["static"] - points[best]
        if point <= 0:
            status = "NOT_SUPPORTED"
        elif lo > 0:
            status = "STRONG_SUPPORT"
        else:
            status = "WEAK_SUPPORT"
        entry["primary"] = {
            "baseline": best, "point": point, "ci_low": lo, "ci_high": hi,
            "status": status,
            "meets_historical_0.005_threshold":
                bool(point >= PRACTICAL_THRESHOLD),
            "threshold": PRACTICAL_THRESHOLD,
            "threshold_note": "historical practical-effect threshold; not "
                              "tuned on test"}
        summary["datasets"][dataset] = entry
    write_json(os.path.join(args.out_root, "gap_a", "bootstrap.json"),
               {ds: {"primary": v["primary"],
                     "comparisons": v["comparisons"]}
                for ds, v in summary["datasets"].items()})
    write_json(os.path.join(args.out_root, "gap_a", "gap_a_summary.json"),
               summary)
    return summary


# ---------------------------------------------------------------- Gap F -----

def load_gap_f(out_root, dataset):
    """fold -> (keys, counts {'static','dynamic'}[n,6,4], flips, trans)."""
    root = os.path.join(out_root, "gap_f", "runs", dataset)
    out = {}
    for fold in FOLDS:
        per = {a: {} for a in ("static", "dynamic")}
        flips = {a: {} for a in ("static", "dynamic")}
        trans = {a: {} for a in ("static", "dynamic")}
        for seed in SEEDS:
            path = os.path.join(root, f"fold{fold}_seed{seed}",
                                "test_predictions.jsonl")
            if not os.path.exists(path):
                continue
            rows = read_jsonl(path)
            by_event = {}
            for r in rows:
                by_event.setdefault(r["event_id"], []).append(r)
            for ev, ev_rows in by_event.items():
                ev_rows.sort(key=lambda r: PRIMARY_CUTOFFS.index(
                    int(r["cutoff"])))
                for a, key in (("static", "static_prediction"),
                               ("dynamic", "dynamic_prediction")):
                    cell = per[a].setdefault(
                        ev, np.zeros((len(PRIMARY_CUTOFFS), 4),
                                     dtype=np.int64))
                    for r in ev_rows:
                        ci = PRIMARY_CUTOFFS.index(int(r["cutoff"]))
                        cell[ci] += counts_of([r["gold"]], [r[key]])
                    seq = [int(r[key]) for r in ev_rows]
                    fl = sum(1 for i in range(1, len(seq))
                             if seq[i - 1] != seq[i])
                    tr = max(len(seq) - 1, 0)
                    flips[a][ev] = flips[a].get(ev, 0) + fl
                    trans[a][ev] = trans[a].get(ev, 0) + tr
        out[fold] = {"per": per, "flips": flips, "trans": trans}
    return out


def gap_f(args):
    summary = {"stage": "final_evidence_gap_f",
               "protocol": "ALL_EVENT held-out (test split)",
               "bootstrap": {"unit": "event", "iterations": N_ITER,
                             "seed": BOOT_SEED,
                             "stratified": "within outer fold",
                             "multiplicity": "preserved"},
               "datasets": {}}
    for dataset in DATASETS:
        data = load_gap_f(args.out_root, dataset)
        stacks = {}
        flip_arr = {}
        for fold, d in data.items():
            keys = sorted(d["per"]["static"].keys())
            if not keys:
                continue
            stacks[fold] = {
                a: (keys, np.stack([d["per"][a][k] for k in keys]))
                for a in ("static", "dynamic")}
            flip_arr[fold] = {
                a: (np.array([d["flips"][a].get(k, 0) for k in keys],
                             dtype=np.int64),
                    np.array([d["trans"][a].get(k, 0) for k in keys],
                             dtype=np.int64))
                for a in ("static", "dynamic")}
        folds = sorted(stacks)
        point = {a: mean_primary_of(np.concatenate(
            [stacks[f][a][1] for f in folds], axis=0))
            for a in ("static", "dynamic")}

        def flip_rate(a):
            f = sum(int(flip_arr[x][a][0].sum()) for x in folds)
            t = sum(int(flip_arr[x][a][1].sum()) for x in folds)
            return (f / t) if t else 0.0
        point_flip = {a: flip_rate(a) for a in ("static", "dynamic")}

        rng = np.random.default_rng(BOOT_SEED)
        d_mf1 = np.empty(N_ITER)
        d_flip = np.empty(N_ITER)
        for i in range(N_ITER):
            tot = {a: [] for a in ("static", "dynamic")}
            fl = {a: [0, 0] for a in ("static", "dynamic")}
            for f in folds:
                n = stacks[f]["static"][1].shape[0]
                idx = rng.integers(0, n, size=n)
                for a in ("static", "dynamic"):
                    tot[a].append(stacks[f][a][1][idx].sum(axis=0))
                    fl[a][0] += int(flip_arr[f][a][0][idx].sum())
                    fl[a][1] += int(flip_arr[f][a][1][idx].sum())
            m = {a: float(macro_f1_from_counts(
                np.sum(tot[a], axis=0)).mean()) for a in tot}
            fr = {a: (fl[a][0] / fl[a][1] if fl[a][1] else 0.0)
                  for a in fl}
            d_mf1[i] = m["dynamic"] - m["static"]
            d_flip[i] = fr["dynamic"] - fr["static"]
        lo_m, hi_m = ci95(d_mf1)
        lo_f, hi_f = ci95(d_flip)
        delta_mf1 = point["dynamic"] - point["static"]
        delta_flip = point_flip["dynamic"] - point_flip["static"]
        if abs(delta_mf1) < 0.005 and lo_m <= 0 <= hi_m:
            status = "DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED"
        elif delta_mf1 <= 0:
            status = "DYNAMIC_V1_NOT_SUPPORTED_CONFIRMED"
        elif delta_mf1 >= 0.005 and lo_m > 0:
            status = "UNEXPECTED_POSITIVE_REQUIRES_RESEARCH_REVIEW"
        else:
            status = "INCONCLUSIVE"
        summary["datasets"][dataset] = {
            "static_macro_f1": point["static"],
            "dynamic_macro_f1": point["dynamic"],
            "delta_macro_f1": delta_mf1,
            "delta_macro_f1_ci": {"point": float(delta_mf1),
                                  "ci_low": lo_m, "ci_high": hi_m},
            "static_flip_rate": point_flip["static"],
            "dynamic_flip_rate": point_flip["dynamic"],
            "delta_flip_rate": delta_flip,
            "delta_flip_rate_ci": {"point": float(delta_flip),
                                   "ci_low": lo_f, "ci_high": hi_f},
            "per_cutoff": {
                str(c): {"static": float(macro_f1_from_counts(
                    np.concatenate([stacks[f]["static"][1][:, ci, :]
                                    for f in folds], axis=0).sum(axis=0))),
                    "dynamic": float(macro_f1_from_counts(
                        np.concatenate([stacks[f]["dynamic"][1][:, ci, :]
                                        for f in folds], axis=0).sum(axis=0)))}
                for ci, c in enumerate(PRIMARY_CUTOFFS)},
            "per_fold": {
                str(f): {"static": mean_primary_of(stacks[f]["static"][1]),
                         "dynamic": mean_primary_of(stacks[f]["dynamic"][1])}
                for f in folds},
            "status": status,
        }
    write_json(os.path.join(args.out_root, "gap_f", "bootstrap.json"),
               {ds: {"delta_macro_f1": v["delta_macro_f1_ci"],
                     "delta_flip_rate": v["delta_flip_rate_ci"],
                     "n_iterations": N_ITER, "seed": BOOT_SEED,
                     "unit": "event", "stratified": "within outer fold",
                     "multiplicity": "preserved"}
                for ds, v in summary["datasets"].items()})
    write_json(os.path.join(args.out_root, "gap_f", "gap_f_summary.json"),
               summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", choices=("a", "f", "both"), default="both")
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args(argv)
    if args.gap in ("a", "both"):
        gap_a(args)
    if args.gap in ("f", "both"):
        gap_f(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
