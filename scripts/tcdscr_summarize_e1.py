#!/usr/bin/env python
"""Aggregate Formal E1 results (plan §37): per-cutoff Accuracy / Macro-F1 /
Weighted-F1 / Rumor-F1 for Random Init vs UMER Init over 5 folds x 3 seeds,
per dataset. Writes formal_e1_summary.json + formal_e1_tables.md.

Aggregation modes:
  per-run  — each (fold, seed) test score; mean +/- std over the 30 runs
  pooled   — per seed, the five folds' test predictions are merged into a
             single whole-dataset prediction set (every event scored once);
             mean +/- std over the 3 seeds
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

DATASETS = ("pheme", "maweibo")
INITS = ("random", "umer")
SEEDS = (2000, 2001, 2002)
FOLDS = (0, 1, 2, 3, 4)
CUTOFFS = ("SOURCE_ONLY", "5", "15", "30", "60", "180", "360", "1440")
METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1", "rumor_f1")


def load_runs(root):
    runs = []
    for dataset in DATASETS:
        ddir = os.path.join(root, dataset)
        if not os.path.isdir(ddir):
            continue
        for name in sorted(os.listdir(ddir)):
            rdir = os.path.join(ddir, name)
            mpath = os.path.join(rdir, "metrics.json")
            if not os.path.exists(mpath):
                continue
            with open(mpath, encoding="utf-8") as fh:
                metrics = json.load(fh)
            with open(os.path.join(rdir, "run_manifest.json"),
                      encoding="utf-8") as fh:
                manifest = json.load(fh)
            preds_path = os.path.join(rdir, "predictions.jsonl")
            preds = []
            with open(preds_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        preds.append(json.loads(line))
            runs.append({"dataset": dataset, "dir": name,
                         "fold": manifest["fold"], "init": manifest["init"],
                         "seed": manifest["seed"], "metrics": metrics,
                         "predictions": preds, "manifest": manifest})
    return runs


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.stdev(values)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e1")
    args = ap.parse_args(argv)

    runs = load_runs(args.root)
    per_run = defaultdict(list)
    pooled = defaultdict(list)
    for run in runs:
        key = (run["dataset"], run["init"])
        for cutoff in CUTOFFS:
            m = run["metrics"].get(cutoff)
            if not m:
                continue
            for mk in METRIC_KEYS:
                per_run[key + (cutoff, mk)].append(m[mk])
    # pooled: merge folds within a seed, then score the merged predictions
    by_seed = defaultdict(lambda: defaultdict(list))
    for run in runs:
        key = (run["dataset"], run["init"], run["seed"])
        for row in run["predictions"]:
            by_seed[key][row["cutoff"]].append((row["gold"], row["pred"]))
    for (dataset, init, seed), by_cutoff in by_seed.items():
        for cutoff, pairs in by_cutoff.items():
            tp1 = sum(1 for g, p in pairs if g == 1 and p == 1)
            fp1 = sum(1 for g, p in pairs if g == 0 and p == 1)
            fn1 = sum(1 for g, p in pairs if g == 1 and p == 0)
            tp0 = sum(1 for g, p in pairs if g == 0 and p == 0)
            fp0 = sum(1 for g, p in pairs if g == 1 and p == 0)
            fn0 = sum(1 for g, p in pairs if g == 0 and p == 1)

            def prf(tp, fp, fn):
                pr = tp / (tp + fp) if tp + fp else 0.0
                re = tp / (tp + fn) if tp + fn else 0.0
                return 2 * pr * re / (pr + re) if pr + re else 0.0

            f1r = prf(tp1, fp1, fn1)
            f1n = prf(tp0, fp0, fn0)
            acc = (tp1 + tp0) / max(len(pairs), 1)
            s1, s0 = tp1 + fn1, tp0 + fn0
            pooled[(dataset, init, cutoff)].append({
                "accuracy": acc,
                "macro_f1": (f1r + f1n) / 2,
                "weighted_f1": (f1r * s1 + f1n * s0) / max(s1 + s0, 1),
                "rumor_f1": f1r,
            })

    summary = {"n_runs": len(runs), "datasets": {}}
    for dataset in DATASETS:
        d_summary = {"runs": 0, "per_run": {}, "pooled": {}}
        for init in INITS:
            for cutoff in CUTOFFS:
                for mk in METRIC_KEYS:
                    vals = per_run.get((dataset, init, cutoff, mk))
                    if vals:
                        mean, sd = mean_std(vals)
                        d_summary["per_run"].setdefault(init, {}) \
                            .setdefault(cutoff, {})[mk] \
                            = {"mean": mean, "std": sd, "n": len(vals)}
                pvals = pooled.get((dataset, init, cutoff))
                if pvals:
                    for mk in METRIC_KEYS:
                        vals = [p[mk] for p in pvals]
                        mean, sd = mean_std(vals)
                        d_summary["pooled"].setdefault(init, {}) \
                            .setdefault(cutoff, {})[mk] \
                            = {"mean": mean, "std": sd, "n_seeds": len(vals)}
            d_summary["runs"] = sum(
                1 for r in runs if r["dataset"] == dataset)
        summary["datasets"][dataset] = d_summary

    lines = ["# Formal E1 — Causal Encoder results", "",
             "Aggregated over 5 folds x 3 seeds (per-run: mean +/- std over"
             " 30 runs; pooled: folds merged per seed, mean +/- std over 3"
             " seeds). Cutoffs: SOURCE_ONLY, 5m..6h primary, 24h diagnostic.",
             ""]
    for dataset in DATASETS:
        if dataset not in summary["datasets"]:
            continue
        d = summary["datasets"][dataset]
        lines += [f"## {dataset}", ""]
        for init in INITS:
            lines += [f"### {init} init (per-run mean +/- std)", "",
                      "| cutoff | accuracy | macro_f1 | weighted_f1 |"
                      " rumor_f1 |", "|---|---|---|---|---|"]
            pc = d["per_run"].get(init, {})
            for cutoff in CUTOFFS:
                row = pc.get(cutoff)
                if not row:
                    continue
                cells = []
                for mk in METRIC_KEYS:
                    m, s = row[mk]["mean"], row[mk]["std"]
                    cells.append(f"{m:.4f} ± {s:.4f}")
                lines.append(f"| {cutoff} | " + " | ".join(cells) + " |")
            lines.append("")

    out_json = os.path.join(args.root, "formal_e1_summary.json")
    out_md = os.path.join(args.root, "formal_e1_tables.md")
    os.makedirs(args.root, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(json.dumps({"n_runs": len(runs), "wrote": [out_json, out_md]}))


if __name__ == "__main__":
    main()
