#!/usr/bin/env python
"""Verify Formal E1 completeness: every run dir has the 4 required files,
metrics cover 8 cutoffs x 4 metrics, predictions cover all expected cutoffs,
and manifest fields (dataset/fold/init/seed) parse correctly."""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "results", "tcdscr", "formal_e1")
CUTOFFS = ("SOURCE_ONLY", "5", "15", "30", "60", "180", "360", "1440")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "rumor_f1")
FILES = ("metrics.json", "predictions.jsonl", "run_manifest.json",
         "history.json")

problems = []
runs = 0
pred_rows = 0
for dataset in ("pheme", "maweibo"):
    ddir = os.path.join(ROOT, dataset)
    names = sorted(os.listdir(ddir))
    for name in names:
        rdir = os.path.join(ddir, name)
        if not os.path.isdir(rdir):
            continue
        runs += 1
        for f in FILES:
            if not os.path.exists(os.path.join(rdir, f)):
                problems.append(f"{dataset}/{name}: missing {f}")
                continue
        mpath = os.path.join(rdir, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                m = json.load(fh)
            for c in CUTOFFS:
                if c not in m:
                    problems.append(f"{dataset}/{name}: metrics missing cutoff {c}")
                else:
                    for mk in METRICS:
                        v = m[c].get(mk)
                        if v is None or not (0.0 <= v <= 1.0):
                            problems.append(
                                f"{dataset}/{name}: bad metric {c}/{mk}={v}")
        mpath = os.path.join(rdir, "run_manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                man = json.load(fh)
            need = {"dataset", "fold", "init", "seed", "hparams", "split",
                    "partition_seed", "best_epoch", "best_val_macro_f1",
                    "init_info"}
            missing = need - set(man)
            if missing:
                problems.append(f"{dataset}/{name}: manifest missing {sorted(missing)}")
        ppath = os.path.join(rdir, "predictions.jsonl")
        if os.path.exists(ppath):
            cutoffs_seen = set()
            n = 0
            with open(ppath, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        n += 1
                        cutoffs_seen.add(row["cutoff"])
            pred_rows += n
            if not set(CUTOFFS[1:]) <= cutoffs_seen:
                problems.append(
                    f"{dataset}/{name}: predictions missing cutoffs "
                    f"{sorted(set(CUTOFFS[1:]) - cutoffs_seen)}")
        hpath = os.path.join(rdir, "history.json")
        if os.path.exists(hpath):
            with open(hpath, encoding="utf-8") as fh:
                h = json.load(fh)
            if not (isinstance(h, list) and h and isinstance(h[0], dict)
                    and "epoch" in h[0] and "train_loss" in h[0]
                    and "val_macro_f1" in h[0]):
                problems.append(f"{dataset}/{name}: history.json format unexpected")

print(f"runs={runs} prediction_rows={pred_rows}")
print(f"problems={len(problems)}")
for p in problems[:50]:
    print(" -", p)
sys.exit(1 if problems else 0)