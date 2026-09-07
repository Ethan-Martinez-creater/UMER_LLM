#!/usr/bin/env python
"""Formal E1 finalization — post-hoc actual epochs-run audit.

The historical E1 run manifests recorded ``epochs_run`` as the *configured*
max epochs (60) although training early-stopped; the true count is always
``len(history.json)``. This script aggregates the true counts over the 60
stored runs (per dataset x init: mean / min / max) without touching any
run artifact, and summarizes how the manifest field should be read.
"""
import argparse
import json
import os
import statistics as st

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
INITS = ("random", "umer")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args(argv)

    by_key = {}
    n_runs = 0
    for dataset in ("pheme", "maweibo"):
        for init in INITS:
            key = (dataset, init)
            by_key[key] = []
            for fold in FOLDS:
                for seed in SEEDS:
                    run_dir = os.path.join(args.e1_root, dataset,
                                           f"fold{fold}_{init}_seed{seed}")
                    hpath = os.path.join(run_dir, "history.json")
                    mpath = os.path.join(run_dir, "run_manifest.json")
                    if not (os.path.exists(hpath) and os.path.exists(mpath)):
                        continue
                    with open(hpath, encoding="utf-8") as fh:
                        history = json.load(fh)
                    with open(mpath, encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    actual = len(history)
                    by_key[key].append(actual)
                    n_runs += 1
                    if actual > manifest["hparams"]["epochs_run"]:
                        raise ValueError(
                            f"{run_dir}: actual {actual} exceeds configured "
                            f"{manifest['hparams']['epochs_run']}")

    out = {"note": ("historical manifests recorded epochs_run as the "
                    "configured max epochs (60); actual epochs run = "
                    "len(history.json). No run artifact was modified."),
           "n_runs": n_runs, "per_key": {}}
    for (dataset, init), vals in sorted(by_key.items()):
        out["per_key"][f"{dataset}/{init}"] = {
            "mean": st.mean(vals), "min": min(vals), "max": max(vals),
            "values": sorted(vals)}
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "epochs_posthoc_audit.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    lines = ["# E1 post-hoc audit — actual epochs run", "",
             f"{out['note']}", ""]
    for (dataset, init), vals in sorted(by_key.items()):
        lines.append(
            f"- {dataset} {init}: actual epochs mean "
            f"{st.mean(vals):.1f}, min {min(vals)}, max {max(vals)} "
            f"(over 15 runs)")
    with open(os.path.join(args.out_root, "epochs_posthoc_audit.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({k: out["per_key"][k] for k in out["per_key"]},
                     indent=1))


if __name__ == "__main__":
    main()