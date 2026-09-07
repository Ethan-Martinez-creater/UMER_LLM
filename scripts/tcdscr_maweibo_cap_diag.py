#!/usr/bin/env python
"""Formal E1 finalization — Ma-Weibo 1021-node cap-aware post-hoc diagnostic.

No retraining. Rebuilds per-event cap status for the 3h / 6h cutoffs with
the frozen snapshot builder (cap_hit = nodes before cap > 1021), then splits
the existing Formal E1 UMER-init test predictions (15 runs) into uncapped /
capped subsets and reports Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1 plus
rumor / nonrumor ratios at three granularities:

- per seed x fold
- pooled per seed (five folds merged, every test event scored once)
- 3-seed mean +/- std over the pooled scores

The diagnostic is descriptive only: max_nodes=1021 stays frozen, no
sampling / top-k / cap change. When the fold-parity audit is not PASS the
output is explicitly flagged UMER_INIT_RESULT_NOT_YET_APPROVED.

Reads:
  /data/jyz/next/llm/results/tcdscr/formal_e1/maweibo/foldN_umer_seedS/*
Writes:
  results/tcdscr/formal_e1_finalization/maweibo_cap_aware_e1.json
  results/tcdscr/formal_e1_finalization/maweibo_cap_aware_e1.md
"""
import argparse
import json
import os
import statistics as st

import tcdscr_common  # noqa: F401  (registers the project package)

from tcdscr.config.schema import config_from_env  # noqa: E402
from tcdscr_common import event_label_registry  # noqa: E402

CUTOFFS = ("180", "360")
SEEDS = (2000, 2001, 2002)
FOLDS = (0, 1, 2, 3, 4)
METRICS = ("accuracy", "macro_f1", "weighted_f1", "rumor_f1")


def classification_metrics(pairs):
    """pairs: (gold, pred) -> four metrics + ratios; same formulas as E1."""
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
    f1r, f1n = f1(tp1, fp1, fn1), f1(tp0, fp0, fn0)
    return {"n": len(pairs),
            "accuracy": acc,
            "macro_f1": (f1r + f1n) / 2,
            "weighted_f1": (f1r * s1 + f1n * s0) / max(s1 + s0, 1),
            "rumor_f1": f1r,
            "rumor_ratio": s1 / max(n, 1),
            "nonrumor_ratio": s0 / max(n, 1)}


def build_cap_status(cfg):
    """{event_id: {cutoff: {"cap_hit": bool, "nodes_before": int}}}."""
    from tcdscr.data.maweibo_adapter import load_event
    from tcdscr.data.snapshot_builder import build_snapshot
    registry = event_label_registry("maweibo", cfg)
    status = {}
    for eid in sorted(registry):
        event = load_event(eid, registry[eid],
                           f"{cfg.raw_dir.rstrip('/')}/{eid}.json")
        entry = {}
        for c in CUTOFFS:
            snap = build_snapshot(event, int(c))
            entry[c] = {"cap_hit": bool(snap["cap_hit"]),
                        "nodes_before": int(snap["num_nodes_before_cap"])}
        status[eid] = entry
    return status


def load_e1_predictions(e1_root):
    """{seed: {(fold, cutoff): [rows]}} for UMER-init runs."""
    out = {}
    for seed in SEEDS:
        out[seed] = {}
        for fold in FOLDS:
            run_dir = os.path.join(e1_root, "maweibo",
                                   f"fold{fold}_umer_seed{seed}")
            path = os.path.join(run_dir, "predictions.jsonl")
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["cutoff"] in CUTOFFS:
                        out[seed].setdefault(
                            (fold, row["cutoff"]), []).append(row)
    return out


def split_by_cap(rows, cap_status, cutoff):
    grouped = {"uncapped": [], "capped": []}
    for r in rows:
        hit = bool(cap_status.get(r["event_id"], {}).get(cutoff, {})
                   .get("cap_hit"))
        grouped["capped" if hit else "uncapped"].append((r["gold"], r["pred"]))
    return {subset: classification_metrics(grouped[subset])
            for subset in grouped}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e1")
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/"
                            "tcdscr/formal_e1_finalization")
    args = ap.parse_args(argv)

    cfg = config_from_env("maweibo")
    print("building per-event cap status for 3h/6h ...", flush=True)
    cap_status = build_cap_status(cfg)
    e1 = load_e1_predictions(args.e1_root)

    out = {"dataset": "maweibo", "cutoffs_min": [180, 360],
           "note": "diagnostic only; max_nodes=1021 frozen; no sampling/"
                   "top-k/cap change; predictions reused from Formal E1 "
                   "UMER-init runs.",
           "per_event_cap": cap_status,
           "per_seed_per_fold": {},
           "per_seed_pooled": {},
           "three_seed_stats": {}}
    for seed in SEEDS:
        out["per_seed_per_fold"][str(seed)] = {}
        for fold in FOLDS:
            scored = {}
            for c in CUTOFFS:
                rows = e1[seed].get((fold, c), [])
                scored[c] = split_by_cap(rows, cap_status, c)
            out["per_seed_per_fold"][str(seed)][str(fold)] = scored
        pooled = {}
        for c in CUTOFFS:
            rows = []
            for fold in FOLDS:
                rows.extend(e1[seed].get((fold, c), []))
            pooled[c] = split_by_cap(rows, cap_status, c)
        out["per_seed_pooled"][str(seed)] = pooled
    # 3-seed mean +/- std over pooled scores
    for c in CUTOFFS:
        out["three_seed_stats"][c] = {}
        for subset in ("uncapped", "capped"):
            out["three_seed_stats"][c][subset] = {}
            for mk in METRICS + ("n", "rumor_ratio", "nonrumor_ratio"):
                out["three_seed_stats"][c][subset][mk] = [
                    out["per_seed_pooled"][str(s)][c][subset][mk]
                    for s in SEEDS]

    out["UMER_INIT_RESULT_NOT_YET_APPROVED"] = True
    out["not_approved_reason"] = (
        "fold-parity audit is not PASS (UMER_INIT_FOLD_PARITY_FAIL); the "
        "UMER-init E1 predictions are reported only as a diagnostic")

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "maweibo_cap_aware_e1.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    md = ["# Ma-Weibo cap-aware E1 diagnostic (UMER-init predictions)", "",
          "**UMER_INIT_RESULT_NOT_YET_APPROVED** — fold-parity audit is not "
          "PASS; numbers below are descriptive only and must not be used to "
          "select the E2 encoder.", "",
          "Pooled over 5 folds per seed (every test event scored once); "
          "3-seed mean ± std over the pooled scores. uncapped = cap_hit "
          "false; capped = cap_hit true.", ""]
    for c in CUTOFFS:
        md += [f"## cutoff {c}m", "",
               "| seed | subset | n | accuracy | macro_f1 | weighted_f1 | "
               "rumor_f1 | rumor_ratio | nonrumor_ratio |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in SEEDS:
            for subset in ("uncapped", "capped"):
                m = out["per_seed_pooled"][str(s)][c][subset]
                md.append(
                    f"| {s} | {subset} | {m['n']} | {m['accuracy']:.4f} | "
                    f"{m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
                    f"{m['rumor_f1']:.4f} | {m['rumor_ratio']:.4f} | "
                    f"{m['nonrumor_ratio']:.4f} |")
        stats_row = "| 3-seed mean±std |"
        for subset in ("uncapped", "capped"):
            v = out["three_seed_stats"][c][subset]
            cells = [f"{subset}", f"{st.mean(v['n']):.1f}"]
            for mk in METRICS + ("rumor_ratio", "nonrumor_ratio"):
                cells.append(f"{st.mean(v[mk]):.4f} ± "
                             f"{st.stdev(v[mk]):.4f}")
            stats_row += " || " + " | ".join(cells)
        md += ["", "| seed | subset | n | accuracy | macro_f1 | weighted_f1 "
                   "| rumor_f1 | rumor_ratio | nonrumor_ratio |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|",
               stats_row, ""]
    with open(os.path.join(args.out_root, "maweibo_cap_aware_e1.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("wrote", os.path.join(args.out_root, "maweibo_cap_aware_e1.json"))
    return out


if __name__ == "__main__":
    main()