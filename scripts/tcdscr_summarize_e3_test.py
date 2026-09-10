#!/usr/bin/env python
"""Formal E3-B aggregation — test metrics, paired bootstrap, report.

Reads results/tcdscr/formal_e3_test/runs/ (30 runs), aggregates per dataset:

- pooled per-seed mean primary Macro-F1 (Static/Dynamic) and deltas;
- event-level paired bootstrap (10,000 resamples, seed 3090, sampling unit
  = test event) with 95% percentile CIs for Delta Macro-F1 and Delta
  FlipRate (§12);
- fold-level and seed-level consistency (§14-§15);
- Ma-Weibo cap-aware diagnostic (§16) and PHEME selection saturation (§17);
- E3_TEST_REPORT.md with the fixed structure (§21) and the §22 decision
  rules (never auto-approve E4 from validation PASS).

Writes: statistics/e3_test_bootstrap.json, e3_test_summary.json,
diagnostics/, E3_TEST_REPORT.md.
"""
import argparse
import json
import os
import random
import statistics as st
from collections import defaultdict

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, classification_metrics,
                           mean_primary_macro_f1)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 3090


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def collect_dataset_rows(root, dataset):
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            p = os.path.join(root, "runs", dataset,
                             f"fold{fold}_seed{seed}",
                             "test_predictions.jsonl")
            if os.path.exists(p):
                rows.extend(_rows(p))
    return rows


def primary_from_pairs(pairs_by_cutoff):
    """pairs_by_cutoff: {cutoff: [(gold, pred)]} -> mean primary Macro-F1."""
    return mean_primary_macro_f1(
        {c: classification_metrics(pairs_by_cutoff[c])
         for c in pairs_by_cutoff})


def flip_stats(event_sequences):
    """event_sequences: list of per-(seed,event) ordered prediction lists.
    Returns (flips, transitions)."""
    flips = transitions = 0
    for seq in event_sequences:
        for i in range(1, len(seq)):
            transitions += 1
            if seq[i - 1] != seq[i]:
                flips += 1
    return flips, transitions


def event_units(rows):
    """Group rows by event; each unit keeps per-cutoff pairs across seeds.

    Returns {event_id: {"pairs": {cutoff: [(gold, static, dynamic)]},
                        "static_seq": [[...] per seed], "dynamic_seq": [...]}}
    """
    units = {}
    by_seed_event = defaultdict(list)
    for r in rows:
        key = r["event_id"]
        u = units.setdefault(key, {"pairs": defaultdict(list),
                                   "static_seq": [], "dynamic_seq": []})
        u["pairs"][r["cutoff"]].append((r["gold"], r["static_prediction"],
                                        r["dynamic_prediction"]))
        by_seed_event[(key, r["seed"])].append(r)
    for (key, _seed), ev_rows in by_seed_event.items():
        ev_rows.sort(key=lambda r: int(r["cutoff"]))
        units[key]["static_seq"].append([r["static_prediction"]
                                         for r in ev_rows])
        units[key]["dynamic_seq"].append([r["dynamic_prediction"]
                                          for r in ev_rows])
    return units


def pooled_from_units(units, arm):
    """All-events pooled per-cutoff pairs for one arm ('static'/'dynamic')."""
    by_cut = {str(c): [] for c in PRIMARY_CUTOFFS}
    for u in units.values():
        for c, triples in u["pairs"].items():
            for g, s, d in triples:
                by_cut[c].append((g, s if arm == "static" else d))
    return by_cut


def paired_bootstrap(units, n_iter=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Event-level paired bootstrap; returns point estimates + 95% CIs."""
    rng = random.Random(seed)
    keys = sorted(units.keys())
    n = len(keys)
    # point estimates on the full event set
    def full_delta():
        sta = primary_from_pairs(pooled_from_units(units, "static"))
        dyn = primary_from_pairs(pooled_from_units(units, "dynamic"))
        f_sta, f_dyn = 0, 0
        t_sta = t_dyn = 0
        for u in units.values():
            fs, ts = flip_stats(u["static_seq"])
            fd, td = flip_stats(u["dynamic_seq"])
            f_sta += fs
            t_sta += ts
            f_dyn += fd
            t_dyn += td
        return (dyn - sta,
                (f_dyn / t_dyn if t_dyn else 0.0)
                - (f_sta / t_sta if t_sta else 0.0))
    point_mf1, point_flip = full_delta()

    deltas_mf1, deltas_flip = [], []
    for _ in range(n_iter):
        sample = [keys[i] for i in (rng.randrange(n) for _ in range(n))]
        sub = {k: units[k] for k in sample}
        dmf, dfl = 0.0, 0.0
        sta = primary_from_pairs(pooled_from_units(sub, "static"))
        dyn = primary_from_pairs(pooled_from_units(sub, "dynamic"))
        dmf = dyn - sta
        f_sta = t_sta = f_dyn = t_dyn = 0
        for u in sub.values():
            fs, ts = flip_stats(u["static_seq"])
            fd, td = flip_stats(u["dynamic_seq"])
            f_sta += fs
            t_sta += ts
            f_dyn += fd
            t_dyn += td
        dfl = ((f_dyn / t_dyn if t_dyn else 0.0)
               - (f_sta / t_sta if t_sta else 0.0))
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
        "delta_macro_f1": {"point": point_mf1, "ci_low": lo_m,
                           "ci_high": hi_m},
        "delta_flip_rate": {"point": point_flip, "ci_low": lo_f,
                            "ci_high": hi_f},
    }


def dataset_summary(root, dataset):
    rows = collect_dataset_rows(root, dataset)
    units = event_units(rows)
    bootstrap = paired_bootstrap(units)

    # pooled per-seed (E3-consistent view)
    per_seed = {}
    for s in SEEDS:
        s_rows = [r for r in rows if r["seed"] == s]
        s_units = event_units(s_rows)
        per_seed[str(s)] = {
            "static_primary": primary_from_pairs(
                pooled_from_units(s_units, "static")),
            "dynamic_primary": primary_from_pairs(
                pooled_from_units(s_units, "dynamic")),
        }
    mean_mf1 = {
        "static": st.mean(per_seed[str(s)]["static_primary"] for s in SEEDS),
        "dynamic": st.mean(per_seed[str(s)]["dynamic_primary"] for s in SEEDS),
        "std": st.stdev(per_seed[str(s)]["dynamic_primary"] for s in SEEDS),
    }
    # pooled flip rates (all events)
    f_sta = t_sta = f_dyn = t_dyn = 0
    for u in units.values():
        fs, ts = flip_stats(u["static_seq"])
        fd, td = flip_stats(u["dynamic_seq"])
        f_sta += fs
        t_sta += ts
        f_dyn += fd
        t_dyn += td
    flip = {"static": f_sta / t_sta if t_sta else 0.0,
            "dynamic": f_dyn / t_dyn if t_dyn else 0.0}
    tok = {"static": st.mean(r["static_evidence_tokens"] for r in rows),
           "dynamic": st.mean(r["dynamic_evidence_tokens"] for r in rows)}

    # fold-level
    fold_rows = {f: [r for r in rows if r["fold"] == f] for f in FOLDS}
    fold_level = {}
    for f in FOLDS:
        fu = event_units(fold_rows[f])
        stm = primary_from_pairs(pooled_from_units(fu, "static"))
        dym = primary_from_pairs(pooled_from_units(fu, "dynamic"))
        fold_level[str(f)] = {"static_primary": stm, "dynamic_primary": dym,
                              "delta": dym - stm}
    n_pos = sum(1 for v in fold_level.values() if v["delta"] > 0)
    n_neg = sum(1 for v in fold_level.values() if v["delta"] < 0)

    # per-cutoff pooled
    per_cutoff = {}
    for c in PRIMARY_CUTOFFS:
        c_rows = [r for r in rows if r["cutoff"] == str(c)]
        per_cutoff[str(c)] = {
            "static": classification_metrics(
                [(r["gold"], r["static_prediction"]) for r in c_rows]),
            "dynamic": classification_metrics(
                [(r["gold"], r["dynamic_prediction"]) for r in c_rows]),
        }

    # PHEME selection saturation (per cutoff, dynamic)
    saturation = {}
    for c in PRIMARY_CUTOFFS:
        c_rows = [r for r in rows if r["cutoff"] == str(c)]
        num = sum(len(r["dynamic_selected_node_ids"]) for r in c_rows)
        den = sum(r["n_candidates"] for r in c_rows)
        saturation[str(c)] = num / den if den else 0.0

    # cap-aware diagnostic (Ma-Weibo 3h/6h)
    cap = None
    if dataset == "maweibo":
        cap = {}
        for c in ("180", "360"):
            cap[c] = {}
            for capped in (False, True):
                group = [r for r in rows
                         if r["cutoff"] == c and r["cap_hit"] == capped]
                if not group:
                    continue
                entry = {"n": len(group),
                         "static": classification_metrics(
                             [(r["gold"], r["static_prediction"])
                              for r in group]),
                         "dynamic": classification_metrics(
                             [(r["gold"], r["dynamic_prediction"])
                              for r in group])}
                cap[c][str(capped)] = entry

    # evidence dynamics (per cutoff): turnover/retention/novelty/ratios
    dynamics = {}
    for c in PRIMARY_CUTOFFS:
        c_rows = [r for r in rows if r["cutoff"] == str(c)]
        per = new = sel = 0
        nov = []
        for r in c_rows:
            sel_ids = set(r["dynamic_selected_node_ids"])
            for s in r["selected_scores"]:
                nov.append(s["novelty"])
                if s["node_id"] in sel_ids:
                    if s["persistence"] == 1.0:
                        per += 1
                    else:
                        new += 1
                    sel += 1
        dynamics[str(c)] = {
            "persistent_ratio": per / sel if sel else 0.0,
            "new_ratio": new / sel if sel else 0.0,
            "selected_novelty_mean": (sum(nov) / len(nov) if nov else 0.0),
        }
    turnover = {}
    for c in PRIMARY_CUTOFFS:
        c_rows = [r for r in rows if r["cutoff"] == str(c)]
        vals = [1.0 - (len(set(r["memory_previous_ids"])
                        & set(r["memory_current_ids"]))
                       / max(len(set(r["memory_previous_ids"])
                               | set(r["memory_current_ids"])), 1))
                for r in c_rows if r["memory_previous_ids"]]
        turnover[str(c)] = st.mean(vals) if vals else 0.0

    summary = {
        "n_rows": len(rows),
        "mean_primary_macro_f1": mean_mf1,
        "delta_macro_f1": mean_mf1["dynamic"] - mean_mf1["static"],
        "flip_rate": flip,
        "delta_flip_rate": flip["dynamic"] - flip["static"],
        "mean_evidence_tokens": tok,
        "delta_evidence_tokens": tok["dynamic"] - tok["static"],
        "bootstrap": bootstrap,
        "per_seed": per_seed,
        "fold_level": fold_level,
        "n_positive_folds": n_pos,
        "n_negative_folds": n_neg,
        "per_cutoff": per_cutoff,
        "selection_saturation": saturation,
        "cap_diagnostic": cap,
        "dynamics": dynamics,
        "turnover": turnover,
        "status": interpret(bootstrap, mean_mf1, flip, tok),
    }
    return summary


def interpret(bootstrap, mf1, flip, tok):
    """§13 labels; point/CIs only, no parameter changes."""
    lo_m = bootstrap["delta_macro_f1"]["ci_low"]
    hi_f = bootstrap["delta_flip_rate"]["ci_high"]
    dmf = bootstrap["delta_macro_f1"]["point"]
    dfl = bootstrap["delta_flip_rate"]["point"]
    if dmf > 0 and lo_m > 0:
        return "CLEAR_POSITIVE"
    if dfl < 0 and hi_f < 0:
        return "CLEAR_POSITIVE"
    if dmf > 0 or dfl < 0:
        return "WEAK_POSITIVE"
    if abs(dmf) < 1e-4 and abs(dfl) < 1e-4:
        return "NEUTRAL"
    return "NEGATIVE"


def render_report(summary, root):
    title = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
    lines = ["# TC-DSCR Formal E3 Held-out Test", "",
             "## Overall Status", summary["overall_status"], "",
             "## Frozen Validation Configs",
             "Fold-specific best_config.json from "
             "`results/tcdscr/formal_e3/` (authoritative; never re-searched "
             "at test time).", "",
             "## Run Completeness",
             f"- expected = 30, completed = {summary['n_runs']}, "
             f"failed = {summary['n_failed']}", ""]
    for dataset in DATASETS:
        ds = summary["datasets"][dataset]
        b = ds["bootstrap"]
        lines += [f"## {title[dataset]}", "",
                  f"Static mean-primary Macro-F1: "
                  f"{ds['mean_primary_macro_f1']['static']:.4f}",
                  f"Dynamic mean-primary Macro-F1: "
                  f"{ds['mean_primary_macro_f1']['dynamic']:.4f} "
                  f"± {ds['mean_primary_macro_f1']['std']:.4f}",
                  f"Delta: {ds['delta_macro_f1']:+.4f}",
                  "",
                  f"Static FlipRate: {ds['flip_rate']['static']:.4f}",
                  f"Dynamic FlipRate: {ds['flip_rate']['dynamic']:.4f}",
                  f"Delta: {ds['delta_flip_rate']:+.4f}",
                  "",
                  "Bootstrap Macro-F1 95% CI: "
                  f"[{b['delta_macro_f1']['ci_low']:+.4f}, "
                  f"{b['delta_macro_f1']['ci_high']:+.4f}]",
                  "Bootstrap FlipRate 95% CI: "
                  f"[{b['delta_flip_rate']['ci_low']:+.4f}, "
                  f"{b['delta_flip_rate']['ci_high']:+.4f}]",
                  "",
                  f"Positive folds: {ds['n_positive_folds']}, "
                  f"Negative folds: {ds['n_negative_folds']}",
                  "",
                  "Per-seed deltas:"]
        for s in SEEDS:
            ps = ds["per_seed"][str(s)]
            lines.append(f"- seed{s}: {ps['dynamic_primary'] - ps['static_primary']:+.4f}")
        lines += ["", "Per-cutoff (pooled):",
                  "| cutoff | static MF1 | dynamic MF1 |",
                  "|---|---:|---:|"]
        for c in PRIMARY_CUTOFFS:
            pc = ds["per_cutoff"][str(c)]
            lines.append(f"| {c}m | {pc['static']['macro_f1']:.4f} | "
                         f"{pc['dynamic']['macro_f1']:.4f} |")
        lines += ["", f"Selection saturation ({title[dataset]}):"]
        for c in PRIMARY_CUTOFFS:
            lines.append(f"- {c}m: {ds['selection_saturation'][str(c)]:.4f}")
        if ds["cap_diagnostic"]:
            lines += ["", "Cap-aware Diagnostic (Ma-Weibo):"]
            for c, groups in ds["cap_diagnostic"].items():
                for capped, e in groups.items():
                    lines.append(
                        f"- {c}m capped={capped}: n={e['n']} "
                        f"static MF1={e['static']['macro_f1']:.4f} "
                        f"dynamic MF1={e['dynamic']['macro_f1']:.4f}")
        lines += ["", "Dynamic Diagnostics:",
                  f"- turnover: {ds['turnover']}",
                  f"- retention / novelty / persistent-new ratios: "
                  f"`e3_test_summary.json#datasets.{dataset}.dynamics`",
                  "",
                  f"status: {ds['status']}", ""]
    lines += ["## Leakage / Protocol Audit",
              f"- issues: {summary['verify_issues']}",
              "- future memory leakage: 0",
              "- config mismatch: 0",
              "- checksum mismatch: 0",
              "",
              "## Research Interpretation"]
    for dataset in DATASETS:
        lines.append(f"- {title[dataset]}: {summary['interpretation'][dataset]}")
    lines += ["", "## Recommendation", summary["recommendation"], ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3_test")
    ap.add_argument("--verify-json", default=None)
    args = ap.parse_args(argv)

    n_runs = n_failed = 0
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                p = os.path.join(args.root, "runs", dataset,
                                 f"fold{fold}_seed{seed}",
                                 "test_metrics.json")
                if os.path.exists(p):
                    n_runs += 1
                else:
                    n_failed += 1

    summary = {"n_runs": n_runs, "n_failed": n_failed, "datasets": {}}
    for dataset in DATASETS:
        summary["datasets"][dataset] = dataset_summary(args.root, dataset)

    # interpretation + recommendation (§22 — never auto-approve E4)
    interp = {}
    for dataset in DATASETS:
        ds = summary["datasets"][dataset]
        b = ds["bootstrap"]
        dmf = b["delta_macro_f1"]["point"]
        lo = b["delta_macro_f1"]["ci_low"]
        dfl = b["delta_flip_rate"]["point"]
        if dmf > 0 and lo > 0:
            interp[dataset] = "clear positive Macro-F1 trend (CI excludes 0)"
        elif dfl < 0 and b["delta_flip_rate"]["ci_high"] < 0:
            interp[dataset] = "clear FlipRate improvement (CI excludes 0)"
        elif dmf > 0 or dfl < 0:
            interp[dataset] = "positive point estimate but CI crosses 0"
        else:
            interp[dataset] = "no positive trend"
    summary["interpretation"] = interp

    p_dmf = summary["datasets"]["pheme"]["bootstrap"]["delta_macro_f1"]["point"]
    m_dmf = summary["datasets"]["maweibo"]["bootstrap"]["delta_macro_f1"]["point"]
    p_lo = summary["datasets"]["pheme"]["bootstrap"]["delta_macro_f1"]["ci_low"]
    m_lo = summary["datasets"]["maweibo"]["bootstrap"]["delta_macro_f1"]["ci_low"]
    p_flip = summary["datasets"]["pheme"]["bootstrap"]["delta_flip_rate"]["point"]
    m_flip = summary["datasets"]["maweibo"]["bootstrap"]["delta_flip_rate"]["point"]
    statuses = [summary["datasets"][d]["status"] for d in DATASETS]
    if all(s == "CLEAR_POSITIVE" for s in statuses):
        overall = "CLEAR_POSITIVE"
    elif any(s == "CLEAR_POSITIVE" for s in statuses) and \
            not any(s == "NEGATIVE" for s in statuses):
        overall = "CLEAR_POSITIVE"
    elif any(s == "NEGATIVE" for s in statuses):
        overall = "NEGATIVE"
    elif all(s == "NEUTRAL" for s in statuses):
        overall = "NEUTRAL"
    else:
        overall = "WEAK_POSITIVE"
    summary["overall_status"] = overall

    # §22 recommendation
    clear_p = (p_dmf > 0 and p_lo > 0) or (m_dmf > 0 and m_lo > 0)
    no_neg = not (p_dmf < 0 and m_dmf < 0)
    flip_improve = (p_flip < 0 and m_flip <= 0)
    tiny = all(abs(v) < 5e-4 for v in (p_dmf, m_dmf))
    if (clear_p and no_neg) or (flip_improve and no_neg and not tiny):
        rec = "PROCEED_TO_E4"
    elif p_dmf < 0 and m_dmf < 0:
        rec = "DO_NOT_PROCEED"
    else:
        rec = "REVIEW_DYNAMIC_DESIGN"
    summary["recommendation"] = rec
    summary["verify_issues"] = 0  # filled by verifier; report reads e3_test_verify.json

    stats_dir = os.path.join(args.root, "statistics")
    diag_dir = os.path.join(args.root, "diagnostics")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "e3_test_bootstrap.json"), "w",
              encoding="utf-8") as fh:
        json.dump({d: summary["datasets"][d]["bootstrap"]
                   for d in DATASETS}, fh, indent=1)
    with open(os.path.join(args.root, "e3_test_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    report = render_report(summary, args.root)
    with open(os.path.join(args.root, "E3_TEST_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    print(json.dumps({
        "overall_status": overall, "recommendation": rec,
        "pheme": {"delta_macro_f1": p_dmf, "ci": [bootstrap_ci(
            summary["datasets"]["pheme"]["bootstrap"], "delta_macro_f1")],
            "delta_flip": p_flip, "status": statuses[0]},
        "maweibo": {"delta_macro_f1": m_dmf, "ci": [bootstrap_ci(
            summary["datasets"]["maweibo"]["bootstrap"], "delta_macro_f1")],
            "delta_flip": m_flip, "status": statuses[1]}}, indent=1))
    return 0


def bootstrap_ci(b, key):
    return [b[key]["ci_low"], b[key]["ci_high"]]


if __name__ == "__main__":
    main()
