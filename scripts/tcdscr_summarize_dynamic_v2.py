#!/usr/bin/env python
"""Dynamic V2 (MF-TSR) validation aggregation (execution protocol §26,
§34-§35, §41, §44).

Reads results/tcdscr/dynamic_v2/runs/<dataset>/fold<f>_seed<s>/ produced by
tcdscr_run_dynamic_v2.py and writes, per dataset x fold, the fold-local
epsilon choice (chosen on validation only, three seeds pooled; test is
never consulted), the frozen best_config.json, the fold deliverables of
§41, stage/pressure diagnostics, the dataset-level readiness gate and the
MF_TSR_VALIDATION_REPORT.md.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_run_e2 import PRIMARY_CUTOFFS  # noqa: E402
from tcdscr_run_dynamic_v2 import (BUDGET_TOKENS, DATASETS, FOLDS,  # noqa: E402
                                   SEEDS)
from tcdscr.models.marginal_set_refiner import EPSILON_GRID  # noqa: E402

EPSILON_ORDER = tuple(str(e) for e in EPSILON_GRID)

DEFAULT_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v2"

DISPLAY_NAME = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
STAGE_CUTOFFS = {"Very Early": ("5", "15"), "Early": ("30", "60"),
                 "Mid": ("180", "360")}
# selection-pressure bins fixed since E3 (saturation = selected/candidates)
PRESSURE_BINS = (("high", lambda s: s < 0.4),
                 ("medium", lambda s: 0.4 <= s < 0.8),
                 ("low", lambda s: s >= 0.8))


def pick_fold_epsilon(per_seed):
    """§26 fold-local epsilon selection, pure function.

    ``per_seed``: {seed: {"static": {"mean_primary_macro_f1": x,
    "flip_rate": y}, "mf_tsr": {"mean_primary_macro_f1": ...,
    "flip_rate": ..., "mean_moves_per_snapshot": ...}}} keyed by str(eps).
    Returns {epsilon: chosen epsilon float} per fold (single fold input ->
    one float).  Rule: highest pooled MF-TSR mean-primary Macro-F1 wins;
    ties (<= 1e-9) break on lower MF-TSR flip rate, then fewer accepted
    moves per snapshot, then the smaller epsilon.  Deterministic.
    """
    eps_stats = {}
    for eps in per_seed:
        seeds = per_seed[eps]
        n = len(seeds)
        mf = sum(s["mean_primary_macro_f1"]["mf_tsr"]
                 for s in seeds) / n
        flip = sum(s["flip_rate"]["mf_tsr"] for s in seeds) / n
        moves = sum(s["set_mechanism"]["mean_moves_per_snapshot"]
                    for s in seeds) / n
        eps_stats[eps] = (mf, flip, moves)
    best = min(eps_stats, key=lambda e: (-eps_stats[e][0], eps_stats[e][1],
                                         eps_stats[e][2], float(e)))
    return best


def load_run(root, dataset, fold, seed):
    rd = os.path.join(root, "runs", dataset, f"fold{fold}_seed{seed}")
    with open(os.path.join(rd, "grid_metrics.json"), encoding="utf-8") as fh:
        grid = json.load(fh)
    rows = []
    with open(os.path.join(rd, "validation_predictions.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return grid, rows


def gate_verdict(delta_mf1, flip_static, flip_mf_tsr):
    """§34 new validation gate: classification path OR stability path."""
    classification = delta_mf1 >= 0.005
    rel_red = ((flip_static - flip_mf_tsr) / flip_static
               if flip_static > 0 else 0.0)
    stability = rel_red >= 0.10 and delta_mf1 >= -0.002
    if classification:
        return "PASS", "classification", rel_red
    if stability:
        return "PASS", "stability", rel_red
    return "FAIL", "MF_TSR_NOT_SUPPORTED", rel_red


def dataset_decision(dataset_verdicts):
    n_pass = sum(1 for v in dataset_verdicts.values() if v["gate"] == "PASS")
    if n_pass == len(dataset_verdicts):
        return "PASS", "START_MF_TSR_TEST"
    if n_pass == 0:
        return "FAIL", "REDESIGN_OR_REMOVE_DYNAMIC"
    return "PARTIAL", "STOP_FOR_RESEARCH_REVIEW"


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def fold_diagnostics(rows_best):
    """Aggregate set-mechanism + distortion metrics from prediction rows."""
    exact = jac = moves = fb = hits = 0
    gains, d_st, d_dyn, surv = [], [], [], []
    add_c = rm_c = sw_c = 0
    by_cut = {str(c): {"exact": [], "jac": [], "st": [], "dyn": []}
              for c in PRIMARY_CUTOFFS}
    for r in rows_best:
        s = set(r["static_selected_node_ids"])
        d = set(r["mf_tsr_selected_node_ids"])
        e = 1.0 if s == d else 0.0
        exact += e
        j = (len(s & d) / len(s | d)) if (s | d) else 1.0
        jac += j
        moves += len(r["accepted_moves"])
        gains.extend(x["relative_gain"] for x in r["accepted_moves"])
        fb += int(r["fallback_to_static"])
        hits += int(r.get("max_step_hit", False))
        for x in r["accepted_moves"]:
            if x["move_type"] == "ADD":
                add_c += 1
            elif x["move_type"] == "REMOVE":
                rm_c += 1
            else:
                sw_c += 1
        d_st.append(r["static_distortion"])
        d_dyn.append(r["mf_tsr_distortion"])
        prev = set(r["memory_previous_ids"])
        if prev:
            surv.append(len(prev & d) / len(prev))
        bc = by_cut.get(r["cutoff"])
        if bc is not None:
            bc["exact"].append(e)
            bc["jac"].append(j)
            bc["st"].append(r["static_distortion"])
            bc["dyn"].append(r["mf_tsr_distortion"])
    n = max(len(rows_best), 1)
    return {
        "n_snapshots": n,
        "exact_match_rate": exact / n,
        "mean_jaccard": jac / n,
        "add_count": add_c, "remove_count": rm_c, "swap_count": sw_c,
        "mean_moves_per_snapshot": moves / n,
        "mean_accepted_relative_gain": mean(gains),
        "fallback_to_static_rate": fb / n,
        "max_step_hit_rate": hits / n,
        "memory_survival_rate": mean(surv),
        "static_mean_distortion": mean(d_st),
        "mf_tsr_mean_distortion": mean(d_dyn),
        "distortion_reduction": mean(d_st) - mean(d_dyn),
        "per_cutoff": {c: {
            "exact_match_rate": mean(v["exact"]),
            "mean_jaccard": mean(v["jac"]),
            "static_mean_distortion": mean(v["st"]),
            "mf_tsr_mean_distortion": mean(v["dyn"]),
            "distortion_reduction": mean(v["st"]) - mean(v["dyn"]),
            "n": len(v["st"])} for c, v in by_cut.items()},
    }


def stage_pressure_summary(rows_best, grid):
    """§33 per-stage/per-pressure Static vs MF-TSR macro-F1 + set-change."""
    mf1_st = {c: grid["static_metrics_per_cutoff"][c] for c in
              grid["static_metrics_per_cutoff"]}
    mf1_dyn = grid["mf_tsr_metrics_per_cutoff"]
    stages = {}
    rows_cut = {str(c): [r for r in rows_best if r["cutoff"] == str(c)]
                for c in PRIMARY_CUTOFFS}
    for stage, cuts in STAGE_CUTOFFS.items():
        present = [c for c in cuts if c in mf1_st]
        if not present:
            continue
        st = mean([mf1_st[c]["macro_f1"] for c in present])
        dy = mean([mf1_dyn[c]["macro_f1"] for c in present])
        ch, nch = 0, 0
        for c in present:
            for r in rows_cut[c]:
                nch += 1
                if set(r["static_selected_node_ids"]) != \
                        set(r["mf_tsr_selected_node_ids"]):
                    ch += 1
        stages[stage] = {"static_macro_f1": st, "mf_tsr_macro_f1": dy,
                         "delta": dy - st,
                         "set_change_rate": ch / nch if nch else 0.0,
                         "n_snapshots": nch}
    pressure = {}
    for r in rows_best:
        sat = (len(r["mf_tsr_selected_node_ids"]) / r["candidate_count"]
               if r["candidate_count"] else 1.0)
        for name, pred in PRESSURE_BINS:
            if pred(sat):
                pressure.setdefault(name, []).append(r)
                break
    out_pressure = {}
    for name, rr in pressure.items():
        ch = sum(1 for r in rr if set(r["static_selected_node_ids"]) !=
                 set(r["mf_tsr_selected_node_ids"]))
        st_pairs = [(r["gold"], r["static_prediction"]) for r in rr]
        dy_pairs = [(r["gold"], r["mf_tsr_prediction"]) for r in rr]
        from tcdscr_run_e2 import classification_metrics
        out_pressure[name] = {
            "n_snapshots": len(rr),
            "static_macro_f1": classification_metrics(st_pairs)["macro_f1"],
            "mf_tsr_macro_f1": classification_metrics(dy_pairs)["macro_f1"],
            "delta": (classification_metrics(dy_pairs)["macro_f1"]
                      - classification_metrics(st_pairs)["macro_f1"]),
            "set_change_rate": ch / len(rr)}
    return {"stages": stages, "pressure": out_pressure}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    root = args.root
    readiness = {"protocol": "all validation events at every cutoff "
                             "(candidate_count = 0 included)",
                 "budget": BUDGET_TOKENS,
                 "epsilon_grid": list(EPSILON_ORDER),
                 "datasets": {}, "diagnostics": {}}
    for dataset in DATASETS:
        ds_out = {"folds": {}, "runs": {}}
        agg = {"static_mf1": [], "mfts_mf1": [], "flip_st": 0,
               "flip_dyn": 0, "n_trans": 0, "static_mf1_by_seed": {},
               "stage": [], "pressure": [], "tok_st": [], "tok_dyn": []}
        for fold in FOLDS:
            grids, rows_by_seed = {}, {}
            for seed in SEEDS:
                grid, rows = load_run(root, dataset, fold, seed)
                grids[seed] = grid
                rows_by_seed[seed] = rows
            per_seed_eps = {}
            for eps in EPSILON_ORDER:
                per_seed_eps[eps] = [grids[s][eps] for s in SEEDS]
            best_eps = pick_fold_epsilon(per_seed_eps)
            fold_dir = os.path.join(root, dataset, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "epsilon_grid.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"fold": fold, "dataset": dataset,
                           "candidates": {
                               eps: {
                                   "mf_tsr_mean_primary_macro_f1": mean([
                                       grids[s][eps]["mean_primary_macro_f1"]
                                       ["mf_tsr"] for s in SEEDS]),
                                   "static_mean_primary_macro_f1": mean([
                                       grids[s][eps]["mean_primary_macro_f1"]
                                       ["static"] for s in SEEDS]),
                                   "mf_tsr_flip_rate": mean([
                                       grids[s][eps]["flip_rate"]
                                       ["mf_tsr"] for s in SEEDS]),
                                   "mean_moves_per_snapshot": mean([
                                       grids[s][eps]["set_mechanism"]
                                       ["mean_moves_per_snapshot"]
                                       for s in SEEDS])}
                               for eps in per_seed_eps},
                           "selected_epsilon": float(best_eps),
                           "rule": "validation-only; highest pooled MF-TSR "
                                   "mean-primary Macro-F1, ties -> lower "
                                   "flip rate -> fewer moves -> smaller "
                                   "epsilon"}, fh, indent=1)
            best_rows = []
            flip_stats = {"static": 0, "mf_tsr": 0, "n": 0}
            seed_metrics = {}
            for seed in SEEDS:
                g = grids[seed][best_eps]
                seed_metrics[str(seed)] = {
                    "static_mean_primary":
                        g["mean_primary_macro_f1"]["static"],
                    "mf_tsr_mean_primary":
                        g["mean_primary_macro_f1"]["mf_tsr"],
                    "flip_rate": g["flip_rate"]}
                agg["static_mf1"].append(g["mean_primary_macro_f1"]["static"])
                agg["mfts_mf1"].append(
                    g["mean_primary_macro_f1"]["mf_tsr"])
                agg["tok_st"].append(
                    g["budget"]["mean_evidence_tokens"]["static"])
                agg["tok_dyn"].append(
                    g["budget"]["mean_evidence_tokens"]["mf_tsr"])
                for r in rows_by_seed[seed]:
                    if str(r["epsilon"]) == best_eps:
                        best_rows.append(r)
                # flips need transition counts per seed
                by_ev = {}
                for r in rows_by_seed[seed]:
                    if str(r["epsilon"]) != best_eps:
                        continue
                    by_ev.setdefault(r["event_id"], []).append(r)
                for evr in by_ev.values():
                    evr.sort(key=lambda x: PRIMARY_CUTOFFS.index(
                        int(x["cutoff"])))
                    for i in range(1, len(evr)):
                        flip_stats["n"] += 1
                        if evr[i - 1]["static_prediction"] != \
                                evr[i]["static_prediction"]:
                            flip_stats["static"] += 1
                        if evr[i - 1]["mf_tsr_prediction"] != \
                                evr[i]["mf_tsr_prediction"]:
                            flip_stats["mf_tsr"] += 1
            agg["flip_st"] += flip_stats["static"]
            agg["flip_dyn"] += flip_stats["mf_tsr"]
            agg["n_trans"] += flip_stats["n"]
            static_by_cut = {}
            mfts_by_cut = {}
            for c in PRIMARY_CUTOFFS:
                vals_st, vals_dy = [], []
                for seed in SEEDS:
                    m = grids[seed][best_eps]["per_cutoff"][str(c)]
                    vals_st.append(m["static"]["macro_f1"])
                    vals_dy.append(m["mf_tsr"]["macro_f1"])
                static_by_cut[str(c)] = {"macro_f1": mean(vals_st)}
                mfts_by_cut[str(c)] = {"macro_f1": mean(vals_dy)}
            # stage/pressure diagnostics need grid-like structure
            diag_grid = {"static_metrics_per_cutoff": static_by_cut,
                         "mf_tsr_metrics_per_cutoff": mfts_by_cut}
            sp = stage_pressure_summary(best_rows, diag_grid)
            agg["stage"].append(sp["stages"])
            agg["pressure"].append(sp["pressure"])
            with open(os.path.join(fold_dir,
                                   "validation_predictions.jsonl"), "w",
                      encoding="utf-8") as fh:
                for r in best_rows:
                    fh.write(json.dumps(r) + "\n")
            with open(os.path.join(fold_dir, "set_dynamics.json"), "w",
                      encoding="utf-8") as fh:
                dyn = fold_diagnostics(best_rows)
                dyn["stage_pressure"] = sp
                json.dump(dyn, fh, indent=1)
            with open(os.path.join(fold_dir, "distortion_metrics.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"fold": fold, "dataset": dataset,
                           "epsilon": float(best_eps),
                           "per_seed": seed_metrics,
                           "static_mean_primary_macro_f1":
                               mean(agg["static_mf1"][-3:]),
                           "mf_tsr_mean_primary_macro_f1":
                               mean(agg["mfts_mf1"][-3:])}, fh, indent=1)
            bc = {
                "dataset": dataset, "fold": fold, "budget": BUDGET_TOKENS,
                "epsilon": float(best_eps),
                "validation_static_macro_f1": mean(
                    agg["static_mf1"][-3:]),
                "validation_mf_tsr_macro_f1": mean(
                    agg["mfts_mf1"][-3:]),
                "static_flip_rate": (flip_stats["static"]
                                     / max(flip_stats["n"], 1)),
                "mf_tsr_flip_rate": (flip_stats["mf_tsr"]
                                     / max(flip_stats["n"], 1)),
                "selection_scope": "validation-only (fold-local, §26)",
            }
            with open(os.path.join(fold_dir, "best_config.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(bc, fh, indent=1)
            ds_out["folds"][f"fold{fold}"] = bc
        ds_out["mean_static_macro_f1"] = mean(agg["static_mf1"])
        ds_out["mean_mf_tsr_macro_f1"] = mean(agg["mfts_mf1"])
        ds_out["delta_macro_f1"] = (ds_out["mean_mf_tsr_macro_f1"]
                                    - ds_out["mean_static_macro_f1"])
        ds_out["mean_static_evidence_tokens"] = mean(agg["tok_st"])
        ds_out["mean_mf_tsr_evidence_tokens"] = mean(agg["tok_dyn"])
        ds_out["flip_rate_static"] = agg["flip_st"] / max(agg["n_trans"], 1)
        ds_out["flip_rate_mf_tsr"] = agg["flip_dyn"] / max(agg["n_trans"], 1)
        gate, path, rel = gate_verdict(ds_out["delta_macro_f1"],
                                       ds_out["flip_rate_static"],
                                       ds_out["flip_rate_mf_tsr"])
        ds_out["relative_flip_reduction"] = rel
        ds_out["gate"] = gate
        ds_out["gate_path"] = path
        readiness["datasets"][dataset] = ds_out
        # pooled stage/pressure across folds
        pooled_stage, pooled_press = {}, {}
        for st_list in agg["stage"]:
            for k, v in st_list.items():
                pooled_stage.setdefault(k, []).append(v)
        for pr_list in agg["pressure"]:
            for k, v in pr_list.items():
                pooled_press.setdefault(k, []).append(v)
        readiness["diagnostics"][dataset] = {
            "stages": {k: {
                "static_macro_f1": mean([x["static_macro_f1"] for x in vs]),
                "mf_tsr_macro_f1": mean([x["mf_tsr_macro_f1"] for x in vs]),
                "delta": mean([x["delta"] for x in vs]),
                "set_change_rate": mean([x["set_change_rate"]
                                         for x in vs])}
                for k, vs in pooled_stage.items()},
            "pressure": {k: {
                "static_macro_f1": mean([x["static_macro_f1"] for x in vs]),
                "mf_tsr_macro_f1": mean([x["mf_tsr_macro_f1"] for x in vs]),
                "delta": mean([x["delta"] for x in vs]),
                "set_change_rate": mean([x["set_change_rate"] for x in vs]),
                "n_snapshots": sum(x["n_snapshots"] for x in vs)}
                for k, vs in pooled_press.items()}}
    overall, rec = dataset_decision(readiness["datasets"])
    readiness["overall"] = overall
    readiness["recommendation"] = rec
    os.makedirs(os.path.join(root, "diagnostics"), exist_ok=True)
    for dataset in DATASETS:
        with open(os.path.join(root, "diagnostics",
                               f"{dataset}_stage_pressure.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(readiness["diagnostics"][dataset], fh, indent=1)
    os.makedirs(os.path.join(root, "readiness"), exist_ok=True)
    with open(os.path.join(root, "readiness", "mf_tsr_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(readiness, fh, indent=1)
    with open(os.path.join(root, "mf_tsr_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(readiness, fh, indent=1)
    write_report(root, readiness)
    print(json.dumps({"overall": overall, "recommendation": rec,
                      "datasets": {d: {
                          "delta": round(v["delta_macro_f1"], 5),
                          "gate": v["gate"], "path": v["gate_path"],
                          "rel_flip_reduction":
                              round(v["relative_flip_reduction"], 4)}
                          for d, v in readiness["datasets"].items()}},
                     indent=1), flush=True)
    return 0


def write_report(root, readiness):
    lines = ["# TC-DSCR Dynamic V2 — MF-TSR Validation", ""]
    lines += ["## Protocol Correction",
              "- scope: all validation events at every cutoff "
              "(candidate_count = 0 included)",
              "- corrected E2 impact: see "
              "results/tcdscr/dynamic_v2_protocol/"
              "e2_corrected_all_event_metrics.md",
              "- old E3 V1 all-event diagnostic: see results/tcdscr/"
              "dynamic_v2_protocol/e3_v1_all_event_diagnostic.json", ""]
    lines += ["## Implementation", "- encoder frozen: True (corrected E2 "
              "provenance SHA-checked)", "- selector frozen: True",
              "- proxy frozen: True", "- new trainable parameters: NONE",
              "- budget: 1024 (frozen)", ""]
    for dataset in DATASETS:
        d = readiness["datasets"][dataset]
        lines += [f"## {DISPLAY_NAME[dataset]}", "", "### Fold epsilon",
                  "| fold | epsilon |", "|---|---:|"]
        for f in FOLDS:
            bc = d["folds"][f"fold{f}"]
            lines.append(f"| {f} | {bc['epsilon']} |")
        lines += ["", "### Validation",
                  "| method | Macro-F1 | FlipRate | tokens |",
                  "|---|---:|---:|---:|",
                  f"| Static | {d['mean_static_macro_f1']:.4f} | "
                  f"{d['flip_rate_static']:.4f} | "
                  f"{d['mean_static_evidence_tokens']:.1f} |",
                  f"| MF-TSR | {d['mean_mf_tsr_macro_f1']:.4f} | "
                  f"{d['flip_rate_mf_tsr']:.4f} | "
                  f"{d['mean_mf_tsr_evidence_tokens']:.1f} |", "",
                  f"delta: {d['delta_macro_f1']:+.5f}",
                  f"relative flip reduction: "
                  f"{d['relative_flip_reduction']:.4f}",
                  f"gate: {d['gate']} ({d['gate_path']})",
                  f"effect: mean-primary Macro-F1 over 15 runs, "
                  f"all-event protocol", ""]
        lines += ["### Mechanism"]
        meas = []
        for f in FOLDS:
            p = os.path.join(root, dataset, f"fold{f}",
                             "set_dynamics.json")
            with open(p, encoding="utf-8") as fh:
                meas.append(json.load(fh))
        m = lambda k: mean([x[k] for x in meas])  # noqa: E731
        lines += [f"- exact match: {m('exact_match_rate'):.4f}",
                  f"- Jaccard: {m('mean_jaccard'):.4f}",
                  f"- ADD: {sum(x['add_count'] for x in meas)}",
                  f"- REMOVE: {sum(x['remove_count'] for x in meas)}",
                  f"- SWAP: {sum(x['swap_count'] for x in meas)}",
                  f"- fallback rate: {m('fallback_to_static_rate'):.4f}",
                  f"- distortion reduction: "
                  f"{m('distortion_reduction'):+.5f}",
                  f"- mean moves/snapshot: {m('mean_moves_per_snapshot'):.2f}",
                  f"- memory survival rate: "
                  f"{m('memory_survival_rate'):.4f}", ""]
    lines += ["## Stage-wise"]
    for dataset in DATASETS:
        lines.append(f"### {DISPLAY_NAME[dataset]}")
        lines += ["| stage | Static MF1 | MF-TSR MF1 | delta | "
                  "set-change |", "|---|---:|---:|---:|---:|"]
        for k, v in readiness["diagnostics"][dataset]["stages"].items():
            lines.append(f"| {k} | {v['static_macro_f1']:.4f} | "
                         f"{v['mf_tsr_macro_f1']:.4f} | "
                         f"{v['delta']:+.4f} | "
                         f"{v['set_change_rate']:.4f} |")
        lines.append("")
    lines += ["## Selection-pressure"]
    for dataset in DATASETS:
        lines.append(f"### {DISPLAY_NAME[dataset]}")
        lines += ["| bin | Static MF1 | MF-TSR MF1 | delta | set-change |",
                  "|---|---:|---:|---:|---:|"]
        for k in ("low", "medium", "high"):
            v = readiness["diagnostics"][dataset]["pressure"].get(k)
            if not v:
                continue
            lines.append(f"| {k} | {v['static_macro_f1']:.4f} | "
                         f"{v['mf_tsr_macro_f1']:.4f} | "
                         f"{v['delta']:+.4f} | "
                         f"{v['set_change_rate']:.4f} |")
        lines.append("")
    v_path = os.path.join(root, "mf_tsr_verify.json")
    n_issues = "not run"
    v_detail = ""
    if os.path.exists(v_path):
        with open(v_path, encoding="utf-8") as fh:
            v = json.load(fh)
        n_issues = v.get("n_issues")
        v_detail = (f" (runs={v.get('runs')}, rows={v.get('rows')}, "
                    f"no_candidate_rows={v.get('no_candidate_rows')})")
    lines += ["## Verifier", f"issues = {n_issues}{v_detail}", "",
              "## Recommendation", f"Overall = {readiness['overall']}",
              f"Recommendation = {readiness['recommendation']}", ""]
    with open(os.path.join(root, "MF_TSR_VALIDATION_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
