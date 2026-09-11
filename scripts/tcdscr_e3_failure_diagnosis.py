#!/usr/bin/env python
"""TC-DSCR E3 Failure Diagnosis + Bootstrap Fix (E3 diagnosis order §1-§22).

Reads ONLY the frozen E3 validation/test predictions (formal_e3/,
formal_e3_test/ — never modified) and re-computes:

1. paired bootstrap with preserved multiplicity (the previous dict-based
   implementation collapsed duplicate resampled events, invalidating the
   CIs — model predictions and point estimates were unaffected);
2. diagnosis Q1-Q11 and the parameter-necessity audit:
   Q1 selection change, Q2 selection->prediction effect,
   Q3 proxy output change (post-hoc frozen inference, no re-selection),
   Q4 score scale, Q5 ranking change, Q6 budget masking,
   Q7 persistence effect, Q8 novelty trade-off, Q9 stage-wise,
   Q10 selection-pressure stratification, Q11 validation->test stability.

No model is trained, no prediction is regenerated, no Qwen model is called.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"))

import torch  # noqa: E402

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,  # noqa: E402
                           classification_metrics, mean_primary_macro_f1)

from tcdscr_run_e3 import (FOLDS, SEEDS, DATASETS, collect_event_data,  # noqa: E402
                           load_frozen_components)
from tcdscr_run_e3_test import load_best_config  # noqa: E402

from tcdscr.evaluation.e3_diagnosis import (  # noqa: E402
    budget_masking_classification, paired_bootstrap, pressure_bin,
    prediction_transition_categories, resample, selection_change_metrics,
    selection_change_prediction_effect, selection_pressure_summary,
    spearman_rank_correlation, topk_set_changed)

E3_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3"
TEST_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3_test"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/e3_failure_diagnosis"

STAGES = {"very_early": ("5", "15"), "early": ("30", "60"),
          "mid": ("180", "360")}


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_test_rows(dataset):
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            p = os.path.join(TEST_ROOT, "runs", dataset,
                             f"fold{fold}_seed{seed}",
                             "test_predictions.jsonl")
            if os.path.exists(p):
                rows.extend(_rows(p))
    return rows


def event_units(rows):
    """{event_id: {"pairs": {cutoff: [(gold, static, dynamic)]},
                   "static_seq": [...], "dynamic_seq": [...]}}"""
    units = {}
    by_seed_event = {}
    for r in rows:
        u = units.setdefault(r["event_id"],
                             {"pairs": {}, "static_seq": [],
                              "dynamic_seq": []})
        u["pairs"].setdefault(r["cutoff"], []).append(
            (r["gold"], r["static_prediction"], r["dynamic_prediction"]))
        by_seed_event.setdefault((r["event_id"], r["seed"]), []).append(r)
    for (eid, _s), ev_rows in by_seed_event.items():
        ev_rows.sort(key=lambda r: int(r["cutoff"]))
        units[eid]["static_seq"].append([r["static_prediction"]
                                         for r in ev_rows])
        units[eid]["dynamic_seq"].append([r["dynamic_prediction"]
                                          for r in ev_rows])
    return units


def bootstrap_fixed(dataset):
    rows = load_test_rows(dataset)
    units = event_units(rows)
    return paired_bootstrap(units, n_iter=10000, seed=3090)


# ---------------------------------------------------------------------------
# Post-hoc frozen inference: candidate scores + proxy logits per run
# ---------------------------------------------------------------------------

def posthoc_run(dataset, fold, seed, device):
    """Recompute candidate scores and proxy logits for one frozen run.

    Reads the run's frozen config + test_predictions rows; re-forwards the
    frozen encoder/selector to obtain per-candidate u and node reprs, and
    the frozen proxy to classify the ALREADY-selected sets (never re-selects).
    Returns a per-(event, cutoff) records list.
    """
    import torch.nn.functional as F

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.dynamic_memory import (dynamic_scores,
                                              novelty_scores,
                                              persistence_flags)
    from tcdscr.models.selector_proxy import classify_selected
    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)

    config, _bc, _p = load_best_config(E3_ROOT, dataset, fold)
    lambda_n, lambda_p, budget = config

    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in events["test"]:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            items.append(build_light_item(ev, snap,
                                          store.get_store(ev, cache)))

    encoder, selector, proxy, _ck = load_frozen_components(
        dataset, fold, seed, E1_ROOT, E2_ROOT, device)
    by_event = collect_event_data(encoder, selector, items, device)

    rows = _rows(os.path.join(TEST_ROOT, "runs", dataset,
                              f"fold{fold}_seed{seed}",
                              "test_predictions.jsonl"))
    row_index = {}
    for r in rows:
        row_index[(r["event_id"], r["cutoff"])] = r

    records = []
    for eid, ev_rows in by_event.items():
        for er in ev_rows:
            c = str(er["cutoff"])
            r = row_index.get((eid, c))
            if r is None:
                continue
            cand_ids = er["cand_node_ids"]
            sem_c = er["sem_c"].float()
            u = er["u"].float()
            mem_ids = r["memory_previous_ids"]
            if mem_ids:
                idx = {nid: i for i, nid in enumerate(er["node_ids"])}
                mem_embs = torch.stack(
                    [er["sem_all"][idx[nid]] for nid in mem_ids])
            else:
                mem_embs = None
            nov = novelty_scores(sem_c, mem_embs)
            per = persistence_flags(cand_ids, mem_ids)
            d = dynamic_scores(u, nov, per, lambda_n, lambda_p)
            u_list = u.tolist()
            d_list = d.tolist()
            nov_list = nov.tolist()
            per_list = per.tolist()
            bonus = [lambda_n * a + lambda_p * b
                     for a, b in zip(nov_list, per_list)]
            nov_contrib = [lambda_n * a for a in nov_list]
            per_contrib = [lambda_p * b for b in per_list]

            order_u = sorted(range(len(cand_ids)),
                             key=lambda i: (-u_list[i], i))
            order_d = sorted(range(len(cand_ids)),
                             key=lambda i: (-d_list[i], i))
            ranked_u = [cand_ids[i] for i in order_u]
            ranked_d = [cand_ids[i] for i in order_d]

            node_repr = er["node_repr"]
            h_source = node_repr[er["src_pos"]]

            def classify(sel_ids):
                if not sel_ids:
                    return proxy.classify(
                        h_source, torch.zeros(768, device=device))
                idx = [er["units_index"][nid] for nid in sel_ids]
                return classify_selected(proxy, h_source, node_repr[idx])

            p_sta = F.softmax(classify(r["static_selected_node_ids"]),
                              dim=-1)[1].item()
            p_dyn = F.softmax(classify(r["dynamic_selected_node_ids"]),
                              dim=-1)[1].item()

            records.append({
                "event_id": eid, "cutoff": c,
                "gold": r["gold"],
                "static_prediction": r["static_prediction"],
                "dynamic_prediction": r["dynamic_prediction"],
                "p_static_rumor": p_sta,
                "p_dynamic_rumor": p_dyn,
                "cand_ids": cand_ids,
                "u": u_list, "novelty": nov_list,
                "persistence": per_list, "bonus": bonus,
                "nov_contrib": nov_contrib, "per_contrib": per_contrib,
                "ranked_static_ids": ranked_u, "ranked_dynamic_ids": ranked_d,
                "static_selected": r["static_selected_node_ids"],
                "dynamic_selected": r["dynamic_selected_node_ids"],
                "n_candidates": len(cand_ids),
                "static_tokens": r["static_evidence_tokens"],
                "dynamic_tokens": r["dynamic_evidence_tokens"],
                "budget": budget,
            })
    return records


def aggregate_posthoc(dataset, device):
    """Run post-hoc over all 15 (fold, seed) runs and aggregate Q3-Q8 stats."""
    q3_delta_p = []
    q3_margin_sta, q3_margin_dyn = [], []
    q4_u, q4_bonus, q4_nov, q4_per = [], [], [], []
    q5_spearman = []
    q5_top = {k: 0 for k in (1, 3, 5, 10)}
    q5_total = 0
    q6_rows = []
    q7_persist_sel = q7_persist_all = q7_new_sel = q7_new_all = 0
    q7_saved = []
    q8_dyn_only = []      # (novelty, u, d, cutoff)
    q8_removed_static = []  # (novelty, u, cutoff)
    per_cutoff_delta_p = {str(c): [] for c in PRIMARY_CUTOFFS}
    per_cutoff_margin = {str(c): {"sta": [], "dyn": []}
                         for c in PRIMARY_CUTOFFS}
    stage_rank = {name: [0, 0] for name in STAGES}
    stage_sel = {name: [0, 0] for name in STAGES}
    stage_pred = {name: [0, 0] for name in STAGES}

    for fold in FOLDS:
        for seed in SEEDS:
            for rec in posthoc_run(dataset, fold, seed, device):
                c = rec["cutoff"]
                dp = abs(rec["p_dynamic_rumor"] - rec["p_static_rumor"])
                q3_delta_p.append(dp)
                per_cutoff_delta_p[c].append(dp)
                q3_margin_sta.append(abs(rec["p_static_rumor"] - 0.5))
                q3_margin_dyn.append(abs(rec["p_dynamic_rumor"] - 0.5))
                per_cutoff_margin[c]["sta"].append(
                    abs(rec["p_static_rumor"] - 0.5))
                per_cutoff_margin[c]["dyn"].append(
                    abs(rec["p_dynamic_rumor"] - 0.5))

                q4_u.extend(rec["u"])
                q4_bonus.extend(rec["bonus"])
                q4_nov.extend(rec["nov_contrib"])
                q4_per.extend(rec["per_contrib"])

                rho_ud = spearman_rank_correlation(
                    rec["u"], [u + b for u, b in
                               zip(rec["u"], rec["bonus"])])
                q5_spearman.append(rho_ud)
                q5_total += 1
                for k in (1, 3, 5, 10):
                    changed = topk_set_changed(rec["ranked_static_ids"],
                                               rec["ranked_dynamic_ids"], k)
                    q5_top.setdefault(k, 0)
                    q5_top[k] += int(changed)

                # Q6 budget-masking record
                ranking_changed = rho_ud < 0.999999
                selection_changed = set(rec["static_selected"]) != \
                    set(rec["dynamic_selected"])
                q6_rows.append((ranking_changed, selection_changed,
                                rec["static_tokens"], rec["dynamic_tokens"],
                                rec["budget"], len(rec["dynamic_selected"]),
                                rec["n_candidates"]))

                # stage-wise counters (Q9)
                stage = ("very_early" if c in ("5", "15")
                         else "early" if c in ("30", "60") else "mid")
                stage_rank[stage][1] += 1
                if ranking_changed:
                    stage_rank[stage][0] += 1
                stage_sel[stage][1] += 1
                if selection_changed:
                    stage_sel[stage][0] += 1
                stage_pred[stage][1] += 1
                if rec["static_prediction"] != rec["dynamic_prediction"]:
                    stage_pred[stage][0] += 1

                # Q7 persistence effect (all folds; lambda_p may be 0)
                per_list = rec["persistence"]
                dyn_sel = set(rec["dynamic_selected"])
                sta_sel = set(rec["static_selected"])
                cand_ids = rec["cand_ids"]
                for i, nid in enumerate(cand_ids):
                    if per_list[i] == 1.0:
                        q7_persist_all += 1
                        if nid in dyn_sel:
                            q7_persist_sel += 1
                    else:
                        q7_new_all += 1
                        if nid in dyn_sel:
                            q7_new_sel += 1
                # persistence-saved: dyn selected, static not, persistence 1
                for i, nid in enumerate(cand_ids):
                    if nid in dyn_sel and nid not in sta_sel and \
                            per_list[i] == 1.0:
                        q7_saved.append((rec["gold"],
                                         rec["static_prediction"],
                                         rec["dynamic_prediction"]))

                # Q8 novelty trade-off: dynamic-only vs removed static
                idx_of = {nid: i for i, nid in enumerate(cand_ids)}
                for nid in (dyn_sel - sta_sel):
                    if nid in idx_of:
                        i = idx_of[nid]
                        q8_dyn_only.append({"novelty": rec["novelty"][i],
                                            "u": rec["u"][i],
                                            "bonus": rec["bonus"][i],
                                            "cutoff": c})
                for nid in (sta_sel - dyn_sel):
                    if nid in idx_of:
                        i = idx_of[nid]
                        q8_removed_static.append(
                            {"novelty": rec["novelty"][i],
                             "u": rec["u"][i], "cutoff": c})

    q4_ratio = (st.pstdev(q4_bonus) / st.pstdev(q4_u)
                if len(q4_u) > 1 and st.pstdev(q4_u) else 0.0)
    mean_abs_ratio = ((sum(abs(b) for b in q4_bonus) / len(q4_bonus))
                      / (sum(abs(x) for x in q4_u) / len(q4_u))
                      if q4_u else 0.0)
    q5_top_rates = {str(k): (q5_top[k] / q5_total if q5_total else 0.0)
                    for k in (1, 3, 5, 10)}

    def pct(vals, p):
        vals = sorted(vals)
        return vals[min(int(p * len(vals)), len(vals) - 1)] if vals else 0.0

    a = b = c = 0
    tot_tokens = {"static": [], "dynamic": []}
    util = []
    for ranking_changed, selection_changed, st_tok, dy_tok, budget, \
            n_sel, n_cand in q6_rows:
        if not ranking_changed and not selection_changed:
            a += 1
        elif ranking_changed and not selection_changed:
            b += 1
        else:
            c += 1
        tot_tokens["static"].append(st_tok)
        tot_tokens["dynamic"].append(dy_tok)
        util.append(dy_tok / budget if budget else 0.0)
    n6 = a + b + c

    q7 = {
        "P_select_given_persistent":
            q7_persist_sel / q7_persist_all if q7_persist_all else 0.0,
        "P_select_given_new": q7_new_sel / q7_new_all if q7_new_all else 0.0,
        "persistence_selection_lift":
            (q7_persist_sel / q7_persist_all if q7_persist_all else 0.0)
            - (q7_new_sel / q7_new_all if q7_new_all else 0.0),
        "n_persistent_candidates": q7_persist_all,
        "n_new_candidates": q7_new_all,
        "n_persistent_selected": q7_persist_sel,
        "n_new_selected": q7_new_sel,
        "persistence_saved_evidence": {
            "n": len(q7_saved),
            "transitions": prediction_transition_categories([
                {"gold": g, "static_prediction": sp,
                 "dynamic_prediction": dp,
                 "static_selected_node_ids": ["x"],
                 "dynamic_selected_node_ids": ["x", "y"]}
                for g, sp, dp in q7_saved]),
        },
    }

    dyn_only_nov = [x["novelty"] for x in q8_dyn_only]
    dyn_only_u = [x["u"] for x in q8_dyn_only]
    rem_nov = [x["novelty"] for x in q8_removed_static]
    rem_u = [x["u"] for x in q8_removed_static]
    q8 = {
        "n_dynamic_only_nodes": len(q8_dyn_only),
        "n_removed_static_nodes": len(q8_removed_static),
        "dynamic_only_mean_novelty": (st.mean(dyn_only_nov)
                                      if dyn_only_nov else 0.0),
        "removed_static_mean_novelty": (st.mean(rem_nov) if rem_nov else 0.0),
        "novelty_gain": ((st.mean(dyn_only_nov) if dyn_only_nov else 0.0)
                         - (st.mean(rem_nov) if rem_nov else 0.0)),
        "dynamic_only_mean_u": (st.mean(dyn_only_u) if dyn_only_u else 0.0),
        "removed_static_mean_u": (st.mean(rem_u) if rem_u else 0.0),
        "utility_delta": ((st.mean(dyn_only_u) if dyn_only_u else 0.0)
                          - (st.mean(rem_u) if rem_u else 0.0)),
    }

    return {
        "q3": {
            "n": len(q3_delta_p),
            "delta_p_mean": st.mean(q3_delta_p) if q3_delta_p else 0.0,
            "delta_p_median": pct(q3_delta_p, 0.5),
            "delta_p_p90": pct(q3_delta_p, 0.90),
            "delta_p_p99": pct(q3_delta_p, 0.99),
            "margin_static_mean": (st.mean(q3_margin_sta)
                                   if q3_margin_sta else 0.0),
            "margin_dynamic_mean": (st.mean(q3_margin_dyn)
                                    if q3_margin_dyn else 0.0),
            "per_cutoff": {c: {"delta_p_mean":
                               (st.mean(per_cutoff_delta_p[c])
                                if per_cutoff_delta_p[c] else 0.0),
                               "margin_static_mean":
                               (st.mean(per_cutoff_margin[c]["sta"])
                                if per_cutoff_margin[c]["sta"] else 0.0),
                               "margin_dynamic_mean":
                               (st.mean(per_cutoff_margin[c]["dyn"])
                                if per_cutoff_margin[c]["dyn"] else 0.0)}
                           for c in per_cutoff_delta_p},
        },
        "q4": {
            "u_mean": st.mean(q4_u) if q4_u else 0.0,
            "u_std": st.pstdev(q4_u) if q4_u else 0.0,
            "u_p10": pct(q4_u, 0.10),
            "u_p50": pct(q4_u, 0.50),
            "u_p90": pct(q4_u, 0.90),
            "bonus_mean": st.mean(q4_bonus) if q4_bonus else 0.0,
            "bonus_std": st.pstdev(q4_bonus) if q4_bonus else 0.0,
            "novelty_contrib_mean": st.mean(q4_nov) if q4_nov else 0.0,
            "persistence_contrib_mean": st.mean(q4_per) if q4_per else 0.0,
            "bonus_to_utility_std_ratio": q4_ratio,
            "mean_abs_bonus_over_mean_abs_u": mean_abs_ratio,
        },
        "q5": {
            "n": q5_total,
            "spearman_mean": (st.mean(q5_spearman)
                              if q5_spearman else 0.0),
            "top1_change_rate": q5_top_rates["1"],
            "top3_set_change_rate": q5_top_rates["3"],
            "top5_set_change_rate": q5_top_rates["5"],
            "top10_set_change_rate": q5_top_rates["10"],
        },
        "q6": {
            "A_unchanged": a, "B_ranking_changed_selection_same": b,
            "C_ranking_changed_selection_changed": c,
            "ranking_changed_but_selection_same_rate": b / n6 if n6 else 0.0,
            "mean_static_tokens": (st.mean(tot_tokens["static"])
                                   if tot_tokens["static"] else 0.0),
            "mean_dynamic_tokens": (st.mean(tot_tokens["dynamic"])
                                    if tot_tokens["dynamic"] else 0.0),
            "mean_budget_utilization": st.mean(util) if util else 0.0,
        },
        "q7": q7,
        "q8": q8,
        "stage_extra": {name: {
            "ranking_change_rate": (stage_rank[name][0] / stage_rank[name][1]
                                    if stage_rank[name][1] else 0.0),
            "selection_change_rate": (stage_sel[name][0] / stage_sel[name][1]
                                      if stage_sel[name][1] else 0.0),
            "prediction_change_rate": (stage_pred[name][0]
                                       / stage_pred[name][1]
                                       if stage_pred[name][1] else 0.0)}
            for name in STAGES},
    }


def selection_pressure_rows(rows):
    return selection_pressure_summary(rows)


def stagewise(rows):
    out = {}
    for name, cuts in STAGES.items():
        g = [r for r in rows if r["cutoff"] in cuts]
        if not g:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": len(g),
            "static_macro_f1": classification_metrics(
                [(r["gold"], r["static_prediction"]) for r in g]),
            "dynamic_macro_f1": classification_metrics(
                [(r["gold"], r["dynamic_prediction"]) for r in g]),
            "selection": selection_change_metrics(g),
            "ranking_change_rate": None,  # needs post-hoc; filled later
            "prediction_change_rate": sum(
                1 for r in g if r["static_prediction"] !=
                r["dynamic_prediction"]) / len(g),
        }
    return out


def validation_stability(dataset):
    folds = {}
    for fold in FOLDS:
        gp = os.path.join(E3_ROOT, dataset, f"fold{fold}",
                          "grid_results.json")
        bc = load_best_config(E3_ROOT, dataset, fold)[1]
        if not os.path.exists(gp):
            continue
        grid = json.load(open(gp, encoding="utf-8"))
        scores = {}
        for key, entry in grid.items():
            dyn = entry.get("mean_primary_macro_f1", {}).get("dynamic")
            if dyn is not None:
                scores[key] = dyn
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_key = ranked[0][0]
        gap = ranked[0][1] - ranked[4][1] if len(ranked) >= 5 else \
            ranked[0][1] - ranked[-1][1]
        test_rows = [r for r in load_test_rows(dataset) if r["fold"] == fold]
        test_delta = None
        if test_rows:
            test_delta = (mean_primary_macro_f1(
                {c: classification_metrics(
                    [(r["gold"], r["dynamic_prediction"]) for r in test_rows
                     if r["cutoff"] == c]) for c in map(str, PRIMARY_CUTOFFS)})
                - mean_primary_macro_f1(
                    {c: classification_metrics(
                        [(r["gold"], r["static_prediction"]) for r in test_rows
                         if r["cutoff"] == c]) for c in map(str, PRIMARY_CUTOFFS)}))
        folds[str(fold)] = {
            "best_config_key": best_key,
            "best_validation_dynamic_mf1": ranked[0][1],
            "top1_minus_top5_gap": gap,
            "validation_best_config": {
                "lambda_n": bc["lambda_n"], "lambda_p": bc["lambda_p"],
                "budget": bc["budget"]},
            "test_delta_macro_f1": test_delta,
            "flat_landscape": gap < 0.005 and test_delta is not None
            and abs(test_delta) < 0.005,
        }
    return folds


def parameter_necessity(dataset):
    cats = {"lambda_n_zero": [], "lambda_p_zero": [], "both_positive": [],
            "both_zero": []}
    for fold in FOLDS:
        bc = load_best_config(E3_ROOT, dataset, fold)[1]
        test_rows = [r for r in load_test_rows(dataset) if r["fold"] == fold]
        delta = None
        if test_rows:
            delta = (mean_primary_macro_f1(
                {c: classification_metrics(
                    [(r["gold"], r["dynamic_prediction"]) for r in test_rows
                     if r["cutoff"] == c]) for c in map(str, PRIMARY_CUTOFFS)})
                - mean_primary_macro_f1(
                    {c: classification_metrics(
                        [(r["gold"], r["static_prediction"]) for r in test_rows
                         if r["cutoff"] == c]) for c in map(str, PRIMARY_CUTOFFS)}))
        entry = {"fold": fold, "lambda_n": bc["lambda_n"],
                 "lambda_p": bc["lambda_p"], "budget": bc["budget"],
                 "test_delta_macro_f1": delta,
                 "selection_change_rate": (sum(
                     1 for r in test_rows if set(r["static_selected_node_ids"])
                     != set(r["dynamic_selected_node_ids"])) / len(test_rows)
                     if test_rows else 0.0)}
        if bc["lambda_n"] == 0 and bc["lambda_p"] == 0:
            cats["both_zero"].append(entry)
        elif bc["lambda_n"] == 0:
            cats["lambda_n_zero"].append(entry)
        elif bc["lambda_p"] == 0:
            cats["lambda_p_zero"].append(entry)
        else:
            cats["both_positive"].append(entry)
    return cats


def _stage_delta(summary, dataset, name):
    s = summary["stagewise"][dataset].get(name, {})
    if s.get("n"):
        return (s["dynamic_macro_f1"]["macro_f1"]
                - s["static_macro_f1"]["macro_f1"])
    return None


def overall_finding_and_recommendation(summary):
    """§18 overall finding (evidence-ordered) + §19 recommendation."""
    reasons = []
    for ds in DATASETS:
        sel = summary["selection"][ds]["overall"]
        q2 = summary["selection"][ds]["selection_to_prediction"]
        ph = summary["posthoc"][ds]
        q4, q5, q6 = ph["q4"], ph["q5"], ph["q6"]
        if q5["spearman_mean"] > 0.98 or q5["top1_change_rate"] < 0.05 \
                or q4["bonus_to_utility_std_ratio"] < 0.1:
            reasons.append(("DYNAMIC_RANKING_RARELY_CHANGES", ds,
                            q4["bonus_to_utility_std_ratio"]))
        if q6["ranking_changed_but_selection_same_rate"] > 0.5 \
                and sel["replacement_rate"] < 0.5:
            reasons.append(("RANKING_CHANGES_BUT_BUDGET_MASKS_IT", ds,
                            q6["ranking_changed_but_selection_same_rate"]))
        if sel["replacement_rate"] > 0.1 and \
                q2["prediction_change_rate_given_selection_change"] < 0.2:
            reasons.append(("SELECTION_CHANGES_BUT_PROXY_INSENSITIVE", ds,
                            q2["prediction_change_rate_given_selection_change"]))
        if sel["replacement_rate"] > 0.1 and \
                q2["transitions"]["net_correction_gain"] <= 0:
            reasons.append(("DYNAMIC_REPLACEMENTS_NOT_USEFUL", ds,
                            q2["transitions"]["net_correction_gain"]))
    counts = {}
    for r in reasons:
        counts[r[0]] = counts.get(r[0], 0) + 1
    order = sorted(counts, key=lambda k: -counts[k])
    overall = "MIXED_CAUSES"
    if len(order) == 1:
        overall = order[0]

    change_low = all(
        summary["selection"][d]["overall"]["replacement_rate"] < 0.1
        for d in DATASETS)
    net_nonpositive = all(
        summary["selection"][d]["selection_to_prediction"]
        ["transitions"]["net_correction_gain"] <= 0 for d in DATASETS)
    subscene_gain = False
    for ds in DATASETS:
        for name in STAGES:
            d = _stage_delta(summary, ds, name)
            if d is not None and d >= 0.005:
                subscene_gain = True
        for grp in ("low", "medium", "high"):
            p = summary["pressure"][ds].get(grp, {})
            if p.get("n", 0) >= 100:
                d = p["dynamic_macro_f1"]["macro_f1"] - \
                    p["static_macro_f1"]["macro_f1"]
                if d >= 0.005:
                    subscene_gain = True
    if change_low and net_nonpositive:
        rec = "REMOVE_DYNAMIC_AS_CORE"
    elif subscene_gain:
        rec = "KEEP_CURRENT_DYNAMIC"
    else:
        rec = "REDESIGN_DYNAMIC"
    return overall, reasons, rec


def render_bootstrap_md(summary):
    lines = ["# E3 Paired Bootstrap (fixed)", "",
             "## Previous bug",
             "The E3_TEST_REPORT bootstrap converted the with-replacement "
             "resampled event list into a dict keyed by event_id, so events "
             "drawn more than once were folded into a single occurrence. "
             "Bootstrap multiplicity was lost and the 95% CIs were invalid.",
             "",
             "## Corrected implementation",
             "Sampling unit = test event; 10,000 iterations; seed = 3090; "
             "each iteration draws n events WITH replacement and every "
             "occurrence contributes its predictions (the sampled list is "
             "never re-keyed into a unique-event dict). Paired Static vs "
             "Dynamic on the same events.", "",
             "Model predictions and point estimates are unaffected.", ""]
    for ds in DATASETS:
        b = summary["bootstrap"][ds]
        lines += [f"## {ds.upper()}", "",
                  f"Delta Macro-F1 point: {b['delta_macro_f1']['point']:+.5f}",
                  f"95% CI: [{b['delta_macro_f1']['ci_low']:+.5f}, "
                  f"{b['delta_macro_f1']['ci_high']:+.5f}]",
                  "",
                  f"Delta FlipRate point: {b['delta_flip_rate']['point']:+.5f}",
                  f"95% CI: [{b['delta_flip_rate']['ci_low']:+.5f}, "
                  f"{b['delta_flip_rate']['ci_high']:+.5f}]", ""]
    return "\n".join(lines)


def render_report(summary, overall, reasons, rec):
    title = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
    lines = ["# TC-DSCR E3 Failure Diagnosis", "",
             "## Overall Finding", overall, ""]
    if reasons:
        lines += ["Evidence-ordered causes:"]
        for r in reasons:
            lines.append(f"- {r[0]} ({r[1]}, signal={r[2]:.4f})")
    lines += ["", "## Bootstrap Fix",
              "- previous bug: with-replacement resample converted into a "
              "unique-event dict (multiplicity lost) -> CIs invalid",
              "- corrected implementation: sampled event list keeps every "
              "occurrence; 10,000 iterations, seed 3090, paired",
              ""]
    for ds in DATASETS:
        b = summary["bootstrap"][ds]
        lines += [f"- {title[ds]} Macro-F1 CI: "
                  f"[{b['delta_macro_f1']['ci_low']:+.5f}, "
                  f"{b['delta_macro_f1']['ci_high']:+.5f}]",
                  f"- {title[ds]} FlipRate CI: "
                  f"[{b['delta_flip_rate']['ci_low']:+.5f}, "
                  f"{b['delta_flip_rate']['ci_high']:+.5f}]"]
    lines += ["", "## 1. Static vs Dynamic Selection Change"]
    for ds in DATASETS:
        sel = summary["selection"][ds]["overall"]
        lines.append(
            f"- {title[ds]}: exact-match rate {sel['exact_match_rate']:.4f}, "
            f"mean Jaccard {sel['jaccard_mean']:.4f}, replacement rate "
            f"{sel['replacement_rate']:.4f}")
    lines += ["", "## 2. Ranking Change"]
    for ds in DATASETS:
        q5 = summary["posthoc"][ds]["q5"]
        lines.append(
            f"- {title[ds]}: Spearman mean {q5['spearman_mean']:.6f}, "
            f"top1 change {q5['top1_change_rate']:.4f}, top5 set change "
            f"{q5['top5_set_change_rate']:.4f}")
    lines += ["", "## 3. Budget Masking"]
    for ds in DATASETS:
        q6 = summary["posthoc"][ds]["q6"]
        lines.append(
            f"- {title[ds]}: ranking-changed-but-selection-same "
            f"{q6['ranking_changed_but_selection_same_rate']:.4f}, "
            f"budget utilization {q6['mean_budget_utilization']:.4f}")
    lines += ["", "## 4. Prediction Impact"]
    for ds in DATASETS:
        q2 = summary["selection"][ds]["selection_to_prediction"]
        lines.append(
            f"- {title[ds]}: change-given-selection-change "
            f"{q2['prediction_change_rate_given_selection_change']:.4f}, "
            f"wrong->correct {q2['transitions']['wrong_to_correct']}, "
            f"correct->wrong {q2['transitions']['correct_to_wrong']}, "
            f"net gain {q2['transitions']['net_correction_gain']}")
    lines += ["", "## 5. Score Scale"]
    for ds in DATASETS:
        q4 = summary["posthoc"][ds]["q4"]
        lines.append(
            f"- {title[ds]}: u std {q4['u_std']:.4f}, bonus std "
            f"{q4['bonus_std']:.4f}, bonus/utility std ratio "
            f"{q4['bonus_to_utility_std_ratio']:.4f}, mean|bonus|/mean|u| "
            f"{q4['mean_abs_bonus_over_mean_abs_u']:.4f}")
    lines += ["", "## 6. Novelty Analysis"]
    for ds in DATASETS:
        q8 = summary["posthoc"][ds]["q8"]
        lines.append(
            f"- {title[ds]}: dynamic-only novelty "
            f"{q8['dynamic_only_mean_novelty']:.4f} vs removed static "
            f"{q8['removed_static_mean_novelty']:.4f} (gain "
            f"{q8['novelty_gain']:+.4f}); utility delta "
            f"{q8['utility_delta']:+.4f}")
    lines += ["", "## 7. Persistence Analysis"]
    for ds in DATASETS:
        q7 = summary["posthoc"][ds]["q7"]
        lines.append(
            f"- {title[ds]}: P(select|persistent) {q7['P_select_given_persistent']:.4f} "
            f"vs P(select|new) {q7['P_select_given_new']:.4f} (lift "
            f"{q7['persistence_selection_lift']:+.4f}), saved-evidence "
            f"transitions {q7['persistence_saved_evidence']['transitions']}")
    lines += ["", "## 8. Stage-wise Analysis"]
    for ds in DATASETS:
        for name in STAGES:
            s = summary["stagewise"][ds].get(name, {})
            if not s.get("n"):
                continue
            d = s["dynamic_macro_f1"]["macro_f1"] - \
                s["static_macro_f1"]["macro_f1"]
            lines.append(
                f"- {title[ds]} {name}: delta {d:+.4f}, ranking-change "
                f"{s.get('ranking_change_rate', 0.0):.4f}, selection-change "
                f"{s.get('selection_change_rate', 0.0):.4f}, pred-change "
                f"{s.get('prediction_change_rate', 0.0):.4f}")
    lines += ["", "## 9. Selection-pressure Analysis"]
    for ds in DATASETS:
        for grp in ("low", "medium", "high"):
            p = summary["pressure"][ds].get(grp, {})
            if not p.get("n"):
                continue
            d = p["dynamic_macro_f1"]["macro_f1"] - \
                p["static_macro_f1"]["macro_f1"]
            lines.append(
                f"- {title[ds]} {grp}-pressure (n={p['n']}): delta {d:+.4f}, "
                f"selection-change {p['selection_change_rate']:.4f}, "
                f"pred-change {p['prediction_change_rate']:.4f}")
    lines += ["", "## 10. Validation-to-Test Stability"]
    for ds in DATASETS:
        for fold in FOLDS:
            v = summary["validation_stability"][ds].get(str(fold), {})
            if not v:
                continue
            td = v["test_delta_macro_f1"]
            td_str = f"{td:+.5f}" if td is not None else "n/a"
            lines.append(
                f"- {title[ds]} fold{fold}: val top1-top5 gap "
                f"{v['top1_minus_top5_gap']:.5f}, test delta {td_str} "
                f"{'FLAT_VALIDATION_LANDSCAPE' if v['flat_landscape'] else ''}")
    lines += ["", "## 11. Parameter Necessity"]
    for ds in DATASETS:
        pn = summary["parameter_necessity"][ds]
        summary_line = (
            f"- {title[ds]}: lambda_n=0 folds "
            f"{[e['fold'] for e in pn['lambda_n_zero']]}, lambda_p=0 folds "
            f"{[e['fold'] for e in pn['lambda_p_zero']]}, both>0 folds "
            f"{[e['fold'] for e in pn['both_positive']]}")
        lines.append(summary_line)
    lines += ["", "## Root Cause Ranking"]
    for i, r in enumerate(reasons, 1):
        lines.append(f"{i}. {r[0]} ({r[1]}, signal={r[2]:.4f})")
    if not reasons:
        lines.append("1. no dominant single cause (MIXED_CAUSES)")
    lines += ["", "## Implications for Method Design",
              "Diagnosis only; no method changes were made in this round.",
              "",
              "## Recommendation", rec, ""]
    return "\n".join(lines)


def load_existing_outputs():
    """Rebuild the summary dict from the already-written analysis JSONs."""
    def rd(name):
        with open(os.path.join(OUT_ROOT, name), encoding="utf-8") as fh:
            return json.load(fh)

    posthoc = {}
    for ds in DATASETS:
        posthoc[ds] = {"q3": rd("proxy_output_change.json")[ds],
                       "q4": rd("score_scale_analysis.json")[ds],
                       "q5": rd("ranking_change_analysis.json")[ds],
                       "q6": rd("budget_masking_analysis.json")[ds],
                       "q7": rd("persistence_effect_analysis.json")[ds],
                       "q8": rd("novelty_tradeoff_analysis.json")[ds]}
    return {"bootstrap": rd("bootstrap_fixed.json"),
            "selection": rd("selection_change_summary.json"),
            "posthoc": posthoc,
            "stagewise": rd("stagewise_analysis.json"),
            "pressure": rd("selection_pressure_analysis.json"),
            "validation_stability": rd("validation_stability_analysis.json"),
            "parameter_necessity": rd("parameter_necessity_diagnostic.json")}


def run_full_diagnosis(_args):
    """Full pipeline: bootstrap + selection + post-hoc + stage/pressure/
    stability/parameter-necessity analyses."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"bootstrap": {}, "selection": {}, "posthoc": {},
               "stagewise": {}, "pressure": {}, "validation_stability": {},
               "parameter_necessity": {}}
    for dataset in DATASETS:
        rows = load_test_rows(dataset)
        summary["bootstrap"][dataset] = bootstrap_fixed(dataset)
        summary["selection"][dataset] = {
            "per_cutoff": {str(c): selection_change_metrics(
                [r for r in rows if r["cutoff"] == str(c)])
                for c in PRIMARY_CUTOFFS},
            "overall": selection_change_metrics(rows),
            "selection_to_prediction": selection_change_prediction_effect(
                rows),
        }
        summary["posthoc"][dataset] = aggregate_posthoc(dataset, device)
        sw = stagewise(rows)
        for name in STAGES:
            extra = summary["posthoc"][dataset]["stage_extra"][name]
            if name in sw and sw[name].get("n"):
                sw[name]["ranking_change_rate"] = extra[
                    "ranking_change_rate"]
                sw[name]["selection_change_rate"] = extra[
                    "selection_change_rate"]
                sw[name]["prediction_change_rate"] = extra[
                    "prediction_change_rate"]
        summary["stagewise"][dataset] = sw
        summary["pressure"][dataset] = selection_pressure_rows(rows)
        summary["validation_stability"][dataset] = validation_stability(
            dataset)
        summary["parameter_necessity"][dataset] = parameter_necessity(
            dataset)
    return summary


def write_outputs(summary):
    with open(os.path.join(OUT_ROOT, "bootstrap_fixed.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary["bootstrap"], fh, indent=1)
    with open(os.path.join(OUT_ROOT, "selection_change_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary["selection"], fh, indent=1)
    with open(os.path.join(OUT_ROOT, "selection_to_prediction_effect.json"),
              "w", encoding="utf-8") as fh:
        json.dump({d: summary["selection"][d]["selection_to_prediction"]
                   for d in DATASETS}, fh, indent=1)
    with open(os.path.join(OUT_ROOT, "proxy_output_change.json"), "w",
              encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q3"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "score_scale_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q4"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "ranking_change_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q5"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "budget_masking_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q6"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "persistence_effect_analysis.json"),
              "w", encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q7"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "novelty_tradeoff_analysis.json"),
              "w", encoding="utf-8") as fh:
        json.dump({d: summary["posthoc"][d]["q8"] for d in DATASETS},
                  fh, indent=1)
    with open(os.path.join(OUT_ROOT, "stagewise_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary["stagewise"], fh, indent=1)
    with open(os.path.join(OUT_ROOT, "selection_pressure_analysis.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary["pressure"], fh, indent=1)
    with open(os.path.join(OUT_ROOT, "validation_stability_analysis.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary["validation_stability"], fh, indent=1)
    with open(os.path.join(OUT_ROOT, "parameter_necessity_diagnostic.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary["parameter_necessity"], fh, indent=1)


def main(argv=None):
    global E3_ROOT, TEST_ROOT, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--e3-root", default=E3_ROOT)
    ap.add_argument("--test-root", default=TEST_ROOT)
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the report from existing JSONs")
    args = ap.parse_args(argv)
    E3_ROOT, TEST_ROOT, OUT_ROOT = args.e3_root, args.test_root, args.out_root
    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.report_only:
        summary = load_existing_outputs()
    else:
        summary = run_full_diagnosis(args)
        write_outputs(summary)

    overall, reasons, rec = overall_finding_and_recommendation(summary)
    with open(os.path.join(OUT_ROOT, "bootstrap_fixed.md"), "w",
              encoding="utf-8") as fh:
        fh.write(render_bootstrap_md(summary))
    with open(os.path.join(OUT_ROOT, "E3_FAILURE_DIAGNOSIS_REPORT.md"),
              "w", encoding="utf-8") as fh:
        fh.write(render_report(summary, overall, reasons, rec))

    print(json.dumps({d: {"bootstrap": summary["bootstrap"][d],
                          "selection": summary["selection"][d]["overall"],
                          "posthoc_q4_q5_q6": {
                              "q4_ratio": summary["posthoc"][d]["q4"]
                              ["bonus_to_utility_std_ratio"],
                              "q5_spearman": summary["posthoc"][d]["q5"]
                              ["spearman_mean"],
                              "q5_top1": summary["posthoc"][d]["q5"]
                              ["top1_change_rate"],
                              "q6_mask_rate": summary["posthoc"][d]["q6"]
                              ["ranking_changed_but_selection_same_rate"]}}
                      for d in DATASETS}, indent=1))
    print(json.dumps({"overall_finding": overall,
                      "recommendation": rec,
                      "reasons": [list(r) for r in reasons]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
