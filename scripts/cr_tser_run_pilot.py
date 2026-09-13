#!/usr/bin/env python
"""CR-TSER pilot orchestrator, gate evaluation and report (plan §26, §36–§38).

Reads the per-dataset artifact namespaces produced by the stage scripts, so
PHEME and Weibo22 results can coexist without overwriting each other:

    manifests/<dataset>/...
    utility_labels/<dataset>/labels.jsonl
    predictor/<dataset>/...
    unseen_reader/<dataset>/rotation_<heldout>.json

Gate discipline (review fixes):

* **P1–P4 are Weibo22-primary.** PHEME contributes only the secondary
  condition and can never produce ``FULL_GO`` (plan §25);
* **P2** consumes reader- and snapshot-matched pairs;
* **P3** requires the plan §24 event-level paired bootstrap of ΔMacro-F1 and
  pools all three LORO rotations instead of reporting only the first;
* **B2 / S6** (legacy TC-DSCR) are PHEME-only and never reach a Weibo22 gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_tser_common as common  # noqa: E402

from cr_tser.config.pilot_config import (LORO_ROTATIONS, READER_KEYS,  # noqa: E402
                                         SIGN_CLASSES)
from cr_tser.evaluation.bootstrap import (class_f1_from_counts,  # noqa: E402
                                          macro_f1_from_counts, spearman_rank)
from cr_tser.evaluation.heterogeneity import (gate_p1,  # noqa: E402
                                              heterogeneity_report)
from cr_tser.evaluation.structural_interaction import (  # noqa: E402
    gate_p2, structural_interaction_report)
from cr_tser.evaluation.unseen_reader import (final_decision, gate_p4,  # noqa: E402
                                              pheme_secondary)
from cr_tser.evaluation.utility_prediction import (  # noqa: E402
    evaluate_utility, gate_p3, macro_f1_delta_bootstrap, merge_event_counts,
    per_event_counts)
from cr_tser.intervention.evidence_units import evidence_key  # noqa: E402
from cr_tser.models.legacy_utility import legacy_arm_enabled  # noqa: E402

PRIMARY_DATASET = "weibo22"
SECONDARY_DATASET = "pheme"
LEGACY_TYPES = ("I2_parent_child", "I3_matched_nonadjacent", "I4_subtree",
                "I5_matched_disconnected")


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def labels_path(out_root, dataset):
    namespaced = os.path.join(out_root, "utility_labels", dataset,
                              "labels.jsonl")
    legacy = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    return namespaced if os.path.exists(namespaced) else legacy


def load_unit_table(out_root, dataset):
    """``{canonical_key: {reader: {...}}}`` from atomic utility labels (§18).

    The key includes dataset, event, cutoff and reply node, so the same reply
    at different cutoffs (or in different datasets) can never collide.
    """
    table = defaultdict(dict)
    for row in _read_jsonl(labels_path(out_root, dataset)):
        if row["intervention_type"] != "I1_atomic":
            continue
        node = row["affected_reply_ids"][0]
        key = evidence_key(dataset, row["event_id"], row["cutoff"], node)
        table[key][row["reader"]] = {
            "utility": row["utility"],
            "correct_before": row["correctness_before"],
            "correct_after": row["correctness_after"],
        }
    return dict(table)


def load_interaction_records(out_root, dataset):
    """Structured-intervention records with member atomic utilities (§12)."""
    rows = _read_jsonl(labels_path(out_root, dataset))
    atomic = {}
    for row in rows:
        if row["intervention_type"] == "I1_atomic":
            atomic[(row["event_id"], row["cutoff"], row["reader"],
                    row["affected_reply_ids"][0])] = row["utility"]
    records = []
    for row in rows:
        if row["intervention_type"] not in LEGACY_TYPES:
            continue
        members = [atomic.get((row["event_id"], row["cutoff"], row["reader"],
                               nid)) for nid in row["affected_reply_ids"]]
        if any(m is None for m in members):
            continue
        records.append({
            "event": row["event_id"], "cutoff": row["cutoff"],
            "reader": row["reader"],
            "type": {"I2_parent_child": "I2", "I3_matched_nonadjacent": "I3",
                     "I4_subtree": "I4",
                     "I5_matched_disconnected": "I5"}[row["intervention_type"]],
            "utility": row["utility"], "members": members})
    return records


def _sign_of(value, row):
    if row["correctness_before"] and not row["correctness_after"]:
        return "HELPFUL"
    if not row["correctness_before"] and row["correctness_after"]:
        return "HARMFUL"
    if value >= 0.05:
        return "HELPFUL"
    if value <= -0.05:
        return "HARMFUL"
    return "NEUTRAL"


def _metrics_from_counts(counts):
    return {
        "macro_f1": macro_f1_from_counts(counts, labels=SIGN_CLASSES),
        "per_class_f1": {c: class_f1_from_counts(counts, c)
                         for c in SIGN_CLASSES},
    }


def utility_prediction_gate(out_root, dataset, split):
    """P3 over **all** LORO rotations with an event-level paired bootstrap."""
    labels = _read_jsonl(labels_path(out_root, dataset))
    eval_ids = set(split["utility_eval"])
    targets = {}
    for row in labels:
        if row["intervention_type"] != "I1_atomic" or \
                row["event_id"] not in eval_ids:
            continue
        node = row["affected_reply_ids"][0]
        key = evidence_key(dataset, row["event_id"], row["cutoff"], node)
        targets[(key, row["reader"])] = row
    b3_counts, base_counts = [], defaultdict(list)
    b3_true_cont, b3_pred_cont = [], []
    b3_true_sign, b3_pred_sign = [], []
    helpful, harmful, active = [], [], []
    rotations_used = []
    for rotation in LORO_ROTATIONS:
        train_readers = list(rotation[:2])
        payload = _read_json(os.path.join(
            out_root, "predictor", dataset,
            f"rotation_{train_readers[0]}_{train_readers[1]}",
            "predictions.json"))
        if payload is None:
            continue
        rotations_used.append(f"{train_readers[0]}+{train_readers[1]}")
        for reader_key in train_readers:
            pred = payload["predictions"].get(reader_key, {})
            pairs = [(k, pred[k[0]]) for k in targets
                     if k[1] == reader_key and k[0] in pred]
            if not pairs:
                continue
            events = [targets[k]["event_id"] for k, _ in pairs]
            y_true_sign = [targets[k]["sign"] for k, _ in pairs]
            y_pred_sign = [_sign_of(v["utility"], targets[k]) for k, v in pairs]
            b3_counts.append(per_event_counts(events, y_true_sign, y_pred_sign))
            b3_true_cont.extend(targets[k]["utility"] for k, _ in pairs)
            b3_pred_cont.extend(v["utility"] for _k, v in pairs)
            b3_true_sign.extend(y_true_sign)
            b3_pred_sign.extend(y_pred_sign)
            helpful.extend(v.get("helpful_score", 0.0) for _k, v in pairs)
            harmful.extend(v.get("harmful_score", 0.0) for _k, v in pairs)
            active.extend(abs(targets[k]["utility"]) >= 0.05 for k, _ in pairs)
            for name in ("B0_text", "B1_scalar_structure"):
                base_pred = payload.get("baselines", {}).get(name, {}) \
                    .get("predictions", {})
                bpairs = [(k, base_pred[k[0]]["utility"]) for k in targets
                          if k[1] == reader_key and k[0] in base_pred]
                if len(bpairs) != len(pairs):
                    continue
                bevents = [targets[k]["event_id"] for k, _ in bpairs]
                bsign = ["HELPFUL" if v >= 0.05 else
                         ("HARMFUL" if v <= -0.05 else "NEUTRAL")
                         for _k, v in bpairs]
                base_counts[name].append(
                    per_event_counts(bevents, y_true_sign, bsign))
    if not b3_counts:
        return None
    merged_b3 = merge_event_counts(b3_counts)
    b3_full = evaluate_utility(
        b3_true_sign, b3_pred_sign, b3_true_cont, b3_pred_cont, helpful,
        harmful, active)
    b3_full.update(_metrics_from_counts(merged_b3))
    baselines = {}
    for name, counts_list in base_counts.items():
        merged = merge_event_counts(counts_list)
        base_metrics = _metrics_from_counts(merged)
        base_metrics["spearman"] = spearman_rank(b3_true_cont, b3_pred_cont)
        baselines[name] = base_metrics
    best_name = max(baselines, key=lambda n: baselines[n]["macro_f1"])
    delta_ci = macro_f1_delta_bootstrap(merged_b3,
                                        merge_event_counts(base_counts[best_name]))
    gate = gate_p3(b3_full, baselines, delta_ci=delta_ci)
    gate["rotations_used"] = rotations_used
    gate["primary_dataset"] = dataset
    return gate


def compute_gates(out_root, datasets=(PRIMARY_DATASET, SECONDARY_DATASET)):
    p0 = _read_json(os.path.join(out_root, "p0", "p0_readiness.json"), {})
    gates = {"P0": {"pass": p0.get("P0") == "P0_PASS",
                    "detail": p0.get("weibo22_reason", "")}}
    reports = {}
    for dataset in datasets:
        split = _read_json(os.path.join(out_root, "manifests", dataset,
                                        "event_split.json"))
        if split is None:
            continue
        unit_table = load_unit_table(out_root, dataset)
        if unit_table:
            het = heterogeneity_report(unit_table, READER_KEYS)
            reports[f"{dataset}_heterogeneity"] = het
            if dataset == PRIMARY_DATASET:
                gates["P1"] = gate_p1(het)
        records = load_interaction_records(out_root, dataset)
        if records:
            inter = structural_interaction_report(records)
            reports[f"{dataset}_interaction"] = inter
            if dataset == PRIMARY_DATASET:
                gates["P2"] = gate_p2(inter)
        if dataset == PRIMARY_DATASET:
            p3 = utility_prediction_gate(out_root, dataset, split)
            if p3:
                gates["P3"] = p3
    rotations, pheme_rotations = [], []
    for dataset in datasets:
        for rotation in LORO_ROTATIONS:
            held = rotation[2]
            record = _read_json(os.path.join(out_root, "unseen_reader",
                                             dataset,
                                             f"rotation_{held}.json"))
            if record:
                (rotations if dataset == PRIMARY_DATASET
                 else pheme_rotations).append(record["delta"])
    if rotations:
        gates["P4"] = gate_p4(rotations)
    pheme = pheme_secondary(pheme_rotations) if pheme_rotations else None
    decision = final_decision(gates, pheme)
    decision["primary_dataset"] = PRIMARY_DATASET
    decision["legacy_diagnostics"] = {
        "b2_s6_enabled_datasets": [d for d in datasets
                                   if legacy_arm_enabled(d)],
        "excluded_from_primary": True,
    }
    return gates, reports, decision


def write_report(out_root, gates, reports, decision, datasets):
    summary = {"primary_dataset": PRIMARY_DATASET,
               "secondary_dataset": SECONDARY_DATASET,
               "gates": gates, "decision": decision, "reports": reports,
               "datasets": list(datasets)}
    common.write_json(os.path.join(out_root, "CR_TSER_PILOT_SUMMARY.json"),
                      summary)
    lines = ["# CR-TSER Feasibility Pilot", "",
             "## Protocol Freeze", "",
             "- seed 7319; split 80/50/15/25; cutoffs 15m/1h/6h; "
             "B_ref=1024; B_pilot=floor(0.5*Tokens(C_ref))",
             "- readers: Qwen3-8B / GLM-4-9B-Chat / InternLM3-8B-Instruct "
             "(frozen, no substitution)",
             f"- primary dataset: `{PRIMARY_DATASET}`; secondary (PHEME) "
             f"cannot produce FULL_GO", ""]
    for dataset in datasets:
        lines += [f"## Dataset: {dataset}", ""]
        for name, report in reports.items():
            if not name.startswith(f"{dataset}_"):
                continue
            lines += [f"### {name}", "", "```json",
                      json.dumps(report, indent=1)[:4000], "```", ""]
    lines += ["## GO / NO-GO Gates (Weibo22 primary)", ""]
    for key in ("P0", "P1", "P2", "P3", "P4"):
        entry = gates.get(key, {})
        lines.append(f"- **{key}**: pass=`{entry.get('pass')}`")
    lines += ["", "## PHEME Secondary Evidence", "",
              f"- enabled B2/S6 legacy diagnostics: "
              f"`{decision['legacy_diagnostics']['b2_s6_enabled_datasets']}`",
              "", "## Final Recommendation", "",
              f"```\n{decision['recommendation']}\n```", "",
              f"- decision: `{decision['decision']}`",
              f"- failed gates: {decision.get('failed_gates', [])}"]
    with open(os.path.join(out_root, "CR_TSER_PILOT_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return summary


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--datasets", default=f"{PRIMARY_DATASET},{SECONDARY_DATASET}")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    gates, reports, decision = compute_gates(out_root, datasets)
    write_report(out_root, gates, reports, decision, datasets)
    print(json.dumps({"recommendation": decision["recommendation"],
                      "decision": decision["decision"],
                      "gates": {k: v.get("pass") for k, v in gates.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
