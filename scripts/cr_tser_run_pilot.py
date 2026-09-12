#!/usr/bin/env python
"""CR-TSER pilot orchestrator, gate evaluation and report (plan §26, §36–§38).

Runs (or re-reads) the pilot stages, computes every pre-registered gate from
the frozen artifacts, writes ``CR_TSER_PILOT_SUMMARY.json`` and the
``CR_TSER_PILOT_REPORT.md`` tables, and emits exactly one recommendation:
``START_FULL_CR_TSER_METHOD_DEVELOPMENT`` / ``STOP_FOR_RESEARCH_REVIEW`` /
``STOP_CR_TSER``. It never redesigns the method (plan §26).
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
from cr_tser.evaluation.heterogeneity import (gate_p1,  # noqa: E402
                                              heterogeneity_report)
from cr_tser.evaluation.structural_interaction import (  # noqa: E402
    gate_p2, structural_interaction_report)
from cr_tser.evaluation.unseen_reader import (final_decision, gate_p4,  # noqa: E402
                                              pheme_secondary)
from cr_tser.evaluation.utility_prediction import (evaluate_utility,  # noqa: E402
                                                   gate_p3)


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


def load_unit_table(out_root, dataset):
    """``{(event,unit): {reader: {...}}}`` from atomic utility labels (§18)."""
    path = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    table = defaultdict(dict)
    for row in _read_jsonl(path):
        if row["intervention_type"] != "I1_atomic":
            continue
        unit = row["affected_reply_ids"][0]
        table[(row["event_id"], unit)][row["reader"]] = {
            "utility": row["utility"],
            "correct_before": row["correctness_before"],
            "correct_after": row["correctness_after"],
        }
    return dict(table)


def load_interaction_records(out_root, dataset):
    """Structured-intervention records with member atomic utilities (§12)."""
    path = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    rows = _read_jsonl(path)
    atomic = {}
    for row in rows:
        if row["intervention_type"] == "I1_atomic":
            atomic[(row["event_id"], row["cutoff"], row["reader"],
                    row["affected_reply_ids"][0])] = row["utility"]
    records = []
    for row in rows:
        if row["intervention_type"] not in ("I2_parent_child",
                                            "I3_matched_nonadjacent",
                                            "I4_subtree",
                                            "I5_matched_disconnected"):
            continue
        members = [atomic.get((row["event_id"], row["cutoff"], row["reader"],
                               nid)) for nid in row["affected_reply_ids"]]
        if any(m is None for m in members):
            continue
        records.append({
            "event": row["event_id"], "reader": row["reader"],
            "type": {"I2_parent_child": "I2", "I3_matched_nonadjacent": "I3",
                     "I4_subtree": "I4",
                     "I5_matched_disconnected": "I5"}[row["intervention_type"]],
            "utility": row["utility"], "members": members})
    return records


def utility_prediction_gate(out_root, dataset, split):
    """P3 from the trained predictor outputs (B0/B1 vs B3, §17/§25)."""
    labels = _read_jsonl(os.path.join(out_root, "utility_labels",
                                      f"{dataset}.jsonl"))
    eval_ids = set(split["utility_eval"])
    targets = {}
    for row in labels:
        if row["intervention_type"] != "I1_atomic" or \
                row["event_id"] not in eval_ids:
            continue
        key = f"{row['event_id']}:{row['cutoff']}:{row['affected_reply_ids'][0]}"
        targets[(key, row["reader"])] = row
    per_rotation = []
    for rotation in LORO_ROTATIONS:
        train_readers = list(rotation[:2])
        payload = _read_json(os.path.join(
            out_root, "predictor", dataset,
            f"rotation_{train_readers[0]}_{train_readers[1]}",
            "predictions.json"))
        if payload is None:
            return None
        for reader_key in train_readers:
            pred = payload["predictions"][reader_key]
            pairs = [(k, v) for k, v in targets.items()
                     if k[1] == reader_key and k[0] in pred]
            if not pairs:
                continue
            y_true_sign = [targets[k]["sign"] for k, _ in pairs]
            y_true_cont = [targets[k]["utility"] for k, _ in pairs]
            y_pred_cont = [pred[k[0]]["utility"] for k, _ in pairs]
            y_pred_sign = [_sign_of(v, targets[k]) for k, v in pairs]
            helpful = [pred[k[0]].get("helpful_score", 0.0) for k, _ in pairs]
            harmful = [pred[k[0]].get("harmful_score", 0.0) for k, _ in pairs]
            active = [abs(targets[k]["utility"]) >= 0.05 for k, _ in pairs]
            b3 = evaluate_utility(y_true_sign, y_pred_sign, y_true_cont,
                                  y_pred_cont, helpful, harmful, active)
            baselines = {}
            for name in ("B0_text", "B1_scalar_structure"):
                base_pred = payload.get("baselines", {}).get(name, {}) \
                    .get("predictions", {})
                if not base_pred:
                    continue
                yb = [base_pred[k[0]]["utility"] for k, _ in pairs
                      if k[0] in base_pred]
                if len(yb) != len(pairs):
                    continue
                yb_sign = ["HELPFUL" if v >= 0.05 else
                           ("HARMFUL" if v <= -0.05 else "NEUTRAL") for v in yb]
                baselines[name] = evaluate_utility(y_true_sign, yb_sign,
                                                   y_true_cont, yb)
            if baselines:
                per_rotation.append(gate_p3(b3, baselines))
    if not per_rotation:
        return None
    return per_rotation[0]


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


def compute_gates(out_root, datasets=("pheme", "weibo22")):
    p0 = _read_json(os.path.join(out_root, "p0", "p0_readiness.json"), {})
    gates = {"P0": {"pass": p0.get("P0") == "P0_PASS",
                    "detail": p0.get("weibo22_reason", "")}}
    reports = {}
    for dataset in datasets:
        split = _read_json(os.path.join(out_root, "manifests",
                                        "event_split.json"))
        if split is None:
            continue
        unit_table = load_unit_table(out_root, dataset)
        if unit_table:
            het = heterogeneity_report(unit_table, READER_KEYS)
            reports[f"{dataset}_heterogeneity"] = het
            if dataset == "weibo22":
                gates["P1"] = gate_p1(het)
        records = load_interaction_records(out_root, dataset)
        if records:
            inter = structural_interaction_report(records)
            reports[f"{dataset}_interaction"] = inter
            if dataset == "weibo22":
                gates["P2"] = gate_p2(inter)
        if dataset == "weibo22":
            p3 = utility_prediction_gate(out_root, dataset, split)
            if p3:
                gates["P3"] = p3
    rotations, pheme_rotations = [], []
    for dataset in datasets:
        for rotation in LORO_ROTATIONS:
            held = rotation[2]
            record = _read_json(os.path.join(out_root, "unseen_reader",
                                             f"rotation_{held}.json"))
            if record:
                (rotations if dataset == "weibo22" else pheme_rotations).append(
                    record["delta"])
    if rotations:
        gates["P4"] = gate_p4(rotations)
    pheme = pheme_secondary(pheme_rotations) if pheme_rotations else None
    decision = final_decision(gates, pheme)
    return gates, reports, decision


def write_report(out_root, gates, reports, decision, datasets):
    summary = {"gates": gates, "decision": decision,
               "reports": reports, "datasets": list(datasets)}
    common.write_json(os.path.join(out_root, "CR_TSER_PILOT_SUMMARY.json"),
                      summary)
    lines = ["# CR-TSER Feasibility Pilot", "",
             "## Protocol Freeze", "",
             "- seed 7319; split 80/50/15/25; cutoffs 15m/1h/6h; "
             "B_ref=1024; B_pilot=floor(0.5*Tokens(C_ref))",
             "- readers: Qwen3-8B / GLM-4-9B-Chat / InternLM3-8B-Instruct "
             "(frozen, no substitution)",
             "", "## P0 Data / Reader Readiness", "",
             f"- `{gates.get('P0', {}).get('detail', 'not run')}`"]
    for name, report in reports.items():
        lines += ["", f"## {name}", "", "```json",
                  json.dumps(report, indent=1)[:4000], "```"]
    lines += ["", "## GO / NO-GO Gates", ""]
    for key in ("P0", "P1", "P2", "P3", "P4"):
        entry = gates.get(key, {})
        lines.append(f"- **{key}**: pass=`{entry.get('pass')}`")
    lines += ["", "## Final Recommendation", "",
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
    ap.add_argument("--datasets", default="pheme,weibo22")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    gates, reports, decision = compute_gates(out_root, datasets)
    summary = write_report(out_root, gates, reports, decision, datasets)
    print(json.dumps({"recommendation": decision["recommendation"],
                      "decision": decision["decision"],
                      "gates": {k: v.get("pass") for k, v in gates.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
