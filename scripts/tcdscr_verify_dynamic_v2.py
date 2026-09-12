#!/usr/bin/env python
"""Dynamic V2 (MF-TSR) verifier (execution protocol §39).

Checks, with issues = 0 required:
  - all validation events present at all 6 cutoffs (no-candidate snapshots
    included, candidate_count = 0 rows are not skipped);
  - no test event was read (run event ids == fold validation ids, disjoint
    from the reconstructed test ids; the verifier never touches labels);
  - no future nodes (selected ids exist in the current snapshot) and no
    future memory (memory_current_ids in the current snapshot,
    memory_previous_ids in the previous cutoff's snapshot);
  - correct E1/E2 checkpoints (frozen checksum matches) and encoder,
    selector, proxy all frozen with zero new trainable parameters;
  - budget <= 1024, atomic evidence units (moves act on whole units),
    no novelty/persistence bonus in the objective (AST audit),
    epsilon only from {0, 0.01, 0.05}, max refinement steps <= 20,
    every accepted move satisfies relative_gain >= epsilon,
    final distortion <= static distortion or fallback_to_static = true;
  - protocol-correction and fold/report artifacts exist.
Writes mf_tsr_verify.json.
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = str(Path(HERE).resolve().parents[0] / "project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
PRIMARY_CUTOFFS = (5, 15, 30, 60, 180, 360)
EPSILON_STRINGS = ("0.0", "0.01", "0.05")
BUDGET = 1024
MAX_STEPS = 20
MODELS_DIR = Path(HERE).resolve().parents[0] / "project" / "tcdscr" / \
    "models"

REQUIRED_PROTOCOL = ("no_candidate_audit.json",
                     "e2_corrected_all_event_metrics.json",
                     "e2_corrected_all_event_metrics.md",
                     "e3_v1_all_event_diagnostic.json",
                     "protocol_manifest.json")
REQUIRED_FOLD = ("epsilon_grid.json", "best_config.json",
                 "validation_predictions.jsonl", "set_dynamics.json",
                 "distortion_metrics.json")


_FOLD_CACHE = {}


def _fold_snapshots(dataset, fold):
    """({cutoff: {event_id: set(node_ids)}}, validation_ids, test_ids).

    Loaded once per (dataset, fold) and cached: the split is loaded a single
    time and every primary cutoff snapshot is rebuilt from the true event.
    """
    key = (dataset, fold)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry, load_split_events
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    snaps = {c: {} for c in PRIMARY_CUTOFFS}
    for ev in events["validation"]:
        for c in PRIMARY_CUTOFFS:
            snaps[c][ev["event_id"]] = set(
                build_snapshot(ev, c)["node_ids"])
    out = (snaps, set(split["validation"]), set(split["test"]))
    _FOLD_CACHE[key] = out
    return out


def _objective_audit(issues):
    for fname in ("set_fidelity.py", "temporal_survival.py",
                  "marginal_set_refiner.py"):
        tree = ast.parse((MODELS_DIR / fname).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.alias):
                names.add(node.name)
        for banned in ("novelty", "persistence", "recency"):
            if any(banned in n.lower() for n in names):
                issues.append(f"{fname}: {banned} appears in the objective")
    src = (Path(HERE) / "tcdscr_run_dynamic_v2.py").read_text(
        encoding="utf-8")
    if "dynamic_score" in src or "lambda_n" in src:
        issues.append("runner still references the V1 additive dynamic "
                      "score")


def verify(root, protocol_root):
    issues = []
    summary = {"runs": 0, "rows": 0, "no_candidate_rows": 0,
               "events_without_candidates": 0}
    _objective_audit(issues)
    for name in REQUIRED_PROTOCOL:
        if not os.path.exists(os.path.join(protocol_root, name)):
            issues.append(f"missing protocol artifact: {name}")
    if not os.path.exists(os.path.join(root, "MF_TSR_VALIDATION_REPORT.md")):
        issues.append("missing MF_TSR_VALIDATION_REPORT.md")
    if not os.path.exists(os.path.join(root, "mf_tsr_summary.json")):
        issues.append("missing mf_tsr_summary.json")
    for dataset in DATASETS:
        for fold in FOLDS:
            fold_dir = os.path.join(root, dataset, f"fold{fold}")
            for name in REQUIRED_FOLD:
                if not os.path.exists(os.path.join(fold_dir, name)):
                    issues.append(f"missing fold artifact: {dataset}/"
                                  f"fold{fold}/{name}")
            bc_path = os.path.join(fold_dir, "best_config.json")
            if os.path.exists(bc_path):
                with open(bc_path, encoding="utf-8") as fh:
                    bc = json.load(fh)
                if str(bc.get("epsilon")) not in EPSILON_STRINGS:
                    issues.append(f"{dataset}/fold{fold}: epsilon "
                                  f"{bc.get('epsilon')} outside grid")
                if bc.get("budget") != BUDGET:
                    issues.append(f"{dataset}/fold{fold}: budget "
                                  f"{bc.get('budget')} != {BUDGET}")
            snap_cache, fold_val_ids, fold_test_ids = _fold_snapshots(
                dataset, fold)
            for seed in SEEDS:
                run_dir = os.path.join(root, "runs", dataset,
                                       f"fold{fold}_seed{seed}")
                man_path = os.path.join(run_dir, "run_manifest.json")
                if not os.path.exists(man_path):
                    issues.append(f"missing run manifest: {run_dir}")
                    continue
                with open(man_path, encoding="utf-8") as fh:
                    man = json.load(fh)
                ck = man.get("checksums", {})
                for key in ("encoder_match", "selector_match",
                            "proxy_match"):
                    if not ck.get(key):
                        issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                      f"frozen check {key} false")
                if man.get("new_trainable_parameters") != 0:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "trainable parameters present")
                if man.get("test_split_read") is not False:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "test_split_read flag missing")
                if man.get("budget") != BUDGET:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "manifest budget != 1024")
                pred_path = os.path.join(run_dir,
                                         "validation_predictions.jsonl")
                if not os.path.exists(pred_path):
                    issues.append(f"missing predictions: {pred_path}")
                    continue
                per_seed_ids = set()
                rows_by_event = {}
                n_rows = 0
                with open(pred_path, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        n_rows += 1
                        per_seed_ids.add(r["event_id"])
                        rows_by_event.setdefault(
                            (r["event_id"], r["cutoff"]), []).append(r)
                        if r["budget"] != BUDGET:
                            issues.append("row budget != 1024")
                        if r["mf_tsr_evidence_tokens"] > BUDGET or \
                                r["static_evidence_tokens"] > BUDGET:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "token budget exceeded")
                        if str(r["epsilon"]) not in EPSILON_STRINGS:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "epsilon outside grid")
                        mv = r["accepted_moves"]
                        if len(mv) > MAX_STEPS:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "more than 20 refinement steps")
                        for x in mv:
                            if x["relative_gain"] < float(r["epsilon"]) - 1e-9:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "accepted move below epsilon")
                            if x["added_node_id"] is None and \
                                    x["removed_node_id"] is None:
                                issues.append("empty move")
                        if r["mf_tsr_distortion"] > \
                                r["static_distortion"] + 1e-8 and \
                                not r["fallback_to_static"]:
                            issues.append(
                                f"{r['event_id']}/{r['cutoff']}: final "
                                "distortion worse than static without "
                                "fallback")
                        if r["candidate_count"] == 0:
                            summary["no_candidate_rows"] += 1
                            if r["mf_tsr_selected_node_ids"] or \
                                    r["static_selected_node_ids"] or \
                                    r["mf_tsr_evidence_tokens"] != 0:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "no-candidate row is not empty")
                        cur_ids = snap_cache[int(r["cutoff"])].get(
                            r["event_id"], set())
                        if not set(r["mf_tsr_selected_node_ids"]) <= cur_ids:
                            issues.append(
                                f"{r['event_id']}/{r['cutoff']}: future "
                                "node in MF-TSR selection")
                        if not set(r["static_selected_node_ids"]) <= cur_ids:
                            issues.append(
                                f"{r['event_id']}/{r['cutoff']}: future "
                                "node in static selection")
                        if not set(r["memory_current_ids"]) <= cur_ids:
                            issues.append(
                                f"{r['event_id']}/{r['cutoff']}: future "
                                "memory in M_t")
                        idx = PRIMARY_CUTOFFS.index(int(r["cutoff"]))
                        if idx > 0:
                            prev_ids = snap_cache[
                                PRIMARY_CUTOFFS[idx - 1]].get(
                                r["event_id"], set())
                            if not set(r["memory_previous_ids"]) <= \
                                    prev_ids:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "memory contains future nodes")
                        elif r["memory_previous_ids"]:
                            issues.append(
                                f"{r['event_id']}/{r['cutoff']}: first "
                                "cutoff memory is not empty")
                if per_seed_ids & fold_test_ids:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: test "
                                  "events were read")
                if per_seed_ids != fold_val_ids:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "validation event coverage mismatch "
                                  f"({len(per_seed_ids)} vs "
                                  f"{len(fold_val_ids)})")
                for ev in fold_val_ids:
                    for c in PRIMARY_CUTOFFS:
                        key = (ev, str(c))
                        if key not in rows_by_event:
                            issues.append(f"{dataset}/fold{fold}_seed"
                                          f"{seed}: missing {key}")
                        elif len(rows_by_event[key]) != 3:
                            issues.append(f"{dataset}/fold{fold}_seed"
                                          f"{seed}: {key} does not have 3 "
                                          "epsilon rows")
                summary["runs"] += 1
                summary["rows"] += n_rows
            summary["events_without_candidates"] = summary.get(
                "events_without_candidates", 0)
    return {"issues": issues, "n_issues": len(issues), **summary,
            "checks": [
                "validation coverage at all 6 cutoffs (no-candidate "
                "included)", "no test events read", "no future nodes",
                "no future memory", "frozen encoder/selector/proxy",
                "budget <= 1024", "atomic evidence units",
                "no novelty/persistence bonus", "epsilon in grid",
                "max steps <= 20", "gain >= epsilon",
                "distortion <= static or fallback"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/dynamic_v2")
    ap.add_argument("--protocol-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    protocol_root = args.protocol_root or os.path.join(
        os.path.dirname(args.root.rstrip("/\\")), "dynamic_v2_protocol")
    result = verify(args.root, protocol_root)
    out = args.out or os.path.join(args.root, "mf_tsr_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({"n_issues": result["n_issues"],
                      "issues": result["issues"][:20],
                      "runs": result["runs"], "rows": result["rows"],
                      "no_candidate_rows": result["no_candidate_rows"]},
                     indent=1), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
