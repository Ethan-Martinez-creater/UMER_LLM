#!/usr/bin/env python
"""MS-TSR (Dynamic V3-A) verifier.

Checks, with issues = 0 required:
  - every validation event present at all 6 cutoffs for all four alphas,
    including candidate_count == 0 snapshots;
  - no test event was read (run ids == fold validation ids, disjoint from
    the reconstructed test ids);
  - no future nodes (selected ids live in the current snapshot) and no
    future memory (M_t in the current snapshot, M_{t-1} in the previous
    cutoff, empty at the first cutoff);
  - frozen corrected-E2 encoder / selector / proxy with before-after
    matching checksums, zero trainable parameters, no test/Qwen flags;
  - budget <= 1024, alpha only from the frozen grid, ADD+REMOVE moves only
    (no SWAP), candidate pool bounded by |S_static| + |M_prev| + 64;
  - sufficiency consistency: compressed non-fallback rows preserve the
    Static decision and satisfy m(S) >= alpha * m_static; fallback rows
    keep the Static set exactly; no-candidate rows are empty;
  - required artifacts exist (fold deliverables, diagnostics, report).
Writes ms_tsr_verify.json.
"""
import argparse
import ast
import json
import os
import re
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
ALPHA_STRINGS = ("0.8", "0.9", "0.95", "1.0")
BUDGET = 1024
TOP_K_POOL = 64
MODELS_DIR = Path(HERE).resolve().parents[0] / "project" / "tcdscr" / \
    "models"
REQUIRED_FOLD = ("alpha_grid.json", "best_config.json",
                 "validation_predictions.jsonl", "sufficiency_dynamics.json",
                 "compression_metrics.json")

_FOLD_CACHE = {}


def _fold_snapshots(dataset, fold):
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


_P_FULL_ALLOWED_CALLS = frozenset((
    "dual_view_agreement", "view", "reshape", "argmax", "int", "bool",
    "float", "str"))


def _p_full_objective_violations(tree):
    """Yield (lineno, detail) for every p_full use that is not gate-only.

    ``p_full_probs`` must be consumed by the dual-view gate or recorded as an
    argmax decision; walking the whole ancestor chain catches arithmetic
    (``p_full_probs * w``), censoring transforms (``.detach()``), and any other
    call that would turn the teacher distribution back into an objective term.
    """
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    violations = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id == "p_full_probs"):
            continue
        cur = parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.BinOp, ast.UnaryOp, ast.BoolOp,
                                ast.AugAssign)):
                violations.append((node.lineno, type(cur).__name__))
                break
            if isinstance(cur, ast.Call):
                fn = cur.func
                name = fn.id if isinstance(fn, ast.Name) else (
                    fn.attr if isinstance(fn, ast.Attribute) else None)
                if name not in _P_FULL_ALLOWED_CALLS:
                    violations.append((node.lineno, f"{name}()"))
                    break
            cur = parents.get(id(cur))
    return violations


def _model_audit(issues):
    """Objective audit.

    V3 minimises Cost(S) subject to Sufficient_t(S; alpha).  Per design §5 the
    full causal encoder is no longer an optimisation target -- only a second
    independent decision view -- so ``p_full_probs`` may appear solely as the
    input of the dual-view gate and in the recorded full prediction.  Any
    teacher-fidelity objective term (KL to p_full and friends), any non-gate
    use of ``p_full_probs`` (arithmetic, loss, gain), or any SWAP move is a
    violation.
    """
    fidelity_terms = ("set_distortion", "batch_set_distortions", "kl_div",
                      "kl_divergence", "KLDivLoss")
    for fname in ("sufficiency.py", "minimal_set_refiner.py"):
        src = (MODELS_DIR / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    names.add(a.name)
        for banned in fidelity_terms:
            if banned in names:
                issues.append(f"{fname}: teacher-fidelity objective term "
                              f"{banned} must not drive V3")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "SWAP":
                issues.append(f"{fname}: SWAP move present in primary V3")
        for lineno, detail in _p_full_objective_violations(tree):
            issues.append(f"{fname}:{lineno}: p_full_probs used outside the "
                          f"dual-view gate ({detail})")


def verify(root):
    issues = []
    summary = {"runs": 0, "rows": 0, "no_candidate_rows": 0}
    _model_audit(issues)
    if not os.path.exists(os.path.join(root, "MS_TSR_VALIDATION_REPORT.md")):
        issues.append("missing MS_TSR_VALIDATION_REPORT.md")
    if not os.path.exists(os.path.join(root, "ms_tsr_summary.json")):
        issues.append("missing ms_tsr_summary.json")
    for dataset in DATASETS:
        for fold in FOLDS:
            fold_dir = os.path.join(root, dataset, f"fold{fold}")
            for name in REQUIRED_FOLD:
                if not os.path.exists(os.path.join(fold_dir, name)):
                    issues.append(f"missing fold artifact: {dataset}/"
                                  f"fold{fold}/{name}")
            snap_cache, fold_val_ids, fold_test_ids = _fold_snapshots(
                dataset, fold)
            for seed in SEEDS:
                run_dir = os.path.join(root, "runs", dataset,
                                       f"fold{fold}_seed{seed}")
                man_path = os.path.join(run_dir, "run_manifest.json")
                pred_path = os.path.join(run_dir,
                                         "validation_predictions.jsonl")
                if not os.path.exists(man_path) or \
                        not os.path.exists(pred_path):
                    issues.append(f"missing run artifacts: {run_dir}")
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
                if man.get("test_split_read") is not False or \
                        man.get("qwen_called") is not False:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "test/Qwen flag is wrong")
                if man.get("budget") != BUDGET:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "manifest budget != 1024")
                per_seed_ids = set()
                counts = {}
                with open(pred_path, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        summary["rows"] += 1
                        per_seed_ids.add(r["event_id"])
                        counts[(r["event_id"], r["cutoff"])] = counts.get(
                            (r["event_id"], r["cutoff"]), 0) + 1
                        astr = str(r["alpha"])
                        if astr not in ALPHA_STRINGS:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "alpha outside the frozen grid")
                        if r["budget"] != BUDGET:
                            issues.append("row budget != 1024")
                        if r["ms_evidence_tokens"] > BUDGET or \
                                r["static_evidence_tokens"] > BUDGET:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "token budget exceeded")
                        for mv in r["accepted_adds"]:
                            if mv.get("move_type", "ADD") != "ADD" or \
                                    "token_cost" not in mv:
                                issues.append("malformed ADD move")
                        for mv in r["accepted_removes"]:
                            if mv.get("move_type", "REMOVE") != "REMOVE":
                                issues.append("malformed REMOVE move")
                        for key in ("accepted_swaps", "swap_count",
                                    "swap_positions"):
                            if key in r:
                                issues.append(f"{r['event_id']}/"
                                              f"{r['cutoff']}: SWAP output "
                                              "field present")
                        cur_ids = snap_cache[int(r["cutoff"])].get(
                            r["event_id"], set())
                        if not set(r["ms_selected_node_ids"]) <= cur_ids:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "future node in MS selection")
                        if not set(r["static_selected_node_ids"]) <= \
                                cur_ids:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "future node in Static selection")
                        if not set(r["memory_current_ids"]) <= cur_ids:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "future memory in M_t")
                        idx = PRIMARY_CUTOFFS.index(int(r["cutoff"]))
                        if idx > 0:
                            prev_ids = snap_cache[
                                PRIMARY_CUTOFFS[idx - 1]].get(
                                r["event_id"], set())
                            if not set(r["memory_previous_ids"]) <= \
                                    prev_ids:
                                issues.append(f"{r['event_id']}/"
                                              f"{r['cutoff']}: memory "
                                              "contains future nodes")
                        elif r["memory_previous_ids"]:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "first cutoff memory not empty")
                        pool_bound = (len(r["static_selected_node_ids"])
                                      + len(r["memory_previous_ids"])
                                      + TOP_K_POOL)
                        if r["pool_size"] > pool_bound:
                            issues.append(f"{r['event_id']}/{r['cutoff']}: "
                                          "pool larger than the frozen "
                                          "bound")
                        if r["candidate_count"] == 0:
                            summary["no_candidate_rows"] += 1
                            if r["ms_selected_node_ids"] or \
                                    r["static_selected_node_ids"] or \
                                    r["ms_evidence_tokens"] != 0:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "no-candidate row is not empty")
                        elif r["fallback_to_static"]:
                            if set(r["ms_selected_node_ids"]) != set(
                                    r["static_selected_node_ids"]):
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "fallback row differs from Static")
                        elif r["compression_attempted"]:
                            if r["static_prediction"] != \
                                    r["ms_prediction"]:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "compressed set does not preserve the "
                                    "Static decision")
                            if r["ms_margin"] < float(r["alpha"]) * \
                                    r["static_margin"] - 1e-6:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "margin retention violated")
                            if "pool_node_ids" in r and not set(
                                    r["ms_selected_node_ids"]) <= set(
                                        r["pool_node_ids"]) and                                     r["compression_attempted"]:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "MS set is not a subset of the frozen "
                                    "candidate pool")
                            if r["ms_evidence_tokens"] > \
                                    r["static_evidence_tokens"] and \
                                    not r["fallback_to_static"]:
                                issues.append(
                                    f"{r['event_id']}/{r['cutoff']}: "
                                    "compression increased tokens")
                if per_seed_ids & fold_test_ids:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: test "
                                  "events were read")
                if per_seed_ids != fold_val_ids:
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "validation coverage mismatch "
                                  f"({len(per_seed_ids)} vs "
                                  f"{len(fold_val_ids)})")
                for ev in fold_val_ids:
                    for c in PRIMARY_CUTOFFS:
                        n = counts.get((ev, str(c)), 0)
                        if n != len(ALPHA_STRINGS):
                            issues.append(
                                f"{dataset}/fold{fold}_seed{seed}: "
                                f"({ev},{c}) has {n} alpha rows, expected "
                                f"{len(ALPHA_STRINGS)}")
                summary["runs"] += 1
    return {"issues": issues, "n_issues": len(issues), **summary,
            "checks": [
                "validation coverage at all 6 cutoffs x all alphas",
                "no-candidate snapshots included", "no test events read",
                "no future nodes", "no future memory",
                "frozen encoder/selector/proxy", "budget <= 1024",
                "alpha in frozen grid", "ADD+REMOVE only (no SWAP)",
                "candidate pool bound",
                "sufficiency consistency (decision + margin)",
                "fallback keeps the Static set"]}


def _refresh_report(root, result):
    """Write the verifier verdict back into MS_TSR_VALIDATION_REPORT.md.

    The report is produced by the summarizer, which runs before the verifier,
    so its ``## Verifier`` line starts out as "not run" and has to be updated
    here rather than left stale.
    """
    path = os.path.join(root, "MS_TSR_VALIDATION_REPORT.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    detail = (f" (runs={result.get('runs')}, rows={result.get('rows')}, "
              f"no_candidate_rows={result.get('no_candidate_rows')})")
    pattern = re.compile(r"^issues = .*$", re.M)
    if not pattern.search(text):
        return
    text = pattern.sub(f"issues = {result['n_issues']}{detail}", text,
                       count=1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/dynamic_v3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = verify(args.root)
    out = args.out or os.path.join(args.root, "ms_tsr_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    _refresh_report(args.root, result)
    print(json.dumps({"n_issues": result["n_issues"],
                      "issues": result["issues"][:20],
                      "runs": result["runs"], "rows": result["rows"],
                      "no_candidate_rows": result["no_candidate_rows"]},
                     indent=1), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
