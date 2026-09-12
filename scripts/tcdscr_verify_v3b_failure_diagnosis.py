#!/usr/bin/env python
"""Verifier for the V3-B failure diagnosis (§29).

Independently re-derives the diagnosis inputs and checks that nothing moved:

  - the frozen V3-B artifacts keep their recorded sha256 (manifest, prompts,
    parsed, raw generations) — no resampling, no regenerated output, no parse
    or gold edits;
  - CC/CW/WC/WW counts and Static/MS Macro-F1 reproduce from the frozen parsed
    rows and match the original V3-B statistics;
  - Qwen citations are remapped from the existing parsed outputs only;
  - node-level features come from the corresponding causal snapshot (no future
    node, text or edge);
  - the cross-fold audit is fold metadata only;
  - no heuristic/selector code was written by the diagnosis step.

Writes ``diagnosis_verify.json`` and refreshes the report verdict line.
Requires ``issues = 0``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tcdscr_diagnose_v3b_failure as D  # noqa: E402

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
PARTITION_SEED = 3090
REQUIRED = (
    "paired_outcome_groups.json", "compression_severity_analysis.json",
    "evidence_count_threshold_analysis.json",
    "removed_evidence_feature_analysis.json",
    "reader_evidence_use_analysis.json",
    "reader_sensitive_gap_analysis.json",
    "proxy_reader_margin_transfer.json",
    "confidence_transition_analysis.json", "label_asymmetry_analysis.json",
    "cutoff_failure_analysis.json",
    "selection_pressure_failure_analysis.json",
    "utility_reader_alignment.json", "ms_removal_reader_alignment.json",
    "structural_role_analysis.json", "text_characteristic_analysis.json",
    "cross_dataset_comparison.json", "cross_fold_development_audit.json",
    "V3B_FAILURE_DIAGNOSIS_REPORT.md",
)
DEFAULT_READER = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"
DEFAULT_DIAG = "/data/jyz/next/llm/results/tcdscr/v3b_failure_diagnosis"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _parsed_pairs(reader_root, ds):
    parsed = load_jsonl(os.path.join(reader_root, "parsed", f"{ds}.jsonl"))
    by = {(r["sample_id"], r["arm"]): r for r in parsed}
    pairs = []
    for sid in sorted({k[0] for k in by}):
        st, ms = by.get((sid, "static")), by.get((sid, "ms"))
        if st is None or ms is None or st["parse_failure"] or ms["parse_failure"]:
            continue
        pairs.append((st, ms))
    return pairs


def _expected_groups(reader_root):
    out = {}
    for ds in DATASETS:
        counts = collections.Counter()
        for st, ms in _parsed_pairs(reader_root, ds):
            counts[D.outcome_group(bool(st["correct"]), bool(ms["correct"]))] += 1
        out[ds] = counts
    return out


def _macro_f1(pairs, arm):
    tp = fp = fn = tn = 0
    for st, ms in pairs:
        row = st if arm == "static" else ms
        g, p = row["gold_label"], row["parsed_label"]
        if g == "RUMOR" and p == "RUMOR":
            tp += 1
        elif g == "NON_RUMOR" and p == "RUMOR":
            fp += 1
        elif g == "RUMOR" and p == "NON_RUMOR":
            fn += 1
        else:
            tn += 1
    return (D._f1(tp, fp, fn) + D._f1(tn, fn, fp)) / 2


def _check_required(diag_root, issues):
    for name in REQUIRED:
        if not os.path.exists(os.path.join(diag_root, name)):
            issues.append(f"missing diagnosis artifact: {name}")
    for name in os.listdir(diag_root):
        if name.endswith(".py"):
            issues.append(f"diagnosis directory contains code: {name}")


def _check_frozen(reader_root, diag_root, issues):
    path = os.path.join(diag_root, "frozen_artifacts.json")
    if not os.path.exists(path):
        issues.append("missing frozen_artifacts.json")
        return
    frozen = load_json(path)
    checks = [("sampling_manifest_sha256",
               os.path.join(reader_root, "sampling_manifest.json"))]
    for ds in DATASETS:
        checks.append((f"parsed_sha256/{ds}",
                       os.path.join(reader_root, "parsed", f"{ds}.jsonl")))
        checks.append((f"raw_generations_sha256/{ds}",
                       os.path.join(reader_root, "raw_generations",
                                    f"{ds}.jsonl")))
        checks.append((f"prompts_sha256/{ds}",
                       os.path.join(reader_root, "prompts", f"{ds}.jsonl")))
    for key, fpath in checks:
        recorded = frozen.get(key.split("/")[0])
        if isinstance(recorded, dict):
            recorded = recorded.get(key.split("/")[1])
        if recorded is None or D.sha256_file(fpath) != recorded:
            issues.append(f"frozen artifact changed: {key}")
    if not frozen.get("no_new_qwen_generation"):
        issues.append("diagnosis does not assert no_new_qwen_generation")


def _check_groups(reader_root, diag_root, issues):
    groups = load_json(os.path.join(diag_root, "paired_outcome_groups.json"))
    expected = _expected_groups(reader_root)
    for ds in DATASETS:
        for g in ("CC", "CW", "WC", "WW"):
            if groups[ds].get(g) != expected[ds].get(g):
                issues.append(f"{ds}: {g} count {groups[ds].get(g)} != "
                              f"recomputed {expected[ds].get(g)}")
        stats_path = os.path.join(reader_root, "statistics", f"{ds}.json")
        stats = load_json(stats_path)
        out = stats["paired_outcomes"]
        mapping = {"CC": "both_correct", "CW": "correct_to_wrong",
                   "WC": "wrong_to_correct", "WW": "both_wrong"}
        for g, key in mapping.items():
            if groups[ds].get(g) != out.get(key):
                issues.append(f"{ds}: {g} does not reproduce the original "
                              f"V3-B summary ({out.get(key)})")


def _check_metrics(reader_root, diag_root, issues):
    cross = load_json(os.path.join(diag_root, "cross_dataset_comparison.json"))
    for ds in DATASETS:
        pairs = _parsed_pairs(reader_root, ds)
        for arm in ("static", "ms"):
            recomputed = _macro_f1(pairs, arm)
            recorded = cross[ds].get(f"{arm}_macro_f1")
            if recorded is None or abs(recomputed - recorded) > 1e-9:
                issues.append(f"{ds}: {arm} Macro-F1 does not reproduce")
        stats = load_json(os.path.join(reader_root, "statistics",
                                       f"{ds}.json"))
        for arm in ("static", "ms"):
            if abs(stats["detection"][arm]["macro_f1"]
                   - cross[ds][f"{arm}_macro_f1"]) > 1e-9:
                issues.append(f"{ds}: {arm} Macro-F1 != original V3-B "
                              "detection")


def _check_citations(reader_root, issues):
    """Qwen citations must be exactly the frozen parsed evidence ids mapped
    through the frozen prompt mapping."""
    for ds in DATASETS:
        prompts = {(r["sample_id"], r["arm"]): r
                   for r in load_jsonl(os.path.join(reader_root, "prompts",
                                                    f"{ds}.jsonl"))}
        for r in load_jsonl(os.path.join(reader_root, "parsed",
                                         f"{ds}.jsonl")):
            if r["parse_failure"]:
                continue
            allowed = prompts[(r["sample_id"], r["arm"])]["evidence_ids"]
            stray = [e for e in r["evidence_ids"] if e not in allowed]
            if stray:
                issues.append(f"{ds}/{r['sample_id']}/{r['arm']}: citation "
                              "outside the frozen prompt mapping")


def _check_node_features(reader_root, diag_root, issues):
    """Sampled node-level features must lie inside their causal snapshot."""
    path = os.path.join(diag_root, "node_level_features.jsonl")
    if not os.path.exists(path):
        issues.append("missing node_level_features.jsonl")
        return
    rows = load_jsonl(path)
    sampled = rows[::max(1, len(rows) // 40)][:40]
    for row in sampled:
        snap_ids = set(row["snapshot_node_ids"])
        for nid in (row["static_evidence_node_ids"]
                    + row["ms_evidence_node_ids"]
                    + row["static_cited_node_ids"]):
            if nid not in snap_ids:
                issues.append(f"{row['sample_id']}: node {nid} outside the "
                              "causal snapshot")
                break
        feats = row["nodes"]
        if not set(feats).issubset(snap_ids):
            issues.append(f"{row['sample_id']}: node features include nodes "
                          "outside the snapshot")


def _check_validation_only(reader_root, diag_root, issues):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry
    manifest = load_json(os.path.join(reader_root, "sampling_manifest.json"))
    for ds in DATASETS:
        cfg = config_from_env(ds)
        registry = event_label_registry(ds, cfg)
        val = {}
        for fold in FOLDS:
            val[fold] = set(build_primary_fold_split(
                registry, fold, seed=PARTITION_SEED)["validation"])
        for s in manifest["samples"]:
            if s["dataset"] != ds:
                continue
            if s["event_id"] not in val[s["fold"]]:
                issues.append(f"{ds}/{s['sample_id']}: not in fold"
                              f"{s['fold']} validation")
                break


def _check_cross_fold_metadata_only(diag_root, issues):
    audit = load_json(os.path.join(diag_root, "cross_fold_development_audit.json"))
    for ds, entry in audit.items():
        for eid, meta in entry.get("per_event", {}).items():
            if set(meta) != {"in_validation_folds", "in_test_folds"}:
                issues.append(f"{ds}/{eid}: cross-fold entry carries more than "
                              "fold metadata")
                break


def _check_no_heuristic(issues):
    src = (os.path.join(HERE, "tcdscr_diagnose_v3b_failure.py"))
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    for banned in ("tcdscr_train_selector", "state_dict()", ".backward(",
                   "optimizer", "minimal_set_refiner import"):
        if banned in text:
            issues.append(f"diagnosis script references training/algorithm "
                          f"machinery: {banned}")
    if re.search(r'\.py",\s*"w"', text):
        issues.append("diagnosis script writes python source")


def _check_structural_consistency(diag_root, issues):
    """§7/§16 structural checks.

    ``snapshot["edge_index"]`` stores node *positions*, so the aggregation must
    map positions to ids; leaf must be ``child_count == 0`` and never the
    undirected degree.  On every checked snapshot the edge count must equal the
    summed child count and equal half the summed undirected degree.
    """
    path = os.path.join(diag_root, "node_level_features.jsonl")
    if not os.path.exists(path):
        issues.append("missing node_level_features.jsonl")
        return
    rows = load_jsonl(path)
    step = max(1, len(rows) // 40)
    for row in rows[::step][:40]:
        c = row.get("consistency")
        if not c:
            issues.append(f"{row['sample_id']}: missing structural "
                          "consistency record")
            continue
        if c["sum_child_count"] != c["n_edges"]:
            issues.append(f"{row['sample_id']}: sum(child_count)="
                          f"{c['sum_child_count']} != edges={c['n_edges']}")
        if c["sum_degree"] != 2 * c["n_edges"]:
            issues.append(f"{row['sample_id']}: sum(degree)="
                          f"{c['sum_degree']} != 2*edges={2 * c['n_edges']}")
        for nid, feats in row.get("nodes", {}).items():
            if feats.get("is_leaf") != (feats.get("child_count", 0) == 0):
                issues.append(f"{row['sample_id']}/{nid}: leaf flag does not "
                              "follow child_count")
                break
        # a reply node may legitimately be a leaf, but the aggregation must not
        # have collapsed: at least one node in the snapshot carries an edge
        if c["n_edges"] > 0 and c["sum_degree"] == 0:
            issues.append(f"{row['sample_id']}: degree collapsed to zero on a "
                          "snapshot with edges")
    struct_path = os.path.join(diag_root, "structural_role_analysis.json")
    if not os.path.exists(struct_path):
        issues.append("missing structural_role_analysis.json")
        return
    struct = load_json(struct_path)
    for ds, groups in struct.items():
        means = [s.get("mean_degree") for s in groups.values()
                 if s and s.get("n")]
        if means and all(m in (None, 0) for m in means):
            issues.append(f"{ds}: mean_degree collapsed across every "
                          "structural group (position/id key bug)")
        for gname, s in groups.items():
            if not s or not s.get("n"):
                continue
            leaf_rate = s.get("is_leaf_rate", s.get("leaf_rate"))
            if leaf_rate is not None and not 0.0 <= leaf_rate <= 1.0:
                issues.append(f"{ds}/{gname}: leaf rate out of range")


def _refresh_report(diag_root, result):
    path = os.path.join(diag_root, "V3B_FAILURE_DIAGNOSIS_REPORT.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    detail = (f" (samples={result['samples']}, groups={result['groups']}, "
              f"root_causes={result['root_cause_flags']})")
    pattern = re.compile(r"^issues = .*$", re.M)
    if not pattern.search(text):
        return
    text = pattern.sub(f"issues = {result['n_issues']}{detail}", text,
                       count=1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def verify(reader_root, diag_root):
    issues = []
    _check_required(diag_root, issues)
    _check_frozen(reader_root, diag_root, issues)
    _check_groups(reader_root, diag_root, issues)
    _check_metrics(reader_root, diag_root, issues)
    _check_citations(reader_root, issues)
    _check_node_features(reader_root, diag_root, issues)
    _check_structural_consistency(diag_root, issues)
    _check_validation_only(reader_root, diag_root, issues)
    _check_cross_fold_metadata_only(diag_root, issues)
    _check_no_heuristic(issues)
    groups = load_json(os.path.join(diag_root,
                                    "paired_outcome_groups.json"))
    summary = load_json(os.path.join(diag_root, "diagnosis_summary.json"))
    return {
        "n_issues": len(issues),
        "issues": issues[:40],
        "samples": sum(groups[ds]["n"] for ds in DATASETS),
        "groups": {ds: {g: groups[ds][g] for g in
                        ("CC", "CW", "WC", "WW")} for ds in DATASETS},
        "root_cause_flags": summary["root_cause_flags"],
        "recommendation": summary["recommendation"],
        "checks": [
            "frozen manifest / prompts / parsed / raw sha256 unchanged",
            "CC/CW/WC/WW reproduce the frozen parsed rows and V3-B summary",
            "Static/MS Macro-F1 reproduce the original detection",
            "citations mapped only from frozen parsed outputs",
            "node features inside the causal snapshot (no future nodes)",
            "edge positions converted to node IDs before structural "
            "aggregation",
            "leaf defined by child_count == 0 (not undirected degree)",
            "sum(child_count) == edge count on checked snapshots",
            "sum(degree) == 2 * edge count on checked snapshots",
            "frozen V3-B artifact hashes unchanged",
            "diagnosis samples are fold-local validation events",
            "cross-fold audit is fold metadata only",
            "diagnosis writes no selector/heuristic code",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-root", default=DEFAULT_READER)
    ap.add_argument("--diag-root", default=DEFAULT_DIAG)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = verify(args.reader_root, args.diag_root)
    out = args.out or os.path.join(args.diag_root, "diagnosis_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    _refresh_report(args.diag_root, result)
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
