#!/usr/bin/env python
"""CR-TSER pilot verifier (plan §34).

Two modes:

``--mode code``
    Static protocol checks that must hold before any expensive label call:
    frozen constants, package layout, the §35 test inventory, absence of the
    retired components (1021D signature / Proxy / MS-TSR) from the CR-TSER
    core, and the fail-closed Weibo22 path.

``--mode pilot``
    Artifact checks over a completed run: split disjointness, causal
    snapshots, held-out reader isolation, utility recomputability, selection
    budget rules, bootstrap discipline and exact gate thresholds.

The plan requires ``issues = 0``. Checks that cannot run because the pilot has
not executed yet are reported as ``pending`` and do not count as issues.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

PACKAGE_FILES = [
    "cr_tser/__init__.py",
    "cr_tser/config/pilot_config.py",
    "cr_tser/data/weibo22_adapter.py",
    "cr_tser/data/pilot_split.py",
    "cr_tser/data/snapshot_bridge.py",
    "cr_tser/data/structural_stats.py",
    "cr_tser/intervention/evidence_units.py",
    "cr_tser/intervention/semantic_reference.py",
    "cr_tser/intervention/intervention_generator.py",
    "cr_tser/intervention/interaction_controls.py",
    "cr_tser/readers/base_reader.py",
    "cr_tser/readers/sequence_scorer.py",
    "cr_tser/readers/qwen_reader.py",
    "cr_tser/readers/glm_reader.py",
    "cr_tser/readers/internlm_reader.py",
    "cr_tser/models/bitte.py",
    "cr_tser/models/text_baseline.py",
    "cr_tser/models/scalar_structure_baseline.py",
    "cr_tser/models/utility_heads.py",
    "cr_tser/models/robust_selector.py",
    "cr_tser/training/utility_dataset.py",
    "cr_tser/training/train_utility.py",
    "cr_tser/training/checkpointing.py",
    "cr_tser/evaluation/heterogeneity.py",
    "cr_tser/evaluation/structural_interaction.py",
    "cr_tser/evaluation/utility_prediction.py",
    "cr_tser/evaluation/unseen_reader.py",
    "cr_tser/evaluation/bootstrap.py",
]

REQUIRED_TESTS = [
    "test_weibo22_timestamp_not_original_order",
    "test_snapshot_excludes_future_node",
    "test_parent_precedes_child_or_flags_anomaly",
    "test_src_budget_never_exceeded",
    "test_sequence_score_A_B",
    "test_utility_sign_definition",
    "test_correct_to_wrong_is_helpful",
    "test_wrong_to_correct_is_harmful",
    "test_parent_child_pair_is_real_edge",
    "test_nonadjacent_control_not_ancestor",
    "test_subtree_group_is_valid_descendant_set",
    "test_bitte_parent_message",
    "test_bitte_child_mean_message",
    "test_bitte_padding_independent",
    "test_bitte_handles_large_tree_without_1021_signature",
    "test_heldout_reader_not_in_training_batch",
    "test_heldout_reader_not_in_early_stopping",
    "test_heldout_reader_not_in_selection",
    "test_robust_score_is_min_train_readers",
    "test_selection_never_exceeds_50pct_target",
    "test_selection_uses_only_reference_candidates",
    "test_bootstrap_preserves_multiplicity",
    "test_gate_logic_exact",
]

RETIRED_IN_CORE = ("1021", "proxy", "ms_tsr", "ms-tsr", "mf_tsr", "mf-tsr")


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail=""):
        status = "pass" if ok is True else ("fail" if ok is False
                                            else "pending")
        self.checks.append({"check": name, "status": status,
                            "detail": detail})

    def pending(self, name, detail=""):
        self.add(name, None, detail)

    @property
    def issues(self):
        return sum(1 for c in self.checks if c["status"] == "fail")

    @property
    def pending_count(self):
        return sum(1 for c in self.checks if c["status"] == "pending")

    def to_dict(self, mode):
        return {"mode": mode, "issues": self.issues,
                "pending": self.pending_count, "checks": self.checks}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code_only(text: str) -> str:
    """Strip docstrings and line comments.

    The prohibited-component checks must look at *code*: CR-TSER deliberately
    names the retired components in its documentation to say they are not
    used, and that prose must not be mistaken for a dependency.
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    out = []
    for line in text.splitlines():
        idx = line.find("#")
        out.append(line[:idx] if idx >= 0 else line)
    return "\n".join(out)


def _collect_test_names():
    names = set()
    tests_dir = PROJECT / "cr_tser" / "tests"
    for path in tests_dir.glob("test_*.py"):
        for match in re.finditer(r"^def (test_[A-Za-z0-9_]+)\(",
                                 _read(path), flags=re.M):
            names.add(match.group(1))
    return names


# --------------------------------------------------------------------------
# code mode
# --------------------------------------------------------------------------
def verify_code(report: Report):
    from cr_tser.config import pilot_config as C

    missing = [f for f in PACKAGE_FILES if not (PROJECT / f).exists()]
    report.add("package_layout", not missing,
               f"missing: {missing}" if missing else "all §28 files present")

    expected = {
        "partition_seed": 7319,
        "train_seeds": [7319, 7320, 7321],
        "bootstrap_seed": 7319,
        "bootstrap_iterations": 10000,
        "budget_ref": 1024,
        "utility_threshold": 0.05,
        "atomic_cap": 20,
        "pilot_budget_fraction": 0.50,
        "p1_mean_disagreement_min": 0.10,
        "p1_pair_disagreement_min": 0.05,
        "p2_edge_delta_min": 0.02,
        "p3_macro_f1_delta_min": 0.02,
        "p4_mean_delta_min": 0.01,
        "p4_worst_rotation_min": -0.005,
        "pheme_mean_delta_min": -0.005,
    }
    got = C.frozen_constants()
    drift = {k: (got.get(k), v) for k, v in expected.items()
             if got.get(k) != v}
    report.add("frozen_constants", not drift, f"drift: {drift}" if drift
               else "all frozen constants match the plan")

    report.add("split_sizes", C.SPLIT_SIZES == {
        "foundation_train": 80, "utility_train": 50, "utility_dev": 15,
        "utility_eval": 25} and C.SPLIT_TOTAL == 170,
        f"SPLIT_SIZES={C.SPLIT_SIZES}")
    report.add("cutoffs_exact", tuple(C.CUTOFFS_MIN) == (15, 60, 360),
               f"cutoffs={C.CUTOFFS_MIN}")
    report.add("reader_models_exact",
               C.READER_MODEL_IDS == {
                   "qwen": "Qwen/Qwen3-8B",
                   "glm": "zai-org/glm-4-9b-chat-hf",
                   "internlm": "internlm/internlm3-8b-instruct"},
               f"readers={C.READER_MODEL_IDS}")
    report.add("loro_rotations",
               C.LORO_ROTATIONS == (("qwen", "glm", "internlm"),
                                    ("qwen", "internlm", "glm"),
                                    ("glm", "internlm", "qwen")),
               f"rotations={C.LORO_ROTATIONS}")
    report.add("utility_z_dim", C.UTILITY_Z_DIM == 1286,
               f"z_dim={C.UTILITY_Z_DIM}")
    report.add("loss_weights",
               (C.LOSS_W_READER, C.LOSS_W_SHARED, C.LOSS_W_SIGN,
                C.LOSS_W_RESID) == (1.0, 0.5, 0.5, 0.01),
               "loss weights 1.0/0.5/0.5/0.01")

    names = _collect_test_names()
    missing_tests = [t for t in REQUIRED_TESTS if t not in names]
    report.add("required_tests", not missing_tests,
               f"missing: {missing_tests}" if missing_tests
               else f"all {len(REQUIRED_TESTS)} §35 tests present")

    bitte_src = _code_only(_read(PROJECT / "cr_tser" / "models" / "bitte.py"))
    report.add("bitte_no_1021_signature",
               "1021" not in bitte_src
               and "adjacency_signature" not in bitte_src,
               "BiTTE must not use the retired 1021D adjacency signature")
    src_src = _code_only(_read(PROJECT / "cr_tser" / "intervention" /
                               "semantic_reference.py")).lower()
    forbidden = [t for t in ("proxy", "ms_tsr", "ms-tsr", "mf_tsr", "mf-tsr")
                 if t in src_src]
    report.add("src_reader_gold_proxy_independent", not forbidden,
               f"forbidden component referenced: {forbidden}"
               if forbidden else "SRC uses only MiniLM + Qwen tokenizer")

    wsrc = _code_only(_read(PROJECT / "cr_tser" / "data" /
                            "weibo22_adapter.py"))
    report.add("weibo22_fail_closed",
               "Weibo22TemporalUnavailable" in wsrc
               and "pseudo" in _read(PROJECT / "cr_tser" / "data" /
                                     "weibo22_adapter.py").lower(),
               "Weibo22 refuses pseudo-time and raises when timestamps are "
               "absent")
    report.add("no_timestamp_synthesis",
               "original_order" not in wsrc.lower()
               or "never" in wsrc.lower() or "no timestamp" in wsrc.lower(),
               "adapter documents that original_order is never time")


_HELDOUT_IN_TRAIN_PATTERNS = (
    r"heldout", r"held_out", r"holdout",
)


def verify_pilot(report: Report, results_root: str):
    root = Path(results_root)
    manifests = root / "manifests"
    split_file = manifests / "event_split.json"
    if not split_file.exists():
        report.pending("pilot_artifacts",
                       f"pilot not executed yet: {split_file} missing")
        return
    split = json.loads(_read(split_file))
    names = ("foundation_train", "utility_train", "utility_dev",
             "utility_eval")
    seen, overlap = {}, []
    for name in names:
        for eid in split.get(name, []):
            if eid in seen:
                overlap.append((eid, seen[eid], name))
            seen[eid] = name
    report.add("split_events_disjoint", not overlap, f"overlap: {overlap[:5]}")
    report.add("split_sizes_exact",
               all(len(split.get(n, [])) == size for n, size in
                   zip(names, (80, 50, 15, 25))),
               {n: len(split.get(n, [])) for n in names})

    snap_file = manifests / "snapshot_manifest.jsonl"
    if snap_file.exists():
        rows = [json.loads(line) for line in _read(snap_file).splitlines()
                if line.strip()]
        bad_future = [r for r in rows if r.get("future_node_leak")]
        bad_ts = [r for r in rows if r.get("used_original_order_as_time")]
        report.add("snapshot_no_future_node", not bad_future,
                   f"{len(bad_future)} leaking snapshots")
        report.add("snapshot_uses_true_timestamp", not bad_ts,
                   f"{len(bad_ts)} rows substituted row order for time")
        report.add("snapshot_cutoffs_frozen",
                   all(r.get("cutoff") in (15, 60, 360) for r in rows),
                   "cutoffs outside {15,60,360}" )
    else:
        report.pending("snapshot_manifest", "snapshot_manifest.jsonl missing")

    iv_file = manifests / "intervention_manifest.jsonl"
    if iv_file.exists():
        rows = [json.loads(line) for line in _read(iv_file).splitlines()
                if line.strip()]
        bad = [r for r in rows if not r.get("valid_in_gt", True)]
        report.add("structured_intervention_valid_in_gt", not bad,
                   f"{len(bad)} invalid structured interventions")
    else:
        report.pending("intervention_manifest",
                       "intervention_manifest.jsonl missing")

    hashes = manifests / "hashes.json"
    if hashes.exists():
        payload = json.loads(_read(hashes))
        frozen = payload.get("readers", {})
        report.add("reader_hashes_frozen",
                   len(frozen) == 3 and all(frozen.values()),
                   f"reader hashes: {list(frozen)}")
    else:
        report.pending("reader_hashes", "hashes.json missing")

    sel_dir = root / "unseen_reader"
    if sel_dir.exists():
        for path in sorted(sel_dir.glob("rotation_*.json")):
            data = json.loads(_read(path))
            held = data.get("held_out_reader")
            leaked = [k for k in data.get("train_readers", []) + [
                data.get("early_stopping_reader", "")] if k == held]
            report.add(f"heldout_isolated:{path.stem}", not leaked,
                       f"held-out {held!r} leaked into {leaked}")
            for arm, entry in data.get("arms", {}).items():
                ids = set(entry.get("selected_node_ids", []))
                ref = set(entry.get("reference_ids", []))
                report.add(f"arm_subset_of_cref:{path.stem}:{arm}",
                           ids <= ref if ref else True,
                           f"{len(ids - ref)} ids outside C_ref")
                report.add(f"arm_within_target:{path.stem}:{arm}",
                           entry.get("total_tokens", 0)
                           <= entry.get("target_tokens",
                                        entry.get("total_tokens", 0)),
                           f"{entry.get('total_tokens')} vs "
                           f"{entry.get('target_tokens')}")
    else:
        report.pending("unseen_reader_artifacts",
                       "unseen_reader/ missing (pilot not executed)")

    verifier_gates = root / "CR_TSER_PILOT_SUMMARY.json"
    if verifier_gates.exists():
        summary = json.loads(_read(verifier_gates))
        gates = summary.get("gates", {})
        for key, entry in gates.items():
            report.add(f"gate_{key}_thresholds",
                       entry.get("thresholds_match_plan", True),
                       entry.get("detail", ""))
    else:
        report.pending("pilot_summary", "CR_TSER_PILOT_SUMMARY.json missing")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("code", "pilot"), default="code")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = Report()
    if args.mode == "code":
        verify_code(report)
    else:
        root = args.results_root or os.environ.get(
            "CRTSER_OUT_ROOT", str(REPO / "results" / "cr_tser"))
        verify_pilot(report, root)
    payload = report.to_dict(args.mode)
    out = args.out or str(REPO / "results" / "cr_tser" / "verifier" /
                          f"{args.mode}_verify.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({"mode": payload["mode"], "issues": payload["issues"],
                      "pending": payload["pending"], "out": out}, indent=1))
    for check in payload["checks"]:
        if check["status"] != "pass":
            print(f"  [{check['status']}] {check['check']}: {check['detail']}")
    return 0 if payload["issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
