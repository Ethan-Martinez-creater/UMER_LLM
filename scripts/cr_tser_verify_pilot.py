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
    _verify_review_fixes(report)
    _verify_semantics(report)


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text).split())))}


def _semantic_fixture():
    units = [{"node_id": f"n{i}", "reply_text": f"reply {i} body",
              "parent_text": "parent body", "parent_id": None,
              "timestamp": i, "elapsed_seconds": i, "snapshot_order": i,
              "depth": 1} for i in range(1, 7)]
    ids = [u["node_id"] for u in units]
    src = {"selected_node_ids": ids, "ranked_node_ids": ids,
           "unit_token_costs": {n: 60 for n in ids}, "total_tokens": 360,
           "budget_ref": 1024, "n_selected": len(ids), "utilization": 0.2,
           "rank_percentile": {n: i / len(ids) for i, n in enumerate(ids)},
           "relevance": {n: 1.0 - i / 10 for i, n in enumerate(ids)}}
    return units, src


def _verify_semantics(report):
    """Execute the review-round contracts instead of grepping for them."""
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # --- normalized Weibo22 path is wired through every stage ---
    try:
        from cr_tser.config.pilot_config import PilotPaths
        has_field = "weibo22_normalized" in PilotPaths().__dict__
        import cr_tser_common as common
        uses = "weibo22_normalized" in _read(REPO / "scripts" /
                                             "cr_tser_common.py")
        stages = {name: "weibo22_normalized" in _script(name) or
                  "load_dataset_events" in _script(name)
                  for name in ("cr_tser_p0_audit.py", "cr_tser_build_manifests.py",
                               "cr_tser_generate_labels.py",
                               "cr_tser_train_predictors.py",
                               "cr_tser_run_selection.py")}
        report.add("weibo22_normalized_path_wired",
                   has_field and uses and all(stages.values()),
                   f"field={has_field} common={uses} stages={stages}")
    except Exception as exc:  # pragma: no cover - defensive
        report.add("weibo22_normalized_path_wired", False, f"error: {exc}")

    # --- P0 boundary failure must block PASS ---
    try:
        import cr_tser_p0_audit as p0
        audit = {"verdict": "WEIBO22_TEMPORAL_READY"}
        readers = {"qwen": {"model_path_exists": True, "loaded": True}}
        bad = {"qwen": {"identical_predictions": True, "boundaries_ok": False}}
        good = {"qwen": {"identical_predictions": True, "boundaries_ok": True}}
        blocked = p0.evaluate_readiness(audit, readers, bad, True, True)
        allowed = p0.evaluate_readiness(audit, readers, good, True, True)
        report.add("p0_boundary_failure_blocks_pass",
                   blocked["P0"] == "P0_FAIL"
                   and allowed["P0"] == "P0_PASS",
                   f"bad={blocked['P0']} good={allowed['P0']}")
    except Exception as exc:
        report.add("p0_boundary_failure_blocks_pass", False, f"error: {exc}")

    # --- selector fail-closed coverage + S6 consumes legacy scores ---
    try:
        from cr_tser.models.robust_selector import (MissingPredictionError,
                                                    build_arm)
        units, src = _semantic_fixture()
        tokenizer = _FakeTokenizer()
        ids = src["selected_node_ids"]
        full = {n: 0.2 for n in ids}
        arm_a = build_arm("S6_legacy_utility_tm", units, src,
                          {"legacy": {n: float(i) for i, n in enumerate(ids)}},
                          tokenizer, ["qwen", "glm"], seed=7319)
        arm_b = build_arm("S6_legacy_utility_tm", units, src,
                          {"legacy": {n: float(len(ids) - i)
                                      for i, n in enumerate(ids)}},
                          tokenizer, ["qwen", "glm"], seed=7319)
        report.add("s6_consumes_legacy_scores",
                   arm_a["selected_node_ids"] != arm_b["selected_node_ids"],
                   "different legacy scores must change the S6 subset")
        partial = {n: 0.2 for n in ids[:-1]}
        try:
            build_arm("S5_cross_reader_robust", units, src,
                      {"qwen": partial, "glm": full}, tokenizer,
                      ["qwen", "glm"], seed=7319)
            raised = False
        except MissingPredictionError:
            raised = True
        report.add("selector_missing_prediction_fails_closed", raised,
                   "S5 must refuse a candidate set without full coverage")
    except Exception as exc:
        report.add("s6_consumes_legacy_scores", False, f"error: {exc}")
        report.add("selector_missing_prediction_fails_closed", False,
                   f"error: {exc}")

    # --- cache identity fail-closed on tokenizer/template substitution ---
    try:
        import cr_tser_generate_labels as labels
        row = {f: "v" for f in labels.FINGERPRINT_FIELDS}
        labels._verify_cached(row, dict(row), "k")
        raised = 0
        for field in ("tokenizer_hash", "chat_template_hash",
                      "reader_identity_hash", "prompt_ids_hash"):
            try:
                labels._verify_cached(row, {**row, field: "other"}, "k")
            except labels.CacheIdentityMismatch:
                raised += 1
        report.add("cache_identity_substitution_fails_closed", raised == 4,
                   f"{raised}/4 substitutions detected")
    except Exception as exc:
        report.add("cache_identity_substitution_fails_closed", False,
                   f"error: {exc}")

    # --- P3 predictions come from the model's own sign head ---
    try:
        import cr_tser_run_pilot as pilot
        rec = pilot._pred_record("HELPFUL", 0.5,
                                 {"utility": 0.3, "probs": [0.1, 0.8, 0.1],
                                  "predicted_sign": "NEUTRAL"}, active=True)
        # predicted sign is the artifact's sign-head argmax, not a threshold on
        # the ground-truth utility, and no correctness transition is read
        src_text = _script("cr_tser_run_pilot.py")
        report.add("p3_predicted_sign_from_sign_head",
                   rec["pred_sign"] == "NEUTRAL"
                   and rec["pred_cont"] == 0.3
                   and "predicted_sign" in src_text
                   and "_sign_of" not in src_text,
                   "P3 must use the model sign head, never correctness")
        # B0/B1/B3 each carry their own artifact predictions
        report.add("p3_per_model_predictions",
                   "per_model_metrics" in src_text
                   and 'collected[name]' in src_text
                   and "B1_scalar_structure" in src_text,
                   "each model contributes independent predictions/metrics")
    except Exception as exc:
        report.add("p3_predicted_sign_from_sign_head", False, f"error: {exc}")
        report.add("p3_per_model_predictions", False, f"error: {exc}")

    # --- label cap does not limit inference candidates ---
    try:
        uds = _pkg_code("cr_tser/training/utility_dataset.py")
        tp = _script_code("cr_tser_train_predictors.py")
        report.add("inference_covers_all_candidates",
                   "infer_rows" in uds and "infer_z" in tp
                   and "infer_keys" in tp and "prediction_coverage" in tp,
                   "label-capped rows must not limit inference candidates")
    except Exception as exc:
        report.add("inference_covers_all_candidates", False, f"error: {exc}")

    # --- S3a/S3b are single-reader trained ---
    report.add("s3_single_reader_trained",
               "def train_single_reader" in _script(
                   "cr_tser_train_predictors.py")
               and "_load_single_predictions" in _script(
                   "cr_tser_run_selection.py")
               and "training_readers != [reader]" in _script(
                   "cr_tser_run_selection.py"),
               "S3a/S3b must come from single-reader-trained artifacts")


def _pkg(rel):
    return _read(PROJECT / rel)


def _pkg_code(rel):
    return _code_only(_pkg(rel))


def _script(name):
    return _read(REPO / "scripts" / name)


def _script_code(name):
    return _code_only(_script(name))


def _verify_review_fixes(report):
    """Checks added by the code-review fix round (plan §3.1.B, §9, §20, §24, §31–§33)."""
    snap = _pkg_code("cr_tser/data/snapshot_bridge.py")
    report.add("cr_tser_snapshot_uncapped",
               "MAX_NODES_CAP = None" in snap and "1021" not in snap
               and '"cap_hit": False' in snap,
               "CR-TSER snapshot must not inherit the TC-DSCR 1021 cap")

    uds = _pkg_code("cr_tser/training/utility_dataset.py")
    report.add("loro_isolation_entrance",
               "def filter_rows_for_readers" in uds
               and "def assert_only_readers" in uds
               and "assert_groups_only_readers(self.groups" in uds,
               "dataset fails closed when a held-out reader is present")
    tp = _script_code("cr_tser_train_predictors.py")
    report.add("loro_isolation_wired",
               "filter_rows_for_readers(all_labels, train_readers)" in tp
               and "held_out_rows_filtered" in tp
               and "assert_groups_only_readers(groups, train_readers" in tp,
               "training entry filters the cache to the two training readers")

    ek = _pkg_code("cr_tser/intervention/evidence_units.py")
    report.add("evidence_key_defined",
               "def evidence_key(" in ek and "def evidence_key_parts(" in ek,
               "canonical evidence key defined exactly once")
    contract_users = {
        "utility_dataset": _pkg_code("cr_tser/training/utility_dataset.py"),
        "train_predictors": tp,
        "run_pilot": _script_code("cr_tser_run_pilot.py"),
        "run_selection": _script_code("cr_tser_run_selection.py"),
    }
    missing = [n for n, src in contract_users.items()
               if "evidence_key" not in src]
    report.add("evidence_key_contract_consistent", not missing,
               f"modules not using the canonical key: {missing}")

    bm = _script_code("cr_tser_build_manifests.py")
    report.add("manifest_immutable_after_labels",
               "def assert_manifests_mutable" in bm
               and "--force cannot bypass this" in _script(
                   "cr_tser_build_manifests.py"),
               "manifests refuse rewriting once any label cache exists")
    body = bm.split("def build_manifests", 1)[-1]
    report.add("viability_before_split",
               body.index("viable_event_ids(events")
               < body.index("build_pilot_split(registry)"),
               "viability filtering must precede the 7319 split")

    si = _pkg_code("cr_tser/evaluation/structural_interaction.py")
    report.add("p2_reader_snapshot_pairing",
               "reader_x_snapshot" in si and "def _pair_deltas" in si
               and "cutoff" in si,
               "P2 pairs inside one reader and snapshot before the event bootstrap")

    up = _pkg_code("cr_tser/evaluation/utility_prediction.py")
    report.add("p3_bootstrap_gate",
               "delta_ci" in up and "missing event-level paired bootstrap" in up
               and "def per_event_counts" in up,
               "P3 requires the event-level paired bootstrap CI to pass")

    rp = _script_code("cr_tser_run_pilot.py")
    report.add("p3_all_rotations",
               "for rotation in LORO_ROTATIONS" in rp
               and "rotations_used" in rp
               and "_counts_by_event(collected" in rp,
               "P3 pools every LORO rotation, not just the first")
    report.add("primary_secondary_separated",
               'PRIMARY_DATASET = "weibo22"' in rp
               and 'SECONDARY_DATASET = "pheme"' in rp
               and "if dataset == PRIMARY_DATASET" in rp
               and "pheme_secondary" in rp,
               "Weibo22 drives P1-P4; PHEME is secondary only")

    lu = _pkg_code("cr_tser/models/legacy_utility.py")
    report.add("b2_s6_pheme_only",
               'LEGACY_DATASET = "pheme"' in lu
               and "def assert_legacy_dataset" in lu
               and "requires_grad_(False)" in lu
               and "no_grad" in lu,
               "legacy TC-DSCR diagnostic is PHEME-only and inference-only")

    ss = _pkg_code("cr_tser/readers/sequence_scorer.py")
    report.add("teacher_forced_tokenization",
               "add_special_tokens=False" in ss
               and "def continuation_boundary" in ss
               and "teacher_forced_logprob_sum" in ss,
               "prompt/continuation tokenized without duplicate special tokens")

    gl = _script_code("cr_tser_generate_labels.py")
    report.add("cache_fingerprint_fail_closed",
               "CacheIdentityMismatch" in gl and "_verify_cached" in gl
               and "FINGERPRINT_FIELDS" in gl,
               "cache hits are fingerprint-validated or refused")
    report.add("prompt_hash_full_chat",
               "_prompt_identity" in gl and "tokenize_prompt" in gl
               and "prompt_ids_hash" in gl,
               "prompt identity covers chat text and tokenized prompt ids")

    for name in ("cr_tser_build_manifests.py", "cr_tser_run_selection.py",
                 "cr_tser_train_predictors.py", "cr_tser_run_pilot.py",
                 "cr_tser_generate_labels.py"):
        src = _script_code(name)
        scoped = bool(re.search(
            r'("manifests"|"unseen_reader"|"utility_labels"|"predictor")'
            r'\s*[,/)]?\s*[,/)]?\s*dataset', src))
        report.add(f"dataset_namespace:{name}", scoped,
                   "artifact paths are dataset-scoped")


_HELDOUT_IN_TRAIN_PATTERNS = (
    r"heldout", r"held_out", r"holdout",
)


def verify_pilot(report: Report, results_root: str):
    root = Path(results_root)
    manifests = root / "manifests"
    split_files = sorted(manifests.glob("*/event_split.json")) \
        if manifests.exists() else []
    if not split_files:
        report.pending("pilot_artifacts",
                       f"pilot not executed yet: no "
                       f"{manifests}/<dataset>/event_split.json")
        return
    names = ("foundation_train", "utility_train", "utility_dev",
             "utility_eval")
    for split_file in split_files:
        dataset = split_file.parent.name
        split = json.loads(_read(split_file))
        seen, overlap = {}, []
        for name in names:
            for eid in split.get(name, []):
                if eid in seen:
                    overlap.append((eid, seen[eid], name))
                seen[eid] = name
        report.add(f"split_events_disjoint:{dataset}", not overlap,
                   f"overlap: {overlap[:5]}")
        report.add(f"split_sizes_exact:{dataset}",
                   all(len(split.get(n, [])) == size for n, size in
                       zip(names, (80, 50, 15, 25))),
                   {n: len(split.get(n, [])) for n in names})

        snap_file = split_file.parent / "snapshot_manifest.jsonl"
        if snap_file.exists():
            rows = [json.loads(line) for line in _read(snap_file).splitlines()
                    if line.strip()]
            bad_future = [r for r in rows if r.get("future_node_leak")]
            bad_ts = [r for r in rows if r.get("used_original_order_as_time")]
            capped = [r for r in rows if r.get("cap_hit")]
            report.add(f"snapshot_no_future_node:{dataset}", not bad_future,
                       f"{len(bad_future)} leaking snapshots")
            report.add(f"snapshot_uses_true_timestamp:{dataset}", not bad_ts,
                       f"{len(bad_ts)} rows substituted row order for time")
            report.add(f"snapshot_uncapped:{dataset}", not capped,
                       f"{len(capped)} capped snapshot rows")
            report.add(f"snapshot_cutoffs_frozen:{dataset}",
                       all(r.get("cutoff") in (15, 60, 360) for r in rows),
                       "cutoffs outside {15,60,360}")
        else:
            report.pending(f"snapshot_manifest:{dataset}",
                           "snapshot_manifest.jsonl missing")

        iv_file = split_file.parent / "intervention_manifest.jsonl"
        if iv_file.exists():
            rows = [json.loads(line) for line in _read(iv_file).splitlines()
                    if line.strip()]
            bad = [r for r in rows if not r.get("valid_in_gt", True)]
            report.add(f"structured_intervention_valid_in_gt:{dataset}", not bad,
                       f"{len(bad)} invalid structured interventions")
        else:
            report.pending(f"intervention_manifest:{dataset}",
                           "intervention_manifest.jsonl missing")

        hashes = split_file.parent / "hashes.json"
        if hashes.exists():
            payload = json.loads(_read(hashes))
            frozen = payload.get("readers", {})
            non_empty = all(entry.get("weight_hash")
                            for entry in frozen.values())
            report.add(f"reader_hashes_frozen:{dataset}",
                       len(frozen) == 3 and non_empty,
                       f"reader hashes: {list(frozen)}")
        else:
            report.pending(f"reader_hashes:{dataset}", "hashes.json missing")

        sel_dir = root / "unseen_reader" / dataset
        if sel_dir.exists():
            for path in sorted(sel_dir.glob("rotation_*.json")):
                data = json.loads(_read(path))
                held = data.get("held_out_reader")
                leaked = [k for k in data.get("train_readers", []) + [
                    data.get("early_stopping_reader", "")] if k == held]
                report.add(f"heldout_isolated:{dataset}:{path.stem}", not leaked,
                           f"held-out {held!r} leaked into {leaked}")
        else:
            report.pending(f"unseen_reader_artifacts:{dataset}",
                           "unseen_reader/<dataset>/ missing")

    summary_file = root / "CR_TSER_PILOT_SUMMARY.json"
    if summary_file.exists():
        summary = json.loads(_read(summary_file))
        report.add("pilot_summary_primary_is_weibo22",
                   summary.get("primary_dataset") == "weibo22",
                   f"primary_dataset={summary.get('primary_dataset')}")
    else:
        report.pending("pilot_summary", "CR_TSER_PILOT_SUMMARY.json missing")
    _verify_pilot_review(report, root)


def _verify_pilot_review(report, root):
    """Artifact-level checks added by the review fix round."""
    from cr_tser.config.pilot_config import LORO_ROTATIONS

    namespaces = {}
    for dataset in ("pheme", "weibo22"):
        split_file = root / "manifests" / dataset / "event_split.json"
        if split_file.exists():
            namespaces[dataset] = json.loads(_read(split_file))
    if namespaces:
        report.add("artifact_namespace_dataset_tagged",
                   all(data.get("dataset") == ds
                       for ds, data in namespaces.items()),
                   f"namespaces present: {sorted(namespaces)}")
        report.add("viability_before_split_recorded",
                   all(data.get("viability_filtered") is not None
                       and data.get("viable_event_count") is not None
                       for data in namespaces.values()),
                   "split manifest must record the viability filter")
    if len(namespaces) == 2:
        report.add("pheme_weibo22_artifacts_separate",
                   namespaces["pheme"].get("dataset") == "pheme"
                   and namespaces["weibo22"].get("dataset") == "weibo22",
                   "both datasets coexist without overwriting")

    for dataset in ("pheme", "weibo22"):
        snapshot_file = root / "manifests" / dataset / "snapshot_manifest.jsonl"
        if snapshot_file.exists():
            rows = [json.loads(line) for line in _read(snapshot_file).splitlines()
                    if line.strip()]
            capped = [r for r in rows if r.get("cap_hit")]
            report.add(f"snapshot_uncapped:{dataset}", not capped,
                       f"{len(capped)} capped snapshot rows")

        label_file = None
        for candidate in (root / "utility_labels" / dataset / "labels.jsonl",
                          root / "utility_labels" / f"{dataset}.jsonl"):
            if candidate.exists():
                label_file = candidate
                break
        if label_file is not None:
            rows = [json.loads(line) for line in _read(label_file).splitlines()
                    if line.strip()][:2000]
            empty = [r for r in rows
                     if not r.get("prompt_hash") or not r.get("reader_hash")]
            report.add(f"cache_hashes_nonempty:{dataset}", not empty,
                       f"{len(empty)} rows with an empty prompt/reader hash")

        for rotation in LORO_ROTATIONS:
            path = (root / "predictor" / dataset /
                    f"rotation_{rotation[0]}_{rotation[1]}" / "predictions.json")
            if not path.exists():
                continue
            data = json.loads(_read(path))
            held = data.get("held_out_reader")
            leaked = [k for k in data.get("predictions", {}) if k == held]
            report.add(f"heldout_absent_from_predictions:{dataset}:{held}",
                       not leaked,
                       f"held-out reader present in predictions: {leaked}")


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
