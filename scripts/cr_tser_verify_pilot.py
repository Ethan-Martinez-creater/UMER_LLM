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
import hashlib
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
    "cr_tser/readers/mistral_reader.py",
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

#: The V1 namespace is read-only history (amendment V2 §22). A V2 run writes
#: only under ``results/cr_tser_v2``; these pins prove the historical V1
#: evidence was not silently rewritten by the dataset migration.
#:
#: Identity is the **canonical LF** digest, not the raw worktree bytes. A
#: Windows checkout (``core.autocrlf=true``) and a Linux checkout carry the
#: same git blob with different bytes, so raw-byte pins can never be satisfied
#: on both hosts. Normalizing CRLF to LF keeps the pin content-based — any
#: real edit still changes the digest — while making it platform-stable.
V1_FROZEN_VERIFIER_DIR = "results/cr_tser/verifier"
V1_FROZEN_ARTIFACTS = {
    "results/cr_tser/verifier/code_verify.json":
        "134b32e2f00765ab221ee710a8def67ede91b009b97da95beecfac4463e22f0e",
    "results/cr_tser/verifier/pilot_verify.json":
        "c81000a97ed82c323c1953049d83fa864f9eba2b83640c6211af810076a061af",
}
V1_FROZEN_VERIFIER_DIR_SHA256 = \
    "aba9f5169d63b39c7e423d886e01ec5e40547f43aa0f46a5c937bf73df4d90c2"


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
def _expected_reader_set(protocol: str):
    """Literal contract: ``(keys, model_ids, loro)`` a protocol must freeze.

    The values are written out here rather than read from the configuration so
    a drifted constant is caught instead of mirrored. ``v1``/``v2`` pin the R0
    set that produced ``results/cr_tser_v2``; ``v2r1`` pins the amended R1 set
    (amendment R1 §2, §10).
    """
    if protocol == "v2r1":
        return (("qwen", "mistral", "internlm"),
                {"qwen": "Qwen/Qwen3-8B",
                 "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
                 "internlm": "internlm/internlm3-8b-instruct"},
                (("qwen", "mistral", "internlm"),
                 ("qwen", "internlm", "mistral"),
                 ("mistral", "internlm", "qwen")))
    return (("qwen", "glm", "internlm"),
            {"qwen": "Qwen/Qwen3-8B",
             "glm": "zai-org/glm-4-9b-chat-hf",
             "internlm": "internlm/internlm3-8b-instruct"},
            (("qwen", "glm", "internlm"),
             ("qwen", "internlm", "glm"),
             ("glm", "internlm", "qwen")))


def _current_reader_set(protocol: str, C):
    """``(keys, model_ids, loro)`` the configuration declares for a protocol."""
    if protocol == "v2r1":
        return C.READER_KEYS, C.READER_MODEL_IDS, C.LORO_ROTATIONS
    return C.READER_KEYS_V2, C.READER_MODEL_IDS_V2, C.LORO_ROTATIONS_V2


def verify_code(report: Report, protocol: str = "v2r1"):
    from cr_tser.config import pilot_config as C

    _keys, _models, _loro = _expected_reader_set(protocol)
    cur_keys, cur_models, cur_loro = _current_reader_set(protocol, C)
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
               tuple(cur_keys) == tuple(_keys)
               and cur_models == _models,
               f"readers={cur_models}")
    report.add("loro_rotations",
               cur_loro == _loro,
               f"rotations={cur_loro}")
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
    adapter_text = _read(PROJECT / "cr_tser" / "data" /
                         "weibo22_adapter.py").lower()
    report.add("no_timestamp_synthesis",
               "original_order_used_as_time" in wsrc
               and ("pseudo-time" in adapter_text
                    or "never become pseudo-time" in adapter_text),
               "adapter never derives time from original_order")
    _verify_review_fixes(report)
    _verify_semantics(report)
    if protocol in ("v2", "v2r1"):
        # The dataset protocol is unchanged by amendment R1, so both
        # namespaces run the same Ma-Weibo/PHEME closure checks.
        _verify_v2_migration(report, protocol)
    if protocol == "v2r1":
        _verify_reader_amendment_r1(report)


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
        import cr_tser_common as common
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
                          tokenizer, ["qwen", "mistral"], seed=7319)
        arm_b = build_arm("S6_legacy_utility_tm", units, src,
                          {"legacy": {n: float(len(ids) - i)
                                      for i, n in enumerate(ids)}},
                          tokenizer, ["qwen", "mistral"], seed=7319)
        report.add("s6_consumes_legacy_scores",
                   arm_a["selected_node_ids"] != arm_b["selected_node_ids"],
                   "different legacy scores must change the S6 subset")
        partial = {n: 0.2 for n in ids[:-1]}
        try:
            build_arm("S5_cross_reader_robust", units, src,
                      {"qwen": partial, "mistral": full}, tokenizer,
                      ["qwen", "mistral"], seed=7319)
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
    _verify_protocol_closure(report)
    _verify_code_freeze_hotfix(report)


def _write_p3_fixture(root, rotations):
    """Minimal P3 artifact tree: one eval event, two readers, N rotations."""
    import os as _os
    _os.makedirs(_os.path.join(root, "manifests", "weibo22"), exist_ok=True)
    with open(_os.path.join(root, "manifests", "weibo22", "event_split.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"dataset": "weibo22", "utility_eval": ["e1"],
                   "utility_train": [], "utility_dev": [],
                   "foundation_train": [], "unused": []}, fh)
    _os.makedirs(_os.path.join(root, "utility_labels", "weibo22"),
                 exist_ok=True)
    rows = [{"dataset": "weibo22", "event_id": "e1", "cutoff": 60,
             "reader": reader, "intervention_type": "I1_atomic",
             "intervention_id": "I1:n1", "affected_reply_ids": ["n1"],
             "utility": utility, "sign": sign, "correctness_before": True,
             "correctness_after": True}
            for reader, utility, sign in (("qwen", 0.4, "HELPFUL"),
                                          ("mistral", -0.3, "HARMFUL"))]
    with open(_os.path.join(root, "utility_labels", "weibo22", "labels.jsonl"),
              "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    key = "weibo22|e1|60|n1"
    for rid, qwen_utility in rotations.items():
        train = rid.split("_")
        d = _os.path.join(root, "predictor", "weibo22", f"rotation_{rid}")
        _os.makedirs(d, exist_ok=True)
        payload = {"dataset": "weibo22", "train_readers": train,
                   "predictions": {
                       train[0]: {key: {"utility": qwen_utility,
                                        "probs": [0.8, 0.1, 0.1],
                                        "predicted_sign": "HELPFUL"}},
                       train[1]: {key: {"utility": 0.1,
                                        "probs": [0.1, 0.8, 0.1],
                                        "predicted_sign": "NEUTRAL"}}},
                   "baselines": {"B0_text": {"predictions": {
                       key: {"utility": 0.2, "probs": [0.4, 0.3, 0.3],
                             "predicted_sign": "HELPFUL"}}}}}
        with open(_os.path.join(d, "predictions.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh)


def _verify_protocol_closure(report):
    """Execute the protocol-closure contracts (final fix round)."""
    import tempfile
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # 1. S6 scores come from the frozen StaticUtilitySelector, not the Proxy
    try:
        import torch
        import torch.nn as nn
        from cr_tser.models.legacy_utility import LegacyPHEMEUtility

        class _Sel(nn.Module):
            calls = 0

            def forward(self, h_cand, event_repr, sem_c, sem_src, struct_c):
                _Sel.calls += 1
                return torch.tensor([3.0, 2.0, 1.0])[:h_cand.shape[0]]

        class _Proxy(nn.Module):
            calls = 0

            def classify(self, h_source, z_sel):
                _Proxy.calls += 1
                return torch.tensor([[9.0, 0.0]])

        util = LegacyPHEMEUtility("pheme", encoder=None, selector=_Sel(),
                                  proxy=_Proxy())
        scores = util.per_unit_scores(torch.zeros(4, 768), torch.zeros(768),
                                      torch.zeros(4, 384), torch.zeros(4, 3),
                                      0, [1, 2, 3])
        report.add("s6_uses_static_utility_selector",
                   scores == {1: 3.0, 2: 2.0, 3: 1.0} and _Sel.calls == 1
                   and _Proxy.calls == 0,
                   f"scores={scores} selector_calls={_Sel.calls} "
                   f"proxy_calls={_Proxy.calls}")
    except Exception as exc:
        report.add("s6_uses_static_utility_selector", False, f"error: {exc}")

    # 2./3. P3 keeps every rotation; incomplete rotations fail closed
    try:
        import cr_tser_run_pilot as pilot
        with tempfile.TemporaryDirectory() as tmp:
            _write_p3_fixture(tmp, {"qwen_mistral": 0.9, "qwen_internlm": -0.9,
                                    "mistral_internlm": 0.3})
            gate = pilot.utility_prediction_gate(tmp, "weibo22",
                                                 {"utility_eval": ["e1"]})
            report.add("p3_keeps_all_rotations",
                       gate is not None and gate.get("n_observations") == 4,
                       f"n_observations={gate and gate.get('n_observations')}")
        with tempfile.TemporaryDirectory() as tmp:
            _write_p3_fixture(tmp, {"qwen_mistral": 0.9})
            gate = pilot.utility_prediction_gate(tmp, "weibo22",
                                                 {"utility_eval": ["e1"]})
            report.add("p3_missing_rotation_fails_closed",
                       gate.get("pass") is False
                       and gate.get("rotations_required") == 3
                       and len(gate.get("rotations_missing", [])) == 2,
                       f"missing={gate.get('rotations_missing')}")
    except Exception as exc:
        report.add("p3_keeps_all_rotations", False, f"error: {exc}")
        report.add("p3_missing_rotation_fails_closed", False, f"error: {exc}")

    # 4. P4 completeness + PHEME fail-open
    try:
        from cr_tser.evaluation.unseen_reader import final_decision, gate_p4
        two = [{"held_out_reader": "internlm", "delta": 0.05,
                "token_target_ok": True},
               {"held_out_reader": "mistral", "delta": 0.05,
                "token_target_ok": True}]
        three = two + [{"held_out_reader": "qwen", "delta": 0.05,
                        "token_target_ok": True}]
        report.add("p4_requires_three_rotations",
                   gate_p4(two)["pass"] is False
                   and gate_p4(three)["pass"] is True,
                   "two rotations must never satisfy P4")
        all_pass = {k: {"pass": True} for k in ("P0", "P1", "P2", "P3", "P4")}
        report.add("missing_pheme_cannot_full_go",
                   final_decision(all_pass)["decision"] != "FULL_GO"
                   and final_decision(all_pass, pheme={"pass": True,
                                                       "complete": True})[
                       "decision"] == "FULL_GO",
                   "PHEME secondary evidence is required for FULL_GO")
    except Exception as exc:
        report.add("p4_requires_three_rotations", False, f"error: {exc}")
        report.add("missing_pheme_cannot_full_go", False, f"error: {exc}")

    # 5./6. Stage A never builds a held-out reader; frozen subsets are immutable
    try:
        import cr_tser_run_selection as selection
        freeze_names = set(selection.freeze_subsets.__code__.co_names)
        score_names = set(selection.score_heldout.__code__.co_names)
        report.add("freeze_stage_excludes_heldout_reader",
                   "build_reader" not in freeze_names
                   and "build_reader" in score_names,
                   "Stage A must not construct a held-out reader")
        with tempfile.TemporaryDirectory() as tmp:
            import os as _os
            _os.makedirs(selection.frozen_dir(tmp, "pheme"), exist_ok=True)
            record = {"stage": "A_freeze_subsets", "dataset": "pheme",
                      "held_out_reader": "internlm", "subsets": [
                          {"arm": "S0_src_full", "selected_node_ids": ["n1"],
                           "total_tokens": 1, "target_tokens": 1}]}
            record["sha256"] = selection._canonical_sha(record)
            path = selection._frozen_path(tmp, "pheme", "internlm")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            with open(path + ".sha256", "w", encoding="utf-8") as fh:
                fh.write(record["sha256"])
            selection.load_frozen_subsets(tmp, "pheme", "internlm")
            record["subsets"][0]["selected_node_ids"] = ["n1", "n2"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            try:
                selection.load_frozen_subsets(tmp, "pheme", "internlm")
                refused = False
            except selection.FrozenSubsetChanged:
                refused = True
            report.add("modified_frozen_subset_rejected", refused,
                       "a tampered subset must not be scored")
    except Exception as exc:
        report.add("freeze_stage_excludes_heldout_reader", False,
                   f"error: {exc}")
        report.add("modified_frozen_subset_rejected", False, f"error: {exc}")

    # 7. normalized coverage cannot produce a false READY
    try:
        from cr_tser.data import weibo22_adapter as adapter
        with tempfile.TemporaryDirectory() as tmp:
            import os as _os
            nodes = [
                {"node_id": "n0", "parent_id": None, "timestamp": 1000,
                 "text": "src", "original_order": 0, "status": "VALID"},
                {"node_id": "n1", "parent_id": "n0", "timestamp": 1060,
                 "text": "reply", "original_order": 1, "status": "VALID"},
                {"node_id": "n2", "parent_id": None, "timestamp": 1120,
                 "text": "orphan reply", "original_order": 2,
                 "status": "VALID"},
            ]
            event = {"event_id": "e1", "label": 1, "source_id": "n0",
                     "source_timestamp": 1000, "nodes": nodes}
            path = _os.path.join(tmp, "events.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
            result = adapter.validate_normalized_export(path)
            report.add("normalized_parent_coverage_not_false_ready",
                       result["valid"] is False
                       and result["parent_coverage"] < 1.0,
                       f"valid={result['valid']} "
                       f"parent_coverage={result['parent_coverage']:.3f}")
    except Exception as exc:
        report.add("normalized_parent_coverage_not_false_ready", False,
                   f"error: {exc}")

    # 8. frozen source fingerprint matches the effective normalized source
    try:
        from cr_tser.data.source_manifest import (SourceIdentityError,
                                                  assert_same_source,
                                                  fingerprint_path)
        with tempfile.TemporaryDirectory() as tmp:
            import os as _os
            path = _os.path.join(tmp, "norm.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}\n")
            first = fingerprint_path(path)
            assert_same_source(first, dict(first), "verify")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}\n{}\n")
            changed = fingerprint_path(path)
            try:
                assert_same_source(first, changed, "verify")
                refused = False
            except SourceIdentityError:
                refused = True
            report.add("manifest_fingerprint_matches_source", refused,
                       "a changed normalized source must be refused")
    except Exception as exc:
        report.add("manifest_fingerprint_matches_source", False,
                   f"error: {exc}")

    # 9. CRTSER_SMOKE resolves without TypeError
    try:
        from cr_tser.config.pilot_config import paths_from_env
        paths = paths_from_env({"CRTSER_SMOKE": "1",
                                "CRTSER_OUT_ROOT": "/tmp/out"})
        report.add("crtser_smoke_env_resolves",
                   paths.out_root.endswith("smoke"),
                   f"out_root={paths.out_root}")
    except Exception as exc:
        report.add("crtser_smoke_env_resolves", False, f"error: {exc}")


def _agg_tree(root, dataset, held_readers, delta=0.05):
    """Formal-shape Stage-B artifacts: identity top level, Δ nested."""
    os.makedirs(os.path.join(root, "manifests", dataset), exist_ok=True)
    with open(os.path.join(root, "manifests", dataset, "event_split.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "utility_eval": []}, fh)
    os.makedirs(os.path.join(root, "unseen_reader", dataset), exist_ok=True)
    for held in held_readers:
        with open(os.path.join(root, "unseen_reader", dataset,
                               f"rotation_{held}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"stage": "B_score_heldout", "dataset": dataset,
                       "held_out_reader": held,
                       "delta": {"delta": delta, "primary_macro_f1": 0.5,
                                 "best_simple_macro_f1": 0.5 - delta,
                                 "token_target_ok": True}}, fh)
    os.makedirs(os.path.join(root, "p0"), exist_ok=True)
    with open(os.path.join(root, "p0", "p0_readiness.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"P0": "P0_PASS"}, fh)


def _verify_aggregator_rotation_identity(report):
    """A lost ``held_out_reader`` must be caught at the aggregator, not just
    inside ``gate_p4`` (the code-freeze hotfix regression)."""
    import tempfile
    held = ("internlm", "mistral", "qwen")
    try:
        import cr_tser_run_pilot as pilot
        with tempfile.TemporaryDirectory() as tmp:
            _agg_tree(tmp, "maweibo", held)
            _agg_tree(tmp, "pheme", held)
            gates, _reports, decision = pilot.compute_gates(tmp)
            p4 = gates.get("P4", {})
            completeness = p4.get("rotation_completeness", {})
            report.add(
                "p4_rotation_identity_preserved",
                completeness.get("complete") is True
                and completeness.get("distinct_held_out") == sorted(held)
                and p4.get("pass") is True,
                f"complete={completeness.get('complete')} "
                f"distinct={completeness.get('distinct_held_out')} "
                f"pass={p4.get('pass')}")
            report.add(
                "pheme_rotation_identity_complete",
                decision.get("pheme_evidence_complete") is True
                and decision.get("pheme_secondary") is True,
                f"pheme_complete={decision.get('pheme_evidence_complete')}")
        with tempfile.TemporaryDirectory() as tmp:
            _agg_tree(tmp, "maweibo", held[:2])
            gates, _reports, _decision = pilot.compute_gates(tmp)
            p4 = gates.get("P4", {})
            report.add(
                "incomplete_rotations_fail_closed",
                p4.get("pass") is False
                and p4.get("rotation_completeness", {}).get("complete")
                is False,
                f"two rotations -> pass={p4.get('pass')}")
    except Exception as exc:
        report.add("p4_rotation_identity_preserved", False, f"error: {exc}")
        report.add("pheme_rotation_identity_complete", False, f"error: {exc}")
        report.add("incomplete_rotations_fail_closed", False, f"error: {exc}")


def _verify_freeze_write_once(report):
    """Execute a real double-freeze: the second Stage A must raise and leave
    the frozen JSON/hash byte-identical."""
    import tempfile
    import types
    import torch
    import cr_tser_common as common
    import cr_tser_run_selection as selection

    originals = []

    def patch(obj, name, value):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    try:
        event = {"event_id": "e1", "label": 1, "source_id": "n0",
                 "source_timestamp": 1000,
                 "nodes": [{"node_id": "n0", "parent_id": None,
                            "timestamp": 1000, "text": "src",
                            "original_order": 0, "status": "VALID"},
                           {"node_id": "n1", "parent_id": "n0",
                            "timestamp": 1060, "text": "reply",
                            "original_order": 1, "status": "VALID"}]}
        item = {"event_id": "e1", "cutoff_minutes": 60, "label": 1,
                "num_nodes": 2, "node_ids": ["n0", "n1"], "source_pos": 0}
        art = {"zero_reply": False, "units": [{"node_id": "n1"}],
               "src": {"selected_node_ids": ["n1"], "total_tokens": 10},
               "snapshot": {"node_ids": ["n0", "n1"]},
               "semantic": torch.zeros(2, 384)}
        arms = {name: {"selected_node_ids": ["n1"], "total_tokens": 5,
                       "target_tokens": 10, "content_hash": "h"}
                for name in ("S0_src_full", "S1_random_tm", "S2_semantic_tm",
                             "S3a_source", "S3b_source", "S4_shared",
                             "S5_cross_reader_robust",
                             "S6_legacy_utility_tm")}
        patch(common, "load_dataset_events", lambda dataset, paths: [event])
        patch(common, "CrSemanticEncoder",
              lambda model, dataset, device="cpu": torch.nn.Module())
        patch(common, "canonical_tokenizer", lambda path: _FakeTokenizer())
        patch(common, "snapshot_artifacts", lambda e, c, enc, tok: art)
        patch(common, "source_fingerprint_for", lambda dataset, paths: {})
        patch(selection, "_load_predictor",
              lambda out_root, dataset, readers: {"predictions": {},
                                                  "prediction_coverage": {}})
        patch(selection, "_load_single_predictions",
              lambda out_root, dataset, reader: {})
        patch(selection, "_legacy_item", lambda e, snapshot, sem_rows: item)
        patch(selection, "all_arms", lambda *a, **k: dict(arms))
        paths = types.SimpleNamespace(semantic_model="m",
                                      canonical_tokenizer="t")
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "manifests", "pheme"),
                        exist_ok=True)
            with open(os.path.join(tmp, "manifests", "pheme",
                                   "event_split.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"dataset": "pheme", "utility_eval": ["e1"]}, fh)
            selection.freeze_subsets("pheme", paths, tmp, max_snapshots=1)
            path = selection._frozen_path(tmp, "pheme", "internlm")
            with open(path, encoding="utf-8") as fh:
                before = fh.read()
            with open(path + ".sha256", encoding="utf-8") as fh:
                before_sha = fh.read()
            patch(selection, "_load_predictor",
                  lambda out_root, dataset, readers: {
                      "predictions": {"changed": True},
                      "prediction_coverage": {"changed": True}})
            try:
                selection.freeze_subsets("pheme", paths, tmp, max_snapshots=1)
                refused = False
            except selection.FrozenSubsetAlreadyExists:
                refused = True
            unchanged = (open(path, encoding="utf-8").read() == before
                         and open(path + ".sha256", encoding="utf-8").read()
                         == before_sha)
            report.add("frozen_subset_write_once", refused and unchanged,
                       f"second freeze refused={refused} unchanged={unchanged}")
    except Exception as exc:
        report.add("frozen_subset_write_once", False, f"error: {exc}")
    finally:
        for obj, name, value in reversed(originals):
            setattr(obj, name, value)


def _verify_b2_artifact_contract(report):
    """The PHEME B2 diagnostic must be produced by the real frozen surface,
    reach the report, and never touch a Weibo22 gate."""
    import tempfile
    import torch
    import torch.nn as nn
    import cr_tser_run_pilot as pilot
    import cr_tser_run_selection as selection
    import tcdscr_run_e2
    from cr_tser.models.legacy_utility import LegacyPHEMEUtility

    class _Sel(nn.Module):
        def forward(self, *args):
            return torch.zeros(1)

    class _Proxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def classify(self, h_source, z_sel):
            self.calls += 1
            return torch.tensor([[0.0, 9.0]])

    class _Scorer(LegacyPHEMEUtility):
        def score_items(self, items, device=None):
            return [{nid: 0.1 for nid in item["node_ids"]
                     if nid != item["node_ids"][item["source_pos"]]}
                    for item in items]

    original = tcdscr_run_e2.encoder_forward_batch
    tcdscr_run_e2.encoder_forward_batch = (
        lambda encoder, items, device, batch_size=32: [
            (torch.zeros(it["num_nodes"], 768), torch.zeros(768),
             torch.zeros(2)) for it in items])
    held = ("internlm", "mistral", "qwen")
    try:
        scorer = _Scorer("pheme", encoder=nn.Module(), selector=_Sel(),
                         proxy=_Proxy())
        item = {"event_id": "e1", "cutoff_minutes": 60, "label": 1,
                "num_nodes": 2, "node_ids": ["n0", "n1"], "source_pos": 0}
        with tempfile.TemporaryDirectory() as tmp:
            payload = selection._write_b2_diagnostic(
                "pheme", tmp, scorer, [item], [["n1"]],
                [{"held_out_reader": reader, "sha256": "s"}
                 for reader in held])
            path = os.path.join(tmp, "unseen_reader", "pheme",
                                "b2_legacy_diagnostic.json")
            with open(path, encoding="utf-8") as fh:
                on_disk = json.load(fh)
            report.add(
                "b2_surface_enters_pheme_artifact",
                payload["metrics"]["accuracy"] == 1.0
                and on_disk["diagnostic"] == "B2_legacy_tcdscr"
                and on_disk["diagnostic_only"] is True
                and on_disk["participates_in_primary_gate"] is False
                and on_disk["frozen_fingerprint"]["s6_score_source"]
                == "StaticUtilitySelector"
                and sorted(on_disk["frozen_subset_sha256"]) == sorted(held),
                f"metrics={payload['metrics']}")

        with tempfile.TemporaryDirectory() as tmp:
            _agg_tree(tmp, "maweibo", held)
            _agg_tree(tmp, "pheme", held)
            before, _r, _d = pilot.compute_gates(tmp)
            with open(os.path.join(tmp, "unseen_reader", "pheme",
                                   "b2_legacy_diagnostic.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"dataset": "pheme",
                           "diagnostic": "B2_legacy_tcdscr",
                           "diagnostic_only": True,
                           "participates_in_primary_gate": False,
                           "metrics": {"accuracy": 0.9, "macro_f1": 0.9,
                                       "n": 4},
                           "rows": [1, 2, 3]}, fh)
            after, _r2, decision = pilot.compute_gates(tmp)
            b2 = decision["legacy_diagnostics"]["b2_legacy_tcdscr"]
            report.add("b2_never_changes_gates",
                       after == before and b2["n_rows"] == 3
                       and "rows" not in b2,
                       f"gates_identical={after == before} "
                       f"n_rows={b2['n_rows']}")

        with tempfile.TemporaryDirectory() as tmp:
            _write_p3_fixture(tmp, {"qwen_mistral": 0.9, "qwen_internlm": -0.9,
                                    "mistral_internlm": 0.3})
            gate = pilot.utility_prediction_gate(tmp, "weibo22",
                                                 {"utility_eval": ["e1"]})
            report.add(
                "b2_excluded_from_p3_comparison",
                "S6_legacy_utility_tm" not in gate["per_model_metrics"]
                and set(gate["per_model_metrics"]) <= {
                    "B3_text_graph", "B0_text", "B1_scalar_structure"},
                f"P3 models={sorted(gate['per_model_metrics'])}")

        from cr_tser.config.pilot_config import SIMPLE_BASELINE_ARMS
        from cr_tser.evaluation.unseen_reader import best_simple_baseline
        metrics = {"S1_random_tm": {"macro_f1": 0.10},
                   "S2_semantic_tm": {"macro_f1": 0.20},
                   "S5_cross_reader_robust": {"macro_f1": 0.50},
                   "S6_legacy_utility_tm": {"macro_f1": 0.99}}
        chosen, _ = best_simple_baseline(metrics)
        report.add("b2_excluded_from_p4_comparison",
                   chosen in SIMPLE_BASELINE_ARMS
                   and chosen != "S6_legacy_utility_tm",
                   f"best simple baseline selected {chosen!r}")
    except Exception as exc:
        report.add("b2_surface_enters_pheme_artifact", False, f"error: {exc}")
        report.add("b2_never_changes_gates", False, f"error: {exc}")
        report.add("b2_excluded_from_p3_comparison", False, f"error: {exc}")
        report.add("b2_excluded_from_p4_comparison", False, f"error: {exc}")
    finally:
        tcdscr_run_e2.encoder_forward_batch = original


def _verify_code_freeze_hotfix(report):
    """Execute the code-freeze hotfix contracts (aggregator, freeze, B2)."""
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    _verify_aggregator_rotation_identity(report)
    _verify_freeze_write_once(report)
    _verify_b2_artifact_contract(report)


def _write_maweibo_fixture(root, n_events=2, replies=2, t0=1000):
    """Synthetic Ma-Weibo composite source (raw JSON dir + label file).

    Only used by the verifier's own tmp namespace; it never touches a formal
    artifact root.
    """
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    label_lines = []
    for i in range(n_events):
        eid = str(1000 + i)
        base_t = t0 + i * 100000
        posts = [{"mid": f"{eid}_s", "parent": None, "t": base_t,
                  "original_text": f"source body {i}"}]
        parent = f"{eid}_s"
        for j in range(replies):
            nid = f"{eid}_{j}"
            posts.append({"mid": nid, "parent": parent,
                          "t": base_t + 60 * (j + 1),
                          "original_text": f"reply {i}-{j}"})
            parent = nid
        with open(os.path.join(raw_dir, eid + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(posts, fh)
        label_lines.append(f"eid:{eid} label:{i % 2}")
    label_file = os.path.join(root, "Weibo.txt")
    with open(label_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(label_lines) + "\n")
    return raw_dir, label_file


def _v2_paths(raw_dir, label_file):
    import types
    return types.SimpleNamespace(maweibo_raw=raw_dir, maweibo_labels=label_file,
                                 pheme_raw="", weibo22_normalized="",
                                 weibo22_raw="")


def _verify_v2_migration(report, protocol: str = "v2"):
    """Execute the amendment-V2 dataset/orchestration contracts.

    Amendment R1 left the dataset protocol untouched, so this runs for both
    ``v2`` and ``v2r1``; only the namespace/reader-set checks are
    protocol-specific.
    """
    import tempfile
    from cr_tser.config import pilot_config as C

    # 1. dataset roles
    report.add("v2_primary_dataset_is_maweibo",
               C.PRIMARY_DATASET == "maweibo"
               and C.SECONDARY_DATASET == "pheme"
               and C.V2_DATASETS == ("maweibo", "pheme"),
               f"primary={C.PRIMARY_DATASET} secondary={C.SECONDARY_DATASET}")

    # 2. Weibo22 must not appear in a V2 execution loop
    offenders = []
    for name in ("cr_tser_build_manifests.py", "cr_tser_generate_labels.py",
                 "cr_tser_train_predictors.py", "cr_tser_run_selection.py"):
        if 'choices=("maweibo", "pheme")' not in _script(name):
            offenders.append(name)
    pilot_src = _script("cr_tser_run_pilot.py")
    report.add("weibo22_absent_from_v2_loops",
               not offenders
               and "PRIMARY_DATASET" in pilot_src
               and 'PRIMARY_DATASET = "weibo22"' not in pilot_src,
               f"offenders={offenders}")

    # 3.-10. bridge + composite source + viability + firewall (executed)
    try:
        import cr_tser_common as common
        import cr_tser_p0_audit as p0
        from cr_tser.config.pilot_config import (V1_RESULTS_ROOT,
                                                 V2_RESULTS_ROOT,
                                                 paths_from_env)
        from cr_tser.data import maweibo_bridge as bridge
        from cr_tser.data.snapshot_bridge import (build_causal_snapshot,
                                                  count_parent_cycles)
        from cr_tser.data.source_manifest import (SourceIdentityError,
                                                  assert_same_source,
                                                  source_fingerprint)
        from cr_tser.intervention.evidence_units import build_evidence_units
        from cr_tser.models.legacy_utility import legacy_arm_enabled

        bridge_src = _pkg_code("cr_tser/data/maweibo_bridge.py")
        forbidden = [t for t in ("selector", "proxy", "checkpoint",
                                 "static_utility", "best_selector")
                     if t in bridge_src.lower()]
        report.add("maweibo_bridge_reuses_audited_adapter",
                   "tcdscr.data import maweibo_adapter" in bridge_src
                   and "audited.load_event" in bridge_src
                   and not forbidden,
                   f"forbidden learned-artifact refs: {forbidden}")

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, label_file = _write_maweibo_fixture(tmp, n_events=2,
                                                         replies=2)
            events = bridge.load_events(raw_dir, label_file)
            posts = json.load(open(os.path.join(raw_dir, "1000.json"),
                                   encoding="utf-8"))
            by_mid = {p["mid"]: p for p in posts}
            timestamp_ok = all(
                node["timestamp"] == by_mid[node["node_id"]]["t"]
                for node in events[0]["nodes"])
            report.add("maweibo_timestamp_from_raw_t", timestamp_ok,
                       "node timestamps must equal the raw t field")

            paths = _v2_paths(raw_dir, label_file)
            fp = source_fingerprint("maweibo", paths)
            report.add("maweibo_fingerprint_is_composite",
                       "raw_json" in fp and "label_file" in fp
                       and bool(fp.get("combined_source_sha256"))
                       and fp["kind"] == "maweibo_composite",
                       f"kind={fp.get('kind')}")
            with open(label_file, "a", encoding="utf-8") as fh:
                fh.write("eid:9999 label:1\n")
            current = source_fingerprint("maweibo", paths)
            try:
                assert_same_source(fp, current, "v2")
                refused = False
            except SourceIdentityError:
                refused = True
            report.add("maweibo_source_replacement_fails_closed", refused,
                       "a changed label file must be refused")

            audit = bridge.audit_maweibo(raw_dir, label_file)
            required = ("raw_event_count", "label_distribution",
                        "source_text_coverage", "reply_text_coverage",
                        "timestamp_coverage", "parent_resolution_coverage",
                        "duplicate_ids", "cycle_count", "multi_root_event_count",
                        "missing_parent_count", "missing_parent_rate",
                        "external_parent_count", "external_parent_rate",
                        "temporal_invalid_node_count",
                        "events_with_ge1_valid_reply_parent_unit",
                        "events_viable_15m", "events_viable_1h",
                        "events_viable_6h", "total_viable_events")
            missing = [f for f in required if f not in audit]
            report.add("maweibo_p0a_fields_complete", not missing,
                       f"missing={missing}")
            report.add("maweibo_fixture_all_events_viable",
                       audit["total_viable_events"] == len(events)
                       and audit["cycle_count"] == 0
                       and audit["duplicate_ids"] == 0,
                       f"viable={audit['total_viable_events']} "
                       f"of {len(events)}")

        # EMPTY_TEXT must not become an evidence unit (amendment §7)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = os.path.join(tmp, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            posts = [
                {"mid": "7_s", "parent": None, "t": 1000,
                 "original_text": "source body"},
                {"mid": "7_a", "parent": "7_s", "t": 1060,
                 "original_text": "reply a"},
                {"mid": "7_b", "parent": "7_a", "t": 1120,
                 "original_text": ""},
                {"mid": "7_c", "parent": "7_a", "t": 1180,
                 "text": "fallback text reply"},
            ]
            with open(os.path.join(raw_dir, "7.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(posts, fh)
            label_file = os.path.join(tmp, "Weibo.txt")
            with open(label_file, "w", encoding="utf-8") as fh:
                fh.write("eid:7 label:1\n")
            event = bridge.load_events(raw_dir, label_file)[0]
            statuses = {n["node_id"]: n["status"] for n in event["nodes"]}
            snapshot = build_causal_snapshot(event, 360)
            # the Ma-Weibo orchestration stamps the eligibility contract; the
            # PHEME path leaves it unset and keeps the V1 behaviour
            snapshot["eligibility"] = common.eligibility_for("maweibo")
            unit_ids = {u["node_id"] for u in build_evidence_units(snapshot)}
            pheme_snapshot = build_causal_snapshot(event, 360)
            pheme_units = {u["node_id"]
                           for u in build_evidence_units(pheme_snapshot)}
            report.add("empty_text_not_an_evidence_unit",
                       statuses["7_b"] == "EMPTY_TEXT"
                       and "7_b" not in unit_ids
                       and "7_c" in unit_ids,
                       f"statuses={statuses} units={sorted(unit_ids)}")
            report.add("pheme_keeps_v1_evidence_units",
                       "7_b" in pheme_units,
                       f"pheme units={sorted(pheme_units)}")

        # viability filter runs before the split and < 170 blocks P0
        from cr_tser.data.pilot_split import viable_event_ids
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, label_file = _write_maweibo_fixture(tmp, n_events=3,
                                                         replies=1)
            events = bridge.load_events(raw_dir, label_file)
            report.add("maweibo_viability_filter_runs",
                       len(viable_event_ids(events, eligibility="v2_strict"))
                       == len(events),
                       "every fixture event has a valid Reply–Parent unit")

        good_readers = {k: {"model_path_exists": True, "loaded": True}
                        for k in C.READER_KEYS}
        good_sanity = {k: {"identical_predictions": True, "boundaries_ok": True}
                       for k in C.READER_KEYS}
        base = {"raw_event_count": 400, "duplicate_ids": 0, "cycle_count": 0,
                "multi_root_event_count": 0, "source_text_coverage": 1.0,
                "source_timestamp_coverage": 1.0, "timestamp_coverage": 1.0,
                "verdict": "MAWEIBO_READY"}
        short = dict(base, total_viable_events=169)
        enough = dict(base, total_viable_events=170)
        r_short = p0.evaluate_readiness_v2(short, {"status": "OK"}, good_readers,
                                           good_sanity, True, True)
        r_enough = p0.evaluate_readiness_v2(enough, {"status": "OK"},
                                            good_readers, good_sanity, True,
                                            True)
        report.add("viable_lt_170_blocks_p0",
                   r_short["P0"] == "P0_FAIL"
                   and r_short["maweibo_viable_ok"] is False
                   and r_enough["P0"] == "P0_PASS",
                   f"169->{r_short['P0']} 170->{r_enough['P0']}")

        report.add("maweibo_b2_s6_forbidden",
                   legacy_arm_enabled("maweibo") is False
                   and legacy_arm_enabled("pheme") is True,
                   "B2/S6 are PHEME-only; Ma-Weibo has neither")

        # 11./12. namespace separation: the default output root always follows
        # the *current* reader protocol, historical roots stay declared
        report.add("v2_namespace_separate_from_v1",
                   V2_RESULTS_ROOT == "results/cr_tser_v2"
                   and V1_RESULTS_ROOT == "results/cr_tser"
                   and V2_RESULTS_ROOT != V1_RESULTS_ROOT,
                   f"v1={V1_RESULTS_ROOT} v2={V2_RESULTS_ROOT}")
        report.add("default_out_root_is_current_namespace",
                   paths_from_env({}).out_root == C.V2R1_RESULTS_ROOT
                   and C.V2R1_RESULTS_ROOT not in (V1_RESULTS_ROOT,
                                                   V2_RESULTS_ROOT),
                   f"out_root={paths_from_env({}).out_root} "
                   f"current={C.V2R1_RESULTS_ROOT}")

        # 13. the V2 P0 audit runs end-to-end on synthetic data only
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, label_file = _write_maweibo_fixture(tmp, n_events=2,
                                                         replies=2)
            out_dir = os.path.join(tmp, "v2p0")
            argv = ["--protocol", "v2", "--maweibo-raw", raw_dir,
                    "--maweibo-labels", label_file, "--out-root", out_dir]
            import contextlib
            import io as _io
            with contextlib.redirect_stdout(_io.StringIO()):
                rc = p0.main_v2(p0.build_parser().parse_args(argv),
                                paths_from_env({}))
            written = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) \
                else []
            report.add("v2_p0_audit_runs_synthetic",
                       rc == 2
                       and "maweibo_audit.json" in written
                       and "p0_readiness.json" in written
                       and "pheme_smoke.json" in written,
                       f"rc={rc} files={written}")

        # 14. no Ma-Weibo learned artifact is referenced by the primary path
        selection_src = _script("cr_tser_run_selection.py")
        report.add("maweibo_primary_path_has_no_legacy_artifacts",
                   "build_legacy_scorer" in selection_src
                   and "legacy_arm_enabled" in selection_src,
                   "the legacy TC-DSCR scorer stays behind the PHEME-only "
                   "legacy gate")
    except Exception as exc:
        for name in ("maweibo_bridge_reuses_audited_adapter",
                     "maweibo_timestamp_from_raw_t",
                     "maweibo_fingerprint_is_composite",
                     "maweibo_source_replacement_fails_closed",
                     "maweibo_p0a_fields_complete",
                     "maweibo_fixture_all_events_viable",
                     "empty_text_not_an_evidence_unit",
                     "maweibo_viability_filter_runs",
                     "viable_lt_170_blocks_p0",
                     "maweibo_b2_s6_forbidden",
                     "v2_namespace_separate_from_v1",
                     "default_out_root_is_v2",
                     "v2_p0_audit_runs_synthetic",
                     "maweibo_primary_path_has_no_legacy_artifacts"):
            report.add(name, False, f"error: {exc}")

    # 15. the scientific constants are untouched by the dataset amendment
    _proto_keys, _proto_models, _proto_loro = _expected_reader_set(protocol)
    _cur_keys, _cur_models, _cur_loro = _current_reader_set(protocol, C)
    report.add("v2_scientific_constants_unchanged",
               C.CUTOFFS_MIN == (15, 60, 360)
               and C.PARTITION_SEED == 7319
               and C.TRAIN_SEEDS == (7319, 7320, 7321)
               and C.SPLIT_SIZES == {"foundation_train": 80,
                                     "utility_train": 50,
                                     "utility_dev": 15, "utility_eval": 25}
               and tuple(_cur_keys) == tuple(_proto_keys)
               and _cur_models == _proto_models
               and _cur_loro == _proto_loro
               and (C.UTILITY_THRESHOLD, C.P1_MEAN_DISAGREEMENT_MIN,
                    C.P2_EDGE_DELTA_MIN, C.P3_MACRO_F1_DELTA_MIN,
                    C.P4_MEAN_DELTA_MIN, C.P4_WORST_ROTATION_MIN,
                    C.PHEME_MEAN_DELTA_MIN)
               == (0.05, 0.10, 0.02, 0.02, 0.01, -0.005, -0.005),
               "the dataset amendment changes dataset roles only, and the "
               "reader amendment changes the reader set only")

    # 16. dataset-aware eligibility: PHEME keeps the V1 evidence-unit contract
    try:
        from cr_tser.config.pilot_config import (PRIMARY_DATASET,
                                                 V2_STRICT_ELIGIBILITY_DATASETS)
        from cr_tser.intervention.evidence_units import build_evidence_units
        from cr_tser.data.snapshot_bridge import valid_reply_parent_units

        def _snapshot(eligibility, statuses, parent_ids, texts):
            return {"node_ids": ["s", "a", "b"], "texts": texts,
                    "parent_ids": parent_ids, "timestamps": [0, 10, 20],
                    "elapsed_seconds": [0, 10, 20], "source_id": "s",
                    "depths": [0, 1, 2], "statuses": statuses,
                    "eligibility": eligibility}

        # reply "a" is VALID but its parent "s" is EMPTY_TEXT
        snap = _snapshot("v2_strict", ["EMPTY_TEXT", "VALID", "VALID"],
                         [None, "s", "a"], ["", "reply a", "reply b"])
        strict_units = {u["node_id"] for u in build_evidence_units(snap)}
        v1_snap = dict(snap, eligibility=None)
        v1_units = {u["node_id"] for u in build_evidence_units(v1_snap)}
        report.add("eligibility_is_dataset_aware",
                   "a" not in strict_units and {"a", "b"} <= v1_units
                   and V2_STRICT_ELIGIBILITY_DATASETS == (PRIMARY_DATASET,),
                   f"strict={sorted(strict_units)} v1={sorted(v1_units)}")

        event = {"event_id": "e", "source_id": "s",
                 "nodes": [
                     {"node_id": "s", "parent_id": None, "timestamp": 0,
                      "text": "", "original_order": 0, "status": "EMPTY_TEXT"},
                     {"node_id": "a", "parent_id": "s", "timestamp": 10,
                      "text": "reply a", "original_order": 1,
                      "status": "VALID"}]}
        report.add("invalid_parent_cannot_form_unit",
                   valid_reply_parent_units(event) == [],
                   "a VALID child must not smuggle an invalid parent")
    except Exception as exc:
        report.add("eligibility_is_dataset_aware", False, f"error: {exc}")
        report.add("invalid_parent_cannot_form_unit", False, f"error: {exc}")

    _verify_v1_historical_immutable(report)


#: Canonical (LF-normalized) digests of the frozen V2 artifacts as they stood
#: when amendment R1 was approved (commit ``6da025c``). Amendment R1 §4 makes
#: ``results/cr_tser_v2/`` immutable history: the new reader set must not have
#: rewritten the manifests, the hashes or the partial GLM cache.
V2_FROZEN_MANIFEST_SHA256 = {
    "results/cr_tser_v2/manifests/maweibo/source.json":
        "80b07954ce3199c57cb25e7ca11b07115dd0e20a84787b97effc1cfed291b783",
    "results/cr_tser_v2/manifests/maweibo/hashes.json":
        "fac953a5aaec6571f2ae90ebeb7a5f017c20673896ff2a65ae14991a2254154b",
    "results/cr_tser_v2/manifests/maweibo/event_split.json":
        "24e18ed10954e8387c49d9a119e78934e2c8d33ccb582aa62e4f2cf201b8dde2",
    "results/cr_tser_v2/manifests/pheme/source.json":
        "1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363",
    "results/cr_tser_v2/manifests/pheme/hashes.json":
        "4b865d5a811efe74a560e921bf888e6a7838ef3a20c854dbaa8b2bcf5b8b08f3",
    "results/cr_tser_v2/manifests/pheme/event_split.json":
        "f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385",
}


def _migration_fixture(src_root: str, dst_root: str, readers=("qwen",
                                                              "internlm",
                                                              "glm")):
    """Synthetic V2/v2r1 namespaces for the migration fail-closed checks."""
    source = {"dataset": "pheme", "kind": "pheme_raw", "path": "/raw",
              "exists": True, "sha256": "aa", "bytes": 10, "n_files": 1}
    readers_record = {
        key: {"model_id": f"org/{key}", "weight_hash": f"w-{key}",
              "tokenizer_hash": f"t-{key}"}
        for key in ("qwen", "mistral", "internlm")}
    hashes = {"dataset": "pheme", "source": source,
              "event_split_sha256": "es", "cutoffs": [15, 60, 360],
              "viable_events": 200, "n_snapshots": 3, "n_interventions": 3,
              "readers": readers_record}
    for root in (src_root, dst_root):
        mdir = os.path.join(root, "manifests", "pheme")
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, "source.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(source, fh)
        with open(os.path.join(mdir, "hashes.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(hashes, fh)
    cache = os.path.join(src_root, "utility_labels", "pheme", "labels.jsonl")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        for i, reader in enumerate(readers):
            row = {"dataset": "pheme", "event_id": f"e{i}", "cutoff": 60,
                   "reader": reader, "intervention_id": "I0",
                   "base_context_hash": f"b{i}", "intervened_context_hash":
                       f"c{i}", "reader_hash": f"w-{reader}",
                   "reader_identity_hash": f"id-{reader}",
                   "tokenizer_hash": f"t-{reader}",
                   "chat_template_hash": f"ct-{reader}",
                   "prompt_hash": f"p{i}", "prompt_ids_hash": f"pi{i}"}
            fh.write(json.dumps(row) + "\n")
    return src_root, dst_root


def _verify_reader_amendment_r1(report):
    """Execute the amendment-R1 reader-set contracts (R1 §10, §13)."""
    import tempfile
    import types as _types

    from cr_tser.config import pilot_config as C
    from cr_tser.config.pilot_config import paths_from_env

    report.add("r1_reader_keys_exact",
               tuple(C.READER_KEYS) == ("qwen", "mistral", "internlm"),
               f"READER_KEYS={C.READER_KEYS}")
    report.add("r1_reader_model_ids_exact",
               C.READER_MODEL_IDS == {
                   "qwen": "Qwen/Qwen3-8B",
                   "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
                   "internlm": "internlm/internlm3-8b-instruct"},
               f"READER_MODEL_IDS={C.READER_MODEL_IDS}")
    report.add("r1_loro_exact",
               C.LORO_ROTATIONS == (("qwen", "mistral", "internlm"),
                                    ("qwen", "internlm", "mistral"),
                                    ("mistral", "internlm", "qwen")),
               f"rotations={C.LORO_ROTATIONS}")
    report.add("r1_protocol_versions",
               C.DATASET_PROTOCOL_VERSION == "v2"
               and C.READER_PROTOCOL_VERSION == "r1"
               and C.PROTOCOL_VERSION == "v2r1",
               f"dataset={C.DATASET_PROTOCOL_VERSION} "
               f"reader={C.READER_PROTOCOL_VERSION} "
               f"protocol={C.PROTOCOL_VERSION}")
    report.add("v2r1_namespace_separate",
               C.V2R1_RESULTS_ROOT == "results/cr_tser_v2r1"
               and len({C.V1_RESULTS_ROOT, C.V2_RESULTS_ROOT,
                        C.V2R1_RESULTS_ROOT}) == 3,
               f"v1={C.V1_RESULTS_ROOT} v2={C.V2_RESULTS_ROOT} "
               f"v2r1={C.V2R1_RESULTS_ROOT}")

    # the retired reader must not be reachable from any v2r1 execution loop
    loop_scripts = ("cr_tser_build_manifests.py", "cr_tser_generate_labels.py",
                    "cr_tser_train_predictors.py", "cr_tser_run_selection.py",
                    "cr_tser_run_pilot.py")
    offenders = [name for name in loop_scripts
                 if "glm" in _script(name).lower()]
    report.add("glm_absent_from_r1_loops",
               "glm" not in C.READER_KEYS
               and not any("glm" in rotation for rotation in C.LORO_ROTATIONS)
               and "glm" not in C.READER_MODEL_IDS
               and not offenders,
               f"offenders={offenders}")
    report.add("glm_kept_only_as_history",
               C.READER_KEYS_V2 == ("qwen", "glm", "internlm")
               and C.KNOWN_READER_MODEL_IDS["glm"]
               == "zai-org/glm-4-9b-chat-hf",
               "the R0 key stays addressable for frozen V2 evidence only")

    # path/env wiring
    paths = paths_from_env({"CRTSER_MISTRAL_MODEL": "/models/mistral"})
    report.add("mistral_path_and_env_resolution",
               paths.reader_path("mistral") == "/models/mistral"
               and paths.reader_path("glm") == ""
               and set(paths.reader_paths()) == set(C.READER_KEYS),
               f"mistral={paths.reader_path('mistral')!r}")

    # the replacement reader must use the shared teacher-forced scorer
    msrc = _code_only(_read(PROJECT / "cr_tser" / "readers" /
                            "mistral_reader.py"))
    report.add("mistral_reader_uses_shared_scorer",
               "candidate_logprobs_hf" in msrc
               and ".generate(" not in msrc
               and "trust_remote_code = False" in msrc
               and "torch_dtype=getattr(torch, self.spec.dtype)" in msrc,
               "Mistral must reuse the shared A/B teacher-forced scorer")

    # the historical V2 namespace must be untouched
    drifted = {}
    for rel, expected in V2_FROZEN_MANIFEST_SHA256.items():
        path = REPO / rel
        if not path.exists():
            drifted[rel] = "missing"
        elif _canonical_sha256(path) != expected:
            drifted[rel] = "changed"
    report.add("v2_history_untouched", not drifted,
               f"drift: {drifted}" if drifted
               else f"{len(V2_FROZEN_MANIFEST_SHA256)} frozen V2 files match")

    # the migration utility must be fail-closed end-to-end
    try:
        import cr_tser_migrate_v2_labels as migration
        with tempfile.TemporaryDirectory() as tmp:
            src_root, dst_root = _migration_fixture(os.path.join(tmp, "v2"),
                                                    os.path.join(tmp, "v2r1"))
            ok = migration.migrate("pheme", src_root, dst_root)
            good = (ok["migrated_rows"] == 2 and ok["skipped_retired_rows"] == 1
                    and ok["readers_migrated"] == ["internlm", "qwen"])

            applied = migration.migrate("pheme", src_root, dst_root, apply=True)
            written = [json.loads(l) for l in
                       open(applied["target_cache"], encoding="utf-8")
                       if l.strip()]
            no_glm = (len(written) == 2
                      and all(r["reader"] != "glm" for r in written))
            try:
                migration.migrate("pheme", src_root, dst_root)
                nonempty_refused = False
            except migration.MigrationRefused:
                nonempty_refused = True

            bad_root = os.path.join(tmp, "drift")
            _migration_fixture(bad_root, os.path.join(tmp, "drift_t"))
            with open(os.path.join(bad_root, "manifests", "pheme",
                                   "hashes.json"), encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["event_split_sha256"] = "tampered"
            with open(os.path.join(bad_root, "manifests", "pheme",
                                   "hashes.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            try:
                migration.migrate("pheme", bad_root,
                                  os.path.join(tmp, "drift_t"))
                drift_refused = False
            except migration.MigrationRefused:
                drift_refused = True

        report.add("r1_migration_utility_fail_closed",
                   good and no_glm and nonempty_refused and drift_refused,
                   f"dry_run={good} no_glm={no_glm} "
                   f"nonempty_refused={nonempty_refused} "
                   f"drift_refused={drift_refused}")
    except Exception as exc:  # pragma: no cover - defensive
        report.add("r1_migration_utility_fail_closed", False,
                   f"error: {exc}")


def _canonical_bytes(data: bytes) -> bytes:
    """CRLF -> LF, so identity does not depend on the checkout's line endings.

    ``core.autocrlf=true`` on Windows and a plain Linux checkout store the
    same git blob with different bytes; the canonical form is what both share.
    """
    return data.replace(b"\r\n", b"\n")


def _canonical_sha256(path) -> str:
    """Canonical (LF-normalized) content digest of one file."""
    return hashlib.sha256(_canonical_bytes(path.read_bytes())).hexdigest()


def _dir_sha256(path: str) -> str:
    """Canonical digest of a directory tree.

    Contents are LF-normalized and relative paths are POSIX-normalized, so the
    digest is identical on a CRLF Windows checkout and an LF Linux checkout.
    """
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode())
            h.update(b"\x00")
            with open(full, "rb") as fh:
                h.update(_canonical_bytes(fh.read()))
            h.update(b"\x00")
    return h.hexdigest()


def _verify_v1_historical_immutable(report):
    """V1 historical artifacts must keep their content (amendment §22).

    The comparison uses the canonical LF digest, so the same repository state
    passes on a CRLF Windows checkout and on an LF Linux checkout, while any
    real content change still fails.
    """
    for rel, want in V1_FROZEN_ARTIFACTS.items():
        path = REPO / rel
        if not path.exists():
            report.add(f"v1_historical_immutable:{path.name}", False,
                       f"{rel} is missing")
            continue
        got = _canonical_sha256(path)
        report.add(f"v1_historical_immutable:{path.name}", got == want,
                   f"canonical_sha256={got[:16]} expected={want[:16]}")
    directory = REPO / V1_FROZEN_VERIFIER_DIR
    if not directory.exists():
        report.add("v1_verifier_dir_immutable", False,
                   f"{V1_FROZEN_VERIFIER_DIR} is missing")
        return
    got = _dir_sha256(str(directory))
    report.add("v1_verifier_dir_immutable",
               got == V1_FROZEN_VERIFIER_DIR_SHA256,
               f"canonical_sha256={got[:16]} expected="
               f"{V1_FROZEN_VERIFIER_DIR_SHA256[:16]}")


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
               "PRIMARY_DATASET" in rp
               and "SECONDARY_DATASET" in rp
               and "if dataset == PRIMARY_DATASET" in rp
               and "pheme_secondary" in rp
               and 'PRIMARY_DATASET = "weibo22"' not in rp,
               "the primary dataset drives P1-P4; PHEME is secondary only")

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
        from cr_tser.config.pilot_config import PRIMARY_DATASET
        report.add("pilot_summary_primary_is_primary_dataset",
                   summary.get("primary_dataset") == PRIMARY_DATASET,
                   f"primary_dataset={summary.get('primary_dataset')}")
    else:
        report.pending("pilot_summary", "CR_TSER_PILOT_SUMMARY.json missing")
    _verify_pilot_review(report, root)


def _verify_pilot_review(report, root):
    """Artifact-level checks added by the review fix round."""
    from cr_tser.config.pilot_config import LORO_ROTATIONS

    namespaces = {}
    for dataset in ("maweibo", "pheme", "weibo22"):
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
    if "maweibo" in namespaces and "pheme" in namespaces:
        report.add("primary_secondary_artifacts_separate",
                   namespaces["maweibo"].get("dataset") == "maweibo"
                   and namespaces["pheme"].get("dataset") == "pheme",
                   "Ma-Weibo primary and PHEME secondary coexist")

    for dataset in ("maweibo", "pheme", "weibo22"):
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
    ap.add_argument("--protocol", choices=("v1", "v2", "v2r1"), default="v2r1",
                    help="v2r1 (default) verifies the amended R1 reader set; "
                         "v2 verifies the frozen R0 namespace; v1 keeps the "
                         "historical checks only")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    from cr_tser.config.pilot_config import (V1_RESULTS_ROOT,
                                             V2_RESULTS_ROOT,
                                             V2R1_RESULTS_ROOT)

    read_root = {"v2r1": V2R1_RESULTS_ROOT,
                 "v2": V2_RESULTS_ROOT}.get(args.protocol, V1_RESULTS_ROOT)
    report = Report()
    if args.mode == "code":
        verify_code(report, args.protocol)
    else:
        root = args.results_root or os.environ.get(
            "CRTSER_OUT_ROOT", str(REPO / read_root))
        verify_pilot(report, root)
    payload = report.to_dict(args.mode)
    # Verifier output always lands in the namespace it verified: the V1 and V2
    # directories are frozen history and must never be rewritten by a re-run
    # (amendment V2 §22, amendment R1 §4).
    v1_run = args.protocol == "v1"
    out_name = (f"v1_{args.mode}_verify.json" if v1_run
                else f"{args.mode}_verify.json")
    write_root = V2_RESULTS_ROOT if v1_run else read_root
    out = args.out or str(REPO / write_root / "verifier" / out_name)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({"mode": payload["mode"],
                      "protocol": args.protocol,
                      "issues": payload["issues"],
                      "pending": payload["pending"], "out": out}, indent=1))
    for check in payload["checks"]:
        if check["status"] != "pass":
            print(f"  [{check['status']}] {check['check']}: {check['detail']}")
    return 0 if payload["issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
