"""BCR-Utility M1 verifier extension (M1 plan §18).

Fail-closed checks over the M1 artifacts, layered on top of the M0 verifier
(``bcr_verify.py --mode m1`` runs both). Every check is either an exact
comparison against a frozen constant or a recomputation from the artifacts;
nothing trusts a summary written by the producing stage.
"""
from __future__ import annotations

import json
import os

from .config import protocol as P
from .probes import fingerprint as fp
from .verifier import Report, _read_json


def _jsonl_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _count_jsonl(path):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


# --------------------------------------------------------------------------
def _check_probe_responses(report, repo_root):
    m1 = P.m1_path(repo_root)
    shards = {f"{d}/{r}": os.path.join(
        m1, "probe_responses", f"probe_responses_{d}_{r}.jsonl")
        for d in P.DATASETS for r in P.READER_KEYS}
    missing = [k for k, p in shards.items() if not os.path.exists(p)]
    report.add("m1_probe_shards_present", not missing,
               f"missing: {missing}" if missing else "6 probe shards present")
    if missing:
        return
    total = 0
    forbidden_hits = []
    identities = {}
    for name, path in shards.items():
        rows = _jsonl_rows(path)
        total += len(rows)
        dataset, reader = name.split("/")
        for row in rows:
            for field in row:
                if fp.field_is_forbidden(field):
                    forbidden_hits.append(f"{name}:{field}")
            if row.get("reader") != reader or row.get("dataset") != dataset:
                forbidden_hits.append(f"{name}: row identity mismatch")
            identities.setdefault(reader, set()).add(
                row.get("reader_identity_hash"))
            if row.get("model_id") != P.READER_MODEL_IDS[reader]:
                forbidden_hits.append(f"{name}: model id drift")
    report.add("m1_probe_rows_exact_864",
               total == P.PROBE_TOTAL_EVALS,
               f"{total} rows (expected {P.PROBE_TOTAL_EVALS})")
    report.add("m1_probe_rows_have_no_utility_or_gold_fields",
               not forbidden_hits,
               "; ".join(sorted(set(forbidden_hits))[:5]) if forbidden_hits
               else "no utility/gold/label fields in any probe response")
    bad = {r: ids for r, ids in identities.items()
           if len(ids) != 1 or not list(ids)[0]}
    report.add("m1_reader_identity_stable_across_datasets", not bad,
               f"{bad}" if bad else
               "one non-empty reader_identity_hash per reader across both "
               "datasets")


def _check_fingerprints(report, repo_root):
    path = P.m1_path(repo_root, "fingerprints", "fingerprints.json")
    data = _read_json(path)
    report.add("m1_fingerprints_present", data is not None,
               "fingerprints.json" if data else "missing")
    audit = _read_json(P.m1_path(repo_root, "fingerprints",
                                 "fingerprint_audit.json"))
    report.add("m1_fingerprint_audit_present", audit is not None,
               "fingerprint_audit.json" if audit else "missing")
    if data is None:
        return
    problems = []
    if sorted(data.get("readers") or {}) != sorted(P.READER_KEYS):
        problems.append("reader set drift")
    for reader in P.READER_KEYS:
        entry = (data.get("readers") or {}).get(reader) or {}
        compact = entry.get("compact") or {}
        if entry.get("model_id") != P.READER_MODEL_IDS[reader]:
            problems.append(f"{reader}: model id")
        vector = compact.get("vector") or []
        if len(vector) != P.FINGERPRINT_DIM:
            problems.append(f"{reader}: dim {len(vector)}")
        if any(not isinstance(v, (int, float)) or v != v or
               v in (float("inf"), float("-inf")) for v in vector):
            problems.append(f"{reader}: non-finite component")
    if data.get("sha256") != fp.fingerprints_digest(data):
        problems.append("compact fingerprint digest mismatch")
    manifest_path = os.path.join(P.bootstrap_dir(repo_root),
                                 P.PROBE_MANIFEST_FILENAME)
    from .data.historical_import import sha256_file
    if data.get("probe_manifest_sha256") != sha256_file(manifest_path):
        problems.append("probe manifest hash mismatch")
    report.add("m1_fingerprints_exact_and_pinned", not problems,
               "; ".join(problems) if problems else
               f"3 x {P.FINGERPRINT_DIM}D compact fingerprints, digest and "
               "probe-manifest hash pinned")
    if audit is not None:
        response_audit = audit.get("response_audit") or {}
        ok = bool(response_audit.get("ok")) and \
            response_audit.get("n_rows") == P.PROBE_TOTAL_EVALS and \
            not response_audit.get("missing") and \
            not response_audit.get("extra") and \
            not response_audit.get("duplicates")
        report.add("m1_fingerprint_audit_clean", ok,
                   json.dumps({k: response_audit.get(k) for k in
                               ("n_rows", "duplicates", "per_reader_counts")}
                              ))


def _check_features(report, repo_root):
    audit = _read_json(P.m1_path(repo_root, "features",
                                 "feature_audit.json"))
    report.add("m1_feature_audit_present", audit is not None,
               "feature_audit.json" if audit else "missing")
    problems = []
    e3_present = []
    fdir = P.m1_path(repo_root, "features")
    if os.path.isdir(fdir):
        e3_present = [n for n in os.listdir(fdir) if n.startswith("e3")]
    verdict = _read_json(P.m1_path(repo_root, P.M1_VERDICT_FILENAME))
    zero_touch_failed = bool(verdict) and \
        verdict.get("zero_touch_passed") is False
    if e3_present and not zero_touch_failed:
        problems.append(f"E3 artifacts exist without a recorded ZERO-TOUCH "
                        f"failure: {e3_present}")
    report.add("m1_no_e3_before_zero_touch_failure", not problems,
               "; ".join(problems) if problems else
               "no E3 artifacts (or E3 only after a recorded ZERO-TOUCH "
               "failure)")
    if audit is None:
        return
    problems = []
    features = audit.get("features") or {}
    for dataset in P.DATASETS:
        entry = features.get(dataset) or {}
        expected = P.FROZEN_ATOMIC_KEYS[dataset]
        if (entry.get("e0") or {}).get("rows") != expected:
            problems.append(f"{dataset}: e0 rows "
                            f"{(entry.get('e0') or {}).get('rows')}")
        if (entry.get("e1") or {}).get("rows") != expected:
            problems.append(f"{dataset}: e1 rows "
                            f"{(entry.get('e1') or {}).get('rows')}")
        if (entry.get("e2") or {}).get("rows") != \
                expected * len(P.READER_KEYS):
            problems.append(f"{dataset}: e2 rows "
                            f"{(entry.get('e2') or {}).get('rows')}")
        extractor = (entry.get("e1") or {}).get("extractor") or {}
        if extractor.get("model_id") != P.NLI_MODEL_ID:
            problems.append(f"{dataset}: NLI extractor "
                            f"{extractor.get('model_id')}")
        for stage in ("e0", "e1", "e2"):
            path = (entry.get(stage) or {}).get("path")
            if path and os.path.exists(path):
                from .data.historical_import import sha256_file
                if sha256_file(path) != (entry.get(stage) or {}).get("sha256"):
                    problems.append(f"{dataset}: {stage} cache digest drift")
    report.add("m1_feature_coverage_and_nli_identity", not problems,
               "; ".join(problems[:6]) if problems else
               "E0/E1 rows == atomic keys, E2 == 3x keys, NLI extractor is "
               "the frozen mDeBERTa, cache digests recomputed")


def _check_loro(report, repo_root):
    evaluation = _read_json(P.m1_path(repo_root, "evaluation",
                                      "evaluation.json"))
    report.add("m1_evaluation_present", evaluation is not None,
               "evaluation.json" if evaluation else "missing")
    if evaluation is None:
        return
    problems = []
    datasets = evaluation.get("datasets") or {}
    primary = datasets.get(P.PRIMARY_DATASET) or {}
    if primary.get("decides_m1_gate") is not True:
        problems.append("primary dataset not marked as gate-deciding")
    secondary = datasets.get(P.SECONDARY_DATASET) or {}
    if secondary:
        if secondary.get("decides_m1_gate") is not False or \
                secondary.get("diagnostic_only") is not True:
            problems.append("PHEME is not diagnostic-only")
    for dataset, result in datasets.items():
        rotations = result.get("rotations") or {}
        if sorted(rotations) != sorted(P.READER_KEYS):
            problems.append(f"{dataset}: rotation set {sorted(rotations)}")
        for rotation in P.LORO_ROTATIONS:
            held = rotation[2]
            rot = rotations.get(held) or {}
            if sorted(rot.get("train_readers") or []) != \
                    sorted([rotation[0], rotation[1]]):
                problems.append(f"{dataset}/{held}: train readers "
                                f"{rot.get('train_readers')}")
            if rot.get("held_out") != held:
                problems.append(f"{dataset}/{held}: held_out drift")
            models = rot.get("models") or {}
            for needed in (P.MODEL_B0, P.MODEL_B3, P.MODEL_B4):
                if needed not in models:
                    problems.append(f"{dataset}/{held}: missing {needed}")
            b4 = models.get(P.MODEL_B4) or {}
            config = b4.get("selected_config") or {}
            if config.get("hidden") != 32:
                problems.append(f"{dataset}/{held}: B4 hidden "
                                f"{config.get('hidden')}")
            if config.get("dropout") not in P.MODEL_GRID["dropout"]:
                problems.append(f"{dataset}/{held}: B4 dropout")
            if config.get("lr") not in P.MODEL_GRID["lr"]:
                problems.append(f"{dataset}/{held}: B4 lr")
            if config.get("weight_decay") not in \
                    P.MODEL_GRID["weight_decay"]:
                problems.append(f"{dataset}/{held}: B4 weight decay")
            if int(b4.get("n_parameters") or 0) >= P.B4_MAX_PARAMETERS:
                problems.append(f"{dataset}/{held}: B4 parameters "
                                f"{b4.get('n_parameters')}")
            folds = b4.get("fold_sizes") or {}
            b0folds = (models.get(P.MODEL_B0) or {}).get("fold_sizes") or {}
            if folds.get("eval") != b0folds.get("eval"):
                problems.append(f"{dataset}/{held}: B0/B4 eval fold sizes "
                                "disagree")
            grid = b4.get("grid_summary") or []
            if len(grid) != 8:
                problems.append(f"{dataset}/{held}: grid size {len(grid)}")
        b2 = result.get("b2_in_domain_diagnostic") or {}
        if b2.get("in_domain") is not True:
            problems.append(f"{dataset}: B2 not marked in-domain")
    report.add("m1_loro_rotations_splits_grid_exact", not problems,
               "; ".join(problems[:6]) if problems else
               "3 frozen rotations per dataset, B0/B3/B4 present, B4 grid "
               "8 configs, parameters < 250k, B2 in-domain only")


def _check_gate(report, repo_root):
    gate = _read_json(P.m1_path(repo_root, "evaluation", P.M1_GATE_FILENAME))
    report.add("m1_gate_present", gate is not None,
               "gate.json" if gate else "missing")
    verdict = _read_json(P.m1_path(repo_root, P.M1_VERDICT_FILENAME))
    if gate is None:
        return
    problems = []
    thresholds = gate.get("thresholds") or {}
    if thresholds.get("mean_delta_min") != P.GATE_MEAN_DELTA_MIN:
        problems.append("mean delta threshold drift")
    if thresholds.get("positive_readers_min") != P.GATE_POSITIVE_READERS_MIN:
        problems.append("positive readers threshold drift")
    if thresholds.get("worst_reader_min") != P.GATE_WORST_READER_MIN:
        problems.append("worst reader threshold drift")
    if thresholds.get("ci_alpha") != P.GATE_CI_ALPHA:
        problems.append("ci alpha drift")
    agg = gate.get("aggregate") or {}
    if sorted((agg.get("per_reader_delta") or {})) != sorted(P.READER_KEYS):
        problems.append("gate readers drift")
    if gate.get("decides_m1_gate") is not True:
        problems.append("gate not marked as Ma-Weibo primary")
    boot = (gate.get("aggregate") or {})
    report.add("m1_gate_thresholds_frozen", not problems,
               "; ".join(problems) if problems else
               f"thresholds {thresholds}; observed mean delta "
               f"{agg.get('mean_delta_macro_f1')}")
    if verdict is not None:
        consistent = verdict.get("zero_touch_passed") == gate.get("passed")
        pheme = verdict.get("pheme_diagnostic") or {}
        report.add("m1_verdict_consistent_and_pheme_non_deciding",
                   consistent and pheme.get("decides_m1_gate") is False,
                   f"zero_touch_passed={verdict.get('zero_touch_passed')} "
                   f"pheme.decides={pheme.get('decides_m1_gate')}")
    else:
        report.add("m1_verdict_consistent_and_pheme_non_deciding", False,
                   "M1_VERDICT.json missing")


def _check_no_m2(report, repo_root):
    root = os.path.join(str(repo_root), P.RESULTS_ROOT)
    problems = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.lower().startswith("m2"):
                problems.append(f"M2 artifact dir: {name}")
    for path in (P.m1_path(repo_root, "evaluation", "evaluation.json"),
                 P.m1_path(repo_root, "features", "feature_audit.json")):
        data = _read_json(path)
        if data is None:
            continue
        text = json.dumps(data).lower()
        for banned in ("phi-4", "gemma"):
            if banned in text:
                problems.append(f"{os.path.basename(path)} mentions {banned}")
    report.add("m1_no_m2_or_reader_panel_expansion", not problems,
               "; ".join(problems) if problems else
               "no M2 artifacts, no Phi/Gemma references")


def _check_light_touch(report, repo_root):
    """B5 / E3 consistency — only when the LIGHT-TOUCH stage ran."""
    evaluation = _read_json(P.m1_path(repo_root, "evaluation",
                                      "evaluation_light_touch.json"))
    verdict = _read_json(P.m1_path(repo_root, P.M1_VERDICT_FILENAME))
    if evaluation is None:
        report.add("m1_light_touch_consistent", True,
                   "LIGHT-TOUCH stage not run (ZERO-TOUCH "
                   f"passed={bool(verdict) and verdict.get('zero_touch_passed')}"
                   if verdict else "no verdict yet",
                   pending=verdict is None or
                   verdict.get("zero_touch_passed") is None)
        return
    problems = []
    if not verdict or verdict.get("zero_touch_passed") is not False:
        problems.append("LIGHT-TOUCH ran without a ZERO-TOUCH failure")
    if evaluation.get("stage") != "m1d_light_touch":
        problems.append(f"stage={evaluation.get('stage')}")
    for dataset in P.DATASETS:
        result = (evaluation.get("datasets") or {}).get(dataset) or {}
        for rotation in P.LORO_ROTATIONS:
            models = ((result.get("rotations") or {}).get(rotation[2])
                      or {}).get("models") or {}
            if P.MODEL_B5 not in models:
                problems.append(f"{dataset}/{rotation[2]}: B5 missing")
        e3_path = P.m1_path(repo_root, "features", f"e3_{dataset}.jsonl")
        if not os.path.exists(e3_path):
            problems.append(f"{dataset}: e3 cache missing")
        else:
            n = _count_jsonl(e3_path)
            expected = P.FROZEN_ATOMIC_KEYS[dataset] * len(P.READER_KEYS)
            if n != expected:
                problems.append(f"{dataset}: e3 rows {n} != {expected}")
    gate = _read_json(P.m1_path(repo_root, "evaluation",
                                "gate_light_touch.json"))
    if gate is None:
        problems.append("gate_light_touch.json missing")
    else:
        thresholds = gate.get("thresholds") or {}
        if thresholds.get("mean_delta_min") != P.GATE_MEAN_DELTA_MIN or \
                thresholds.get("worst_reader_min") != P.GATE_WORST_READER_MIN \
                or thresholds.get("positive_readers_min") != \
                P.GATE_POSITIVE_READERS_MIN:
            problems.append("light-touch gate thresholds drift")
        if gate.get("passed") != verdict.get("light_touch_passed"):
            problems.append("verdict light_touch_passed inconsistent")
    outcome = (verdict or {}).get("final_outcome")
    if outcome not in ("M1_CONDITIONAL_GO", "M1_NO_GO"):
        problems.append(f"final_outcome={outcome}")
    report.add("m1_light_touch_consistent", not problems,
               "; ".join(problems[:6]) if problems else
               f"E3 rows exact (keys x {len(P.READER_KEYS)}), B5 rotations "
               f"complete, gate thresholds frozen, outcome={outcome}")


def verify(repo_root: str) -> dict:
    report = Report()
    _check_probe_responses(report, repo_root)
    _check_fingerprints(report, repo_root)
    _check_features(report, repo_root)
    _check_loro(report, repo_root)
    _check_gate(report, repo_root)
    _check_light_touch(report, repo_root)
    _check_no_m2(report, repo_root)
    return {"checks": report.checks, "issues": report.issues,
            "pending": report.pending, "m1": {"mode": "m1"}}
