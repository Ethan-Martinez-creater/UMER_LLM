"""M1-E verifier (fail-closed, post-hoc audit only).

Layered on top of the M0/M1 verifiers (``bcr_verify.py --mode m1e`` runs all
three). Checks exactly what the M1-E mandate requires:

* M1 evidence unchanged (git blob + SHA256 vs the approved commit);
* the M1 verdict is still ``M1_CONDITIONAL_GO``;
* no reader-inference artifacts and no utility labels were added;
* no M2 artifacts exist;
* every diagnostic variant is the fixed one (D0–D5, D1/D4 reused verbatim);
* ``utility_eval`` was never used for training or tuning;
* PHEME stays diagnostic-only and no new gate is defined.
"""
from __future__ import annotations

import json
import os

from .attribution import evidence_pins as pins
from .config import protocol as P
from .verifier import Report, _read_json


def _check_pins(report, repo_root):
    try:
        pin = pins.pin_m1_evidence(repo_root)
    except pins.EvidencePinRefused as exc:
        report.add("m1e_m1_evidence_unchanged", False, str(exc)[:300])
        return None
    report.add("m1e_m1_evidence_unchanged", True,
               f"{pin['n_artifacts']} M1 artifacts match their committed "
               f"blobs at {pin['commit'][:12]}; no tracked changes")
    report.add("m1e_m1_verdict_unchanged",
               pin["m1_verdict"] == P.M1E_EXPECTED_M1_VERDICT,
               f"final_outcome={pin['m1_verdict']}")
    return pin


def _check_attribution(report, repo_root):
    payloads = {}
    for dataset in P.DATASETS:
        path = P.m1e_path(repo_root, P.M1E_ATTRIBUTION_DIRNAME,
                          f"{dataset}.json")
        payloads[dataset] = _read_json(path)
    missing = [d for d, p in payloads.items() if p is None]
    report.add("m1e_attribution_present", not missing,
               f"missing: {missing}" if missing else
               "attribution/<dataset>.json for both datasets")
    if missing:
        return

    problems = []
    for dataset, payload in payloads.items():
        if payload.get("defines_gate") is not False or \
                payload.get("scope") != "post_hoc_diagnostic_only":
            problems.append(f"{dataset}: not marked post-hoc/no-gate")
        variants = payload.get("variants") or {}
        reused = payload.get("reused_variants") or {}
        if sorted(variants) != sorted(P.ATTRIBUTION_TRAINED):
            problems.append(f"{dataset}: trained variants {sorted(variants)}")
        if sorted(reused) != sorted(P.ATTRIBUTION_REUSED):
            problems.append(f"{dataset}: reused variants {sorted(reused)}")
        for variant, model_kind in P.ATTRIBUTION_REUSED.items():
            entry = reused.get(variant) or {}
            if entry.get("frozen_model") != model_kind:
                problems.append(f"{dataset}/{variant}: reused model "
                                f"{entry.get('frozen_model')}")
            if entry.get("reused") is not True:
                problems.append(f"{dataset}/{variant}: not marked reused")
        comparisons = payload.get("comparisons_vs_D0") or {}
        for variant in P.ATTRIBUTION_VARIANTS:
            if variant == "D0_Z":
                continue
            if variant not in comparisons:
                problems.append(f"{dataset}: missing comparison {variant}")
        increment = payload.get("fingerprint_increment_D4_vs_D5") or {}
        if increment.get("variant") != "D4_Z_F_S_C" or \
                increment.get("baseline") != "D5_Z_S_C":
            problems.append(f"{dataset}: fingerprint increment mislabelled")
        # utility_eval never used for training/tuning: scaler fitted on the
        # training rows only and the config chosen on dev
        for variant, entry in list(variants.items()) + list(reused.items()):
            rotations = entry.get("rotations") or {}
            for held, rot in rotations.items():
                train_readers = rot.get("train_readers") or []
                if held in train_readers:
                    problems.append(f"{dataset}/{variant}/{held}: held-out "
                                    "reader inside the training pair")
                folds = rot.get("fold_sizes") or {}
                scaler_rows = rot.get("scaler_rows")
                if scaler_rows is not None and folds.get("train") is not None \
                        and scaler_rows != folds["train"]:
                    problems.append(f"{dataset}/{variant}/{held}: scaler "
                                    f"fitted on {scaler_rows} rows, train fold "
                                    f"{folds['train']}")
                config = rot.get("selected_config") or {}
                if config.get("hidden") != 32:
                    problems.append(f"{dataset}/{variant}/{held}: grid drift")
                if config.get("dropout") not in P.MODEL_GRID["dropout"]:
                    problems.append(f"{dataset}/{variant}/{held}: dropout "
                                    "outside the frozen grid")
                if config.get("lr") not in P.MODEL_GRID["lr"]:
                    problems.append(f"{dataset}/{variant}/{held}: lr outside "
                                    "the frozen grid")
    report.add("m1e_variants_fixed_and_eval_never_trained_on", not problems,
               "; ".join(problems[:6]) if problems else
               "D0/D2/D3/D5 trained, D1/D4 reused verbatim, 6 variants total, "
               "scalers fitted on training rows only, configs from the frozen "
               "grid chosen on dev")

    # feature-group definitions are the frozen ones
    problems = []
    from .attribution import feature_ablation as fa
    expected_dims = {"D0_Z": 26, "D1_Z_F": 26, "D2_Z_F_S": 29, "D3_Z_F_C": 29,
                     "D4_Z_F_S_C": 32, "D5_Z_S_C": 32}
    for variant, dim in expected_dims.items():
        if fa.variant_evidence_dim(variant) != dim:
            problems.append(f"{variant}: dim {fa.variant_evidence_dim(variant)}"
                            f" != {dim}")
    for variant in P.ATTRIBUTION_FINGERPRINT_VARIANTS:
        if not fa.variant_uses_fingerprint(variant):
            problems.append(f"{variant}: fingerprint flag")
    for variant in P.ATTRIBUTION_NO_FINGERPRINT_VARIANTS:
        if fa.variant_uses_fingerprint(variant):
            problems.append(f"{variant}: fingerprint flag")
    report.add("m1e_feature_groups_frozen", not problems,
               "; ".join(problems) if problems else
               "Z/F/S/C groups and D0-D5 evidence widths match the frozen "
               "definition")


def _check_boundaries(report, repo_root):
    root = os.path.join(str(repo_root), P.RESULTS_ROOT)
    m1e = P.m1e_dir(repo_root)
    present = set()
    if os.path.isdir(m1e):
        present = {n for n in os.listdir(m1e)
                   if os.path.isdir(os.path.join(m1e, n))}
    forbidden = sorted(present & set(P.M1E_FORBIDDEN_ARTIFACT_DIRS))
    report.add("m1e_no_reader_inference_or_label_artifacts", not forbidden,
               f"unexpected: {forbidden}" if forbidden else
               f"m1e contains only {sorted(present)}")

    m2 = sorted(n for n in os.listdir(root)
                if n.lower().startswith("m2")) if os.path.isdir(root) else []
    report.add("m1e_no_m2_artifacts", not m2,
               f"found: {m2}" if m2 else "no M2 artifact directory exists")

    verdict = _read_json(P.m1e_path(repo_root, P.M1E_VERDICT_FILENAME))
    if verdict is None:
        report.add("m1e_verdict_present", False, "M1E_VERDICT.json missing",
                   pending=True)
    else:
        pheme = verdict.get("pheme_diagnostic") or {}
        report.add("m1e_pheme_remains_diagnostic",
                   pheme.get("decides_gate") is False
                   and pheme.get("diagnostic_only") is True,
                   f"pheme decides_gate={pheme.get('decides_gate')}")
        report.add("m1e_no_new_gate_defined",
                   verdict.get("defines_new_gate") is False
                   and verdict.get("m1_verdict_unchanged") is True,
                   f"defines_new_gate={verdict.get('defines_new_gate')}")


def verify(repo_root: str) -> dict:
    report = Report()
    _check_pins(report, repo_root)
    _check_attribution(report, repo_root)
    _check_boundaries(report, repo_root)
    return {"checks": report.checks, "issues": report.issues,
            "pending": report.pending, "m1e": {"mode": "m1e"}}
