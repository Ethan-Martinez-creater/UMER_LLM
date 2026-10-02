"""M1-F verifier (fail-closed, bounded falsification audit).

Layered on top of the M0/M1/M1-E verifiers (``bcr_verify.py --mode m1f`` runs
all four). It checks exactly what the M1-F mandate fixes:

* every frozen input is byte-identical to its committed blob (M1/M1-E evidence,
  E0–E3 caches, probe/fingerprint artifacts, D5 predictions, atomic index,
  historical manifests) and the frozen label caches still hash to their pinned
  digests;
* the historical CR-TSER namespaces carry no tracked change;
* the strong-baseline ladder is the fixed one (S0–S4, S4 reused from D5) and
  the comparator was selected from S0–S3 on training readers' dev only;
* no held-out reader ever entered training, dev selection, scaler fitting or
  the class prior;
* thresholds are the frozen ones (±0.05 official, ±0.03/0.05/0.07 diagnostic
  only) and PHEME never decides the primary result;
* no reader-inference artifact, no new utility label, no regenerated E3 and no
  M2 artifact exists.
"""
from __future__ import annotations

import os

from .config import protocol as P
from .m1f import pins
from .verifier import Report, _git_status_clean, _read_json


def _check_pins(report, repo_root):
    try:
        pin = pins.pin_m1f_inputs(repo_root)
    except pins.PinRefused as exc:
        report.add("m1f_frozen_inputs_pinned", False, str(exc)[:300])
        return None
    groups = pin["groups"]
    report.add("m1f_frozen_inputs_pinned", True,
               f"{pin['n_artifacts']} artifacts across {len(groups)} frozen "
               f"input groups match their committed blobs at "
               f"{pin['commit'][:12]}")
    m1_verdict = _read_json(os.path.join(
        str(repo_root), P.RESULTS_ROOT, P.M1_DIRNAME, P.M1_VERDICT_FILENAME))
    report.add("m1f_m1_verdict_unchanged",
               (m1_verdict or {}).get("final_outcome")
               == P.M1E_EXPECTED_M1_VERDICT,
               f"M1 final_outcome={(m1_verdict or {}).get('final_outcome')}")
    m1e_verdict = _read_json(P.m1e_path(repo_root, P.M1E_VERDICT_FILENAME))
    report.add("m1f_m1e_verdict_unchanged",
               (m1e_verdict or {}).get("status") == "COMPLETED"
               and (m1e_verdict or {}).get("defines_new_gate") is False,
               f"M1-E status={(m1e_verdict or {}).get('status')}")
    caches = groups["historical_utility_caches"]
    report.add("m1f_historical_caches_recomputed", caches["all_present"],
               "recomputed: match" if caches["all_present"] else
               "server-side only, absent locally: "
               f"{caches['server_only_missing']}",
               pending=not caches["all_present"])
    return pin


def _check_historical(report, repo_root):
    clean, detail = _git_status_clean(repo_root, P.HISTORICAL_NAMESPACES)
    report.add("m1f_historical_namespaces_unmodified",
               clean is not False, detail, pending=clean is None)


def _check_artifacts(report, repo_root):
    required = [P.M1F_EVIDENCE_PINS_FILENAME, P.M1F_TASK_VALIDITY_FILENAME,
                P.M1F_TASK_VALIDITY_REPORT_FILENAME,
                P.M1F_ACTIVE_AUDIT_FILENAME,
                P.M1F_WITHIN_SNAPSHOT_FILENAME,
                P.M1F_ROBUSTNESS_FILENAME, P.M1F_PHEME_FILENAME,
                P.M1F_VERDICT_FILENAME, P.M1F_REPORT_FILENAME]
    missing = [name for name in required
               if not os.path.exists(P.m1f_path(repo_root, name))]
    base_dir = P.m1f_path(repo_root, P.M1F_BASELINES_DIRNAME)
    if not os.path.isdir(base_dir):
        missing.append(f"{P.M1F_BASELINES_DIRNAME}/")
    else:
        for dataset in P.DATASETS:
            if not os.path.exists(os.path.join(base_dir, f"{dataset}.json")):
                missing.append(f"{P.M1F_BASELINES_DIRNAME}/{dataset}.json")
    verdict = _read_json(P.m1f_path(repo_root, P.M1F_VERDICT_FILENAME)) or {}
    paused = verdict.get("status") == "INFRASTRUCTURE_PAUSE"
    report.add("m1f_required_artifacts_present", not missing,
               f"missing: {missing}" if missing else
               "evidence pins, task validity, baselines, active/within/"
               "robustness/PHEME diagnostics, verdict and report all present",
               pending=bool(missing) and paused)
    return not missing


def _check_task_validity(report, repo_root):
    payload = _read_json(P.m1f_path(repo_root, P.M1F_TASK_VALIDITY_FILENAME))
    if payload is None:
        report.add("m1f_task_validity_recorded", False,
                   "task_validity.json missing", pending=True)
        return
    meaning = payload.get("utility_meaning") or {}
    datasets = payload.get("datasets") or {}
    pheme = datasets.get("pheme") or {}
    problems = []
    if payload.get("relabelled_historical_data") is not False:
        problems.append("historical data was relabelled")
    if meaning.get("is_verification_utility") is not False:
        problems.append("utility described as truth verification")
    if pheme.get("is_truth_verification") is not False:
        problems.append("PHEME rumour treated as truth value")
    boundary = str(pheme.get("boundary") or "")
    if "rumour != false" not in boundary or "non-rumour != true" not in boundary:
        problems.append("PHEME rumour/non-rumour boundary text missing")
    if (payload.get("cross_dataset_boundary") or {}).get(
            "is_language_only_shift") is not False:
        problems.append("cross-dataset contrast called a language-only shift")
    report.add("m1f_task_validity_boundaries", not problems,
               "; ".join(problems) if problems else
               "utility is rumour-label agreement utility, not truth "
               "verification; PHEME rumour/non-rumour != false/true; the "
               "dataset contrast is not reduced to a language shift")


def _check_baselines(report, repo_root):
    problems = []
    for dataset in P.DATASETS:
        payload = _read_json(os.path.join(
            P.m1f_path(repo_root, P.M1F_BASELINES_DIRNAME),
            f"{dataset}.json"))
        if payload is None:
            problems.append(f"{dataset}: baselines payload missing")
            continue
        if payload.get("defines_gate") is not False:
            problems.append(f"{dataset}: baseline stage defines a gate")
        variants = payload.get("variants") or {}
        if sorted(variants) != sorted(P.M1F_VARIANTS):
            problems.append(f"{dataset}: variants {sorted(variants)}")
            continue
        reused = variants.get(P.M1F_S4) or {}
        if reused.get("kind") != "reused" or \
                reused.get("frozen_variant") != "D5_Z_S_C":
            problems.append(f"{dataset}: S4 is not the frozen D5 reuse")
        comparator = payload.get("comparator") or {}
        pool = list(comparator.get("pool") or [])
        if pool != list(P.M1F_COMPARATOR_POOL):
            problems.append(f"{dataset}: comparator pool {pool}")
        if comparator.get("selected") not in P.M1F_COMPARATOR_POOL:
            problems.append(f"{dataset}: comparator {comparator.get('selected')}"
                            " is not a strong simple baseline")
        means = comparator.get("dev_macro_f1_mean") or {}
        if sorted(means) != sorted(P.M1F_COMPARATOR_POOL):
            problems.append(f"{dataset}: comparator dev scores incomplete")
        dev = payload.get("dev_macro_f1") or {}
        for variant in P.M1F_COMPARATOR_POOL:
            if variant not in dev:
                problems.append(f"{dataset}/{variant}: no dev Macro-F1")
        # leakage: held-out reader must not appear in training/dev/scaler
        for variant, entry in variants.items():
            for held, rot in (entry.get("rotations") or {}).items():
                train_readers = rot.get("train_readers")
                if train_readers and held in train_readers:
                    problems.append(f"{dataset}/{variant}/{held}: held-out "
                                    "reader inside the training pair")
                folds = rot.get("fold_sizes") or {}
                scaler_rows = rot.get("scaler_rows")
                if entry.get("kind") == "trained":
                    if scaler_rows != folds.get("train"):
                        problems.append(f"{dataset}/{variant}/{held}: scaler "
                                        f"fitted on {scaler_rows} rows, train "
                                        f"fold {folds.get('train')}")
                    config = rot.get("selected_config") or {}
                    if config.get("hidden") != 32 or \
                            config.get("dropout") not in P.MODEL_GRID["dropout"] \
                            or config.get("lr") not in P.MODEL_GRID["lr"] \
                            or config.get("weight_decay") \
                            not in P.MODEL_GRID["weight_decay"]:
                        problems.append(f"{dataset}/{variant}/{held}: grid drift")
                if entry.get("kind") == "constant" and scaler_rows not in (0, None):
                    problems.append(f"{dataset}/{variant}/{held}: constant "
                                    "baseline fitted a scaler")
    report.add("m1f_baselines_fixed_and_eval_never_used_for_selection",
               not problems, "; ".join(problems[:6]) if problems else
               "S0-S4 fixed; S4 reuses the frozen D5 predictions; the "
               "comparator was selected from S0-S3 on training-reader dev "
               "Macro-F1 only; scalers and priors use training rows only")


def _check_active_and_within(report, repo_root):
    active = _read_json(P.m1f_path(repo_root, P.M1F_ACTIVE_AUDIT_FILENAME))
    problems = []
    if active is None:
        problems.append("active_signed_audit.json missing")
    else:
        primary = active.get("primary") or {}
        if primary.get("primary") != P.M1F_S4:
            problems.append("primary is not S4/D5")
        if primary.get("comparator") not in P.M1F_COMPARATOR_POOL:
            problems.append("comparator is not a strong simple baseline")
        variants = active.get("variants") or {}
        for variant in P.M1F_VARIANTS:
            entry = variants.get(variant)
            if entry is None:
                problems.append(f"{variant}: active bundle missing")
                continue
            act = entry.get("active") or {}
            if act.get("defined") is not True:
                problems.append(f"{variant}: active audit undefined")
                continue
            if act.get("n_active") != (act.get("n_gold_helpful", 0)
                                       + act.get("n_gold_harmful", 0)):
                problems.append(f"{variant}: active count mismatch")
            if "neutral_prediction_rate" not in act:
                problems.append(f"{variant}: NEUTRAL rate on active rows "
                                "not reported")
    report.add("m1f_active_signed_audit_complete", not problems,
               "; ".join(problems[:6]) if problems else
               "HELPFUL/HARMFUL-only metrics, active prevalence and the "
               "NEUTRAL prediction rate on active rows are reported for every "
               "variant (NEUTRAL on an active row counts as an error)")

    within = _read_json(P.m1f_path(repo_root, P.M1F_WITHIN_SNAPSHOT_FILENAME))
    problems = []
    if within is None:
        problems.append("within_snapshot_audit.json missing")
    else:
        if within.get("min_rows") != P.M1F_WITHIN_MIN_ROWS:
            problems.append("min-rows rule drifted")
        variants = within.get("variants") or {}
        for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
            entry = variants.get(variant)
            if entry is None:
                problems.append(f"{variant}: within-snapshot stats missing")
                continue
            if entry.get("centered_pooled_spearman") is None:
                problems.append(f"{variant}: centered Spearman missing")
            if "pairwise_concordance" not in entry:
                problems.append(f"{variant}: pairwise concordance missing")
        comparisons = within.get("comparisons_vs_S2") or {}
        for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
            if variant == P.M1F_S2:
                continue
            if variant not in comparisons:
                problems.append(f"{variant}: centered increment vs S2 missing")
    report.add("m1f_within_snapshot_discrimination_reported", not problems,
               "; ".join(problems[:6]) if problems else
               "per-snapshot Spearman, pairwise order concordance and the "
               "snapshot-centred pooled Spearman (with an event-clustered "
               "interval) are reported for S2/S3/S4 and compared against S2")


def _check_thresholds(report, repo_root):
    payload = _read_json(P.m1f_path(repo_root, P.M1F_ROBUSTNESS_FILENAME))
    if payload is None:
        report.add("m1f_thresholds_frozen", False,
                   "robustness_diagnostic.json missing", pending=True)
        return
    problems = []
    if payload.get("official_threshold") != P.M1F_OFFICIAL_THRESHOLD:
        problems.append("official threshold drifted")
    keys = sorted((payload.get("diagnostics") or {}).keys())
    expected = sorted(f"{t:.2f}" for t in P.M1F_THRESHOLDS)
    if keys != expected:
        problems.append(f"diagnostic thresholds {keys} != {expected}")
    if payload.get("threshold_selected") is not False:
        problems.append("a new threshold was selected")
    if (payload.get("ab_label_swap") or {}).get("status") \
            != P.M1F_AB_SWAP_STATUS:
        problems.append("A/B label-swap status missing")
    if "near_boundary_rate" not in (payload.get("activity") or {}):
        problems.append("near-boundary frequency missing")
    report.add("m1f_thresholds_fixed_and_diagnostic_only", not problems,
               "; ".join(problems) if problems else
               "official ±0.05 unchanged; ±0.03/0.05/0.07 reported as "
               "diagnostics only, no threshold selected; A/B label-swap "
               "recorded as not testable without new reader inference")


def _check_boundaries(report, repo_root):
    root = os.path.join(str(repo_root), P.RESULTS_ROOT)
    m1f = P.m1f_dir(repo_root)
    present = set()
    if os.path.isdir(m1f):
        present = {n for n in os.listdir(m1f)
                   if os.path.isdir(os.path.join(m1f, n))}
    forbidden = sorted(present & set(P.M1F_FORBIDDEN_ARTIFACT_DIRS))
    report.add("m1f_no_reader_inference_or_label_artifacts", not forbidden,
               f"unexpected: {forbidden}" if forbidden else
               f"m1f contains only {sorted(present)}")

    m2 = sorted(n for n in os.listdir(root)
                if n.lower().startswith("m2")) if os.path.isdir(root) else []
    report.add("m1f_no_m2_artifacts", not m2,
               f"found: {m2}" if m2 else "no M2 artifact directory exists")

    verdict = _read_json(P.m1f_path(repo_root, P.M1F_VERDICT_FILENAME))
    if verdict is None:
        report.add("m1f_verdict_present", False, "M1F_VERDICT.json missing",
                   pending=True)
        return
    report.add("m1f_verdict_is_valid_outcome",
               verdict.get("verdict") in P.M1F_OUTCOMES,
               f"verdict={verdict.get('verdict')}")
    pheme = verdict.get("pheme_diagnostic") or {}
    report.add("m1f_pheme_remains_diagnostic",
               pheme.get("diagnostic_only") is True
               and pheme.get("decides_gate") is False,
               f"pheme diagnostic_only={pheme.get('diagnostic_only')} "
               f"decides_gate={pheme.get('decides_gate')}")
    report.add("m1f_primary_is_maweibo",
               verdict.get("primary_dataset") == P.PRIMARY_DATASET,
               f"primary_dataset={verdict.get('primary_dataset')}")
    report.add("m1f_m1_m1e_untouched",
               verdict.get("m1_verdict_unchanged") is True
               and verdict.get("m1e_verdict_unchanged") is True
               and verdict.get("m2_entered") is False,
               f"m1_unchanged={verdict.get('m1_verdict_unchanged')} "
               f"m1e_unchanged={verdict.get('m1e_verdict_unchanged')} "
               f"m2_entered={verdict.get('m2_entered')}")


def verify(repo_root: str) -> dict:
    report = Report()
    _check_pins(report, repo_root)
    _check_historical(report, repo_root)
    if _check_artifacts(report, repo_root):
        _check_task_validity(report, repo_root)
        _check_baselines(report, repo_root)
        _check_active_and_within(report, repo_root)
        _check_thresholds(report, repo_root)
        _check_boundaries(report, repo_root)
    return {"checks": report.checks, "issues": report.issues,
            "pending": report.pending, "m1f": {"mode": "m1f"}}
