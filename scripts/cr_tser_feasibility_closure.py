#!/usr/bin/env python
"""CR-TSER V2R1 feasibility closure after the P2 failure (plan §25, §26).

This round is not an experiment. It re-verifies the frozen P1/P2 artifacts
against the frozen protocol, checks that nothing moved, and writes the formal
closure:

    P1 = PASS
    P2 = FAIL
    P3 = NOT RUN
    P4 = NOT RUN
    CR_TSER_V2R1_FEASIBILITY = NO_GO
    reason = STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED

Every number in the closure is read out of the gate artifacts, and every
threshold is compared against the frozen configuration, so the closure cannot
disagree with the evidence it closes. A single mismatch is a refusal, not a
warning. No P3/P4 artifact is created or implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (REPO / "scripts", REPO / "project"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cr_tser.config.pilot_config import (BOOTSTRAP_ITERATIONS,  # noqa: E402
                                         BOOTSTRAP_SEED, CUTOFFS_MIN,
                                         P1_MEAN_DISAGREEMENT_MIN,
                                         P1_PAIR_DISAGREEMENT_MIN,
                                         P1_PAIRS_REQUIRED, P2_EDGE_DELTA_MIN,
                                         PARTITION_SEED, PRIMARY_DATASET,
                                         PROTOCOL_VERSION, READER_KEYS,
                                         READER_MODEL_IDS, SECONDARY_DATASET,
                                         SPLIT_SIZES, UTILITY_THRESHOLD,
                                         V2R1_RESULTS_ROOT)

GATES = ("p1_maweibo.json", "p2_maweibo.json", "p1_pheme_diagnostic.json",
         "p2_pheme_diagnostic.json")
CLOSURE_DIR = os.path.join(V2R1_RESULTS_ROOT, "closure")

#: Artifacts that must NOT exist: this round produces no P3/P4 evidence.
FORBIDDEN_ARTIFACTS = ("predictor", "unseen_reader")

REASON_CODE = "STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED"
REASON = (
    "The frozen pilot did not establish the required structural-interaction "
    "effect: the observed edge gap was positive but below the pre-registered "
    "threshold and its event-bootstrap confidence interval crossed zero."
)
NOT_RUN = ("predictor_training", "b0_b1_b3_training", "stage_a", "stage_b",
           "held_out_reader_evaluation", "P3", "P4",
           "final_selector_experiments", "final_pilot_report")


class ClosureRefused(RuntimeError):
    """Raised when the closure would misstate the frozen evidence."""


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expect(condition, message, problems):
    if not condition:
        problems.append(message)


def build_closure(root: str = V2R1_RESULTS_ROOT,
                  baseline: str = "") -> dict:
    """Assemble the closure, refusing if any frozen fact does not hold."""
    gates_dir = os.path.join(root, "gates")
    problems = []

    artifacts, digests = {}, {}
    for name in GATES:
        path = os.path.join(gates_dir, name)
        if not os.path.exists(path):
            problems.append(f"missing gate artifact {name}")
            continue
        artifacts[name] = _read_json(path)
        digests[name] = _sha256(path)
    if problems:
        raise ClosureRefused("; ".join(problems))

    p1 = artifacts["p1_maweibo.json"]
    p2 = artifacts["p2_maweibo.json"]
    p1_diag = artifacts["p1_pheme_diagnostic.json"]
    p2_diag = artifacts["p2_pheme_diagnostic.json"]

    # -- protocol, readers, dataset roles ---------------------------------
    for name, payload in artifacts.items():
        _expect(payload.get("protocol") == PROTOCOL_VERSION,
                f"{name}: protocol {payload.get('protocol')!r}", problems)
        _expect(sorted(payload.get("readers_present") or [])
                == sorted(READER_KEYS),
                f"{name}: readers {payload.get('readers_present')}", problems)
        _expect(not payload.get("retired_readers_present"),
                f"{name}: a retired reader is present", problems)
    _expect(p1.get("dataset") == PRIMARY_DATASET
            and p2.get("dataset") == PRIMARY_DATASET,
            "the primary artifacts are not on Ma-Weibo", problems)
    _expect(p1.get("decides_primary_gate") is True
            and p2.get("decides_primary_gate") is True,
            "a primary artifact does not decide a gate", problems)
    for name, payload in (("p1_pheme", p1_diag), ("p2_pheme", p2_diag)):
        _expect(payload.get("dataset") == SECONDARY_DATASET,
                f"{name}: dataset {payload.get('dataset')!r}", problems)
        _expect(payload.get("diagnostic_only") is True
                and payload.get("decides_primary_gate") is False,
                f"{name} is not diagnostic-only", problems)

    # -- verdicts ---------------------------------------------------------
    _expect(p1.get("verdict") == "P1_PASS",
            f"P1 verdict is {p1.get('verdict')!r}", problems)
    _expect(p2.get("verdict") == "P2_FAIL",
            f"P2 verdict is {p2.get('verdict')!r}", problems)
    _expect(p1["gate"].get("pass") is True, "P1 gate is not passing", problems)
    _expect(p2["gate"].get("pass") is False, "P2 gate is not failing",
            problems)
    _expect(p1_diag.get("verdict") == "DIAGNOSTIC_ONLY"
            and p2_diag.get("verdict") == "DIAGNOSTIC_ONLY",
            "a PHEME artifact is not DIAGNOSTIC_ONLY", problems)

    # -- frozen thresholds, read from the artifacts and compared to config --
    expected_thresholds = {
        "utility_threshold": UTILITY_THRESHOLD,
        "p1_mean_disagreement_min": P1_MEAN_DISAGREEMENT_MIN,
        "p1_pair_disagreement_min": P1_PAIR_DISAGREEMENT_MIN,
        "p1_pairs_required": P1_PAIRS_REQUIRED,
        "p2_edge_delta_min": P2_EDGE_DELTA_MIN,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for name, payload in (("p1_maweibo", p1), ("p2_maweibo", p2)):
        frozen = payload.get("frozen_statistics", {})
        for key, want in expected_thresholds.items():
            if key in frozen:
                _expect(frozen[key] == want,
                        f"{name}: {key}={frozen[key]!r} != {want!r}", problems)
    # the gate record names its thresholds differently from the config
    for gate_key, config_key in (("mean_threshold",
                                  "p1_mean_disagreement_min"),
                                 ("pair_threshold",
                                  "p1_pair_disagreement_min"),
                                 ("pairs_required", "p1_pairs_required")):
        _expect(p1["gate"].get(gate_key) == expected_thresholds[config_key],
                f"p1_maweibo gate {gate_key} drifted", problems)
    _expect(p2["gate"].get("edge_threshold") == P2_EDGE_DELTA_MIN,
            "p2_maweibo gate edge_threshold drifted", problems)
    for name, payload in (("p2_maweibo", p2), ("p2_pheme", p2_diag)):
        _expect(payload["report"].get("matching_unit") == "reader_x_snapshot"
                and payload["report"].get("bootstrap_unit") == "event",
                f"{name}: matching/bootstrap unit drifted", problems)
    _expect(p1["coverage"].get("split_seed") == PARTITION_SEED
            and p1["coverage"].get("split_sizes") == dict(SPLIT_SIZES)
            and p1["coverage"].get("cutoffs") == list(CUTOFFS_MIN),
            "the coverage block does not match the frozen split/cutoffs",
            problems)

    # -- frozen input digests --------------------------------------------
    from cr_tser_verify_pilot import V2R1_FROZEN_LABELS_SHA256
    from cr_tser_verify_pilot import V2R1_FROZEN_MANIFEST_SHA256
    labels_seen = {}
    for name, payload in artifacts.items():
        frozen = payload.get("frozen_inputs", {})
        for dataset, want in V2R1_FROZEN_LABELS_SHA256.items():
            got = (frozen.get("labels_sha256") or {}).get(dataset)
            _expect(got == want, f"{name}: {dataset} cache digest {got!r}",
                    problems)
            labels_seen[dataset] = got
        for rel, want in V2R1_FROZEN_MANIFEST_SHA256.items():
            short = rel.split("manifests/", 1)[1]
            got = (frozen.get("manifest_sha256") or {}).get(short)
            _expect(got == want, f"{name}: manifest digest {short}", problems)

    # -- P3/P4 must be absent --------------------------------------------
    absent = {}
    for name in FORBIDDEN_ARTIFACTS:
        path = os.path.join(root, name)
        absent[name] = not os.path.exists(path)
        _expect(absent[name], f"{name}/ exists under {root}", problems)
    stale = sorted(p for p in os.listdir(gates_dir)
                   if p.startswith(("p3_", "p4_")))
    _expect(not stale, f"P3/P4 gate artifacts present: {stale}", problems)

    if problems:
        raise ClosureRefused(
            "the frozen evidence does not support a clean closure: "
            + "; ".join(problems))

    edge = p2["report"]["edge"]
    subtree = p2["report"]["subtree"]
    closure = {
        "artifact": "CR_TSER_V2R1_FEASIBILITY_CLOSURE",
        "baseline_commit": baseline,
        "protocol": PROTOCOL_VERSION,
        "round": "P2_failure_closure",
        "reader_keys": list(READER_KEYS),
        "reader_model_ids": {k: READER_MODEL_IDS[k] for k in READER_KEYS},
        "dataset_roles": {
            "primary_decision_dataset": PRIMARY_DATASET,
            "secondary_dataset": SECONDARY_DATASET,
            "secondary_role": "diagnostic_only",
            "pheme_decides_any_gate": False,
        },
        "frozen_thresholds": {
            **expected_thresholds,
            "cutoffs": list(CUTOFFS_MIN),
            "partition_seed": PARTITION_SEED,
            "split_sizes": dict(SPLIT_SIZES),
            "matching_unit": "reader_x_snapshot",
            "bootstrap_unit": "event",
        },
        "P1": {
            "verdict": "P1_PASS",
            "dataset": PRIMARY_DATASET,
            "mean_disagreement": p1["report"]["macro_mean_disagreement"],
            "pairs": [
                {"reader_a": pair["reader_a"], "reader_b": pair["reader_b"],
                 "n_active": pair["n_active"],
                 "disagreement": pair["disagreement"],
                 "utility_spearman": pair["utility_spearman"],
                 "sign_contingency": pair["sign_contingency"]}
                for pair in p1["report"]["pairs"]],
            "active_counts": p1["active_counts"],
            "jointly_active_pairs": p1["jointly_active_pairs"],
            "coverage": p1["coverage"],
            "gate": p1["gate"],
            "artifact": f"gates/{GATES[0]}",
            "artifact_sha256": digests[GATES[0]],
        },
        "P2": {
            "verdict": "P2_FAIL",
            "dataset": PRIMARY_DATASET,
            "delta_edge": edge["delta"],
            "edge_ci_low": edge["ci_low"],
            "edge_ci_high": edge["ci_high"],
            "edge_n_events": edge["n_events"],
            "edge_n_matched_pairs": edge["n_matched_pairs"],
            "delta_subtree": subtree["delta"],
            "subtree_ci_low": subtree["ci_low"],
            "subtree_ci_high": subtree["ci_high"],
            "subtree_n_events": subtree["n_events"],
            "subtree_n_matched_pairs": subtree["n_matched_pairs"],
            "slot_means": p2["slot_means"],
            "record_counts": p2["record_counts"],
            "coverage": p2["coverage"],
            "gate": p2["gate"],
            "artifact": f"gates/{GATES[1]}",
            "artifact_sha256": digests[GATES[1]],
        },
        "pheme_diagnostic": {
            "role": "diagnostic_only",
            "decides_primary_gate": False,
            "P1": {
                "verdict": p1_diag["verdict"],
                "mean_disagreement":
                    p1_diag["report"]["macro_mean_disagreement"],
                "diagnostic_gate": p1_diag.get("diagnostic_gate"),
                "artifact": f"gates/{GATES[2]}",
                "artifact_sha256": digests[GATES[2]],
            },
            "P2": {
                "verdict": p2_diag["verdict"],
                "delta_edge": p2_diag["report"]["edge"]["delta"],
                "edge_ci_low": p2_diag["report"]["edge"]["ci_low"],
                "edge_ci_high": p2_diag["report"]["edge"]["ci_high"],
                "delta_subtree": p2_diag["report"]["subtree"]["delta"],
                "diagnostic_gate": p2_diag.get("diagnostic_gate"),
                "artifact": f"gates/{GATES[3]}",
                "artifact_sha256": digests[GATES[3]],
            },
        },
        "P3": "NOT_RUN",
        "P4": "NOT_RUN",
        "final_feasibility_verdict": "NO_GO",
        "reason_code": REASON_CODE,
        "reason": REASON,
        "gate_artifacts": {name: digests[name] for name in GATES},
        "frozen_inputs": {
            "labels_sha256": labels_seen,
            "manifest_sha256": V2R1_FROZEN_MANIFEST_SHA256,
            "manifests_unchanged": True,
            "utility_caches_unchanged": True,
            "historical_namespace_untouched": "results/cr_tser_v2",
        },
        "not_run": list(NOT_RUN),
        "p3_p4_artifacts_absent": absent,
    }
    return closure


def _write_markdown(closure: dict, path: str) -> str:
    p1, p2, diag = closure["P1"], closure["P2"], closure["pheme_diagnostic"]
    thresholds = closure["frozen_thresholds"]
    lines = [
        "# CR-TSER V2R1 — feasibility closure",
        "",
        f"Baseline commit: `{closure['baseline_commit']}`  ",
        f"Protocol: `{closure['protocol']}`  ",
        f"Readers: {', '.join(closure['reader_keys'])}  ",
        f"Primary decision dataset: `{closure['dataset_roles']['primary_decision_dataset']}`  ",
        f"Secondary (diagnostic only): `{closure['dataset_roles']['secondary_dataset']}`",
        "",
        "## Final verdict",
        "",
        "```text",
        "P1 = PASS",
        "P2 = FAIL",
        "P3 = NOT RUN",
        "P4 = NOT RUN",
        "",
        f"CR_TSER_V2R1_FEASIBILITY = {closure['final_feasibility_verdict']}",
        f"reason = {closure['reason_code']}",
        "```",
        "",
        closure["reason"],
        "",
        "The observed direction was positive — parent-child removal cost more "
        "than matched non-adjacent removal — but the frozen pilot did not "
        "establish the pre-registered structural-interaction effect, so no "
        "structural claim is made.",
        "",
        "## P1 — reader heterogeneity (Ma-Weibo, primary) = PASS",
        "",
        f"- mean disagreement **{p1['mean_disagreement']:.4f}** against a "
        f"threshold of {thresholds['p1_mean_disagreement_min']:.2f}",
        f"- pairs at or above {thresholds['p1_pair_disagreement_min']}: "
        f"{p1['gate']['pairs_above_pair_threshold']} of "
        f"{len(p1['pairs'])}, with {p1['gate']['pairs_required']} required",
        "",
        "| pair | jointly active | disagreement | utility Spearman |",
        "|---|---|---|---|",
    ]
    for pair in p1["pairs"]:
        lines.append(
            f"| {pair['reader_a']} + {pair['reader_b']} | {pair['n_active']} | "
            f"**{pair['disagreement']:.4f}** | {pair['utility_spearman']:.4f} |")
    lines += [
        "",
        f"- active counts: "
        + ", ".join(f"{k} {v['active']}/{v['atomic_rows']}"
                    for k, v in p1["active_counts"].items()
                    if isinstance(v, dict)),
        f"- coverage: {p1['coverage']['events_with_labels']}/"
        f"{p1['coverage']['labelled_pool_events']} labelled events, "
        f"{p1['coverage']['event_cutoff_pairs']} event×cutoff pairs, cutoffs "
        f"{p1['coverage']['cutoffs']}",
        f"- artifact: `{p1['artifact']}` (`{p1['artifact_sha256']}`)",
        "",
        "## P2 — structured interaction (Ma-Weibo, primary) = FAIL",
        "",
        f"- **Δedge = {p2['delta_edge']:.6f}** against a pre-registered "
        f"threshold of {thresholds['p2_edge_delta_min']}",
        f"- 95% event-bootstrap CI = "
        f"**({p2['edge_ci_low']:.6f}, {p2['edge_ci_high']:.6f})** over "
        f"{p2['edge_n_events']} events and {p2['edge_n_matched_pairs']} matched "
        f"(reader, snapshot) pairs",
        f"- the CI lower bound is not above 0, so the second condition of the "
        f"frozen gate also fails",
        f"- confirmatory Δsubtree = {p2['delta_subtree']:.6f}, 95% CI "
        f"({p2['subtree_ci_low']:.6f}, {p2['subtree_ci_high']:.6f}) over "
        f"{p2['subtree_n_events']} events",
        "",
        "| slot | role | matched pairs | mean \\|Interaction\\| |",
        "|---|---|---|---|",
        f"| I2 parent-child | treatment | {p2['slot_means']['pc']['n_pairs']} | "
        f"{p2['slot_means']['pc']['mean_abs_interaction']:.5f} |",
        f"| I3 matched-nonadjacent | control | "
        f"{p2['slot_means']['na']['n_pairs']} | "
        f"{p2['slot_means']['na']['mean_abs_interaction']:.5f} |",
        f"| I4 subtree | treatment | {p2['slot_means']['sub']['n_pairs']} | "
        f"{p2['slot_means']['sub']['mean_abs_interaction']:.5f} |",
        f"| I5 matched-disconnected | control | "
        f"{p2['slot_means']['disc']['n_pairs']} | "
        f"{p2['slot_means']['disc']['mean_abs_interaction']:.5f} |",
        "",
        f"- artifact: `{p2['artifact']}` (`{p2['artifact_sha256']}`)",
        "",
        "## PHEME — diagnostic only",
        "",
        f"- `diagnostic_only = true`, `decides_primary_gate = false`; PHEME "
        f"decided no gate",
        f"- P1 mean disagreement {diag['P1']['mean_disagreement']:.4f}; P2 "
        f"Δedge {diag['P2']['delta_edge']:.6f}, 95% CI "
        f"({diag['P2']['edge_ci_low']:.6f}, {diag['P2']['edge_ci_high']:.6f}), "
        f"Δsubtree {diag['P2']['delta_subtree']:.6f}",
        f"- artifacts: `{diag['P1']['artifact']}`, `{diag['P2']['artifact']}`",
        "",
        "## Frozen protocol (unchanged by this round)",
        "",
        "```text",
        f"utility_threshold        = {thresholds['utility_threshold']}",
        f"p1_mean_disagreement_min = {thresholds['p1_mean_disagreement_min']}",
        f"p1_pair_disagreement_min = {thresholds['p1_pair_disagreement_min']}",
        f"p1_pairs_required        = {thresholds['p1_pairs_required']}",
        f"p2_edge_delta_min        = {thresholds['p2_edge_delta_min']}",
        f"bootstrap                = event-level, "
        f"{thresholds['bootstrap_iterations']} iterations, "
        f"seed {thresholds['bootstrap_seed']}",
        f"matching_unit            = {thresholds['matching_unit']}",
        f"cutoffs                  = {thresholds['cutoffs']}",
        f"partition_seed           = {thresholds['partition_seed']}",
        f"split_sizes              = {thresholds['split_sizes']}",
        "```",
        "",
        "## Immutability",
        "",
        "| object | status |",
        "|---|---|",
    ]
    for dataset, digest in closure["frozen_inputs"]["labels_sha256"].items():
        lines.append(f"| `utility_labels/{dataset}/labels.jsonl` | "
                     f"`{digest}` (unchanged) |")
    lines += [
        "| 10 v2r1 manifest files | unchanged |",
        "| `results/cr_tser_v2/` | untouched (read-only history) |",
        "",
        "## Not run",
        "",
        "No predictor training, B0/B1/B3 training, Stage A/B, held-out-reader "
        "evaluation, P3, P4, final selector experiment or final pilot report "
        "was run, and no P3/P4 artifact exists. No threshold, reader, dataset, "
        "split, cutoff, intervention group or utility label was changed in "
        "response to the P2 result.",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--baseline-commit", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.out_root or os.environ.get("CRTSER_OUT_ROOT",
                                           V2R1_RESULTS_ROOT)
    baseline = args.baseline_commit or _head_commit()
    try:
        closure = build_closure(root, baseline)
    except ClosureRefused as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, indent=1))
        return 3
    os.makedirs(CLOSURE_DIR, exist_ok=True)
    json_path = os.path.join(CLOSURE_DIR, "FEASIBILITY_CLOSURE.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(closure, fh, indent=1, ensure_ascii=False)
    md_path = _write_markdown(closure, os.path.join(
        CLOSURE_DIR, "FEASIBILITY_CLOSURE.md"))
    print(json.dumps({
        "P1": closure["P1"]["verdict"],
        "P2": closure["P2"]["verdict"],
        "P3": closure["P3"],
        "P4": closure["P4"],
        "final_feasibility_verdict": closure["final_feasibility_verdict"],
        "reason_code": closure["reason_code"],
        "json": json_path,
        "markdown": md_path,
    }, indent=1))
    return 0


def _head_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover - git unavailable
        return ""


if __name__ == "__main__":
    sys.exit(main())
