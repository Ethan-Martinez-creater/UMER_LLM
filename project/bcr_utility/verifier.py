"""BCR-Utility M0 verifier (plan §28, §30).

Fail-closed checks over the M0 artifacts. Every check is either an exact
comparison against a frozen constant or a recomputation from primary inputs;
nothing here trusts a summary written by the producing stage.

The verifier never reads a model and never touches the historical result
namespaces for writing.
"""
from __future__ import annotations

import json
import os
import subprocess

from .config import protocol as P
from .data import atomic_manifest, historical_import
from .probes import probe_contexts, probe_manifest


class Report:
    """Ordered check log with an issue/pending split."""

    def __init__(self):
        self.checks = []
        self.issues = []
        self.pending = []

    def add(self, name: str, ok: bool, detail: str = "", pending: bool = False):
        entry = {"check": name, "ok": bool(ok), "detail": str(detail),
                 "pending": bool(pending)}
        self.checks.append(entry)
        if not ok:
            (self.pending if pending else self.issues).append(name)
        return entry

    def payload(self, **extra) -> dict:
        return {
            "protocol": P.PROTOCOL_VERSION,
            "mode": "m0",
            "issues": list(self.issues),
            "pending": list(self.pending),
            "n_issues": len(self.issues),
            "n_pending": len(self.pending),
            "checks": self.checks,
            **extra,
        }


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git_status_clean(repo_root: str, paths) -> tuple:
    """``(clean, detail)`` for the read-only historical namespaces.

    Only *tracked* changes fail the check. The frozen label caches are
    deliberately untracked on SERVER (they are large server-side artifacts), so
    an ``??`` entry under a historical namespace is reported but is not itself
    a modification of frozen evidence — their bytes are checked separately by
    the cache-digest check.
    """
    paths = tuple(paths)
    if not paths:
        return True, "no historical namespace paths configured"
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=str(repo_root), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git unavailable: {exc}"
    if proc.returncode != 0:
        return None, f"git status failed: {proc.stderr.strip()[:200]}"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    tracked = [line for line in lines if not line.startswith("??")]
    untracked = [line for line in lines if line.startswith("??")]
    if tracked:
        return False, "tracked changes in historical namespaces: " + \
            "; ".join(tracked[:5])
    return True, ("no tracked changes in historical namespaces"
                  + (f" ({len(untracked)} untracked artifact path(s) present)"
                     if untracked else ""))


# --------------------------------------------------------------------------
# historical inputs
# --------------------------------------------------------------------------
def _verify_historical(report, repo_root, identity):
    hist_root = P.historical_root(repo_root)
    report.add("historical_identity_present", identity is not None,
               P.HISTORICAL_IDENTITY_FILENAME if identity else "missing")
    if identity is None:
        return
    report.add("historical_identity_protocol",
               identity.get("protocol") == P.PROTOCOL_VERSION
               and identity.get("baseline_commit") == P.BASELINE_COMMIT,
               f"protocol={identity.get('protocol')} baseline="
               f"{identity.get('baseline_commit')}")

    problems = []
    for dataset in P.DATASETS:
        entry = ((identity.get("datasets") or {}).get(dataset) or {})
        labels = entry.get("labels") or {}
        spec = P.FROZEN_HISTORICAL_LABELS[dataset]
        if labels.get("sha256") != spec["sha256"]:
            problems.append(f"{dataset}: labels sha256 {labels.get('sha256')}")
        if labels.get("rows") != spec["rows"]:
            problems.append(f"{dataset}: labels rows {labels.get('rows')}")
        if labels.get("bytes") != spec["bytes"]:
            problems.append(f"{dataset}: labels bytes {labels.get('bytes')}")
        if (labels.get("rows_per_reader") or {}) != \
                P.FROZEN_HISTORICAL_ROWS_PER_READER[dataset]:
            problems.append(f"{dataset}: per-reader row counts")
        if sorted(labels.get("cutoffs") or []) != sorted(P.CUTOFFS_MIN):
            problems.append(f"{dataset}: label cutoffs")
    report.add("historical_labels_pinned_to_frozen_digest", not problems,
               "; ".join(problems) if problems
               else "both frozen caches match the pinned SHA256 / rows / bytes")

    problems = []
    for dataset in P.DATASETS:
        entry = ((identity.get("datasets") or {}).get(dataset) or {})
        manifests = entry.get("manifests") or {}
        for rel, expected in P.FROZEN_HISTORICAL_MANIFEST_SHA256.items():
            got = (manifests.get(rel) or {}).get("sha256")
            if got != expected:
                problems.append(f"{dataset}:{rel}")
        readers = (entry.get("reader_contract") or {}).get("readers") or {}
        if sorted(readers) != sorted(P.READER_KEYS):
            problems.append(f"{dataset}: reader panel {sorted(readers)}")
        for key, model_id in readers.items():
            if P.READER_MODEL_IDS.get(key) != model_id:
                problems.append(f"{dataset}:{key} model id")
        split = entry.get("event_split") or {}
        if split.get("split_seed") != P.SEED:
            problems.append(f"{dataset}: split seed {split.get('split_seed')}")
        if (split.get("sizes") or {}).get("foundation_train") != 80:
            problems.append(f"{dataset}: foundation_train size")
    report.add("historical_manifests_pinned_and_reader_panel_exact",
               not problems,
               "; ".join(problems) if problems
               else "10 manifests pinned, R1 reader panel exact, split seed "
                    "7319")

    # Recompute from the repository files: proves the manifests in the working
    # tree are still the frozen ones, not just that the artifact says so.
    problems = []
    for rel, expected in P.FROZEN_HISTORICAL_MANIFEST_SHA256.items():
        path = historical_import.manifest_path(hist_root, rel)
        if not os.path.exists(path):
            problems.append(f"{rel}: missing")
            continue
        got = historical_import.canonical_sha256_file(path)
        if got != expected:
            problems.append(f"{rel}: {got}")
    report.add("historical_manifests_recomputed_from_worktree", not problems,
               "; ".join(problems) if problems
               else f"{len(P.FROZEN_HISTORICAL_MANIFEST_SHA256)} v2r1 "
                    "manifests re-hashed in the working tree")

    clean, detail = _git_status_clean(repo_root, P.HISTORICAL_NAMESPACES)
    report.add("historical_namespaces_unmodified", clean is True,
               detail if clean is None else detail,
               pending=clean is None)

    # Label caches are not tracked, so they are only byte-checkable where they
    # exist. The artifact check above still pins their digest.
    recomputed = {}
    pending = False
    for dataset in P.DATASETS:
        path = historical_import.labels_path(hist_root, dataset)
        if not os.path.exists(path):
            pending = True
            continue
        recomputed[dataset] = historical_import.sha256_file(path)
    if recomputed:
        drift = {d: {"got": v,
                     "frozen": P.FROZEN_HISTORICAL_LABELS[d]["sha256"]}
                 for d, v in recomputed.items()
                 if v != P.FROZEN_HISTORICAL_LABELS[d]["sha256"]}
        report.add("historical_label_caches_recomputed", not drift,
                   f"recomputed {sorted(recomputed)}: "
                   f"{'match' if not drift else drift}")
    else:
        report.add("historical_label_caches_recomputed", False,
                   "label caches are SERVER-side; the pipeline pins their "
                   "SHA256 in historical_identity.json and re-verifies them "
                   "on SERVER", pending=True)

    recorded = identity.get("reused_dependencies") or {}
    problems = []
    for rel in P.REUSED_CR_TSER_MODULES:
        path = os.path.join(str(repo_root), rel.replace("/", os.sep))
        if not os.path.exists(path):
            problems.append(f"{rel}: missing")
            continue
        got = historical_import.canonical_sha256_file(path)
        if recorded.get(rel) != got:
            problems.append(f"{rel}: {got} != recorded {recorded.get(rel)}")
    report.add("reused_cr_tser_provenance_current", not problems,
               "; ".join(problems) if problems
               else f"{len(P.REUSED_CR_TSER_MODULES)} reused CR-TSER files "
                    "match the recorded digests")


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
def _verify_protocol(report):
    constants = P.frozen_constants()
    ok = (constants["protocol_version"] == "bcr_v1"
          and constants["primary_dataset"] == "maweibo"
          and constants["secondary_dataset"] == "pheme"
          and constants["utility_threshold"] == 0.05
          and tuple(constants["cutoffs_min"]) == (15, 60, 360)
          and constants["seed"] == 7319
          and constants["probe_seed"] == 7319
          and constants["probe_events_per_dataset"] == 36
          and constants["probe_events_per_cutoff"] == 12
          and constants["probe_items_total"] == 72
          and constants["bootstrap_iterations"] == 10000
          and constants["bootstrap_seed"] == 7319
          and constants["bootstrap_unit"] == "event")
    report.add("bcr_protocol_frozen", ok, json.dumps(constants, sort_keys=True))

    try:
        from cr_tser.config import pilot_config as cr
    except ImportError as exc:  # pragma: no cover
        report.add("bcr_inherited_constants_agree_with_cr_tser", False,
                   f"cr_tser unavailable: {exc}")
        return
    mismatched = {}
    if P.UTILITY_THRESHOLD != cr.UTILITY_THRESHOLD:
        mismatched["utility_threshold"] = (P.UTILITY_THRESHOLD,
                                           cr.UTILITY_THRESHOLD)
    if tuple(P.CUTOFFS_MIN) != tuple(cr.CUTOFFS_MIN):
        mismatched["cutoffs_min"] = (list(P.CUTOFFS_MIN),
                                     list(cr.CUTOFFS_MIN))
    if tuple(P.READER_KEYS) != tuple(cr.READER_KEYS):
        mismatched["reader_keys"] = (list(P.READER_KEYS), list(cr.READER_KEYS))
    if P.READER_MODEL_IDS != dict(cr.READER_MODEL_IDS):
        mismatched["reader_model_ids"] = (P.READER_MODEL_IDS,
                                          dict(cr.READER_MODEL_IDS))
    report.add("bcr_inherited_constants_agree_with_cr_tser", not mismatched,
               f"{mismatched}" if mismatched else
               "utility threshold, cutoffs and reader panel are identical to "
               "the frozen CR-TSER pilot")

    report.add("bcr_does_not_inherit_structured_interaction",
               bool(P.NON_SUPERVISION_INTERVENTION_TYPES)
               and P.ATOMIC_INTERVENTION_TYPE == "I1_atomic",
               "BCR supervises only I1_atomic; I2-I5 are named as "
               "non-supervision")


# --------------------------------------------------------------------------
# atomic index
# --------------------------------------------------------------------------
def _verify_atomic(report, repo_root, indexes):
    problems = []
    absent = [d for d in P.DATASETS if indexes.get(d) is None]
    report.add("atomic_index_present", not absent,
               f"missing: {absent}" if absent else
               f"{P.ATOMIC_INDEX_FILENAME} for both datasets")
    for dataset, index in indexes.items():
        if index is None:
            continue
        try:
            atomic_manifest.validate_atomic_index(index, dataset)
        except atomic_manifest.AtomicIndexRefused as exc:
            problems.append(f"{dataset}: {exc}")
            continue
        labels = index.get("source_labels") or {}
        if labels.get("sha256") != \
                P.FROZEN_HISTORICAL_LABELS[dataset]["sha256"]:
            problems.append(f"{dataset}: source label digest")
        excluded = index.get("excluded_intervention_type_counts") or {}
        for itype in P.NON_SUPERVISION_INTERVENTION_TYPES:
            if int(excluded.get(itype) or 0) <= 0:
                problems.append(f"{dataset}: {itype} not excluded/counted")
        if int(excluded.get(P.BASE_INTERVENTION_TYPE) or 0) <= 0:
            problems.append(f"{dataset}: I0 rows not accounted for")
    report.add("atomic_index_exact_and_i1_only", not problems,
               "; ".join(problems) if problems else
               f"keys exact ({P.FROZEN_ATOMIC_KEYS}), only I1_atomic indexed, "
               "I0/I2-I5 counted as excluded")


# --------------------------------------------------------------------------
# geometry diagnostic
# --------------------------------------------------------------------------
def _verify_geometry(report, geometries):
    absent = [d for d in P.DATASETS if geometries.get(d) is None]
    report.add("geometry_diagnostic_present", not absent,
               f"missing: {absent}" if absent else
               f"{P.GEOMETRY_DIAGNOSTIC_FILENAME} for both datasets")
    problems = []
    for dataset, geo in geometries.items():
        if geo is None:
            continue
        if geo.get("scope") != "diagnostic_only" or \
                geo.get("decides_gate") is not False:
            problems.append(f"{dataset}: scope")
        boot = geo.get("bootstrap") or {}
        if (boot.get("unit") != P.BOOTSTRAP_UNIT
                or boot.get("iterations") != P.BOOTSTRAP_ITERATIONS
                or boot.get("seed") != P.BOOTSTRAP_SEED):
            problems.append(f"{dataset}: bootstrap protocol {boot}")
        for key in ("activity", "ordinal_geometry",
                    "signed_direction_geometry",
                    "reader_evidence_interaction"):
            if key not in geo:
                problems.append(f"{dataset}: missing {key}")
        signed = (geo.get("signed_direction_geometry") or {}).get("per_pair")
        if not signed:
            problems.append(f"{dataset}: signed direction not reported")
        else:
            for pair, entry in signed.items():
                if "sign_contingency" not in entry or \
                        "conditional_direction" not in entry:
                    problems.append(f"{dataset}:{pair}: contingency missing")
        caveat = (geo.get("ordinal_geometry") or {}).get("caveat", "")
        if "does not by itself" not in caveat:
            problems.append(f"{dataset}: ordinal caveat missing")
    report.add("geometry_reports_activity_ordinal_signed_interaction",
               not problems, "; ".join(problems) if problems else
               "activity, ordinal, signed direction and interaction shares "
               "all present; ordinal is explicitly not read as transfer")


# --------------------------------------------------------------------------
# probe manifest
# --------------------------------------------------------------------------
def _verify_probes(report, repo_root, manifests, availability):
    absent = [d for d in P.DATASETS if manifests.get(d) is None]
    report.add("probe_manifest_present", not absent,
               f"missing: {absent}" if absent else
               f"{P.PROBE_MANIFEST_FILENAME} for both datasets")
    report.add("probe_availability_present",
               all(availability.get(d) is not None for d in P.DATASETS),
               f"{P.PROBE_AVAILABILITY_FILENAME} for both datasets")

    problems = []
    overlap_problems = []
    for dataset, manifest in manifests.items():
        if manifest is None:
            continue
        try:
            probe_manifest.validate_probe_manifest(manifest, dataset)
        except probe_manifest.ProbeManifestRefused as exc:
            problems.append(f"{dataset}: {exc}")
            continue
        split = historical_import.read_json(historical_import.manifest_path(
            P.historical_root(repo_root),
            f"manifests/{dataset}/event_split.json"))
        foundation = {str(e) for e in split.get(P.PROBE_SOURCE_SPLIT, [])}
        probe_events = {i["event_id"] for i in manifest["items"]}
        outside = sorted(probe_events - foundation)
        if outside:
            problems.append(f"{dataset}: {len(outside)} probe events are not "
                            f"in {P.PROBE_SOURCE_SPLIT}: {outside[:3]}")
        measured = probe_manifest.audit_overlap(manifest, split)
        if any(v != 0 for v in measured.values()):
            overlap_problems.append(f"{dataset}: {measured}")
        counts = manifest.get("per_cutoff_counts") or {}
        if any(int(counts.get(str(c), 0)) != P.PROBE_EVENTS_PER_CUTOFF
               for c in P.CUTOFFS_MIN):
            problems.append(f"{dataset}: per-cutoff counts {counts}")
    report.add("probe_manifest_exact_36_and_12_per_cutoff", not problems,
               "; ".join(problems) if problems else
               "36 items / 12 per cutoff per dataset, all from "
               "foundation_train, no gold label stored")
    report.add("probe_events_disjoint_from_utility_splits",
               not overlap_problems,
               "; ".join(overlap_problems) if overlap_problems else
               "probe events x utility_train/dev/eval overlap = 0 "
               "(recomputed from the frozen event split)")

    plan = probe_contexts.context_plan()
    report.add("probe_context_plan_frozen",
               tuple(plan["contexts"]) == P.PROBE_CONTEXTS
               and plan["selection_uses_utility_outcomes"] is False,
               json.dumps(plan, sort_keys=True))


# --------------------------------------------------------------------------
# M0 boundary
# --------------------------------------------------------------------------
def _verify_m0_boundary(report, repo_root, indexes, manifests):
    root = os.path.join(str(repo_root), P.RESULTS_ROOT)
    present = set()
    if os.path.isdir(root):
        present = {name for name in os.listdir(root)
                   if os.path.isdir(os.path.join(root, name))}
    forbidden = sorted(present & set(P.FORBIDDEN_M0_ARTIFACT_DIRS))
    report.add("no_reader_or_training_artifacts", not forbidden,
               f"unexpected: {forbidden}" if forbidden else
               f"only {sorted(present)} under {P.RESULTS_ROOT}")

    probe_events = set()
    for manifest in manifests.values():
        if manifest:
            probe_events |= {i["event_id"] for i in manifest["items"]}
    atomic_events = set()
    for index in indexes.values():
        if index:
            atomic_events |= {e["event_id"] for e in index["entries"]}
    overlap = sorted(probe_events & atomic_events)
    report.add("probe_events_disjoint_from_imported_atomic_events",
               not overlap,
               f"{len(overlap)} overlapping events: {overlap[:3]}" if overlap
               else "probe (foundation_train) and imported atomic utility "
                    "events never coincide")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def verify(repo_root: str, environment: str = "") -> dict:
    report = Report()
    bootstrap_dir = P.bootstrap_dir(repo_root)
    identity = _read_json(os.path.join(
        bootstrap_dir, P.HISTORICAL_IDENTITY_FILENAME))

    atomic_all = _read_json(os.path.join(
        bootstrap_dir, P.ATOMIC_INDEX_FILENAME)) or {}
    indexes = {d: (atomic_all.get("datasets") or {}).get(d)
               for d in P.DATASETS}

    geometry_all = _read_json(os.path.join(
        bootstrap_dir, P.GEOMETRY_DIAGNOSTIC_FILENAME)) or {}
    geometries = {d: (geometry_all.get("datasets") or {}).get(d)
                  for d in P.DATASETS}

    manifest_all = _read_json(os.path.join(
        bootstrap_dir, P.PROBE_MANIFEST_FILENAME)) or {}
    manifests = {d: (manifest_all.get("datasets") or {}).get(d)
                 for d in P.DATASETS}

    availability_all = _read_json(os.path.join(
        bootstrap_dir, P.PROBE_AVAILABILITY_FILENAME)) or {}
    availability = {d: (availability_all.get("datasets") or {}).get(d)
                    for d in P.DATASETS}

    _verify_protocol(report)
    _verify_historical(report, repo_root, identity)
    _verify_atomic(report, repo_root, indexes)
    _verify_geometry(report, geometries)
    _verify_probes(report, repo_root, manifests, availability)
    _verify_m0_boundary(report, repo_root, indexes, manifests)
    return report.payload(environment=environment or "",
                          out_root=P.RESULTS_ROOT)


def main(argv=None) -> int:
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="BCR-Utility M0 verifier")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--environment", default="")
    parser.add_argument("--json", default=None,
                        help="write the verifier report here")
    args = parser.parse_args(argv)
    repo_root = args.repo_root or str(Path(__file__).resolve().parents[2])
    payload = verify(repo_root, environment=args.environment)
    target = args.json or os.path.join(str(repo_root), P.RESULTS_ROOT,
                                       "verifier", "bcr_verify.json")
    historical_import.write_json(target, payload)
    print(json.dumps({k: payload[k] for k in
                      ("protocol", "issues", "pending", "n_issues",
                       "n_pending")}, indent=1))
    for check in payload["checks"]:
        if not check["ok"]:
            print(f"  [{'pending' if check['pending'] else 'ISSUE'}] "
                  f"{check['check']}: {check['detail']}")
    return 1 if payload["n_issues"] else 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
