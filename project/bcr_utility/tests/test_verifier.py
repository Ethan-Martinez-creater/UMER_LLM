"""BCR M0 verifier: the pipeline passes, and every drift fails closed."""
from __future__ import annotations

import json

from bcr_utility import verifier
from bcr_utility.config import protocol as P

from .conftest import _write


def boot(synth_repo):
    synth_repo.seed_bootstrap()
    return str(synth_repo.root)


def failing(payload, name):
    """Names of the checks that failed with ``name`` among them."""
    failed = [c["check"] for c in payload["checks"] if not c["ok"]]
    assert name in failed, [
        (c["check"], c["detail"]) for c in payload["checks"]
        if c["check"] in (name, "atomic_index_exact_and_i1_only",
                          "probe_manifest_exact_36_and_12_per_cutoff",
                          "probe_events_disjoint_from_imported_atomic_events")
    ] + [f"failed={failed}"]
    return failed


def test_verifier_passes_on_the_frozen_pipeline(synth_repo):
    payload = verifier.verify(boot(synth_repo))
    assert payload["issues"] == [], payload["issues"]
    assert payload["protocol"] == "bcr_v1"
    assert payload["mode"] == "m0"
    names = {c["check"] for c in payload["checks"]}
    for expected in ("historical_manifests_recomputed_from_worktree",
                     "historical_labels_pinned_to_frozen_digest",
                     "bcr_protocol_frozen",
                     "atomic_index_exact_and_i1_only",
                     "geometry_reports_activity_ordinal_signed_interaction",
                     "probe_manifest_exact_36_and_12_per_cutoff",
                     "probe_events_disjoint_from_utility_splits",
                     "no_reader_or_training_artifacts",
                     "probe_events_disjoint_from_imported_atomic_events"):
        assert expected in names


def test_verifier_reports_missing_artifacts(synth_repo):
    payload = verifier.verify(str(synth_repo.root))
    failing(payload, "historical_identity_present")
    failing(payload, "atomic_index_present")
    failing(payload, "probe_manifest_present")


def test_verifier_detects_atomic_key_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.ATOMIC_INDEX_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["maweibo"]["entries"] = \
        payload["datasets"]["maweibo"]["entries"][:-1]
    _write(path, payload)
    failing(verifier.verify(root), "atomic_index_exact_and_i1_only")


def test_verifier_detects_source_label_repin(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.ATOMIC_INDEX_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["pheme"]["source_labels"]["sha256"] = "0" * 64
    _write(path, payload)
    failing(verifier.verify(root), "atomic_index_exact_and_i1_only")


def test_verifier_detects_probe_count_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.PROBE_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["maweibo"]["items"] = \
        payload["datasets"]["maweibo"]["items"][:-1]
    _write(path, payload)
    failing(verifier.verify(root), "probe_manifest_exact_36_and_12_per_cutoff")


def test_verifier_detects_a_probe_event_outside_foundation_train(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.PROBE_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["maweibo"]["items"][0]["event_id"] = "u0"
    _write(path, payload)
    failing(verifier.verify(root), "probe_manifest_exact_36_and_12_per_cutoff")


def test_verifier_detects_a_gold_label_in_a_probe_item(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.PROBE_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["pheme"]["items"][0]["label"] = 1
    _write(path, payload)
    failing(verifier.verify(root), "probe_manifest_exact_36_and_12_per_cutoff")


def test_verifier_detects_geometry_scope_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.GEOMETRY_DIAGNOSTIC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["maweibo"]["scope"] = "gate"
    payload["datasets"]["maweibo"]["decides_gate"] = True
    _write(path, payload)
    failing(verifier.verify(root),
            "geometry_reports_activity_ordinal_signed_interaction")


def test_verifier_detects_a_dropped_signed_direction_report(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.GEOMETRY_DIAGNOSTIC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["pheme"].pop("signed_direction_geometry")
    _write(path, payload)
    failing(verifier.verify(root),
            "geometry_reports_activity_ordinal_signed_interaction")


def test_verifier_detects_geometry_bootstrap_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.GEOMETRY_DIAGNOSTIC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["maweibo"]["bootstrap"]["iterations"] = 100
    _write(path, payload)
    failing(verifier.verify(root),
            "geometry_reports_activity_ordinal_signed_interaction")


def test_verifier_detects_a_manifest_edit(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.historical / "manifests" / "maweibo" / "source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["kind"] = "tampered"
    _write(path, payload)
    failing(verifier.verify(root),
            "historical_manifests_recomputed_from_worktree")


def test_verifier_detects_label_cache_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.historical / "utility_labels" / "maweibo" / "labels.jsonl"
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"dataset": "maweibo"}) + "\n")
    payload = verifier.verify(root)
    failed = [c["check"] for c in payload["checks"] if not c["ok"]]
    assert "historical_label_caches_recomputed" in failed or \
        "historical_labels_pinned_to_frozen_digest" in failed


def test_verifier_detects_a_forbidden_m0_namespace(synth_repo):
    root = boot(synth_repo)
    (synth_repo.root / P.RESULTS_ROOT / "probe_responses").mkdir(
        parents=True, exist_ok=True)
    failing(verifier.verify(root), "no_reader_or_training_artifacts")


def test_verifier_detects_probe_atomic_event_overlap(synth_repo):
    from bcr_utility.data import atomic_manifest as am

    root = boot(synth_repo)
    manifest_path = synth_repo.bootstrap / P.PROBE_MANIFEST_FILENAME
    probe_event = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "datasets"]["maweibo"]["items"][0]["event_id"]
    path = synth_repo.bootstrap / P.ATOMIC_INDEX_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["datasets"]["maweibo"]["entries"]:
        entry["event_id"] = probe_event
        entry["key"] = entry["key"].replace("|u0|", f"|{probe_event}|").replace(
            "|u1|", f"|{probe_event}|")
    payload["datasets"]["maweibo"]["keys_sha256"] = am.keys_digest(
        payload["datasets"]["maweibo"]["entries"])
    _write(path, payload)
    failing(verifier.verify(root),
            "probe_events_disjoint_from_imported_atomic_events")


def test_verifier_detects_identity_protocol_drift(synth_repo):
    root = boot(synth_repo)
    path = synth_repo.bootstrap / P.HISTORICAL_IDENTITY_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_commit"] = "0" * 40
    _write(path, payload)
    failing(verifier.verify(root), "historical_identity_protocol")


def test_verifier_main_writes_the_report(synth_repo, capsys):
    root = boot(synth_repo)
    assert verifier.main(["--repo-root", root]) == 0
    capsys.readouterr()
    target = synth_repo.root / P.RESULTS_ROOT / "verifier" / "bcr_verify.json"
    assert target.exists()
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["n_issues"] == 0


def test_verifier_main_fails_nonzero_on_issues(synth_repo):
    root = boot(synth_repo)
    (synth_repo.root / P.RESULTS_ROOT / "models").mkdir(parents=True,
                                                        exist_ok=True)
    assert verifier.main(["--repo-root", root]) == 1
