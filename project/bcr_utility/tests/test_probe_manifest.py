"""Frozen probe manifest: exact counts, balance, determinism (§6.1, §11.7)."""
from __future__ import annotations

import copy

import pytest

from bcr_utility.config import protocol as P
from bcr_utility.probes import probe_manifest as pm

from .conftest import WRITERS  # noqa: F401  (keeps the fixture import path used)


def build(synth_repo, dataset="maweibo", seed=P.PROBE_SEED):
    entry = synth_repo.availability(dataset)["datasets"][dataset]
    return pm.build_probe_manifest(
        synth_repo.split(dataset), dataset, entry["availability"],
        entry["gold_labels_for_balance_only"], seed=seed)


def test_manifest_has_exactly_36_items_12_per_cutoff(synth_repo):
    manifest = build(synth_repo)
    assert manifest["n_items"] == P.PROBE_EVENTS_PER_DATASET
    assert manifest["n_events"] == P.PROBE_EVENTS_PER_DATASET
    assert manifest["per_cutoff_counts"] == {"15": 12, "60": 12, "360": 12}
    for cutoff in P.CUTOFFS_MIN:
        assert sum(1 for i in manifest["items"]
                   if i["cutoff"] == cutoff) == P.PROBE_EVENTS_PER_CUTOFF


def test_manifest_items_are_event_disjoint(synth_repo):
    manifest = build(synth_repo)
    events = [i["event_id"] for i in manifest["items"]]
    assert len(events) == len(set(events))
    for cutoff in P.CUTOFFS_MIN:
        group = {i["event_id"] for i in manifest["items"]
                 if i["cutoff"] == cutoff}
        assert len(group) == P.PROBE_EVENTS_PER_CUTOFF


def test_manifest_is_label_balanced_and_stores_no_labels(synth_repo):
    manifest = build(synth_repo)
    assert manifest["label_balance"] == {
        "15": {"0": 6, "1": 6}, "60": {"0": 6, "1": 6},
        "360": {"0": 6, "1": 6}}
    assert manifest["total_label_balance"] == {"0": 18, "1": 18}
    for item in manifest["items"]:
        assert "label" not in item and "gold" not in item
        assert set(item) == {"event_id", "cutoff", "n_visible_units",
                             "src_selected", "n_visible_nodes"}


def test_manifest_is_deterministic(synth_repo):
    first = build(synth_repo)
    second = build(synth_repo)
    assert first == second
    assert [i["event_id"] for i in first["items"]] == \
        [i["event_id"] for i in second["items"]]


def test_only_foundation_events_are_used(synth_repo):
    manifest = build(synth_repo)
    foundation = set(synth_repo.split("maweibo")["foundation_train"])
    assert {i["event_id"] for i in manifest["items"]} <= foundation


def test_overlap_with_utility_splits_is_zero(synth_repo):
    manifest = build(synth_repo)
    audit = pm.audit_overlap(manifest, synth_repo.split("maweibo"))
    assert audit == {"utility_train": 0, "utility_dev": 0, "utility_eval": 0}


def test_missing_availability_entry_is_refused(synth_repo):
    entry = synth_repo.availability("maweibo")["datasets"]["maweibo"]
    availability = dict(entry["availability"])
    availability.pop(sorted(availability)[0])
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(synth_repo.split("maweibo"), "maweibo",
                                availability,
                                entry["gold_labels_for_balance_only"])


def test_missing_gold_labels_are_refused(synth_repo):
    entry = synth_repo.availability("maweibo")["datasets"]["maweibo"]
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(synth_repo.split("maweibo"), "maweibo",
                                entry["availability"], {})


def test_too_few_eligible_events_is_refused(synth_repo):
    entry = synth_repo.availability("maweibo")["datasets"]["maweibo"]
    availability = {}
    eligible = sorted(entry["availability"])[:10]
    for eid, per_cutoff in entry["availability"].items():
        availability[eid] = {
            str(c): (per_cutoff[str(c)] if eid in eligible
                     else {"n_units": 0, "src_selected": 0,
                           "n_visible_nodes": 0})
            for c in P.CUTOFFS_MIN}
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(synth_repo.split("maweibo"), "maweibo",
                                availability,
                                entry["gold_labels_for_balance_only"])


def test_events_without_src_units_are_ineligible(synth_repo):
    entry = synth_repo.availability("maweibo")["datasets"]["maweibo"]
    availability = {
        eid: {str(c): {"n_units": v[str(c)]["n_units"], "src_selected": 0,
                       "n_visible_nodes": v[str(c)]["n_visible_nodes"]}
              for c in P.CUTOFFS_MIN}
        for eid, v in entry["availability"].items()}
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(synth_repo.split("maweibo"), "maweibo",
                                availability,
                                entry["gold_labels_for_balance_only"])


def test_empty_foundation_split_is_refused(synth_repo):
    split = dict(synth_repo.split("maweibo"), foundation_train=[])
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(split, "maweibo", {}, {})


def test_candidate_pool_is_recorded(synth_repo):
    manifest = build(synth_repo)
    assert manifest["candidate_pool"] == {"15": 80, "60": 80, "360": 80}


def test_validate_accepts_and_rejects(synth_repo):
    manifest = build(synth_repo)
    pm.validate_probe_manifest(manifest, "maweibo")

    dropped = copy.deepcopy(manifest)
    dropped["items"] = dropped["items"][:-1]
    with pytest.raises(pm.ProbeManifestRefused):
        pm.validate_probe_manifest(dropped, "maweibo")

    labelled = copy.deepcopy(manifest)
    labelled["items"][0]["label"] = 1
    with pytest.raises(pm.ProbeManifestRefused):
        pm.validate_probe_manifest(labelled, "maweibo")

    starved = copy.deepcopy(manifest)
    starved["items"][0]["src_selected"] = 0
    with pytest.raises(pm.ProbeManifestRefused):
        pm.validate_probe_manifest(starved, "maweibo")

    duplicated = copy.deepcopy(manifest)
    duplicated["items"][1]["event_id"] = duplicated["items"][0]["event_id"]
    with pytest.raises(pm.ProbeManifestRefused):
        pm.validate_probe_manifest(duplicated, "maweibo")


def test_availability_digest_is_order_independent(synth_repo):
    import json
    entry = synth_repo.availability("maweibo")["datasets"]["maweibo"]
    availability = entry["availability"]
    reordered = json.loads(json.dumps(availability, sort_keys=True))
    reordered = {k: reordered[k] for k in sorted(reordered, reverse=True)}
    assert pm.availability_digest(availability) == \
        pm.availability_digest(reordered)
    tampered = dict(availability)
    first = sorted(tampered)[0]
    per_cutoff = dict(tampered[first])
    per_cutoff["15"] = dict(per_cutoff["15"], n_units=99)
    tampered[first] = per_cutoff
    assert pm.availability_digest(availability) != \
        pm.availability_digest(tampered)


def test_foundation_labels_helper():
    split = {"foundation_train": ["a", "b"]}
    assert pm.foundation_labels(split, "maweibo", {"a": 0, "b": 1}) == \
        {"a": 0, "b": 1}
    with pytest.raises(pm.ProbeManifestRefused):
        pm.foundation_labels(split, "maweibo", {"a": 0})


def test_unknown_dataset_is_refused(synth_repo):
    with pytest.raises(pm.ProbeManifestRefused):
        pm.build_probe_manifest(synth_repo.split("maweibo"), "weibo22", {}, {})
