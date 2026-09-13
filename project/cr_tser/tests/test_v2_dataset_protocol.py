"""Amendment V2 dataset-protocol tests (synthetic, no real data).

Covers the Ma-Weibo primary migration: the bridge reuses the audited TC-DSCR
adapter (timestamps only from the raw ``t`` field, ``original_text`` with a
``text`` fallback), the V2 eligibility statuses, the composite source
fingerprint, viability filtering before the split, and the Ma-Weibo B2/S6
firewall. Nothing here reads a real dataset, a reader checkpoint or the GPU.
"""
from __future__ import annotations

import json
import types

import pytest

from ..config.pilot_config import (PRIMARY_DATASET, REJECTED_PRIMARY_CANDIDATE,
                                   SECONDARY_DATASET, V1_RESULTS_ROOT,
                                   V2_MIN_VIABLE_EVENTS, V2_RESULTS_ROOT)


def _write_raw(root, eid: str, posts) -> str:
    path = root / "raw"
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{eid}.json"
    target.write_text(json.dumps(posts), encoding="utf-8")
    return str(path)


def _write_labels(root, rows) -> str:
    target = root / "Weibo.txt"
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(target)


def _chain(eid: str, base_t: int = 1000, replies: int = 2, label: int = 1):
    posts = [{"mid": f"{eid}_s", "parent": None, "t": base_t,
              "original_text": "source body"}]
    parent = f"{eid}_s"
    for j in range(replies):
        nid = f"{eid}_{j}"
        posts.append({"mid": nid, "parent": parent,
                      "t": base_t + 60 * (j + 1),
                      "original_text": f"reply {j}"})
        parent = nid
    return posts


def _paths(raw_dir, label_file):
    return types.SimpleNamespace(maweibo_raw=raw_dir,
                                 maweibo_labels=label_file, pheme_raw="",
                                 weibo22_normalized="", weibo22_raw="")


# --------------------------------------------------------------------------
# dataset roles
# --------------------------------------------------------------------------
def test_v2_dataset_roles():
    assert PRIMARY_DATASET == "maweibo"
    assert SECONDARY_DATASET == "pheme"
    assert REJECTED_PRIMARY_CANDIDATE == "weibo22"
    assert V2_RESULTS_ROOT == "results/cr_tser_v2"
    assert V1_RESULTS_ROOT == "results/cr_tser"
    assert V2_RESULTS_ROOT != V1_RESULTS_ROOT


def test_v1_default_out_root_moved_to_v2():
    from ..config.pilot_config import paths_from_env
    assert paths_from_env({}).out_root == V2_RESULTS_ROOT


# --------------------------------------------------------------------------
# audited-adapter reuse and the V2 statuses
# --------------------------------------------------------------------------
def test_bridge_reuses_audited_adapter_and_raw_timestamp(tmp_path):
    from ..data import maweibo_bridge as bridge
    raw = _write_raw(tmp_path, "1000", _chain("1000"))
    labels = _write_labels(tmp_path, ["eid:1000 label:1"])
    events = bridge.load_events(raw, labels)
    assert len(events) == 1
    event = events[0]
    posts = json.loads((tmp_path / "raw" / "1000.json").read_text("utf-8"))
    by_mid = {p["mid"]: p for p in posts}
    # timestamp is exactly the raw t field, never an index/order
    for node in event["nodes"]:
        assert node["timestamp"] == by_mid[node["node_id"]]["t"]
    assert event["source_timestamp"] == by_mid["1000_s"]["t"]


def test_bridge_prefers_original_text_and_falls_back(tmp_path):
    from ..data import maweibo_bridge as bridge
    posts = [{"mid": "6_s", "parent": None, "t": 1000,
              "text": "fallback source"},
             {"mid": "6_a", "parent": "6_s", "t": 1060,
              "original_text": "original reply"}]
    raw = _write_raw(tmp_path, "6", posts)
    labels = _write_labels(tmp_path, ["eid:6 label:1"])
    event = bridge.load_events(raw, labels)[0]
    texts = {n["node_id"]: n["text"] for n in event["nodes"]}
    assert texts["6_s"] == "fallback source"       # field absent -> fallback
    assert texts["6_a"] == "original reply"        # field present -> preferred


def test_external_parent_is_audited_and_never_a_unit(tmp_path):
    from ..data import maweibo_bridge as bridge
    from ..data.snapshot_bridge import build_causal_snapshot
    from ..intervention.evidence_units import build_evidence_units
    posts = [{"mid": "9_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "9_a", "parent": "ghost", "t": 1060,
              "original_text": "orphan reply"}]
    raw = _write_raw(tmp_path, "9", posts)
    labels = _write_labels(tmp_path, ["eid:9 label:1"])
    event = bridge.load_events(raw, labels)[0]
    statuses = {n["node_id"]: n["status"] for n in event["nodes"]}
    assert statuses["9_a"] == "EXTERNAL_PARENT"
    # never attached to the source, never a Reply–Parent unit
    assert bridge.valid_reply_parent_units(event) == []
    assert not bridge.event_viable(event)
    snapshot = build_causal_snapshot(event, 360)
    snapshot["eligibility"] = "v2_strict"     # Ma-Weibo contract
    unit_ids = {u["node_id"] for u in build_evidence_units(snapshot)}
    assert "9_a" not in unit_ids
    assert all(pid is None for nid, pid in
               zip(snapshot["node_ids"], snapshot["parent_ids"])
               if nid == "9_a")
    # PHEME keeps the V1 behaviour on the same snapshot
    pheme_snapshot = build_causal_snapshot(event, 360)
    pheme_units = {u["node_id"] for u in build_evidence_units(pheme_snapshot)}
    assert "9_a" in pheme_units


def test_empty_text_is_audited_and_not_a_unit(tmp_path):
    from ..data import maweibo_bridge as bridge
    from ..data.snapshot_bridge import build_causal_snapshot
    from ..intervention.evidence_units import build_evidence_units
    posts = [{"mid": "7_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "7_a", "parent": "7_s", "t": 1060,
              "original_text": "reply a"},
             {"mid": "7_b", "parent": "7_a", "t": 1120, "original_text": ""}]
    raw = _write_raw(tmp_path, "7", posts)
    labels = _write_labels(tmp_path, ["eid:7 label:1"])
    event = bridge.load_events(raw, labels)[0]
    statuses = {n["node_id"]: n["status"] for n in event["nodes"]}
    assert statuses["7_b"] == "EMPTY_TEXT"
    snapshot = build_causal_snapshot(event, 360)
    snapshot["eligibility"] = "v2_strict"     # Ma-Weibo contract
    unit_ids = {u["node_id"] for u in build_evidence_units(snapshot)}
    assert "7_b" not in unit_ids and "7_a" in unit_ids
    # PHEME keeps the V1 behaviour on the same snapshot
    pheme_snapshot = build_causal_snapshot(event, 360)
    pheme_units = {u["node_id"] for u in build_evidence_units(pheme_snapshot)}
    assert "7_b" in pheme_units


def test_temporal_invalid_node_not_in_snapshot(tmp_path):
    from ..data import maweibo_bridge as bridge
    from ..data.snapshot_bridge import build_causal_snapshot
    posts = [{"mid": "8_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "8_a", "parent": "8_s", "t": 900,
              "original_text": "earlier than source"}]
    raw = _write_raw(tmp_path, "8", posts)
    labels = _write_labels(tmp_path, ["eid:8 label:1"])
    event = bridge.load_events(raw, labels)[0]
    statuses = {n["node_id"]: n["status"] for n in event["nodes"]}
    assert statuses["8_a"] == "TEMPORAL_INVALID_NODE"
    snapshot = build_causal_snapshot(event, 360)
    assert "8_a" not in snapshot["node_ids"]


def test_duplicate_node_id_invalidates_event(tmp_path):
    from ..data import maweibo_bridge as bridge
    posts = [{"mid": "5_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "5_a", "parent": "5_s", "t": 1060,
              "original_text": "r"},
             {"mid": "5_a", "parent": "5_s", "t": 1120,
              "original_text": "dup"}]
    raw = _write_raw(tmp_path, "5", posts)
    labels = _write_labels(tmp_path, ["eid:5 label:1"])
    assert bridge.load_events(raw, labels) == []      # fail closed
    audit = bridge.audit_maweibo(raw, labels)
    assert audit["invalid_event_count"] == 1
    assert audit["duplicate_ids"] >= 1
    assert audit["total_viable_events"] == 0


def test_multi_root_event_is_invalid(tmp_path):
    from ..data import maweibo_bridge as bridge
    posts = [{"mid": "4_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "4_b", "parent": None, "t": 1060,
              "original_text": "second root"}]
    raw = _write_raw(tmp_path, "4", posts)
    labels = _write_labels(tmp_path, ["eid:4 label:1"])
    assert bridge.load_events(raw, labels) == []
    audit = bridge.audit_maweibo(raw, labels)
    assert audit["multi_root_event_count"] == 1


# --------------------------------------------------------------------------
# viability (amendment §8)
# --------------------------------------------------------------------------
def test_viability_requires_a_valid_unit_inside_a_cutoff(tmp_path):
    from ..data import maweibo_bridge as bridge
    # reply at +370s: outside 15m (900s)? no -> inside; use a far reply
    posts = [{"mid": "1_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "1_a", "parent": "1_s", "t": 1000 + 3600 * 7,
              "original_text": "reply after 7h"}]
    raw = _write_raw(tmp_path, "1", posts)
    labels = _write_labels(tmp_path, ["eid:1 label:1"])
    event = bridge.load_events(raw, labels)[0]
    flags = bridge.viable_cutoffs(event)
    assert flags == {"15": False, "60": False, "360": False}
    assert not bridge.event_viable(event)


def test_cycle_makes_event_unviable(tmp_path):
    from ..data import maweibo_bridge as bridge
    from ..data.snapshot_bridge import count_parent_cycles
    posts = [{"mid": "3_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "3_a", "parent": "3_b", "t": 1060,
              "original_text": "a"},
             {"mid": "3_b", "parent": "3_a", "t": 1120,
              "original_text": "b"}]
    raw = _write_raw(tmp_path, "3", posts)
    labels = _write_labels(tmp_path, ["eid:3 label:1"])
    event = bridge.load_events(raw, labels)[0]
    assert count_parent_cycles(event) >= 1
    assert not bridge.event_viable(event)


def test_viability_filter_precedes_split(tmp_path):
    from ..data import maweibo_bridge as bridge
    from ..data.pilot_split import viable_event_ids
    raw = _write_raw(tmp_path, "1000", _chain("1000", replies=2))
    posts = [{"mid": "2000_s", "parent": None, "t": 1000,
              "original_text": "src"},
             {"mid": "2000_a", "parent": "2000_s", "t": 1000 + 3600 * 9,
              "original_text": "too late"}]
    raw2 = _write_raw(tmp_path, "2000", posts)
    labels = _write_labels(tmp_path, ["eid:1000 label:1", "eid:2000 label:0"])
    events = bridge.load_events(raw2, labels)
    viable = viable_event_ids(events, eligibility="v2_strict")
    assert viable == ["1000"]


# --------------------------------------------------------------------------
# composite source fingerprint (amendment §25)
# --------------------------------------------------------------------------
def test_composite_fingerprint_binds_raw_and_labels(tmp_path):
    from ..data.source_manifest import (SourceIdentityError,
                                        assert_same_source,
                                        source_fingerprint)
    raw = _write_raw(tmp_path, "1000", _chain("1000"))
    labels = _write_labels(tmp_path, ["eid:1000 label:1"])
    paths = _paths(raw, labels)
    frozen = source_fingerprint("maweibo", paths)
    assert frozen["kind"] == "maweibo_composite"
    assert frozen["raw_json"]["n_files"] == 1
    assert frozen["label_file"]["n_files"] == 1
    assert frozen["combined_source_sha256"]
    assert_same_source(frozen, dict(frozen), "test")

    with open(labels, "a", encoding="utf-8") as fh:
        fh.write("eid:2000 label:0\n")
    with pytest.raises(SourceIdentityError):
        assert_same_source(frozen, source_fingerprint("maweibo", paths),
                           "test")

    import os
    with open(labels, "w", encoding="utf-8") as fh:
        fh.write("eid:1000 label:1\n")
    os.remove(os.path.join(raw, "1000.json"))
    with pytest.raises(SourceIdentityError):
        assert_same_source(frozen, source_fingerprint("maweibo", paths),
                           "test")


# --------------------------------------------------------------------------
# P0 readiness and the firewall
# --------------------------------------------------------------------------
def test_viable_below_170_blocks_p0():
    import cr_tser_p0_audit as p0
    from ..config.pilot_config import READER_KEYS
    readers = {k: {"model_path_exists": True, "loaded": True}
               for k in READER_KEYS}
    sanity = {k: {"identical_predictions": True, "boundaries_ok": True}
              for k in READER_KEYS}
    base = {"raw_event_count": 400, "duplicate_ids": 0, "cycle_count": 0,
            "multi_root_event_count": 0, "source_text_coverage": 1.0,
            "source_timestamp_coverage": 1.0, "timestamp_coverage": 1.0,
            "verdict": "MAWEIBO_READY"}
    short = p0.evaluate_readiness_v2(dict(base, total_viable_events=169),
                                     {"status": "OK"}, readers, sanity, True,
                                     True)
    assert short["P0"] == "P0_FAIL" and short["maweibo_viable_ok"] is False
    enough = p0.evaluate_readiness_v2(dict(base, total_viable_events=170),
                                      {"status": "OK"}, readers, sanity, True,
                                      True)
    assert enough["P0"] == "P0_PASS"


def test_pheme_smoke_failure_blocks_p0():
    import cr_tser_p0_audit as p0
    from ..config.pilot_config import READER_KEYS
    readers = {k: {"model_path_exists": True, "loaded": True}
               for k in READER_KEYS}
    sanity = {k: {"identical_predictions": True, "boundaries_ok": True}
              for k in READER_KEYS}
    audit = {"raw_event_count": 400, "duplicate_ids": 0, "cycle_count": 0,
             "multi_root_event_count": 0, "source_text_coverage": 1.0,
             "source_timestamp_coverage": 1.0, "timestamp_coverage": 1.0,
             "total_viable_events": 400}
    r = p0.evaluate_readiness_v2(audit, {"status": "PHEME_RAW_MISSING"},
                                 readers, sanity, True, True)
    assert r["P0"] == "P0_FAIL" and r["pheme_smoke_ok"] is False


def test_reader_missing_blocks_p0():
    import cr_tser_p0_audit as p0
    from ..config.pilot_config import READER_KEYS
    readers = {k: {"model_path_exists": False, "loaded": False}
               for k in READER_KEYS}
    sanity = {k: {"identical_predictions": True, "boundaries_ok": True}
              for k in READER_KEYS}
    audit = {"raw_event_count": 400, "duplicate_ids": 0, "cycle_count": 0,
             "multi_root_event_count": 0, "source_text_coverage": 1.0,
             "source_timestamp_coverage": 1.0, "timestamp_coverage": 1.0,
             "total_viable_events": 400}
    r = p0.evaluate_readiness_v2(audit, {"status": "OK"}, readers, sanity,
                                 True, True)
    assert r["P0"] == "P0_FAIL" and r["readers_ready"] is False


def test_maweibo_has_no_b2_s6():
    from ..models.legacy_utility import legacy_arm_enabled
    assert legacy_arm_enabled("maweibo") is False
    assert legacy_arm_enabled("pheme") is True


def test_common_loads_maweibo_through_the_bridge(tmp_path):
    import cr_tser_common as common
    raw = _write_raw(tmp_path, "1000", _chain("1000", replies=1))
    labels = _write_labels(tmp_path, ["eid:1000 label:1"])
    events = common.load_dataset_events("maweibo", _paths(raw, labels))
    assert [e["event_id"] for e in events] == ["1000"]
    registry = common.dataset_registry("maweibo", _paths(raw, labels))
    assert registry == {"1000": 1}


def test_audit_reports_every_required_field(tmp_path):
    from ..data import maweibo_bridge as bridge
    raw = _write_raw(tmp_path, "1000", _chain("1000", replies=2))
    labels = _write_labels(tmp_path, ["eid:1000 label:1"])
    audit = bridge.audit_maweibo(raw, labels)
    for field in ("raw_event_count", "label_distribution",
                  "source_text_coverage", "reply_text_coverage",
                  "timestamp_coverage", "parent_resolution_coverage",
                  "duplicate_ids", "cycle_count", "multi_root_event_count",
                  "missing_parent_count", "missing_parent_rate",
                  "external_parent_count", "external_parent_rate",
                  "temporal_invalid_node_count",
                  "events_with_ge1_valid_reply_parent_unit",
                  "events_viable_15m", "events_viable_1h",
                  "events_viable_6h", "total_viable_events"):
        assert field in audit, field
    assert audit["total_viable_events"] == 1
    assert audit["cycle_count"] == 0
    assert audit["duplicate_ids"] == 0
    assert audit["label_distribution"] == {"1": 1}
