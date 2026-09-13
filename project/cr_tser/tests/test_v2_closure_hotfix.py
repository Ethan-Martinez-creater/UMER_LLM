"""V2-M0 closure-hotfix regression tests (synthetic, no real data).

Pins the five fixes: dataset-aware evidence-unit eligibility (PHEME keeps V1),
the strict Ma-Weibo Reply–Parent contract (a ``VALID`` child may not smuggle an
invalid parent into a unit), PHEME smoke fail-closed behaviour, the V2
aggregator/report dataset semantics, and the V1 historical namespace.
"""
from __future__ import annotations

import json
import types

import pytest


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_raw(root, eid: str, posts) -> str:
    path = root / "raw"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{eid}.json").write_text(json.dumps(posts), encoding="utf-8")
    return str(path)


def _write_labels(root, rows) -> str:
    target = root / "Weibo.txt"
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(target)


def _pheme_paths(tmp_path):
    raw = tmp_path / "pheme_raw"
    raw.mkdir(exist_ok=True)
    return types.SimpleNamespace(pheme_raw=str(raw))


def _pheme_event(source_text="source body", reply_text="a reply",
                 parent="s", reply_status="VALID", reply_ts=1060):
    return {
        "event_id": "e1", "label": 1, "source_id": "s",
        "source_timestamp": 1000,
        "nodes": [
            {"node_id": "s", "parent_id": None, "timestamp": 1000,
             "text": source_text, "original_order": 0, "status": "VALID"},
            {"node_id": "a", "parent_id": parent, "timestamp": reply_ts,
             "text": reply_text, "original_order": 1,
             "status": reply_status},
        ],
    }


def _snapshot(eligibility, statuses, parent_ids, texts, timestamps=None):
    n = len(statuses)
    return {"node_ids": ["s"] + [f"n{i}" for i in range(1, n)],
            "texts": texts, "parent_ids": parent_ids,
            "timestamps": timestamps or list(range(0, 10 * n, 10)),
            "elapsed_seconds": [0] * n, "source_id": "s",
            "depths": list(range(n)), "statuses": statuses,
            "eligibility": eligibility}


# --------------------------------------------------------------------------
# 1. eligibility is dataset-aware; PHEME keeps V1 behaviour
# --------------------------------------------------------------------------
def test_same_non_valid_reply_excluded_only_for_maweibo():
    from ..intervention.evidence_units import build_evidence_units
    from ..config.pilot_config import (PRIMARY_DATASET,
                                       V2_STRICT_ELIGIBILITY_DATASETS)
    # reply n1 is VALID; its parent (the source) is EMPTY_TEXT
    statuses = ["EMPTY_TEXT", "VALID"]
    parents = [None, "s"]
    texts = ["", "reply"]
    strict = _snapshot("v2_strict", statuses, parents, texts)
    v1 = _snapshot(None, statuses, parents, texts)
    assert {u["node_id"] for u in build_evidence_units(strict)} == set()
    assert {u["node_id"] for u in build_evidence_units(v1)} == {"n1"}
    assert V2_STRICT_ELIGIBILITY_DATASETS == (PRIMARY_DATASET,)
    assert PRIMARY_DATASET == "maweibo"


def test_eligibility_for_is_explicit():
    import cr_tser_common as common
    assert common.eligibility_for("maweibo") == "v2_strict"
    assert common.eligibility_for("pheme") is None
    assert common.eligibility_for("weibo22") is None


# --------------------------------------------------------------------------
# 2. strict Ma-Weibo Reply–Parent eligibility
# --------------------------------------------------------------------------
def test_valid_child_with_empty_text_parent_has_no_unit():
    from ..data.snapshot_bridge import build_causal_snapshot, \
        valid_reply_parent_units
    from ..intervention.evidence_units import build_evidence_units
    event = {"event_id": "e", "label": 1, "source_id": "s",
             "source_timestamp": 1000,
             "nodes": [
                 {"node_id": "s", "parent_id": None, "timestamp": 1000,
                  "text": "source", "original_order": 0, "status": "VALID"},
                 # valid child, but this parent has empty text
                 {"node_id": "p", "parent_id": "s", "timestamp": 1030,
                  "text": "   ", "original_order": 1, "status": "EMPTY_TEXT"},
                 {"node_id": "c", "parent_id": "p", "timestamp": 1060,
                  "text": "valid reply", "original_order": 2,
                  "status": "VALID"}]}
    assert valid_reply_parent_units(event) == []
    snapshot = build_causal_snapshot(event, 360)
    snapshot["eligibility"] = "v2_strict"
    assert {u["node_id"] for u in build_evidence_units(snapshot)} == set()


def test_valid_child_with_invalid_status_parent_has_no_unit():
    from ..data.snapshot_bridge import valid_reply_parent_units
    event = {"event_id": "e", "label": 1, "source_id": "s",
             "source_timestamp": 1000,
             "nodes": [
                 {"node_id": "s", "parent_id": None, "timestamp": 1000,
                  "text": "source", "original_order": 0, "status": "VALID"},
                 {"node_id": "p", "parent_id": "s", "timestamp": 1030,
                  "text": "external parent text", "original_order": 1,
                  "status": "EXTERNAL_PARENT"},
                 {"node_id": "c", "parent_id": "p", "timestamp": 1060,
                  "text": "valid reply", "original_order": 2,
                  "status": "VALID"}]}
    assert valid_reply_parent_units(event) == []


def test_valid_child_with_valid_textual_parent_has_unit():
    from ..data.snapshot_bridge import (build_causal_snapshot,
                                        valid_reply_parent_units)
    from ..intervention.evidence_units import build_evidence_units
    event = {"event_id": "e", "label": 1, "source_id": "s",
             "source_timestamp": 1000,
             "nodes": [
                 {"node_id": "s", "parent_id": None, "timestamp": 1000,
                  "text": "source", "original_order": 0, "status": "VALID"},
                 {"node_id": "p", "parent_id": "s", "timestamp": 1030,
                  "text": "parent reply", "original_order": 1,
                  "status": "VALID"},
                 {"node_id": "c", "parent_id": "p", "timestamp": 1060,
                  "text": "valid reply", "original_order": 2,
                  "status": "VALID"}]}
    assert len(valid_reply_parent_units(event)) == 2   # p (parent s) and c
    snapshot = build_causal_snapshot(event, 360)
    snapshot["eligibility"] = "v2_strict"
    assert {u["node_id"] for u in build_evidence_units(snapshot)} == {"p", "c"}
    assert all(u["parent_text"] for u in build_evidence_units(snapshot))


def test_invalid_parent_does_not_add_to_viable_pool(tmp_path):
    import types as _t
    from ..data import maweibo_bridge as bridge
    from ..data.pilot_split import viable_event_ids
    # event 1: the only candidate reply's parent has empty text -> NO unit
    posts_bad = [{"mid": "1_s", "parent": None, "t": 1000,
                  "original_text": "source"},
                 {"mid": "1_p", "parent": "1_s", "t": 1030,
                  "original_text": ""},
                 {"mid": "1_c", "parent": "1_p", "t": 1060,
                  "original_text": "valid reply"}]
    _write_raw(tmp_path, "1", posts_bad)
    # event 2: a clean chain -> a unit
    posts_good = [{"mid": "2_s", "parent": None, "t": 1000,
                   "original_text": "source"},
                  {"mid": "2_c", "parent": "2_s", "t": 1060,
                   "original_text": "reply"}]
    _write_raw(tmp_path, "2", posts_good)
    labels = _write_labels(tmp_path, ["eid:1 label:1", "eid:2 label:0"])
    events = bridge.load_events(str(tmp_path / "raw"), labels)
    assert [e["event_id"] for e in events] == ["1", "2"]
    assert viable_event_ids(events, eligibility="v2_strict") == ["2"]
    # the V1 (PHEME) rule is unaffected: both events have a visible reply
    assert viable_event_ids(events) == ["1", "2"]


# --------------------------------------------------------------------------
# 3. PHEME smoke fail-closed
# --------------------------------------------------------------------------
def _patch_pheme(monkeypatch, event, raw="ok"):
    from tcdscr.data import pheme_adapter
    monkeypatch.setattr(pheme_adapter, "event_ids",
                        lambda r: [("e1", "topic", 1, "f")])
    monkeypatch.setattr(pheme_adapter, "load_event",
                        lambda topic, label, folder: event)


def test_pheme_smoke_no_source_text_fails(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    _patch_pheme(monkeypatch, _pheme_event(source_text="   "))
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "PHEME_SMOKE_FAIL"
    assert "no_source_text" in out["failures"]


def test_pheme_smoke_no_reply_text_fails(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    _patch_pheme(monkeypatch, _pheme_event(reply_text=""))
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "PHEME_SMOKE_FAIL"
    assert "no_reply_text" in out["failures"]


def test_pheme_smoke_no_parent_relation_fails(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    _patch_pheme(monkeypatch, _pheme_event(parent="ghost"))
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "PHEME_SMOKE_FAIL"
    assert "no_parent_relation" in out["failures"]


def test_pheme_smoke_missing_cutoff_build_fails(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    import cr_tser.data.snapshot_bridge as sb
    real = sb.build_causal_snapshot

    def flaky(event, cutoff):
        if int(cutoff) == 60:
            raise RuntimeError("boom")
        return real(event, cutoff)

    monkeypatch.setattr(sb, "build_causal_snapshot", flaky)
    _patch_pheme(monkeypatch, _pheme_event())
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "PHEME_SMOKE_FAIL"
    assert any(f.startswith("cutoff_build_failed:60") for f in out["failures"])


def test_pheme_smoke_future_leakage_fails(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    import cr_tser.data.snapshot_bridge as sb
    real = sb.build_causal_snapshot

    def leaky(event, cutoff):
        snap = real(event, cutoff)
        snap["timestamps"] = [ts + 10 ** 6 for ts in snap["timestamps"]]
        return snap

    monkeypatch.setattr(sb, "build_causal_snapshot", leaky)
    _patch_pheme(monkeypatch, _pheme_event())
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "PHEME_SMOKE_FAIL"
    assert "future_leakage" in out["failures"]


def test_pheme_smoke_full_contract_is_ok(monkeypatch, tmp_path):
    import cr_tser_p0_audit as p0
    _patch_pheme(monkeypatch, _pheme_event())
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "OK"
    assert out["failures"] == []
    assert len(out["snapshots_built"]) == 3
    assert out["future_leakage"] == []


def test_pheme_smoke_does_not_require_every_reply_to_have_a_parent(
        monkeypatch, tmp_path):
    """A few unresolved parents are normal; one resolved relation suffices."""
    import cr_tser_p0_audit as p0
    from tcdscr.data import pheme_adapter
    event = _pheme_event()
    event["nodes"].append(
        {"node_id": "b", "parent_id": "ghost", "timestamp": 1090,
         "text": "orphan reply", "original_order": 2, "status": "VALID"})
    _patch_pheme(monkeypatch, event)
    out = p0.pheme_smoke(_pheme_paths(tmp_path))
    assert out["status"] == "OK"
    assert out["parent_relation_available"] is True


# --------------------------------------------------------------------------
# 4. V2 aggregator / report semantics
# --------------------------------------------------------------------------
def test_v2_report_is_maweibo_primary(tmp_path):
    import cr_tser_run_pilot as pilot
    root = str(tmp_path)
    import os
    os.makedirs(os.path.join(root, "p0"), exist_ok=True)
    with open(os.path.join(root, "p0", "p0_readiness.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"protocol": "v2", "P0": "P0_FAIL",
                   "maweibo_source_integrity": True,
                   "maweibo_viable_events": 12, "maweibo_viable_required": 170,
                   "pheme_smoke": "PHEME_RAW_MISSING", "readers_ready": False,
                   "ab_sanity_ok": None,
                   # a V1 field that must never be read by the V2 aggregator
                   "weibo22_reason": "SHOULD_NOT_APPEAR"}, fh)
    gates, reports, decision = pilot.compute_gates(root)
    assert "SHOULD_NOT_APPEAR" not in gates["P0"]["detail"]
    assert "maweibo_integrity" in gates["P0"]["detail"]
    assert decision["primary_dataset"] == "maweibo"

    pilot.write_report(root, gates, reports, decision, ("maweibo", "pheme"))
    text = open(os.path.join(root, "CR_TSER_PILOT_REPORT.md"),
                encoding="utf-8").read()
    assert "Ma-Weibo primary" in text
    assert "Weibo22 primary" not in text
    assert "rejected primary candidate" in text
    assert "Ma-Weibo primary gate" in text

    summary = json.load(open(os.path.join(root, "CR_TSER_PILOT_SUMMARY.json"),
                             encoding="utf-8"))
    assert summary["primary_dataset"] == "maweibo"
    assert summary["secondary_dataset"] == "pheme"


def test_v1_historical_namespace_is_frozen():
    from ..config.pilot_config import V1_RESULTS_ROOT, V2_RESULTS_ROOT
    import hashlib
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    paths = [
        (os.path.join(repo, "results", "cr_tser", "verifier",
                      "code_verify.json"),
         "edc76e509262d12602d2c1f422ab07883950c25bb0e3ea5d5fbcd15cf2e68a27"),
    ]
    for path, want in paths:
        assert os.path.exists(path), path
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == want, f"{path} was rewritten"
    assert V2_RESULTS_ROOT != V1_RESULTS_ROOT
