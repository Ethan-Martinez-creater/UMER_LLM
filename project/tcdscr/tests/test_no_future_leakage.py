"""Unit tests: no-future-leakage integration over synthetic events (§28)."""
from ..data.snapshot_builder import build_snapshot
from ..data.structural_features import build_snapshot_features
from ..evaluation.leakage_scanner import scan_prompt
from ..context.evidence_unit import build_evidence_units, render_evidence
from .conftest import simple_event


def test_features_contain_no_future_nodes():
    event = simple_event()
    snap = build_snapshot(event, 60)
    feats = build_snapshot_features(snap, _sem(snap))
    future_ids = {n["node_id"] for n in event["nodes"]
                  if n["timestamp"] > 1000 + 3600}
    assert "n2" in future_ids
    assert not future_ids & set(feats["node_ids"])
    for child, _parent in feats["edge_index"].t().tolist():
        assert feats["node_ids"][child] not in future_ids


def test_prompt_never_contains_future_text():
    event = simple_event()
    snap = build_snapshot(event, 10)
    units = build_evidence_units(snap)
    evidence = "\n\n".join(render_evidence(u, i + 1)
                            for i, u in enumerate(units))
    prompt = f"SOURCE CLAIM\nsource claim\n\nSELECTED SOCIAL EVIDENCE\n{evidence}"
    report = scan_prompt(prompt, event, snap)
    assert report["pass"], report
    assert report["checked_nodes"] == len(event["nodes"])
    # and a prompt that does embed a future node's text must fail the scan
    bad_prompt = prompt + "\nreply two"
    assert not scan_prompt(bad_prompt, event, snap)["pass"]


def _sem(snap):
    import torch
    gen = torch.Generator().manual_seed(7)
    return torch.rand(len(snap["node_ids"]), 384, generator=gen)
