"""Unit tests: Ma-Weibo adapter (plan §28 — test_maweibo_timestamp,
test_maweibo_clean_text)."""
from ..data.maweibo_adapter import iter_events
from ..data.text_cleaning import clean_text_weibo


def test_maweibo_timestamp(maweibo_tree):
    raw, labels = maweibo_tree
    events = {e["event_id"]: e for e in iter_events(raw, labels)}
    ev = events["30001"]
    # t is an absolute unix second; t0 = source post timestamp
    assert ev["source_timestamp"] == 1400000000
    ts = {n["node_id"]: n["timestamp"] for n in ev["nodes"]}
    assert ts["9002"] - ev["source_timestamp"] == 600


def test_maweibo_clean_text(maweibo_tree):
    raw, labels = maweibo_tree
    events = {e["event_id"]: e for e in iter_events(raw, labels)}
    ev = events["30001"]
    node = next(n for n in ev["nodes"] if n["node_id"] == "9001")
    # adapter keeps raw text preferring original_text; cleaning is applied
    # downstream with clean_text_weibo before encoding (frozen §3.2/§7)
    assert node["text"] == "<b>源帖</b> http://t.cn/x 断言"
    cleaned = clean_text_weibo(node["text"])
    assert cleaned == "源帖 断言"
    assert "http" not in cleaned and "<" not in cleaned


def test_maweibo_multiroot_rejected(maweibo_tree, tmp_path):
    import json
    import os
    raw, labels = maweibo_tree
    bad = os.path.join(raw, "39999.json")
    with open(bad, "w", encoding="utf-8") as fh:
        json.dump([{"mid": "1", "parent": None, "t": 100,
                    "original_text": "a"},
                   {"mid": "2", "parent": None, "t": 200,
                    "original_text": "b"}], fh)
    with open(labels, "a", encoding="utf-8") as fh:
        fh.write("eid:39999 label:1\n")
    it = iter_events(raw, labels)
    raised = False
    try:
        for _ in it:
            pass
    except Exception:
        raised = True
    assert raised  # frozen audit says multi-root events do not exist
    os.remove(bad)


def test_maweibo_text_fallback_count(maweibo_tree, tmp_path):
    """delta-fix §34/§35: original_text -> text fallback is audited, never
    silent."""
    import json
    import os
    from ..data.maweibo_adapter import (aggregate_text_fallback, iter_events,
                                        load_event, parse_labels)
    raw, labels = maweibo_tree
    mixed = os.path.join(raw, "35555.json")
    with open(mixed, "w", encoding="utf-8") as fh:
        json.dump([
            {"mid": "1", "parent": None, "t": 1000,
             "original_text": "原文", "text": "未被使用"},
            {"mid": "2", "parent": "1", "t": 1100, "text": "只有 text 字段"},
            {"mid": "3", "parent": "1", "t": 1200, "original_text": "  "},
        ], fh, ensure_ascii=False)
    labels_map = parse_labels(labels)
    ev = load_event("35555", 1, mixed)
    counts = ev["text_source_counts"]
    assert counts["total_nodes"] == 3
    # field presence decides the branch (historical behavior): node 1 and
    # node 3 carry an original_text field (node 3's is blank), node 2 only
    # has text -> fallback
    assert counts["original_text_used"] == 2
    assert counts["text_fallback_count"] == 1
    assert counts["empty_text_count"] == 1     # node 3's resolved text is blank

    # aggregate over the dataset iterator
    events = list(iter_events(raw, labels))
    agg = aggregate_text_fallback(events)
    assert agg["events"] == len(events)
    assert agg["total_nodes"] == sum(
        e["text_source_counts"]["total_nodes"] for e in events)
    assert agg["text_fallback_count"] == sum(
        e["text_source_counts"]["text_fallback_count"] for e in events)
    assert abs(agg["fallback_rate"] - agg["text_fallback_count"]
               / max(agg["total_nodes"], 1)) < 1e-12
    os.remove(mixed)
