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
