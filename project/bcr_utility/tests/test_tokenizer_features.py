"""M1 E2 tokenizer-feature tests (synthetic whitespace tokenizers)."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..features import tokenizer_features as tf


class FakeTokenizer:
    """Whitespace tokenizer with the HF call contract."""

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, text, add_special_tokens=True):
        n = len(str(text).split()) * self.scale
        return {"input_ids": list(range(n))}


def _unit():
    return {"node_id": "n1", "reply_text": "one two three",
            "parent_text": "four five", "parent_id": "n0",
            "timestamp": 1060, "elapsed_seconds": 60, "snapshot_order": 1,
            "depth": 1}


def test_e2_feature_dict_counts_and_ratios():
    unit = _unit()
    feats = tf.e2_feature_dict(unit, FakeTokenizer(), FakeTokenizer())
    assert tuple(feats) == P.E2_FEATURE_NAMES
    assert feats["reply_tokens"] == pytest.approx(3.0)
    assert feats["parent_tokens"] == pytest.approx(2.0)
    # rendered unit: "[E0]" + Reply/Parent/Observed lines
    assert feats["unit_tokens"] > feats["reply_tokens"]
    assert feats["unit_vs_canonical"] == pytest.approx(1.0)
    assert 0.0 < feats["reply_frac"] < 1.0


def test_e2_reader_tokenizer_differences_are_visible():
    unit = _unit()
    narrow = tf.e2_feature_dict(unit, FakeTokenizer(scale=1),
                                FakeTokenizer())
    wide = tf.e2_feature_dict(unit, FakeTokenizer(scale=2),
                              FakeTokenizer())
    assert wide["reply_tokens"] == pytest.approx(2 * narrow["reply_tokens"])
    assert wide["unit_vs_canonical"] == pytest.approx(
        2 * narrow["unit_vs_canonical"])


def test_e2_missing_parent_token_counts_zero():
    unit = dict(_unit())
    unit["parent_text"] = None
    feats = tf.e2_feature_dict(unit, FakeTokenizer(), FakeTokenizer())
    assert feats["parent_tokens"] == 0.0


def test_e2_rows_cover_keys_times_readers():
    units = {"k1": _unit(), "k2": dict(_unit(), node_id="n2")}
    tokenizers = {r: FakeTokenizer() for r in P.READER_KEYS}
    rows = tf.e2_rows(units, tokenizers, FakeTokenizer())
    assert len(rows) == 2 * len(P.READER_KEYS)
    audit = tf.validate_e2_rows(rows, ["k1", "k2"])
    assert audit["ok"] is True
    with pytest.raises(tf.FeatureExtractionRefused, match="missing"):
        tf.validate_e2_rows(rows[:-1], ["k1", "k2"])
