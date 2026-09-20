"""M1 E1 NLI-feature tests (synthetic; the model itself is never loaded)."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..features import nli_features as nf


def test_e1_feature_dict_order_and_bounds():
    probs = {pair: [0.7, 0.2, 0.1] for pair in P.E1_PAIRS}
    feats = nf.e1_feature_dict(probs)
    assert tuple(feats) == P.E1_FEATURE_NAMES
    assert feats["nli_source_reply_entailment"] == pytest.approx(0.7)
    assert feats["nli_reply_parent_contradiction"] == pytest.approx(0.1)


def test_e1_feature_dict_refuses_bad_probs():
    with pytest.raises(nf.NLIRefused):
        nf.e1_feature_dict({pair: [0.5, 0.5] for pair in P.E1_PAIRS})
    with pytest.raises(nf.NLIRefused):
        nf.e1_feature_dict({pair: [float("nan"), 0.2, 0.1]
                            for pair in P.E1_PAIRS})


def _fake_nli_probs(nli, pairs, batch_size=32):
    # deterministic: entailment-heavy for (a, b) where a is shorter
    out = []
    for premise, hypothesis in pairs:
        if len(premise) <= len(hypothesis):
            out.append([0.6, 0.3, 0.1])
        else:
            out.append([0.1, 0.3, 0.6])
    return out


def test_e1_rows_with_fake_model_and_missing_parent(monkeypatch):
    monkeypatch.setattr(nf, "nli_probs", _fake_nli_probs)
    text_rows = [
        {"key": "k1", "source_text": "SRC", "reply_text": "a longer reply",
         "parent_text": "parent"},
        {"key": "k2", "source_text": "SRC", "reply_text": "reply",
         "parent_text": None},
    ]
    rows = nf.e1_rows(text_rows, nli={"fake": True}, clean=lambda t: t)
    assert [r["key"] for r in rows] == ["k1", "k2"]
    assert rows[0]["e1"]["nli_source_reply_entailment"] == pytest.approx(0.6)
    assert rows[1]["e1"]["nli_source_parent_entailment"] == 0.0
    assert rows[1]["e1"]["nli_reply_parent_contradiction"] == 0.0
    audit = nf.validate_e1_rows(rows, ["k1", "k2"])
    assert audit["ok"] is True


def test_e1_rows_clean_is_applied(monkeypatch):
    seen = []

    def clean(text):
        seen.append(text)
        return text.strip().lower()

    monkeypatch.setattr(nf, "nli_probs", _fake_nli_probs)
    rows = nf.e1_rows([{"key": "k", "source_text": " SRC ",
                        "reply_text": " REPLY ", "parent_text": " PAR "}],
                      nli={"fake": True}, clean=clean)
    assert " SRC " in seen and " REPLY " in seen
    assert rows[0]["e1"]["nli_source_reply_neutral"] == pytest.approx(0.3)


def test_validate_e1_refuses_extra_keys():
    rows = [{"key": "k1", "e1": {n: 0.1 for n in P.E1_FEATURE_NAMES}}]
    with pytest.raises(nf.NLIRefused, match="unexpected"):
        nf.validate_e1_rows(rows, ["k2"])
