"""M1-D E3 compatibility-feature tests (synthetic, tiny fake LM)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ..config import protocol as P                        # noqa: E402
from ..features import compatibility_features as cf       # noqa: E402


class _FakeLM(torch.nn.Module):
    """Uniform-logit LM: every token has log-prob -log(V) everywhere."""

    def __init__(self, vocab=11):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        logits = torch.zeros(ids.shape[0], ids.shape[1], self.vocab)
        return type("Out", (), {"logits": logits})


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [0] if add_special_tokens else []
        ids += [1 + (len(w) % 9) for w in str(text).split()]
        return {"input_ids": ids}


def test_mean_token_nll_is_exactly_log_vocab():
    model = _FakeLM(vocab=11)
    nll = cf.mean_token_nll(model, _FakeTokenizer(), "ctx", "three word unit")
    assert nll == pytest.approx(math.log(11))
    nll_ctx = cf.mean_token_nll(model, _FakeTokenizer(), "given context",
                                "one")
    assert nll_ctx == pytest.approx(math.log(11))


def test_mean_token_nll_refuses_empty_continuation():
    with pytest.raises(cf.CompatibilityRefused, match="nothing"):
        cf.mean_token_nll(_FakeLM(), _FakeTokenizer(), "ctx", "")


def test_e3_feature_dict_order_and_gap():
    feats = cf.e3_feature_dict(0.5, 0.9, 4.0, 6.0, 5.25)
    assert tuple(feats) == P.E3_FEATURE_NAMES
    assert feats["nll_gap"] == pytest.approx(0.75)
    with pytest.raises(cf.CompatibilityRefused, match="not finite"):
        cf.e3_feature_dict(0.5, 0.9, float("nan"), 6.0, 5.0)


def test_source_only_metrics():
    margin, entropy = cf.source_only_metrics(1.0, -1.0, 0.9)
    assert margin == pytest.approx(2.0)
    assert 0.0 < entropy < 1.0


def test_validate_e3_rows_coverage():
    rows = []
    for key in ("k1", "k2"):
        for reader in P.READER_KEYS:
            rows.append({"key": key, "reader": reader,
                         "e3": {n: 0.5 for n in P.E3_FEATURE_NAMES}})
    audit = cf.validate_e3_rows(rows, ["k1", "k2"])
    assert audit["ok"] is True and audit["rows"] == 2 * len(P.READER_KEYS)
    with pytest.raises(cf.CompatibilityRefused, match="missing"):
        cf.validate_e3_rows(rows[:-1], ["k1", "k2"])
