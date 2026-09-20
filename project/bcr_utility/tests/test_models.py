"""M1 model architecture and trainer tests (tiny synthetic data)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ..config import protocol as P                       # noqa: E402
from ..models import baselines as bl                     # noqa: E402
from ..models.conditioned_utility import build_conditioned_net  # noqa: E402


def test_evidence_net_shapes_and_parameter_count():
    model = bl.build_evidence_net(20, hidden=32, dropout=0.0)
    x = torch.randn(5, 20)
    u, logits = model(x)
    assert u.shape == (5,)
    assert logits.shape == (5, len(P.SIGN_CLASSES))
    assert bl.count_parameters(model) < P.B4_MAX_PARAMETERS


def test_evidence_net_extra_dim_for_reader_one_hot():
    model = bl.build_evidence_net(20, hidden=32, dropout=0.0, extra_dim=3)
    x = torch.randn(4, 20)
    extra = torch.zeros(4, 3)
    extra[:, 1] = 1.0
    u, logits = model(x, extra)
    assert logits.shape == (4, 3)


def test_conditioned_net_is_below_the_parameter_cap():
    in_dim = len(P.E0_FEATURE_NAMES) + len(P.E1_FEATURE_NAMES) \
        + len(P.E2_FEATURE_NAMES)
    model = build_conditioned_net(in_dim, P.FINGERPRINT_DIM, embed=32,
                                  dropout=0.1)
    assert bl.count_parameters(model) < P.B4_MAX_PARAMETERS
    x_e = torch.randn(7, in_dim)
    x_r = torch.randn(7, P.FINGERPRINT_DIM)
    u, logits = model(x_e, x_r)
    assert u.shape == (7,)
    assert logits.shape == (7, 3)


def test_conditioned_net_refuses_oversized():
    with pytest.raises(Exception, match="250000|parameters"):
        build_conditioned_net(200000, P.FINGERPRINT_DIM, embed=32)


def test_sign_class_weights_inverse_frequency():
    weights = bl.sign_class_weights(["NEUTRAL"] * 8 + ["HELPFUL",
                                                       "HARMFUL"])
    assert weights[0] == pytest.approx(10 / 3)          # HELPFUL 1/10
    assert weights[1] == pytest.approx(10 / 24)         # NEUTRAL 8/10
    assert weights[2] == pytest.approx(10 / 3)          # HARMFUL 1/10
    assert bl.sign_class_weights(["NEUTRAL"] * 4)[0] == 0.0


def _linear_rows(n=64, dim=4):
    rows = []
    for i in range(n):
        x = [((i * 37 + j * 11) % 13 - 6) / 6.0 for j in range(dim)]
        u = x[0] * 0.5
        sign = 0 if u > 0.05 else (2 if u < -0.05 else 1)
        rows.append({"x": x, "u": u, "s": sign,
                     "sign": P.SIGN_CLASSES[sign], "utility": u})
    return rows


def test_train_model_learns_a_linear_signal():
    rows = _linear_rows()
    model = bl.build_evidence_net(4, hidden=32, dropout=0.0)
    config = {"hidden": 32, "dropout": 0.0, "lr": 3e-3, "weight_decay": 0.0}
    result = bl.train_model(model, rows[:48], rows[48:], config, 7319,
                            [1.0, 1.0, 1.0], max_epochs=150, patience=20)
    assert result["best_dev_macro_f1"] > 0.55
    assert result["n_parameters"] == bl.count_parameters(model)
    pred = bl.predict(model, rows[48:])
    assert len(pred["utility"]) == len(rows[48:])
    assert len(pred["sign_probs"][0]) == 3


def test_train_model_is_deterministic_per_seed():
    rows = _linear_rows()
    config = {"hidden": 32, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0}
    m1 = bl.build_evidence_net(4, hidden=32)
    r1 = bl.train_model(m1, rows[:48], rows[48:], config, 7319,
                        [1.0, 1.0, 1.0], max_epochs=5, patience=2)
    m2 = bl.build_evidence_net(4, hidden=32)
    r2 = bl.train_model(m2, rows[:48], rows[48:], config, 7319,
                        [1.0, 1.0, 1.0], max_epochs=5, patience=2)
    assert r1["best_dev_macro_f1"] == r2["best_dev_macro_f1"]
    p1 = bl.predict(m1, rows[:4])["utility"]
    p2 = bl.predict(m2, rows[:4])["utility"]
    assert p1 == pytest.approx(p2)


def test_average_predictions_mean_and_renormalise():
    runs = [{"utility": [0.1, -0.2],
             "sign_probs": [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]},
            {"utility": [0.3, 0.0],
             "sign_probs": [[0.7, 0.1, 0.2], [0.1, 0.6, 0.3]]}]
    avg = bl.average_predictions(runs)
    assert avg["utility"] == pytest.approx([0.2, -0.1])
    assert avg["sign_probs"][0] == pytest.approx([0.6, 0.2, 0.2])
    assert sum(avg["sign_probs"][1]) == pytest.approx(1.0)
