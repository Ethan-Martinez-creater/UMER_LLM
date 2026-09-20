"""M1 LORO orchestration tests (tiny synthetic end-to-end)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ..config import protocol as P                        # noqa: E402
from ..evaluation import unseen_reader as lor             # noqa: E402
from ..probes import fingerprint as fp                    # noqa: E402
from . import m1_synth as S                               # noqa: E402

FAST = {"max_epochs": 4, "patience": 2, "batch_size": 256}


def _dataset(name="maweibo"):
    entries = S.synth_atomic_entries(name)
    table = lor.build_feature_table(
        name, entries, S.synth_e0_rows(entries), S.synth_e1_rows(entries),
        S.synth_e2_rows(entries))
    return entries, table, S.synth_split(), S.synth_fingerprints()


def test_build_feature_table_dimensions():
    entries, table, _split, _fp = _dataset()
    assert len(table) == len(entries)
    row = next(iter(table.values()))
    assert len(row["e0e1"]) == len(P.E0_FEATURE_NAMES) \
        + len(P.E1_FEATURE_NAMES) == 20
    assert len(row["struct"]) == 10
    assert sorted(row["e2"]) == sorted(P.READER_KEYS)
    assert len(row["e2"]["qwen"]) == 6


def test_make_rows_feature_widths_per_model():
    entries, table, split, _fp = _dataset()
    train_events = split["utility_train"][:3]
    b0 = lor.make_rows(entries, table, ["qwen"], train_events, P.MODEL_B0)
    b1 = lor.make_rows(entries, table, ["qwen"], train_events, P.MODEL_B1)
    b4 = lor.make_rows(entries, table, ["qwen"], train_events, P.MODEL_B4)
    assert len(b0[0]["x"]) == 20
    assert len(b1[0]["x"]) == 30
    assert len(b4[0]["x"]) == 26
    assert {r["reader"] for r in b0} == {"qwen"}


def test_fit_feature_scaler_zero_variance_gets_std_one():
    rows = [{"x": [1.0, 2.0, 5.0]}, {"x": [3.0, 2.0, 5.0]},
            {"x": [5.0, 2.0, 5.0]}]
    scaler = lor.fit_feature_scaler(rows)
    assert scaler["mean"] == pytest.approx([3.0, 2.0, 5.0])
    assert scaler["std"][0] == pytest.approx(2.0)   # sample std of [1,3,5]
    assert scaler["std"][1] == 1.0
    assert scaler["std"][2] == 1.0
    scaled = lor.apply_scaler(rows, scaler)
    assert scaled[1]["x"][0] == pytest.approx(0.0)


def test_grid_configs_exactly_eight():
    configs = lor.grid_configs()
    assert len(configs) == 8
    assert all(c["hidden"] == 32 for c in configs)
    assert {c["dropout"] for c in configs} == {0.0, 0.1}
    assert {c["lr"] for c in configs} == {1e-3, 3e-4}
    assert {c["weight_decay"] for c in configs} == {0.0, 1e-4}


def test_b3_picks_nearest_and_transfers_labels_directly():
    entries, table, split, fingerprints = _dataset()
    rotation = P.LORO_ROTATIONS[0]      # train mistral+internlm, hold qwen
    eval_rows = lor.make_rows(entries, table, ["qwen"],
                              split["utility_eval"], P.MODEL_B0)
    out = lor.b3_predict(fingerprints, ["mistral", "internlm"], "qwen",
                         entries, eval_rows)
    assert out["nearest_reader"] in ("mistral", "internlm")
    assert len(out["distances"]) == 2
    nearest = out["nearest_reader"]
    lookup = {e["key"]: e for e in entries}
    for row, u, probs in zip(eval_rows, out["prediction"]["utility"],
                             out["prediction"]["sign_probs"]):
        entry = lookup[row["key"]]
        assert u == pytest.approx(entry["utility"][nearest])
        expected = [0.0, 0.0, 0.0]
        expected[["HELPFUL", "NEUTRAL", "HARMFUL"].index(
            entry["sign"][nearest])] = 1.0
        assert probs == expected


def test_run_rotation_end_to_end_leakage_free():
    entries, table, split, fingerprints = _dataset()
    rotation = P.LORO_ROTATIONS[0]
    out = lor.run_rotation("maweibo", rotation, entries, table, split,
                           fingerprints, models=(P.MODEL_B0, P.MODEL_B4),
                           train_kwargs=FAST)
    assert out["held_out"] == "qwen"
    assert sorted(out["train_readers"]) == ["internlm", "mistral"]
    for kind in (P.MODEL_B0, P.MODEL_B4):
        record = out["models"][kind]
        assert record["selected_config"]["hidden"] == 32
        assert record["fold_sizes"]["eval"] > 0
        assert 0.0 <= record["eval_metrics"]["macro_f1"] <= 1.0
        assert record["n_parameters"] < P.B4_MAX_PARAMETERS
    b3 = out["models"][P.MODEL_B3]
    assert b3["nearest_reader"] in ("mistral", "internlm")
    assert out["disagreement_subset"]["n"] <= \
        out["disagreement_subset"]["of"]


def test_run_dataset_and_gate_aggregation():
    entries, table, split, fingerprints = _dataset()
    result, aggregate, cache = lor.run_dataset(
        "maweibo", entries, table, split, fingerprints,
        models=(P.MODEL_B0, P.MODEL_B4), train_kwargs=FAST)
    agg = result["aggregate"]
    assert sorted(agg["per_reader_delta"]) == sorted(P.READER_KEYS)
    assert agg["positive_readers"] in (0, 1, 2, 3)
    assert agg["worst_reader_delta"] == min(agg["per_reader_delta"].values())
    assert aggregate["n_readers"] == 3
    gate = lor.decide_primary_gate(result)
    assert set(gate["checks"]) == {"mean_delta_gte_0.03", "ci_low_gt_0",
                                   "positive_readers_gte_2",
                                   "worst_reader_gte_-0.05"}
    assert gate["passed"] == all(gate["checks"].values())


def test_decide_primary_gate_threshold_logic():
    def fake(mean, ci_low, deltas):
        return {"aggregate": {
            "mean_delta_macro_f1": mean, "ci_low": ci_low, "ci_high": 0.5,
            "per_reader_delta": dict(zip(P.READER_KEYS, deltas)),
            "positive_readers": sum(1 for d in deltas if d > 0),
            "worst_reader_delta": min(deltas)}}

    assert lor.decide_primary_gate(
        fake(0.05, 0.01, (0.05, 0.04, -0.01)))["passed"] is True
    assert lor.decide_primary_gate(
        fake(0.02, 0.01, (0.05, 0.04, -0.01)))["passed"] is False
    assert lor.decide_primary_gate(
        fake(0.05, -0.01, (0.05, 0.04, -0.01)))["passed"] is False
    assert lor.decide_primary_gate(
        fake(0.05, 0.01, (0.05, -0.02, -0.03)))["passed"] is False
    assert lor.decide_primary_gate(
        fake(0.05, 0.01, (0.05, 0.04, -0.06)))["passed"] is False


def test_run_b2_in_domain_diagnostic():
    entries, table, split, _fp = _dataset()
    out = lor.run_b2_in_domain("maweibo", entries, table, split,
                               train_kwargs=FAST)
    assert out["in_domain"] is True
    assert out["kind"] == P.MODEL_B2
    assert 0.0 <= out["eval_metrics"]["macro_f1"] <= 1.0


def _dataset_e3(name="maweibo"):
    entries = S.synth_atomic_entries(name)
    table = lor.build_feature_table(
        name, entries, S.synth_e0_rows(entries), S.synth_e1_rows(entries),
        S.synth_e2_rows(entries), e3_rows=S.synth_e3_rows(entries))
    return entries, table, S.synth_split(), S.synth_fingerprints()


def test_b5_feature_width_and_missing_e3_refused():
    entries, table, split, _fp = _dataset_e3()
    b5 = lor.make_rows(entries, table, ["qwen"], split["utility_train"][:2],
                       P.MODEL_B5)
    assert len(b5[0]["x"]) == 20 + 6 + 6
    # without E3 rows a B5 table cannot be built at all
    entries_z = S.synth_atomic_entries("maweibo")
    with pytest.raises(lor.LoroRefused, match="E3"):
        lor.build_feature_table("maweibo", entries_z,
                                S.synth_e0_rows(entries_z),
                                S.synth_e1_rows(entries_z),
                                S.synth_e2_rows(entries_z),
                                e3_rows=[])


def test_run_dataset_light_touch_b5_uses_same_protocol():
    entries, table, split, fingerprints = _dataset_e3()
    result, aggregate, _cache = lor.run_dataset(
        "maweibo", entries, table, split, fingerprints,
        models=(P.MODEL_B0, P.MODEL_B5), primary_model=P.MODEL_B5,
        train_kwargs=FAST)
    assert result["primary_comparison"] == [P.MODEL_B5, P.MODEL_B0]
    for held, rot in result["rotations"].items():
        assert P.MODEL_B5 in rot["models"]
        assert rot["models"][P.MODEL_B5]["selected_config"]["hidden"] == 32
    assert aggregate["n_readers"] == 3
    gate = lor.decide_primary_gate(result)
    assert isinstance(gate["passed"], bool)
