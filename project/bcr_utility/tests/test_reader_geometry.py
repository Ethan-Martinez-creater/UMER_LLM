"""Reader-utility geometry diagnostics (plan §11.6, §1.2)."""
from __future__ import annotations

import numpy as np
import pytest

from bcr_utility.config import protocol as P
from bcr_utility.data import atomic_manifest as am
from bcr_utility.evaluation import reader_geometry as rg

from .conftest import SYNTH_UTILITY, WRITERS


@pytest.fixture
def maweibo_index(synth_repo):
    return am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])


def test_activity_counts_every_synthetic_key(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    activity = report["activity"]["per_reader"]
    for reader in WRITERS:
        assert activity[reader]["n_keys"] == 6
        assert activity[reader]["n_active"] == 6
        assert activity[reader]["active_rate"] == 1.0
    assert activity["mistral"]["sign_counts"] == {
        "HELPFUL": 0, "NEUTRAL": 0, "HARMFUL": 6}
    assert activity["qwen"]["sign_counts"] == {
        "HELPFUL": 6, "NEUTRAL": 0, "HARMFUL": 0}


def test_jointly_active_pairs(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    pairs = report["activity"]["per_pair"]
    assert set(pairs) == {"qwen|mistral", "qwen|internlm", "mistral|internlm"}
    for pair in pairs:
        assert pairs[pair]["n_jointly_active"] == 6
        assert pairs[pair]["jointly_active_rate"] == 1.0


def test_signed_direction_separates_agreeing_and_flipping_pairs(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    signed = report["signed_direction_geometry"]["per_pair"]

    agree = signed["qwen|internlm"]
    assert agree["sign_agreement_rate"] == 1.0
    assert agree["directional_agreement_rate"] == 1.0
    assert agree["sign_contingency"]["HELPFUL:HELPFUL"] == 6

    flip = signed["qwen|mistral"]
    assert flip["sign_agreement_rate"] == 0.0
    assert flip["sign_disagreement_rate"] == 1.0
    assert flip["sign_contingency"]["HELPFUL:HARMFUL"] == 6
    assert flip["n_directional"] == 6


def test_conditional_direction_is_row_normalised(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    conditional = report["signed_direction_geometry"]["per_pair"][
        "qwen|mistral"]["conditional_direction"]
    helpful = conditional["HELPFUL"]
    assert helpful["n"] == 6
    assert helpful["p_sign_b"]["HELPFUL"] == pytest.approx(0.0)
    assert helpful["p_sign_b"]["HARMFUL"] == pytest.approx(1.0)
    assert sum(helpful["p_sign_b"].values()) == pytest.approx(1.0)


def test_ordinal_geometry_follows_the_synthetic_magnitudes(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    ordinal = report["ordinal_geometry"]["per_pair"]
    assert ordinal["qwen|internlm"]["spearman_abs_utility"] == 1.0
    assert ordinal["qwen|mistral"]["spearman_abs_utility"] == 1.0
    assert ordinal["qwen|internlm"]["pearson_abs_utility"] > 0.9
    assert ordinal["qwen|mistral"]["spearman_signed_utility"] == -1.0


def test_ordinal_geometry_alone_would_miss_the_flip(maweibo_index):
    """The P1 phenomenon: identical magnitude ordering, opposite sign."""
    report = rg.geometry_report(maweibo_index, iterations=20)
    ordinal = report["ordinal_geometry"]["per_pair"]["qwen|mistral"]
    signed = report["signed_direction_geometry"]["per_pair"]["qwen|mistral"]
    assert ordinal["spearman_abs_utility"] == 1.0
    assert signed["sign_agreement_rate"] < 1.0
    assert "does not by itself" in report["ordinal_geometry"]["caveat"]


def test_interaction_shares_add_up(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    interaction = report["reader_evidence_interaction"]
    total = (interaction["reader_share"] + interaction["evidence_share"]
             + interaction["interaction_share"])
    assert total == pytest.approx(1.0, abs=1e-9)
    assert interaction["n_keys"] == 6
    assert interaction["n_readers"] == 3
    ci = interaction["interaction_share_event_ci"]
    assert ci["iterations"] == 20
    assert ci["seed"] == P.BOOTSTRAP_SEED
    assert ci["n_events"] == 2


def test_interaction_shares_are_zero_for_an_additive_matrix():
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    shares = rg.interaction_shares(matrix)
    assert shares["ss_residual"] == pytest.approx(0.0, abs=1e-12)
    assert shares["interaction_share"] == pytest.approx(0.0, abs=1e-12)


def test_interaction_shares_are_large_for_a_pure_interaction():
    matrix = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 4.0, 1.0]])
    shares = rg.interaction_shares(matrix)
    assert shares["interaction_share"] > 0.5


def test_bootstrap_draws_match_the_frozen_cr_tser_resampling():
    """A BCR interval must resample the same events as a CR-TSER interval."""
    from cr_tser.evaluation.bootstrap import mean, paired_event_bootstrap

    payload = {f"e{i}": [float(i), float(i + 1)] for i in range(5)}
    iterations, seed = 50, P.BOOTSTRAP_SEED
    cr = paired_event_bootstrap(
        payload, lambda payloads: mean([v for p in payloads for v in p]),
        iterations=iterations, seed=seed)

    events = sorted(payload)
    rows = rg.Rows([np.arange(i, i + 1) * 2 + np.array([0, 1])
                    for i in range(len(events))])
    values = np.array([v for e in events for v in payload[e]])
    bcr = [float(values[rows.expand(draw)].mean())
           for draw in rg.bootstrap_draws(len(events), iterations, seed)]

    assert cr["samples"] == pytest.approx(bcr)
    assert cr["observed"] == pytest.approx(values.mean())


def test_geometry_report_is_explicitly_diagnostic(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    assert report["protocol"] == P.PROTOCOL_VERSION
    assert report["scope"] == "diagnostic_only"
    assert report["decides_gate"] is False
    assert report["bootstrap"] == {"unit": "event",
                                   "iterations": 20,
                                   "seed": P.BOOTSTRAP_SEED}
    assert report["reader_keys"] == list(P.READER_KEYS)
    assert report["n_evidence_keys"] == 6
    assert report["n_events"] == 2
    assert "GO/NO-GO" in report["interpretation"]["m0_scope"]
    assert report["signed_direction_geometry"]["caveat"] == rg.SIGNED_CAVEAT


def test_geometry_has_no_gate_verdict(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    blob = str(report)
    for verdict in ("PASS", "FAIL", "GO_NO_GO", "NO_GO"):
        assert verdict not in blob.replace("GO/NO-GO", "")


def test_geometry_refuses_a_foreign_reader_panel(maweibo_index):
    maweibo_index = dict(maweibo_index, reader_keys=["qwen", "glm", "internlm"])
    with pytest.raises(ValueError):
        rg.geometry_report(maweibo_index, iterations=5)


def test_geometry_reflects_the_imported_utilities(maweibo_index):
    report = rg.geometry_report(maweibo_index, iterations=20)
    activity = report["activity"]["per_reader"]
    for reader in WRITERS:
        expected = sum(SYNTH_UTILITY["maweibo"][reader].values()) * 3 / 6
        assert activity[reader]["mean_utility"] == pytest.approx(expected)


def test_pearson_helper_handles_degenerate_input():
    assert rg._pearson([1.0], [2.0]) != rg._pearson([1.0], [2.0])  # NaN
    assert rg._pearson([1.0, 1.0], [2.0, 3.0]) != \
        rg._pearson([1.0, 1.0], [2.0, 3.0])  # zero variance -> NaN
    assert rg._json_number(float("nan")) is None
    assert rg._json_number(float("inf")) is None
    assert rg._json_number(0.5) == 0.5
