"""The frozen ``bcr_v1`` protocol surface (plan §11.2, §30)."""
from __future__ import annotations

import pytest

from bcr_utility.config import protocol as P


def test_protocol_identity_is_frozen():
    assert P.PROTOCOL_VERSION == "bcr_v1"
    assert P.BASELINE_COMMIT == "518a821d8d839fb7857be99acb91fde71b80c846"
    assert P.PRIMARY_DATASET == "maweibo"
    assert P.SECONDARY_DATASET == "pheme"
    assert P.DATASETS == ("maweibo", "pheme")


def test_frozen_constants_match_the_plan():
    constants = P.frozen_constants()
    assert constants["utility_threshold"] == 0.05
    assert tuple(constants["cutoffs_min"]) == (15, 60, 360)
    assert constants["seed"] == 7319
    assert constants["bootstrap_seed"] == 7319
    assert constants["bootstrap_iterations"] == 10000
    assert constants["bootstrap_unit"] == "event"
    assert constants["probe_events_per_dataset"] == 36
    assert constants["probe_events_per_cutoff"] == 12
    assert constants["probe_items_total"] == 72
    assert constants["probe_source_split"] == "foundation_train"
    assert tuple(constants["probe_contexts"]) == ("P0", "P1", "P2", "P3")
    assert constants["frozen_atomic_keys"] == {"maweibo": 2549, "pheme": 2107}


def test_namespaces_are_separate_from_history():
    assert P.RESULTS_ROOT == "results/bcr_utility_v1"
    assert P.HISTORICAL_RESULTS_ROOT == "results/cr_tser_v2r1"
    for namespace in ("results/cr_tser", "results/cr_tser_v2",
                      "results/cr_tser_v2r1"):
        assert namespace in P.HISTORICAL_NAMESPACES
    assert P.RESULTS_ROOT not in P.HISTORICAL_NAMESPACES


def test_bcr_does_not_inherit_the_structured_interaction_hypothesis():
    assert P.ATOMIC_INTERVENTION_TYPE == "I1_atomic"
    assert "I2_parent_child" in P.NON_SUPERVISION_INTERVENTION_TYPES
    assert "I4_subtree" in P.NON_SUPERVISION_INTERVENTION_TYPES


def test_inherited_values_agree_with_the_frozen_cr_tser_pilot():
    from cr_tser.config import pilot_config as cr

    assert P.UTILITY_THRESHOLD == cr.UTILITY_THRESHOLD
    assert tuple(P.CUTOFFS_MIN) == tuple(cr.CUTOFFS_MIN)
    assert tuple(P.READER_KEYS) == tuple(cr.READER_KEYS)
    assert dict(P.READER_MODEL_IDS) == dict(cr.READER_MODEL_IDS)
    assert P.INHERITED_FROM_CR_TSER["budget_ref"] == cr.BUDGET_REF
    assert P.INHERITED_FROM_CR_TSER["atomic_cap"] == cr.ATOMIC_CAP


def test_frozen_label_and_manifest_digests_are_pinned():
    assert P.FROZEN_HISTORICAL_LABELS["maweibo"]["sha256"] == (
        "6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3")
    assert P.FROZEN_HISTORICAL_LABELS["pheme"]["sha256"] == (
        "773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15")
    assert P.FROZEN_HISTORICAL_LABELS["maweibo"]["rows"] == 9606
    assert P.FROZEN_HISTORICAL_LABELS["pheme"]["rows"] == 8637
    assert len(P.FROZEN_HISTORICAL_MANIFEST_SHA256) == 10
    for digest in P.FROZEN_HISTORICAL_MANIFEST_SHA256.values():
        assert len(digest) == 64


def test_reused_dependency_list_is_explicit():
    assert "project/cr_tser/data/snapshot_bridge.py" in P.REUSED_CR_TSER_MODULES
    assert "project/cr_tser/readers/sequence_scorer.py" in \
        P.REUSED_CR_TSER_MODULES
    assert "scripts/cr_tser_common.py" in P.REUSED_CR_TSER_MODULES


def test_canonical_bytes_normalises_line_endings():
    assert P.canonical_bytes(b"a\r\nb\r\n") == b"a\nb\n"


def test_path_helpers_anchor_to_the_repo_root(tmp_path):
    def posix(value):
        return str(value).replace("\\", "/")

    assert posix(P.bootstrap_dir(tmp_path)).endswith(
        "results/bcr_utility_v1/bootstrap")
    assert posix(P.historical_root(tmp_path)).endswith("results/cr_tser_v2r1")
    assert posix(P.result_path(tmp_path, "x.json")).endswith("bootstrap/x.json")


@pytest.mark.parametrize("name", ["M1", "M2", "M3", "M4", "M5"])
def test_later_phases_are_not_declared(name):
    assert not hasattr(P, f"PHASE_{name}")
