"""Atomic evidence index: only I1, fail-closed on every inconsistency (§11.4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcr_utility.config import protocol as P
from bcr_utility.data import atomic_manifest as am

from .conftest import WRITERS, SYNTH_UTILITY, atomic_entries


def i1_row(dataset="maweibo", event="u0", cutoff=15, node="n1", reader="qwen",
           utility=0.4, sign=None, before=True, after=True):
    if sign is None:
        sign = ("HELPFUL" if utility >= P.UTILITY_THRESHOLD else
                "HARMFUL" if utility <= -P.UTILITY_THRESHOLD else "NEUTRAL")
    return {
        "dataset": dataset, "event_id": event, "cutoff": cutoff,
        "reader": reader, "intervention_id": f"I1:{node}",
        "intervention_type": "I1_atomic", "affected_reply_ids": [node],
        "utility": utility, "sign": sign, "correct_before": before,
        "correct_after": after,
    }


def write_rows(tmp_path, rows, name="labels.jsonl"):
    path = Path(tmp_path) / name
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


def test_extract_atomic_index_structure(synth_repo):
    path = str(synth_repo.historical / "utility_labels" / "maweibo"
               / "labels.jsonl")
    extracted = am.extract_atomic_index(path, "maweibo")
    entries = extracted["entries"]
    assert len(entries) == len(atomic_entries("maweibo"))
    assert extracted["n_i1_rows"] == len(entries) * len(WRITERS)
    for entry in entries:
        assert entry["key"].startswith("maweibo|")
        assert sorted(entry["utility"]) == sorted(P.READER_KEYS)
        assert sorted(entry["sign"]) == sorted(P.READER_KEYS)
    keys = [e["key"] for e in entries]
    as_tuples = [tuple(k.split("|")[1:]) for k in keys]
    assert as_tuples == sorted(as_tuples,
                               key=lambda t: (t[0], int(t[1]), t[2]))


def test_structured_interventions_are_counted_but_excluded(synth_repo):
    path = str(synth_repo.historical / "utility_labels" / "maweibo"
               / "labels.jsonl")
    extracted = am.extract_atomic_index(path, "maweibo")
    excluded = extracted["excluded_intervention_type_counts"]
    assert excluded["I0_base"] == len(WRITERS)
    for itype in P.NON_SUPERVISION_INTERVENTION_TYPES:
        assert excluded[itype] == len(WRITERS)
    indexed_types = {"I1_atomic"}
    assert "I2_parent_child" not in indexed_types


def test_sign_mismatch_is_refused(tmp_path):
    path = write_rows(tmp_path, [
        i1_row(reader=r, sign="NEUTRAL")
        for r in WRITERS])
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(path, "maweibo")


def test_duplicate_reader_row_is_refused(tmp_path):
    rows = [i1_row(reader=r) for r in WRITERS]
    rows.append(i1_row(reader="qwen"))
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_multi_reply_atomic_row_is_refused(tmp_path):
    row = i1_row()
    row["affected_reply_ids"] = ["n1", "n2"]
    rows = [row] + [i1_row(reader=r) for r in WRITERS if r != "qwen"]
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_non_finite_utility_is_refused(tmp_path):
    rows = [i1_row(reader=r) for r in WRITERS]
    rows[0]["utility"] = float("nan")
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_missing_reader_is_refused(tmp_path):
    rows = [i1_row(reader="qwen"), i1_row(reader="mistral")]
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_unknown_reader_is_refused(tmp_path):
    rows = [i1_row(reader=r) for r in WRITERS] + [i1_row(reader="glm")]
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_non_boolean_correctness_is_refused(tmp_path):
    rows = [i1_row(reader=r) for r in WRITERS]
    rows[0]["correct_before"] = 1
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, rows), "maweibo")


def test_empty_cache_is_refused(tmp_path):
    with pytest.raises(am.AtomicIndexRefused):
        am.extract_atomic_index(write_rows(tmp_path, [i1_row()]),
                                "maweibo")


def test_build_atomic_index_count_gate(synth_repo, monkeypatch):
    labels_sha = synth_repo.labels["maweibo"]["sha256"]
    monkeypatch.setitem(P.FROZEN_ATOMIC_KEYS, "maweibo", 999)
    with pytest.raises(am.AtomicIndexRefused):
        am.build_atomic_index(str(synth_repo.historical), "maweibo",
                              labels_sha)


def test_build_atomic_index_digest_gate(synth_repo):
    with pytest.raises(am.AtomicIndexRefused):
        am.build_atomic_index(str(synth_repo.historical), "maweibo", "0" * 64)


def test_build_atomic_index_payload(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    assert index["n_evidence_keys"] == len(atomic_entries("maweibo"))
    assert index["cutoffs"] == list(P.CUTOFFS_MIN)
    assert index["reader_keys"] == list(P.READER_KEYS)
    assert index["utility_threshold"] == P.UTILITY_THRESHOLD
    assert index["atomic_intervention_type"] == "I1_atomic"
    assert index["n_events"] == 2
    assert len(index["keys_sha256"]) == 64
    am.validate_atomic_index(index, "maweibo")


def test_validate_rejects_wrong_key_count(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    index["entries"] = index["entries"][:-1]
    with pytest.raises(am.AtomicIndexRefused):
        am.validate_atomic_index(index, "maweibo")


def test_validate_rejects_sign_drift(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    entry = index["entries"][0]
    entry["sign"]["qwen"] = "NEUTRAL"
    with pytest.raises(am.AtomicIndexRefused):
        am.validate_atomic_index(index, "maweibo")


def test_validate_rejects_threshold_drift(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    index["utility_threshold"] = 0.5
    with pytest.raises(am.AtomicIndexRefused):
        am.validate_atomic_index(index, "maweibo")


def test_utility_matrix_is_stable_and_complete(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    keys, readers, columns = am.utility_matrix(index)
    assert readers == list(P.READER_KEYS)
    assert len(keys) == len(index["entries"])
    assert all(len(columns[r]) == len(keys) for r in readers)
    first = index["entries"][0]
    assert columns["qwen"][0] == first["utility"]["qwen"]


def test_utility_matrix_reflects_the_synthetic_values(synth_repo):
    index = am.build_atomic_index(
        str(synth_repo.historical), "maweibo",
        synth_repo.labels["maweibo"]["sha256"])
    keys, _readers, columns = am.utility_matrix(index)
    for i, key in enumerate(keys):
        _ds, event, _cutoff, _node = key.split("|")
        for reader in WRITERS:
            assert columns[reader][i] == pytest.approx(
                SYNTH_UTILITY["maweibo"][reader][event])


def test_event_of_key():
    assert am.event_of_key("maweibo|42|15|n1") == "42"
    with pytest.raises(am.AtomicIndexRefused):
        am.event_of_key("maweibo|42|15")


def test_load_atomic_index_missing_is_refused(tmp_path):
    with pytest.raises(am.AtomicIndexRefused):
        am.load_atomic_index(str(Path(tmp_path) / "nope.json"))
