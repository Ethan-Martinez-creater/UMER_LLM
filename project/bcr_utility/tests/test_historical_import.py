"""Historical read-only importer: every mismatch must fail closed (plan §2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcr_utility.config import protocol as P
from bcr_utility.data import historical_import as hi

from .conftest import WRITERS


def test_inspect_labels_matches_the_synthetic_cache(synth_repo):
    for dataset in P.DATASETS:
        info = hi.inspect_labels(str(synth_repo.historical), dataset)
        assert info["sha256"] == synth_repo.labels[dataset]["sha256"]
        assert info["rows"] == synth_repo.labels[dataset]["rows"]
        assert info["rows_per_reader"] == {r: info["rows"] // 3
                                           for r in WRITERS}
        assert info["cutoffs"] == list(P.CUTOFFS_MIN)
        assert info["intervention_type_counts"]["I1_atomic"] > 0
        assert info["intervention_type_counts"]["I2_parent_child"] > 0


def test_label_digest_drift_is_refused(synth_repo):
    path = Path(hi.labels_path(str(synth_repo.historical), "maweibo"))
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "maweibo")


def test_label_byte_drift_is_refused(synth_repo):
    spec = P.FROZEN_HISTORICAL_LABELS["pheme"]
    tampered = dict(spec, bytes=spec["bytes"] + 1)
    original = P.FROZEN_HISTORICAL_LABELS
    P.FROZEN_HISTORICAL_LABELS = dict(original, pheme=tampered)
    try:
        with pytest.raises(hi.HistoricalImportRefused):
            hi.inspect_labels(str(synth_repo.historical), "pheme")
    finally:
        P.FROZEN_HISTORICAL_LABELS = original


def test_row_count_drift_is_refused(synth_repo, monkeypatch):
    spec = dict(P.FROZEN_HISTORICAL_LABELS["maweibo"], rows=999)
    monkeypatch.setitem(P.FROZEN_HISTORICAL_LABELS, "maweibo", spec)
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "maweibo")


def test_per_reader_row_drift_is_refused(synth_repo, monkeypatch):
    monkeypatch.setitem(P.FROZEN_HISTORICAL_ROWS_PER_READER, "maweibo",
                        {"qwen": 1, "mistral": 1, "internlm": 1})
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "maweibo")


def test_unknown_reader_is_refused(synth_repo):
    path = Path(hi.labels_path(str(synth_repo.historical), "maweibo"))
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["reader"] = "glm"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    monkeypatch_pin(synth_repo, "maweibo")
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "maweibo")


def test_unexpected_intervention_family_is_refused(synth_repo):
    path = Path(hi.labels_path(str(synth_repo.historical), "maweibo"))
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["intervention_type"] = "I9_unknown"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    monkeypatch_pin(synth_repo, "maweibo")
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "maweibo")


def test_cutoff_drift_is_refused(synth_repo):
    path = Path(hi.labels_path(str(synth_repo.historical), "pheme"))
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["cutoff"] == 360:
            row["cutoff"] = 720
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    monkeypatch_pin(synth_repo, "pheme")
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(synth_repo.historical), "pheme")


def monkeypatch_pin(synth_repo, dataset):
    """Re-pin the synthetic cache so only the *content* check can fail."""
    path = synth_repo.historical / "utility_labels" / dataset / "labels.jsonl"
    P.FROZEN_HISTORICAL_LABELS[dataset] = dict(
        P.FROZEN_HISTORICAL_LABELS[dataset],
        sha256=hi.sha256_file(str(path)), bytes=path.stat().st_size)
    P.FROZEN_HISTORICAL_ROWS_PER_READER[dataset] = {
        r: sum(1 for line in open(path, encoding="utf-8")
               if line.strip() and json.loads(line)["reader"] == r)
        for r in WRITERS}


def test_missing_label_cache_is_refused(tmp_path):
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(tmp_path), "maweibo")


def test_inspect_manifests_pins_every_file(synth_repo):
    manifests = hi.inspect_manifests(str(synth_repo.historical))
    assert set(manifests) == set(P.FROZEN_HISTORICAL_MANIFEST_SHA256)
    for rel, entry in manifests.items():
        assert entry["sha256"] == P.FROZEN_HISTORICAL_MANIFEST_SHA256[rel]


def test_manifest_drift_is_refused(synth_repo):
    path = synth_repo.historical / "manifests" / "maweibo" / "source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["kind"] = "tampered"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_manifests(str(synth_repo.historical))


def test_event_split_contract(synth_repo):
    split = hi.inspect_event_split(str(synth_repo.historical), "maweibo")
    assert split["split_seed"] == P.SEED
    assert split["sizes"]["foundation_train"] == 80
    assert len(split["viable_event_ids"]) == 84


def test_event_split_seed_drift_is_refused(synth_repo):
    path = synth_repo.historical / "manifests" / "pheme" / "event_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["split_seed"] = 1
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    P.FROZEN_HISTORICAL_MANIFEST_SHA256[
        "manifests/pheme/event_split.json"] = hi.canonical_sha256_file(
        str(path))
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_event_split(str(synth_repo.historical), "pheme")


def test_reader_contract_is_exact(synth_repo):
    contract = hi.inspect_readers(str(synth_repo.historical), "maweibo")
    assert sorted(contract["readers"]) == sorted(P.READER_KEYS)
    for key, model_id in contract["readers"].items():
        assert model_id == P.READER_MODEL_IDS[key]
    assert tuple(contract["cutoffs"]) == P.CUTOFFS_MIN


def test_reader_panel_drift_is_refused(synth_repo):
    path = synth_repo.historical / "manifests" / "maweibo" / "hashes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readers"]["glm"] = {"model_id": "zai-org/glm-4-9b-chat-hf"}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    P.FROZEN_HISTORICAL_MANIFEST_SHA256[
        "manifests/maweibo/hashes.json"] = hi.canonical_sha256_file(str(path))
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_readers(str(synth_repo.historical), "maweibo")


def test_build_historical_identity_structure(synth_repo):
    identity = hi.build_historical_identity(
        str(synth_repo.root), str(synth_repo.historical), environment="TEST")
    assert identity["protocol"] == P.PROTOCOL_VERSION
    assert identity["baseline_commit"] == P.BASELINE_COMMIT
    assert set(identity["datasets"]) == set(P.DATASETS)
    for dataset in P.DATASETS:
        entry = identity["datasets"][dataset]
        assert set(entry) == {"labels", "manifests", "event_split",
                              "reader_contract"}
        assert entry["labels"]["sha256"] == \
            P.FROZEN_HISTORICAL_LABELS[dataset]["sha256"]
    assert identity["read_only_namespaces"] == list(P.HISTORICAL_NAMESPACES)
    assert identity["frozen_constants"]["protocol_version"] == "bcr_v1"


def test_reused_dependency_digests_cover_the_repo():
    from .conftest import REPO_DIR
    digests = hi.reused_dependency_digests(str(REPO_DIR))
    assert set(digests) == set(P.REUSED_CR_TSER_MODULES)
    for rel, digest in digests.items():
        assert len(digest) == 64, rel


def test_missing_reused_dependency_is_refused(tmp_path):
    with pytest.raises(hi.HistoricalImportRefused):
        hi.reused_dependency_digests(str(tmp_path))


def test_canonical_hash_is_line_ending_independent(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert hi.sha256_file(str(lf)) != hi.sha256_file(str(crlf))
    assert hi.canonical_sha256_file(str(lf)) == \
        hi.canonical_sha256_file(str(crlf))


def test_write_json_is_lf_and_round_trips(tmp_path):
    target = tmp_path / "out" / "x.json"
    hi.write_json(str(target), {"a": 1})
    assert b"\r\n" not in target.read_bytes()
    assert hi.read_json(str(target)) == {"a": 1}


def test_unknown_dataset_is_refused(tmp_path):
    with pytest.raises(hi.HistoricalImportRefused):
        hi.inspect_labels(str(tmp_path), "weibo22")
