"""V2 Reader Protocol Amendment R1 tests (synthetic; no real models or data).

Pins the reader-set migration (GLM -> Mistral-7B-Instruct-v0.3), the v2r1
namespace, the shared-scorer wiring of the new reader and the fail-closed
V2 -> v2r1 cache-migration utility. Nothing here loads a checkpoint, reads a
real dataset or touches the GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ..config import pilot_config as C

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import cr_tser_migrate_v2_labels as migration  # noqa: E402

R1_KEYS = ("qwen", "mistral", "internlm")
R0_KEYS = ("qwen", "glm", "internlm")
MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
GLM_ID = "zai-org/glm-4-9b-chat-hf"


# --------------------------------------------------------------------------
# 1. the frozen R1 reader set
# --------------------------------------------------------------------------
def test_reader_keys_are_the_r1_set():
    assert tuple(C.READER_KEYS) == R1_KEYS
    assert tuple(C.READER_KEYS_V2) == R0_KEYS
    assert "glm" not in C.READER_KEYS


def test_reader_model_ids_exact():
    assert C.READER_MODEL_IDS == {
        "qwen": "Qwen/Qwen3-8B",
        "mistral": MISTRAL_ID,
        "internlm": "internlm/internlm3-8b-instruct"}
    assert C.READER_MODEL_IDS_V2["glm"] == GLM_ID


def test_loro_rotations_exact():
    assert C.LORO_ROTATIONS == (("qwen", "mistral", "internlm"),
                                ("qwen", "internlm", "mistral"),
                                ("mistral", "internlm", "qwen"))
    assert {r[2] for r in C.LORO_ROTATIONS} == {"internlm", "mistral", "qwen"}
    assert not any("glm" in r for r in C.LORO_ROTATIONS)


def test_glm_absent_from_r1_execution_loops():
    for name in ("cr_tser_build_manifests.py", "cr_tser_generate_labels.py",
                 "cr_tser_train_predictors.py", "cr_tser_run_selection.py",
                 "cr_tser_run_pilot.py"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8").lower()
        assert "glm" not in text, f"{name} still reaches the retired reader"


def test_glm_stays_addressable_as_history():
    assert C.KNOWN_READER_KEYS == ("qwen", "glm", "mistral", "internlm")
    assert C.KNOWN_READER_MODEL_IDS["glm"] == GLM_ID
    assert C.KNOWN_READER_MODEL_IDS["mistral"] == MISTRAL_ID


def test_heldout_reader_contract_follows_r1():
    from ..evaluation.unseen_reader import EXPECTED_HELDOUT_READERS
    assert tuple(EXPECTED_HELDOUT_READERS) == ("internlm", "mistral", "qwen")


# --------------------------------------------------------------------------
# 2. path/env wiring and the Mistral reader
# --------------------------------------------------------------------------
def test_mistral_path_and_env_resolution():
    from ..config.pilot_config import paths_from_env
    paths = paths_from_env({"CRTSER_MISTRAL_MODEL": "/models/mistral"})
    assert paths.reader_path("mistral") == "/models/mistral"
    assert paths.reader_path("glm") == ""          # history only, never wired
    assert set(paths.reader_paths()) == set(R1_KEYS)
    with pytest.raises(ValueError):
        paths.reader_path("llama")


def test_reader_spec_accepts_r1_and_history_keys():
    from ..readers.base_reader import ReaderSpec
    assert ReaderSpec("mistral", "/models/mistral").model_id == MISTRAL_ID
    assert ReaderSpec("glm", "/models/glm").model_id == GLM_ID
    with pytest.raises(ValueError):
        ReaderSpec("llama", "/models/llama")


def test_mistral_reader_reuses_the_shared_scorer():
    import inspect

    from ..readers import base_reader
    from ..readers.base_reader import SYSTEM_PROMPT, build_messages
    from ..readers.internlm_reader import InternLMReader
    from ..readers.mistral_reader import MistralReader

    source = (REPO / "project" / "cr_tser" / "readers" /
              "mistral_reader.py").read_text(encoding="utf-8")
    assert "candidate_logprobs_hf" in source
    assert ".generate(" not in source
    assert "trust_remote_code = False" in source
    assert "torch_dtype=getattr(torch, self.spec.dtype)" in source

    # registered in the shared factory, sharing the frozen prompt + scorer
    registry = inspect.getsource(base_reader.build_reader)
    assert 'key == "mistral"' in registry and "MistralReader" in registry
    assert (MistralReader.candidate_logprobs.__code__.co_names
            == InternLMReader.candidate_logprobs.__code__.co_names)
    assert build_messages("prompt") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "prompt"}]


# --------------------------------------------------------------------------
# 3. namespaces
# --------------------------------------------------------------------------
def test_v2r1_namespace_is_separate():
    from ..config.pilot_config import paths_from_env
    assert C.V2R1_RESULTS_ROOT == "results/cr_tser_v2r1"
    assert len({C.V1_RESULTS_ROOT, C.V2_RESULTS_ROOT,
                C.V2R1_RESULTS_ROOT}) == 3
    assert paths_from_env({}).out_root == C.V2R1_RESULTS_ROOT
    assert C.DATASET_PROTOCOL_VERSION == "v2"
    assert C.READER_PROTOCOL_VERSION == "r1"
    assert C.PROTOCOL_VERSION == "v2r1"


def test_v2_history_manifests_are_pinned():
    import cr_tser_verify_pilot as V
    assert V.V2_FROZEN_MANIFEST_SHA256
    for rel, expected in V.V2_FROZEN_MANIFEST_SHA256.items():
        path = REPO / rel
        assert path.exists(), f"{rel} missing from the frozen V2 namespace"
        assert V._canonical_sha256(path) == expected, rel


def test_v2r1_verifier_protocol_is_selectable():
    import cr_tser_verify_pilot as V
    parser = V.build_parser()
    for protocol in ("v1", "v2", "v2r1"):
        assert parser.parse_args(["--protocol", protocol]).protocol == protocol
    assert parser.parse_args([]).protocol == "v2r1"
    r1 = V._expected_reader_set("v2r1")
    assert r1[1]["mistral"] == MISTRAL_ID and "glm" not in r1[1]
    assert V._expected_reader_set("v2")[1]["glm"] == GLM_ID


# --------------------------------------------------------------------------
# 4. fail-closed V2 -> v2r1 cache migration (synthetic)
# --------------------------------------------------------------------------
def _source_record():
    return {"dataset": "pheme", "kind": "pheme_raw", "path": "/raw",
            "exists": True, "sha256": "aa", "bytes": 10, "n_files": 1}


def _hashes(event_split="es"):
    return {"dataset": "pheme", "source": _source_record(),
            "event_split_sha256": event_split, "cutoffs": [15, 60, 360],
            "viable_events": 200, "n_snapshots": 3, "n_interventions": 3,
            "readers": {k: {"model_id": f"org/{k}", "weight_hash": f"w-{k}",
                            "tokenizer_hash": f"t-{k}"}
                        for k in R1_KEYS}}


def _row(reader, i=0, **overrides):
    row = {"dataset": "pheme", "event_id": f"e{i}", "cutoff": 60,
           "reader": reader, "intervention_id": "I0",
           "base_context_hash": f"b{i}", "intervened_context_hash": f"c{i}",
           "reader_hash": f"w-{reader}",
           "reader_identity_hash": f"id-{reader}",
           "tokenizer_hash": f"t-{reader}",
           "chat_template_hash": f"ct-{reader}",
           "prompt_hash": f"p{i}", "prompt_ids_hash": f"pi{i}"}
    row.update(overrides)
    return row


def _tree(root, rows=None, hashes=None, source=None):
    mdir = root / "manifests" / "pheme"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "source.json").write_text(
        json.dumps(source or _source_record()), encoding="utf-8")
    (mdir / "hashes.json").write_text(json.dumps(hashes or _hashes()),
                                      encoding="utf-8")
    if rows is not None:
        cache = root / "utility_labels" / "pheme"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "labels.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def test_migration_accepts_a_valid_synthetic_cache(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0), _row("internlm", 1),
                                  _row("glm", 2)])
    dst = _tree(tmp_path / "v2r1")
    report = migration.migrate("pheme", str(src), str(dst))
    assert report["source_rows"] == 3
    assert report["migrated_rows"] == 2
    assert report["skipped_retired_rows"] == 1
    assert report["readers_migrated"] == ["internlm", "qwen"]
    assert report["duplicate_keys"] == 0
    assert report["dry_run"] is True
    assert not (dst / "utility_labels" / "pheme" / "labels.jsonl").exists()


def test_migration_writes_only_qwen_and_internlm(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0), _row("internlm", 1),
                                  _row("glm", 2)])
    dst = _tree(tmp_path / "v2r1")
    report = migration.migrate("pheme", str(src), str(dst), apply=True)
    written = [json.loads(l) for l in
               open(report["target_cache"], encoding="utf-8") if l.strip()]
    assert {r["reader"] for r in written} == {"qwen", "internlm"}
    assert all(r["reader"] != "glm" for r in written)
    assert report["written_bytes"] > 0

    # the migrated rows are byte-identical to their source rows
    source_rows = [json.loads(l) for l in
                   open(src / "utility_labels" / "pheme" / "labels.jsonl",
                        encoding="utf-8") if l.strip()
                   and json.loads(l)["reader"] in ("qwen", "internlm")]
    assert written == source_rows


def test_migration_rejects_a_glm_only_cache(tmp_path):
    src = _tree(tmp_path / "v2", [_row("glm", 0)])
    dst = _tree(tmp_path / "v2r1")
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_duplicate_keys(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0), _row("qwen", 0)])
    dst = _tree(tmp_path / "v2r1")
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_manifest_drift(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0)])
    dst = _tree(tmp_path / "v2r1", hashes=_hashes(event_split="other"))
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_source_drift(tmp_path):
    source = dict(_source_record(), sha256="bb")
    src = _tree(tmp_path / "v2", [_row("qwen", 0)])
    dst = _tree(tmp_path / "v2r1", source=source)
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_reader_identity_drift(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0, reader_hash="w-other")])
    dst = _tree(tmp_path / "v2r1")
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_a_nonempty_target_cache(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0)])
    dst = _tree(tmp_path / "v2r1", [_row("internlm", 1)])
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(dst))


def test_migration_rejects_a_missing_target_namespace(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0)])
    empty = tmp_path / "v2r1"
    empty.mkdir()
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(empty))


def test_migration_rejects_identical_source_and_target(tmp_path):
    src = _tree(tmp_path / "v2", [_row("qwen", 0)])
    with pytest.raises(migration.MigrationRefused):
        migration.migrate("pheme", str(src), str(src))
