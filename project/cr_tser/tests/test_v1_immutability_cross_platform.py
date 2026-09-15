"""Cross-platform V1 immutability regression tests (synthetic).

The V1 historical pins must be satisfied by the same repository state on a
CRLF Windows checkout (``core.autocrlf=true``) and on an LF Linux checkout,
while any real content edit must still fail. Identity is therefore the
canonical LF digest, not the raw worktree bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import cr_tser_verify_pilot as V  # noqa: E402


def _mirror_v1(dest: Path, line_ending: bytes = b"\n") -> None:
    """Copy the pinned V1 files into ``dest`` using a chosen line ending."""
    for rel in V.V1_FROZEN_ARTIFACTS:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")
        if line_ending == b"\r\n":
            data = data.replace(b"\n", b"\r\n")
        target.write_bytes(data)


def _status(report):
    return {c["check"]: c["status"] for c in report.checks}


# --------------------------------------------------------------------------
# canonical identity
# --------------------------------------------------------------------------
def test_canonical_bytes_normalizes_crlf():
    assert V._canonical_bytes(b"a\r\nb\r\n") == b"a\nb\n"
    assert V._canonical_bytes(b"a\nb\n") == b"a\nb\n"
    assert V._canonical_bytes(b"a\rb") == b"a\rb"


def test_canonical_sha256_is_line_ending_independent(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n "issues": 0\n}\n')
    crlf.write_bytes(b'{\r\n "issues": 0\r\n}\r\n')
    assert V._canonical_sha256(lf) == V._canonical_sha256(crlf)


def test_canonical_sha256_detects_content_change(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(b'{\n "issues": 0\n}\n')
    b.write_bytes(b'{\r\n "issues": 1\r\n}\r\n')
    assert V._canonical_sha256(a) != V._canonical_sha256(b)


def test_dir_digest_ignores_line_endings(tmp_path):
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.json").write_bytes(b"{}\n")
    (root / "sub" / "b.json").write_bytes(b'{\n "x": 1\n}\n')
    first = V._dir_sha256(str(root))
    (root / "sub" / "b.json").write_bytes(b'{\r\n "x": 1\r\n}\r\n')
    assert V._dir_sha256(str(root)) == first


def test_dir_digest_detects_content_change(tmp_path):
    root = tmp_path / "tree"
    root.mkdir(parents=True)
    (root / "a.json").write_bytes(b'{\n "x": 1\n}\n')
    first = V._dir_sha256(str(root))
    (root / "a.json").write_bytes(b'{\n "x": 2\n}\n')
    assert V._dir_sha256(str(root)) != first


# --------------------------------------------------------------------------
# the live pins
# --------------------------------------------------------------------------
def test_pinned_hashes_equal_canonical_hashes_of_real_files():
    for rel, want in V.V1_FROZEN_ARTIFACTS.items():
        assert V._canonical_sha256(REPO / rel) == want, rel


def test_v1_verifier_dir_pin_matches_live_tree():
    got = V._dir_sha256(str(REPO / V.V1_FROZEN_VERIFIER_DIR))
    assert got == V.V1_FROZEN_VERIFIER_DIR_SHA256


# --------------------------------------------------------------------------
# the check itself, on both line endings and under tampering
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_v1_immutability_passes_on_either_line_ending(tmp_path, monkeypatch,
                                                      ending):
    _mirror_v1(tmp_path, ending)
    monkeypatch.setattr(V, "REPO", tmp_path)
    report = V.Report()
    V._verify_v1_historical_immutable(report)
    status = _status(report)
    assert status, "no V1 checks ran"
    assert all(s == "pass" for s in status.values()), status


def test_v1_immutability_fails_on_tampered_artifact(tmp_path, monkeypatch):
    _mirror_v1(tmp_path, b"\n")
    target = tmp_path / "results/cr_tser/verifier/code_verify.json"
    data = target.read_bytes()
    tampered = data.replace(b'"issues": 0', b'"issues": 7')
    assert tampered != data, "fixture did not contain the tamper token"
    target.write_bytes(tampered)
    monkeypatch.setattr(V, "REPO", tmp_path)
    report = V.Report()
    V._verify_v1_historical_immutable(report)
    status = _status(report)
    assert status["v1_historical_immutable:code_verify.json"] == "fail"
    assert status["v1_verifier_dir_immutable"] == "fail"


def test_v1_immutability_fails_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "REPO", tmp_path)
    report = V.Report()
    V._verify_v1_historical_immutable(report)
    assert all(s == "fail" for s in _status(report).values())
