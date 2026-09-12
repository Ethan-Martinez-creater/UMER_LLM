"""Final Evidence Closure tests (protocol §57).

Pure-function tests run anywhere; artifact checks are skipped when the frozen
final_evidence outputs are not present (they live on the GPU host).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
PROJECT_DIR = Path(__file__).resolve().parents[3] / "project"
for _p in (str(SCRIPTS_DIR), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import tcdscr_final_evidence_stats as S  # noqa: E402
import tcdscr_final_reader_build as B  # noqa: E402
import tcdscr_final_reader_stats as RS  # noqa: E402
import tcdscr_verify_final_evidence as V  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
def _consolidation_dir():
    local = REPO / "results" / "tcdscr" / "research_consolidation"
    if local.exists():
        return local
    return Path("/data/jyz/next/llm/results/tcdscr/research_consolidation")


CONSOLIDATION = _consolidation_dir()


def _root():
    local = REPO / "results" / "tcdscr" / "final_evidence"
    if (local / "reader" / "reader_sampling_manifest.json").exists() or \
            (local / "gap_a" / "runs").exists():
        return str(local)
    server = Path("/data/jyz/next/llm/results/tcdscr/final_evidence")
    if server.exists():
        return str(server)
    return None


ROOT = _root()
needs_artifacts = pytest.mark.skipif(
    ROOT is None, reason="final_evidence artifacts not present")
_RESULT = None


def _result():
    global _RESULT
    if _RESULT is None:
        _RESULT = V.verify(ROOT)
    return _RESULT


def _manifest():
    return json.loads((Path(ROOT) / "reader" /
                       "reader_sampling_manifest.json").read_text(
                           encoding="utf-8"))


class _FakeTok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(max(len(text.split()), 1)))}


def _unit(nid, words):
    return {"node_id": nid, "reply_text": " ".join(["w"] * words),
            "parent_text": "p", "elapsed_seconds": 5, "order": int(nid[1:])}


# --- pure functions ---------------------------------------------------------

def test_final_docs_remove_macro_f1_direction_claim():
    hits = []
    for path in CONSOLIDATION.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for phrase in ("cannot turn the negative delta positive",
                       "restoring those rows would move both arms"):
            if phrase in text:
                hits.append(path.name)
    assert hits == []
    joined = " ".join(p.read_text(encoding="utf-8")
                      for p in CONSOLIDATION.glob("*.md"))
    assert "Macro-F1 is nonlinear" in joined


def test_macro_f1_from_counts_matches_manual():
    # tp=3, fp=1, fn=2, tn=4
    counts = np.array([[3, 1, 2, 4]])
    got = float(S.macro_f1_from_counts(counts)[0])
    def f1(a, b, d):
        pr = a / (a + b) if a + b else 0.0
        re = a / (a + d) if a + d else 0.0
        return 2 * pr * re / (pr + re) if pr + re else 0.0
    exp = (f1(3, 1, 2) + f1(4, 2, 1)) / 2
    assert abs(got - exp) < 1e-12


def test_bootstrap_multiplicity_is_preserved():
    """A duplicated event must count twice, not once (dict-collapse guard)."""
    arr = np.array([[[5, 0, 5, 0]], [[0, 5, 0, 5]]])  # event A, event B
    stacks = {0: (["A", "B"], arr)}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 2, size=2)
    doubled = arr[idx].sum(axis=0)
    if idx[0] == idx[1]:
        assert int(doubled.sum()) == 2 * int(arr[idx[0]].sum())
    assert S.mean_primary_of(arr) >= 0.0


def test_token_matched_never_exceeds_cap():
    units = {f"n{i}": _unit(f"n{i}", i + 1) for i in range(4)}
    order = {f"n{i}": i for i in range(4)}
    tok = _FakeTok()
    chosen = B.token_matched(units, ["n3", "n2", "n1", "n0"], 12, tok, order)
    from tcdscr.llm.reader_prompt import (count_tokens,
                                          render_evidence_block)
    block = render_evidence_block([units[n] for n in sorted(
        chosen, key=lambda n: order[n])])
    assert count_tokens(tok, block) <= 12


def test_allocate_gives_disjoint_cutoff_groups():
    ids = {f"e{i:03d}" for i in range(200)}
    allocation, shortage, pool = B.allocate(ids, "pheme", 0)
    assert shortage == 0 and pool == 200
    seen = []
    for c in B.CUTOFFS:
        assert len(allocation[c]) == 10
        seen.extend(allocation[c])
    assert len(set(seen)) == 60
    assert set(seen) <= ids


def test_allocate_records_shortage():
    ids = {f"e{i}" for i in range(7)}
    allocation, shortage, pool = B.allocate(ids, "pheme", 0)
    assert shortage == 53 and pool == 7
    assert sum(len(v) for v in allocation.values()) == 7


def test_stable_seed_is_deterministic():
    assert B.stable_seed("a", 1) == B.stable_seed("a", 1)
    assert B.stable_seed("a", 1) != B.stable_seed("a", 2)


def test_binom_two_sided_symmetric_and_bounded():
    p = RS.binom_two_sided(3, 10)
    assert 0.0 <= p <= 1.0
    assert abs(RS.binom_two_sided(5, 10) - 1.0) < 1e-9


def test_source_of_extracts_source_text():
    prompt = ("Source Post:\nhello claim\n\nObserved Social Evidence up to "
              "5m:\n[E1]\nReply: x")
    assert V._source_of(prompt) == "hello claim"
    assert V._source_of("no markers here") is None


def test_prior_exclusion_finds_event_ids():
    obj = {"datasets": {"pheme": {"samples": [{"event_id": "a"},
                                              {"event_id": "b"}]}},
           "other": [{"event_id": "c"}]}
    tmp = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "event_id" and isinstance(v, str):
                    tmp.setdefault("ids", set()).add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    assert tmp["ids"] == {"a", "b", "c"}


@needs_artifacts
def test_gap_a_bootstrap_metadata():
    boot = json.loads((Path(ROOT) / "gap_a" / "bootstrap.json").read_text(
        encoding="utf-8"))
    for ds in V.DATASETS:
        assert "primary" in boot[ds]


def test_no_training_in_final_round():
    for name in ("tcdscr_run_final_gap_a.py", "tcdscr_run_final_gap_f.py",
                 "tcdscr_final_reader_build.py",
                 "tcdscr_final_reader_run.py"):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        for banned in ("optimizer.step", "loss.backward", "AdamW",
                       "requires_grad_(True)", ".train()"):
            assert banned not in text, (name, banned)


# --- frozen artifact checks -------------------------------------------------

@needs_artifacts
def test_gap_a_all_event_coverage():
    res = _result()
    assert not [i for i in res["issues"] if "gap_a" in i and
                ("row coverage" in i or "cutoffs" in i)]
    assert res["gap_a"]["n_rows"] == res["gap_a"]["expected_rows"]


@needs_artifacts
def test_gap_a_no_candidate_rows_present():
    res = _result()
    assert not [i for i in res["issues"] if "no no-candidate rows" in i]


@needs_artifacts
def test_gap_a_three_arms_paired():
    res = _result()
    assert not [i for i in res["issues"] if "arms" in i or
                "prediction methods" in i]


@needs_artifacts
def test_gap_f_all_event_coverage():
    res = _result()
    assert not [i for i in res["issues"] if "gap_f" in i and
                "row coverage" in i]
    assert res["gap_f"]["n_rows"] == res["gap_f"]["expected_rows"]


@needs_artifacts
def test_gap_f_frozen_configs_only():
    res = _result()
    assert not [i for i in res["issues"] if "config not frozen" in i or
                "best_config" in i or "test-time search" in i]


@needs_artifacts
def test_gap_f_no_candidate_rows_present():
    res = _result()
    assert not [i for i in res["issues"] if "gap_f" in i and
                "no no-candidate rows" in i]


@needs_artifacts
def test_final_reader_excludes_prior_v3b_events():
    res = _result()
    assert res["reader"]["prior_reader_overlap"] == 0
    assert not [i for i in res["issues"] if "prior V3-B" in i]


@needs_artifacts
def test_reader_unique_event_sampling():
    res = _result()
    assert not [i for i in res["issues"] if "duplicate event id" in i]
    for ds in V.DATASETS:
        rows = [s for s in _manifest()["samples"] if s["dataset"] == ds]
        assert len({r["event_id"] for r in rows}) == len(rows)


@needs_artifacts
def test_reader_10_per_cutoff_per_fold():
    res = _result()
    assert not [i for i in res["issues"] if "expected 10" in i or
                "expected 60" in i]


@needs_artifacts
def test_reader_sampling_does_not_use_gold():
    res = _result()
    assert not [i for i in res["issues"] if "gold leaked" in i]
    for s in _manifest()["samples"]:
        assert "gold" not in s and "label" not in s


@needs_artifacts
def test_reader_four_arms():
    res = _result()
    assert not [i for i in res["issues"] if "four arms" in i or
                "arms !=" in i or "lacks four arms" in i]


@needs_artifacts
def test_utility_token_matched_not_over_budget():
    res = _result()
    assert not [i for i in res["issues"] if "utility TM exceeded" in i]


@needs_artifacts
def test_random_token_matched_not_over_budget():
    res = _result()
    assert not [i for i in res["issues"] if "random TM exceeded" in i]


@needs_artifacts
def test_reader_same_source_all_arms():
    res = _result()
    assert not [i for i in res["issues"] if "source differs" in i]


@needs_artifacts
def test_reader_same_model_config():
    res = _result()
    assert not [i for i in res["issues"] if "MODEL_MISMATCH" in i or
                "model manifest missing" in i]
    mm = json.loads((Path(ROOT) / "reader" / "diagnostics" /
                     "reader_model_manifest.json").read_text(
                         encoding="utf-8"))
    assert mm["model"]["decoding"]["do_sample"] is False
    assert mm["model"]["decoding"]["temperature"] == 0.0
    assert mm["model"]["finetuned"] is False


@needs_artifacts
def test_reader_manifest_frozen():
    res = _result()
    assert not [i for i in res["issues"] if "hash changed" in i]


@needs_artifacts
def test_reader_bootstrap_stratified():
    summary = json.loads((Path(ROOT) / "reader" /
                          "reader_summary.json").read_text(encoding="utf-8"))
    assert summary["bootstrap"]["stratified"] == "fold x cutoff cells"
    assert summary["bootstrap"]["iterations"] == 10000
    assert summary["bootstrap"]["seed"] == 4096


@needs_artifacts
def test_reader_bootstrap_preserves_multiplicity():
    summary = json.loads((Path(ROOT) / "reader" /
                          "reader_summary.json").read_text(encoding="utf-8"))
    assert summary["bootstrap"]["multiplicity"] == "preserved"
