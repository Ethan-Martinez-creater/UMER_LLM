"""V3-B frozen-Qwen reader-transfer tests (§50).

Everything here runs without a GPU: the sampler, the prompt formatter, the
parser, the statistics and the runner's retry path are all exercised with
synthetic inputs.  The only real artifacts touched are the frozen V3-A
``best_config.json`` files, to pin alpha = 0.8.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_summarize_v3_reader as summ  # noqa: E402
import tcdscr_v3_reader_sample as sampler  # noqa: E402
import tcdscr_run_v3_reader as runner  # noqa: E402
import tcdscr_verify_v3_reader_transfer as verify  # noqa: E402

from ..llm.reader_parser import (citation_stats,  # noqa: E402
                                 parse_reader_output, reason_invalid_refs)
from ..llm.reader_prompt import (build_arm_prompt,  # noqa: E402
                                 build_user_prompt, evidence_id_map,
                                 order_units_by_snapshot,
                                 render_evidence_block)

SCRATCH = Path(__file__).resolve().parent / "_v3b_scratch"


@contextlib.contextmanager
def _scratch():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    try:
        yield SCRATCH
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def _units(specs):
    return [{"node_id": n, "reply_text": r, "parent_text": p,
             "elapsed_seconds": e, "depth": 1, "order": i}
            for i, (n, r, p, e) in enumerate(specs)]


def _v3row(fold=0, event_id="e1", cutoff=5, alpha=0.8, seed=2000,
           candidate_count=10, static_selected_count=8, gold=1,
           static_ids=("n1",), ms_ids=("n1",), fallback=False):
    return {
        "dataset": "pheme", "fold": fold, "event_id": event_id,
        "cutoff": str(cutoff), "alpha": alpha, "seed": seed,
        "gold": gold, "candidate_count": candidate_count,
        "static_selected_count": static_selected_count,
        "static_selected_node_ids": list(static_ids),
        "ms_selected_node_ids": list(ms_ids),
        "static_evidence_tokens": len(static_ids),
        "ms_evidence_tokens": len(ms_ids),
        "static_margin": 1.0, "ms_margin": 1.0,
        "dual_view_agree": True, "fallback_to_static": fallback,
    }


def _make_runs(root, dataset, rows_by_fold):
    for fold in sampler.FOLDS:
        folder = root / dataset / \
            f"fold{fold}_seed{sampler.CONTEXT_SOURCE_SEED}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = rows_by_fold.get(fold, [])
        (folder / "validation_predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _pair(sample_id="s1", gold="RUMOR",
          static_label="RUMOR", ms_label="RUMOR",
          static_social_tokens=100, ms_social_tokens=30,
          static_total_input_tokens=200, ms_total_input_tokens=130,
          fallback_to_static=False, static_margin=1.0, cutoff=30,
          pressure_bin="LOW", static_confidence=0.8, ms_confidence=0.7,
          static_correct=1, ms_correct=1, dataset="pheme", fold=0):
    return {
        "sample_id": sample_id, "dataset": dataset, "event_id": "e1",
        "cutoff": cutoff, "fold": fold, "gold": 1 if gold == "RUMOR" else 0,
        "gold_label": gold, "pressure_bin": pressure_bin,
        "fallback_to_static": fallback_to_static, "static_margin": static_margin,
        "ms_margin": static_margin, "dual_view_agree": True,
        "static_label": static_label, "ms_label": ms_label,
        "static_confidence": static_confidence, "ms_confidence": ms_confidence,
        "static_correct": static_correct, "ms_correct": ms_correct,
        "static_social_tokens": static_social_tokens,
        "ms_social_tokens": ms_social_tokens,
        "static_total_input_tokens": static_total_input_tokens,
        "ms_total_input_tokens": ms_total_input_tokens,
        "static_n_evidence": 3, "ms_n_evidence": 1,
    }


# ------------------------------------------------- 1-3 sampling / §6, §7

def test_reader_sampling_unique_event_cutoff():
    with _scratch() as root:
        _make_runs(root, "pheme", {
            0: [_v3row(fold=0, event_id="e1", cutoff=5)],
            1: [_v3row(fold=1, event_id="e1", cutoff=5)],
        })
        val = {f: {"e1"} for f in sampler.FOLDS}
        test = {f: set() for f in sampler.FOLDS}
        rows, conflicts = sampler.load_population("pheme", str(root), val,
                                                  test)
        assert len(rows) == 1
        assert len(conflicts) == 1
        assert rows[0]["fold"] == 0
        assert abs(float(rows[0]["alpha"]) - 0.8) < 1e-9
        assert int(rows[0]["seed"]) == 2000


def test_reader_sampling_uses_validation_only():
    with _scratch() as root:
        _make_runs(root, "pheme", {0: [_v3row(fold=0, event_id="evil")]})
        val = {f: {"e1"} for f in sampler.FOLDS}
        test = {f: {"evil"} for f in sampler.FOLDS}
        with pytest.raises(RuntimeError, match="TEST_PROTOCOL_VIOLATION"):
            sampler.load_population("pheme", str(root), val, test)
    with _scratch() as root:
        _make_runs(root, "pheme", {0: [_v3row(fold=0, event_id="stray")]})
        val = {f: {"e1"} for f in sampler.FOLDS}
        test = {f: set() for f in sampler.FOLDS}
        with pytest.raises(RuntimeError, match="not in fold0 validation"):
            sampler.load_population("pheme", str(root), val, test)


def test_reader_sampling_seed_fixed():
    assert sampler.SAMPLE_SEED == 3090
    assert sampler.CONTEXT_SOURCE_SEED == 2000
    rows = [_v3row(event_id=f"e{i}", cutoff=c)
            for c in sampler.CUTOFFS for i in range(3)]
    a, alloc_a, dev_a, spare_a = sampler.stratified_sample(rows)
    b, alloc_b, dev_b, spare_b = sampler.stratified_sample(rows)
    assert [r["event_id"] + str(r["cutoff"]) for r in a] == \
        [r["event_id"] + str(r["cutoff"]) for r in b]
    assert alloc_a == alloc_b and dev_a == dev_b and spare_a == spare_b


def test_stratified_sample_backfills_underpopulated_cutoff():
    rows = [_v3row(event_id=f"early{i}", cutoff=5) for i in range(2)]
    rows += [_v3row(event_id=f"late{i}", cutoff=360) for i in range(400)]
    chosen, allocation, deviation, spare = sampler.stratified_sample(
        rows, per_cutoff=50, total=100)
    assert allocation["5"] == 2
    assert allocation["360"] == 98
    assert spare == 0
    assert len(chosen) == 100
    assert deviation["5"] == -48


def test_pressure_bin_thresholds():
    assert sampler.pressure_bin(0, 0) == "NO_CANDIDATE"
    assert sampler.pressure_bin(10, 8) == "LOW"
    assert sampler.pressure_bin(10, 4) == "MEDIUM"
    assert sampler.pressure_bin(10, 3) == "HIGH"


def test_arm_order_deterministic_and_both_orders_seen():
    orders = {sampler.arm_order("pheme", f"e{i}", 5)
              for i in range(40)}
    assert orders == {"static_first", "ms_first"}
    assert sampler.arm_order("pheme", "e1", 5) == \
        sampler.arm_order("pheme", "e1", 5)


# ------------------------------------------- 4-7 paired prompt construction

def test_static_ms_pair_same_source():
    u1 = _units([("n1", "r1", "p1", 10)])
    u2 = _units([("n2", "r2", "p2", 20)])
    static_prompt = build_user_prompt("SRC TEXT", 30,
                                      render_evidence_block(u1))
    ms_prompt = build_user_prompt("SRC TEXT", 30, render_evidence_block(u2))
    assert verify._source_text(static_prompt) == "SRC TEXT"
    assert verify._source_text(static_prompt) == \
        verify._source_text(ms_prompt)


def test_static_ms_pair_only_context_differs():
    u1 = _units([("n1", "reply one", "p1", 10)])
    u2 = _units([("n2", "reply two", "p2", 20)])
    static_prompt = build_user_prompt("SRC", 30, render_evidence_block(u1))
    ms_prompt = build_user_prompt("SRC", 30, render_evidence_block(u2))
    hs, bs, ts = verify._split_prompt(static_prompt)
    hm, bm, tm = verify._split_prompt(ms_prompt)
    assert hs == hm and ts == tm
    assert bs != bm
    assert "reply one" in bs and "reply two" in bm


def test_ms_prompt_matches_frozen_v3_selection():
    units = _units([("n1", "r1", "p1", 10), ("n2", "r2", "p2", 20),
                    ("n3", "r3", "p3", 30)])
    selected = ["n3", "n1"]
    ordered = order_units_by_snapshot(units, selected)
    mapping = evidence_id_map(ordered)
    assert [mapping["E1"], mapping["E2"]] == ["n1", "n3"]
    assert sorted(mapping.values()) == sorted(selected)
    block = render_evidence_block(ordered)
    assert "Observed: +10s" in block and "Observed: +30s" in block
    assert block.index("+10s") < block.index("+30s")


def test_fallback_prompts_identical():
    units = _units([("n1", "r1", "p1", 10), ("n2", "r2", "p2", 20)])
    ids = ["n1", "n2"]
    a = build_user_prompt("SRC", 30,
                          render_evidence_block(
                              order_units_by_snapshot(units, ids)))
    b = build_user_prompt("SRC", 30,
                          render_evidence_block(
                              order_units_by_snapshot(units, ids)))
    assert a == b


# ------------------------------------------------------ 8-10 parser / §18

def test_json_parser_valid_output():
    obj, errors = parse_reader_output(
        '{"label": "RUMOR", "confidence": 0.73, '
        '"evidence_ids": ["E2", "E5"], "reason": "because"}')
    assert errors == []
    assert obj["label"] == "RUMOR"
    assert obj["confidence"] == pytest.approx(0.73)
    assert obj["evidence_ids"] == ["E2", "E5"]
    fenced, errors = parse_reader_output(
        '```json\n{"label": "NON_RUMOR", "confidence": 0, '
        '"evidence_ids": [], "reason": "none"}\n```')
    assert errors == [] and fenced["label"] == "NON_RUMOR"
    assert fenced["confidence"] == 0.0
    bad, errors = parse_reader_output('{"classification": "NON_RUMOR", '
                                      '"confidence": 0.5}')
    assert bad is None and errors


class _ScriptedReader:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.outputs.pop(0)


def test_json_parser_one_retry_only():
    good = ('{"label": "NON_RUMOR", "confidence": 0.5, '
            '"evidence_ids": [], "reason": "ok"}')
    reader = _ScriptedReader(["not json at all", good])
    res = runner.run_arm(reader, {"prompt": "P"}, 1)
    assert res["parse_status"] == "retry_ok"
    assert res["retry_used"] is True
    assert len(reader.calls) == 2
    assert reader.calls[1].endswith(runner.RETRY_SUFFIX)
    reader2 = _ScriptedReader(["nope", "still nope"])
    res2 = runner.run_arm(reader2, {"prompt": "P"}, 1)
    assert res2["parse_status"] == "PARSE_FAILURE"
    assert res2["parsed"] is None
    assert len(reader2.calls) == 2


def test_invalid_evidence_id_detected():
    stats = citation_stats(["E1", "E9"], ["E1", "E2"])
    assert stats["valid"] == ["E1"]
    assert stats["invalid"] == ["E9"]
    assert stats["n_cited"] == 2
    assert reason_invalid_refs("see E9 and E1", ["E1"]) == ["E9"]
    assert reason_invalid_refs("no ids here", ["E1"]) == []


def test_context_overflow_is_flagged_not_truncated():
    assert runner.token_limit_exceeded(100, 50) is True
    assert runner.token_limit_exceeded(50, 50) is False
    assert runner.token_limit_exceeded(10 ** 9, None) is False
    overflow = {"raw_output": None, "raw_output_retry": None,
                "retry_used": False, "retry_prompt_hash": None,
                "parsed": None, "errors": ["CONTEXT_OVERFLOW"],
                "parse_status": "CONTEXT_OVERFLOW", "latency_s": 0.0}
    row = runner.parsed_row(
        {"sample_id": "s", "dataset": "pheme", "event_id": "e",
         "cutoff": "5", "fold": 0, "gold": 1}, "static",
        {"evidence_ids": {}}, overflow)
    assert row["parse_failure"] is True
    assert row["parsed_label"] is None
    assert row["correct"] is None


# ------------------------------------------------ 11-12 statistics / §32

def test_bootstrap_preserves_multiplicity():
    draws = summ.resample_indices(5, 200, 3090)
    assert len(draws) == 200
    assert all(len(d) == 5 for d in draws)
    assert all(i < 5 for d in draws for i in d)
    assert any(len(set(d)) < len(d) for d in draws), \
        "resampling must draw with replacement"
    assert any(d.count(x) > 1 for d in draws for x in set(d))
    pairs = [_pair(sample_id=f"s{i}",
                   static_label="RUMOR" if i % 3 else "NON_RUMOR",
                   ms_label="NON_RUMOR" if i % 5 == 0 else "RUMOR")
             for i in range(40)]
    a = summ.paired_bootstrap_delta(pairs, iters=200, seed=3090)
    b = summ.paired_bootstrap_delta(pairs, iters=200, seed=3090)
    assert a == b
    assert a["ci_low"] <= a["ci_high"]


def test_context_token_reduction_excludes_zero_static_context():
    pairs = [_pair(sample_id="zero", static_social_tokens=0,
                   ms_social_tokens=0, static_total_input_tokens=50,
                   ms_total_input_tokens=50),
             _pair(sample_id="full", static_social_tokens=100,
                   ms_social_tokens=25)]
    mech = summ.compression_metrics(pairs)
    assert mech["n_eligible"] == 1
    assert mech["n_excluded_zero_static"] == 1
    assert mech["mean_social_token_reduction"] == pytest.approx(0.75)


def test_mcnemar_and_ece_sanity():
    assert summ.binom_two_sided_p(0, 0) == 1.0
    assert summ.binom_two_sided_p(5, 5) == pytest.approx(1.0)
    assert summ.binom_two_sided_p(0, 10) < 0.01
    ece = summ.expected_calibration_error(
        [(0.9, True)] * 9 + [(0.1, False)] * 1)
    assert 0.0 <= ece <= 0.2


def test_paired_outcomes_and_disagreement_decomposition():
    pairs = [_pair(sample_id="a", gold="RUMOR", static_label="RUMOR",
                   ms_label="RUMOR"),
             _pair(sample_id="b", gold="RUMOR", static_label="NON_RUMOR",
                   ms_label="RUMOR"),
             _pair(sample_id="c", gold="RUMOR", static_label="RUMOR",
                   ms_label="NON_RUMOR"),
             _pair(sample_id="d", gold="RUMOR", static_label="NON_RUMOR",
                   ms_label="NON_RUMOR")]
    out = summ.paired_outcomes(pairs)
    assert out["both_correct"] == 1 and out["both_wrong"] == 1
    assert out["wrong_to_correct"] == 1 and out["correct_to_wrong"] == 1
    assert out["net_correction_gain"] == 0
    assert out["disagreement"] == 2
    assert out["label_changed_static_to_correct"] == 1
    assert out["label_changed_static_to_wrong"] == 1


# ---------------------------------------------- 13-15 protocol invariants

def test_qwen_not_used_for_selection():
    src = (SCRIPTS_DIR / "tcdscr_v3_reader_sample.py").read_text(
        encoding="utf-8")
    for banned in ("generate(", "QwenRumorLLM", "AutoModelForCausalLM",
                   "raw_generations", "parsed/"):
        assert banned not in src, f"sampler must not touch {banned}"
    assert "AutoTokenizer" in src, "sampler only needs the tokenizer"


def _v3_artifacts_root():
    """Frozen V3-A artifacts, in the checkout or on the training server."""
    local = Path(__file__).resolve().parents[3] / "results" / "tcdscr" / \
        "dynamic_v3"
    if local.exists():
        return local
    server = Path("/data/jyz/next/llm/results/tcdscr/dynamic_v3")
    if server.exists():
        return server
    return None


def test_alpha_frozen_08():
    assert sampler.ALPHA == 0.8
    assert sampler.BUDGET == 1024
    root = _v3_artifacts_root()
    if root is None:
        pytest.skip("frozen V3-A artifacts not present")
    for ds in ("pheme", "maweibo"):
        for fold in range(5):
            path = root / ds / f"fold{fold}" / "best_config.json"
            assert path.exists(), f"missing {path}"
            best = json.loads(path.read_text(encoding="utf-8"))
            assert best["alpha"] == pytest.approx(0.8)
            assert best["budget"] == 1024


def test_reader_gate_logic():
    gate, conds, label = summ.dataset_gate(-0.001, 0.40, 0.001, -0.005)
    assert gate == "PASS" and label == "CLEAR_TRANSFER"
    assert all(conds.values())
    gate, conds, label = summ.dataset_gate(-0.001, 0.40, 0.001, -0.02)
    assert gate == "PASS" and label == "WEAK_TRANSFER"
    gate, conds, label = summ.dataset_gate(-0.02, 0.40, 0.001, -0.02)
    assert gate == "FAIL" and label == "TRANSFER_FAIL"
    assert not conds["reader_non_inferiority"]
    gate, _, label = summ.dataset_gate(-0.001, 0.10, 0.0, -0.005)
    assert gate == "FAIL" and label == "TRANSFER_FAIL"
    gate, _, label = summ.dataset_gate(-0.001, 0.40, 0.05, -0.005)
    assert gate == "FAIL" and label == "TRANSFER_FAIL"
    gate, conds, _ = summ.dataset_gate(-0.001, 0.40, None, -0.005)
    assert gate == "PASS" and conds["grounding_integrity"]


def test_overall_decision():
    assert summ.overall_decision({"pheme": {"gate": "PASS"},
                                  "maweibo": {"gate": "PASS"}})[1] == \
        "START_V3_C_HELD_OUT_TEST"
    assert summ.overall_decision({"pheme": {"gate": "PASS"},
                                  "maweibo": {"gate": "FAIL"}})[1] == \
        "STOP_FOR_RESEARCH_REVIEW"
    assert summ.overall_decision({"pheme": {"gate": "FAIL"},
                                  "maweibo": {"gate": "FAIL"}})[1] == \
        "PROXY_READER_TRANSFER_FAIL"
    assert summ.overall_decision({})[1] == "PROXY_READER_TRANSFER_FAIL"


def test_verifier_ordering_accepts_prompts_written_before_manifest():
    """Prompts and manifest are frozen together, so only the generations must
    come last; comparing prompts against the manifest was a verifier bug."""
    with _scratch() as root:
        (root / "prompts").mkdir()
        (root / "raw_generations").mkdir()
        for ds in ("pheme", "maweibo"):
            (root / "prompts" / f"{ds}.jsonl").write_text("", encoding="utf-8")
            (root / "raw_generations" / f"{ds}.jsonl").write_text(
                "", encoding="utf-8")
        (root / "sampling_manifest.json").write_text("{}", encoding="utf-8")
        base = time.time() - 1000
        for ds in ("pheme", "maweibo"):
            os.utime(root / "prompts" / f"{ds}.jsonl", (base, base))
        os.utime(root / "sampling_manifest.json", (base + 10, base + 10))
        for ds in ("pheme", "maweibo"):
            os.utime(root / "raw_generations" / f"{ds}.jsonl",
                     (base + 20, base + 20))
        issues = []
        verify._check_manifest_precedes_generations(str(root), issues)
        assert issues == []
        os.utime(root / "raw_generations" / "pheme.jsonl",
                 (base - 100, base - 100))
        issues = []
        verify._check_manifest_precedes_generations(str(root), issues)
        assert any("predate the frozen manifest" in i for i in issues)
