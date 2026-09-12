"""V3-B failure-diagnosis tests (§30).

Pure-function tests plus two reproduction checks against the frozen V3-B
artifacts.  No GPU and no Qwen model are needed: the diagnosis step only reads
existing generations.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_diagnose_v3b_failure as D  # noqa: E402
import tcdscr_verify_v3b_failure_diagnosis as V  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / "_v3b_diag_scratch"


def _reader_root():
    local = Path(__file__).resolve().parents[3] / "results" / "tcdscr" / \
        "dynamic_v3_reader"
    if local.exists():
        return local
    server = Path("/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader")
    if server.exists():
        return server
    return None


@contextlib.contextmanager
def _scratch():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    try:
        yield SCRATCH
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def _sample(gold="RUMOR", st="RUMOR", ms="RUMOR", st_ok=True, ms_ok=True,
            dataset="pheme", loss_rate=0.0, reduction=0.5):
    return {
        "dataset": dataset, "gold_label": gold, "static_label": st,
        "ms_label": ms, "static_correct": st_ok, "ms_correct": ms_ok,
        "group": D.outcome_group(st_ok, ms_ok),
        "token_reduction_ratio": reduction,
        "evidence_count_reduction_ratio": reduction,
        "reader_evidence_loss_rate": loss_rate,
        "reader_evidence_loss_count": int(loss_rate > 0),
        "static_cited_node_ids": ["a"], "ms_evidence_node_ids": ["b"],
        "static_selected_count": 4, "ms_selected_count": 2,
        "evidence_count_delta": 2, "cutoff": 30, "fold": 0,
        "pressure_bin": "LOW", "static_margin": 2.0, "ms_margin": 1.9,
        "proxy_margin_retention": 0.95,
        "qwen_confidence_delta": -0.05, "transition_code": 0,
        "static_confidence": 0.7, "ms_confidence": 0.65,
        "static_cited_removed_rate": 0.0, "static_cited_removed_count": 0,
        "has_reader_evidence_loss": loss_rate > 0,
        "removed_node_ids": [], "retained_node_ids": [],
        "static_evidence_node_ids": ["a"], "ms_only_node_ids": [],
    }


# ------------------------------------------------ reproduction / §4, §32

def test_outcome_group_counts_reproduce_v3b():
    root = _reader_root()
    if root is None:
        pytest.skip("frozen V3-B artifacts not present")
    expected = V._expected_groups(str(root))
    groups = {}
    for ds in ("pheme", "maweibo"):
        stats = json.loads((root / "statistics" / f"{ds}.json").read_text(
            encoding="utf-8"))
        out = stats["paired_outcomes"]
        groups[ds] = {"CC": out["both_correct"], "CW": out["correct_to_wrong"],
                      "WC": out["wrong_to_correct"],
                      "WW": out["both_wrong"]}
        assert expected[ds]["CC"] == groups[ds]["CC"]
        assert expected[ds]["CW"] == groups[ds]["CW"]
        assert expected[ds]["WC"] == groups[ds]["WC"]
        assert expected[ds]["WW"] == groups[ds]["WW"]
        assert sum(groups[ds].values()) == 300
    assert groups["pheme"]["CW"] == 17 and groups["pheme"]["WC"] == 23
    assert groups["maweibo"]["CW"] == 13 and groups["maweibo"]["WC"] == 10


def test_diagnosis_uses_frozen_manifest():
    src = (SCRIPTS_DIR / "tcdscr_diagnose_v3b_failure.py").read_text(
        encoding="utf-8")
    # the diagnosis writes only into the diagnosis output root
    assert "reader_root, \"w\"" not in src
    assert "reader_root, 'w'" not in src
    assert "open(os.path.join(args.reader_root" not in src
    assert "sampling_manifest.json\"), \"w\"" not in src


# --------------------------------------------------- definitions §8, §9

def test_reader_evidence_loss_definition():
    assert D.reader_evidence_loss(["a", "b"], ["b", "c"]) == ["a"]
    assert D.reader_evidence_loss(["a", "b"], ["a", "b"]) == []
    assert D.reader_evidence_loss([], ["a"]) == []
    assert D.reader_evidence_loss(["a", "a"], ["b"]) == ["a", "a"]


def test_cited_removed_mapping():
    assert D.cited_removed_mapping(["a", "b"], ["b"]) == ["a"]
    assert D.cited_removed_mapping(["a"], ["a", "z"]) == []
    assert D.cited_removed_mapping([], ["a"]) == []


# ------------------------------------------------------- bins §6, §11

def test_evidence_count_bins_fixed():
    assert [D.evidence_count_bin(c) for c in (0, 1, 2, 3, 4, 5, 8, 9, 40)] == \
        ["0", "1", "2", "3-4", "3-4", "5-8", "5-8", "9+", "9+"]
    assert [D.selected_count_delta_bin(d)
            for d in (0, -1, 1, 2, 3, 5, 6, 10, 11)] == \
        ["0", "0", "1-2", "1-2", "3-5", "3-5", "6-10", "6-10", "11+"]


def test_proxy_margin_retention_bins():
    assert D.margin_retention_bin(None) is None
    assert D.margin_retention_bin(0.5) == "<0.80"
    assert D.margin_retention_bin(0.80) == "0.80-0.90"
    assert D.margin_retention_bin(0.90) == "0.90-0.95"
    assert D.margin_retention_bin(0.95) == "0.95-1.00"
    assert D.margin_retention_bin(1.0) == "0.95-1.00"
    assert D.margin_retention_bin(1.2) == ">1.00"


# ------------------------------------------------------- label asymmetry

def test_label_asymmetry_counts():
    samples = [_sample(gold="RUMOR", st_ok=True, ms_ok=False),
               _sample(gold="RUMOR", st_ok=True, ms_ok=True),
               _sample(gold="NON_RUMOR", st="NON_RUMOR", ms="RUMOR",
                       st_ok=True, ms_ok=False)]
    out = D.analysis_label_asymmetry({"pheme": samples, "maweibo": []})
    rum = out["pheme"]["RUMOR"]
    non = out["pheme"]["NON_RUMOR"]
    assert rum["n"] == 2 and rum["cw"] == 1 and rum["wc"] == 0
    assert rum["static_accuracy"] == pytest.approx(1.0)
    assert rum["ms_accuracy"] == pytest.approx(0.5)
    assert rum["delta_accuracy"] == pytest.approx(-0.5)
    assert non["n"] == 1
    assert non["confusion_transitions"] == {"NON_RUMOR->RUMOR": 1}


# ------------------------------------------------------ cross-fold §23

def test_cross_fold_audit():
    val = {0: {"e1", "e2"}, 1: {"e3"}, 2: set(), 3: set(), 4: set()}
    test = {0: {"e9"}, 1: {"e1"}, 2: set(), 3: set(), 4: set()}
    own = {"e1": {0}, "e2": {0}, "e3": {1}}
    entry = D.summarize_fold_membership(["e1", "e2", "e3"], val, test, own)
    assert entry["n_sampled_events"] == 3
    assert entry["n_events_in_any_other_fold_test"] == 1
    assert entry["rate_events_in_any_other_fold_test"] == pytest.approx(1 / 3)
    assert entry["per_event"]["e1"]["in_test_folds"] == [1]
    assert entry["per_event"]["e1"]["in_validation_folds"] == [0]


# ------------------------------------------- no new generation / no rule

def test_no_new_qwen_generation():
    for name in ("tcdscr_diagnose_v3b_failure.py",
                 "tcdscr_verify_v3b_failure_diagnosis.py"):
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        for banned in (".generate(", "QwenRumorLLM", "AutoModelForCausalLM",
                       "from_pretrained(cfg.qwen_model_path, torch_dtype"):
            assert banned not in src, f"{name} must not generate: {banned}"
    diag = (SCRIPTS_DIR / "tcdscr_diagnose_v3b_failure.py").read_text(
        encoding="utf-8")
    assert "max_new_tokens" not in diag
    assert "raw_generations" in diag  # only read for hashing


def test_no_heuristic_written_to_selector():
    issues = []
    V._check_no_heuristic(issues)
    assert issues == []
    diag = (SCRIPTS_DIR / "tcdscr_diagnose_v3b_failure.py").read_text(
        encoding="utf-8")
    for token in ("if fallback_to_static", "rule_", "policy_",
                  "minimal_set_refiner", "sufficiency import"):
        assert token not in diag.replace(
            "requires fold-local", ""), token or "heuristic token present"
    assert "candidate risk signals" in diag.lower() or \
        "Candidate Risk Signals" in diag


def test_statistics_helpers():
    assert D.roc_auc([3, 2, 1, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert D.roc_auc([0, 1, 2, 3], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert D.roc_auc([1, 1, 1], [1, 1, 0]) == pytest.approx(0.5)
    assert D.roc_auc([1, 1], [1, 1]) is None
    assert D.roc_auc([1, 1], [0, 0]) is None
    assert D.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert D.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert D.spearman([1, 1, 1], [1, 2, 3]) is None
    assert D.fisher_exact_2x2(10, 2, 2, 10) < 0.01
    assert D.fisher_exact_2x2(5, 5, 5, 5) == pytest.approx(1.0)
    assert D.odds_ratio(1, 0, 1, 1) is None
    assert D.quantiles([1, 2, 3, 4])["median"] == 3
    assert D.quantiles([])["n"] == 0
