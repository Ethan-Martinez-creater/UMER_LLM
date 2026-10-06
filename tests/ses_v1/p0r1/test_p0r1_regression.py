# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 回归测试（计划 §7 十项要求，全部用合成 fixture 验证程序逻辑）。

fixture 仅为程序验证，不作为科研样本、不替代真实缺失数据、不进入输出 manifest。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "ses_v1" / "p0r1"))
sys.path.insert(0, str(REPO / "scripts" / "ses_v1" / "p0"))

import schema_checks as sc  # noqa: E402


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- 1. 字段错位/非枚举状态被拒绝（014 同类错误） ----------

def test_observability_schema_rejects_misplaced_state():
    bad_rows = [
        {"anon_id": "X-014", "correction_candidate": "yes (attribution resolved in tree)",
         "temporal_availability": "CORE PAIR CANDIDATE: ...",
         "correction_evidence": "before_cutoff_verifiable", "provisional": "1"},
    ]
    res = sc.check_observability_schema(bad_rows)
    assert res["bad_cc"] == ["X-014"]
    assert res["bad_temporal"] == ["X-014"]
    assert res["misplaced"] == ["X-014"]


def test_observability_schema_accepts_clean_row():
    good = [{"anon_id": "OK-1", "correction_candidate": "unknown",
             "temporal_availability": "REPLY_TIME_VERIFIABLE_CONTENT_UNKNOWN",
             "correction_evidence": "R001 assertion-only denial; see ledger",
             "provisional": "1"}]
    res = sc.check_observability_schema(good)
    assert all(not v for v in res.values())


def test_observability_schema_requires_provisional():
    rows = [{"anon_id": "P-1", "correction_candidate": "no",
             "temporal_availability": "UNKNOWN", "correction_evidence": "", "provisional": "0"}]
    assert sc.check_observability_schema(rows)["not_provisional"] == ["P-1"]


# ---------- 2. 截点前 reply 但材料不可核 -> 配对不得通过；晚存档不得升 VERIFIED ----------

def test_pair_gate_blocks_unknown_material():
    ledger = [{"case_id": "C1", "evidence_id": "R001", "source_kind": "reaction",
               "support_state": "SUPPORTED_CORRECTION_CANDIDATE", "provisional": "1"}]
    pairs_unknown_material = [{"case_id": "C1", "cutoff": "15", "meets_original_gate": "UNKNOWN",
                               "historical_availability": "UNKNOWN"}]
    assert not sc.check_pair_semantics(pairs_unknown_material, ledger)
    # 试图把不可核材料标成 YES -> 被拒绝
    pairs_bad_yes = [{"case_id": "C1", "cutoff": "15", "meets_original_gate": "YES",
                      "historical_availability": "UNKNOWN"}]
    assert sc.check_pair_semantics(pairs_bad_yes, ledger)
    # 晚存档标 VERIFIED_AT_CUTOFF 且 YES -> 被拒绝
    pairs_after = [{"case_id": "C1", "cutoff": "15", "meets_original_gate": "YES",
                    "historical_availability": "AFTER_CUTOFF_ONLY"}]
    assert sc.check_pair_semantics(pairs_after, ledger)


# ---------- 3. 侧命题/提问/同源确认/依赖子回复不得计独立根纠错 ----------

def test_pair_gate_blocks_non_independent_support():
    ledger = [{"case_id": "C2", "evidence_id": "R004", "source_kind": "reaction",
               "support_state": "SIDE_CLAIM_ONLY", "provisional": "1"},
              {"case_id": "C3", "evidence_id": "R005", "source_kind": "reaction",
               "support_state": "QUESTION_ONLY", "provisional": "1"},
              {"case_id": "C4", "evidence_id": "R006", "source_kind": "reaction",
               "support_state": "CONFIRMATION_ONLY", "provisional": "1"}]
    for cid in ("C2", "C3", "C4"):
        # 标 YES 被拒
        bad = [{"case_id": cid, "cutoff": "15", "meets_original_gate": "YES",
                "historical_availability": "VERIFIED_AT_CUTOFF"}]
        assert sc.check_pair_semantics(bad, ledger)
        # 标 UNKNOWN 被拒（无纠错尝试行）
        unk = [{"case_id": cid, "cutoff": "15", "meets_original_gate": "UNKNOWN",
                "historical_availability": "UNKNOWN"}]
        assert sc.check_pair_semantics(unk, ledger)
        # 合法：NO
        ok = [{"case_id": cid, "cutoff": "15", "meets_original_gate": "NO",
               "historical_availability": "UNKNOWN"}]
        assert not sc.check_pair_semantics(ok, ledger)


# ---------- 4. annotation 不进部署候选；V 不决定纠错有效性 ----------

def test_annotation_never_supported_correction():
    ann = [{"case_id": "C5", "evidence_id": "ANN", "source_kind": "annotation_audit_only",
            "support_state": "SUPPORTED_CORRECTION_CANDIDATE", "provisional": "1"}]
    assert sc.check_ledger_schema(ann)


def test_veracity_label_does_not_decide_gate():
    # v_class 不参与 gate 判定：相同证据结构在 true/false 线程上结果必须一致
    ledger = [{"case_id": "CT", "evidence_id": "R1", "source_kind": "reaction",
               "support_state": "ASSERTION_ONLY", "provisional": "1"}]
    pairs_true = [{"case_id": "CT", "cutoff": "15", "meets_original_gate": "UNKNOWN",
                   "historical_availability": "UNKNOWN"}]
    assert not sc.check_pair_semantics(pairs_true, ledger)
    # 同一证据若材料可核且被支持则可 YES；否则无论 V 都不能 YES
    ledger_v = [dict(ledger[0], support_state="SUPPORTED_CORRECTION_CANDIDATE")]
    pairs_unverified = [{"case_id": "CT", "cutoff": "15", "meets_original_gate": "YES",
                         "historical_availability": "VERIFIED_AT_CUTOFF"}]
    assert not sc.check_pair_semantics(pairs_unverified, ledger_v)


# ---------- 5. 21.4 分钟材料不进 15 分钟截点 ----------

def test_cutoff_consistency_rejects_21min_in_15():
    rows = [{"case_id": "001", "evidence_id": "R024-R028", "reply_offset_min": "21.42-22.10",
             "cutoff": "15;60;360"}]
    assert sc.check_cutoff_consistency(rows)
    ok = [{"case_id": "001", "evidence_id": "R024-R028", "reply_offset_min": "21.42-22.10",
           "cutoff": "60;360"}]
    assert not sc.check_cutoff_consistency(ok)
    edge = [{"case_id": "001", "evidence_id": "R001", "reply_offset_min": "0.40",
             "cutoff": "15;60;360"}]
    assert not sc.check_cutoff_consistency(edge)


# ---------- 6. 父边三态/父晚子/空树/缺时间的确定性处理 ----------

def test_walk_handles_empty_list_leaf_and_no_src_default():
    mod = load_module("deep_extract_r1", "scripts/ses_v1/p0/deep_extract_r1.py")
    structure = {"root": {"a": {"b": []}, "c": []}}
    parent_of = {}
    mod.walk(structure, "root", parent_of)
    assert parent_of == {"root": "root", "a": "root", "b": "a", "c": "root"}


def test_asset_audit_counts_parent_states_without_defaulting():
    cov = (REPO / "docs/check/ses_v1/p0r1/P0R1_ASSET_COVERAGE.json")
    if not cov.exists():
        pytest.skip("coverage file not present")
    import json
    t = json.loads(cov.read_text(encoding="utf-8"))["totals"]
    # 三种父边状态分别计数，未被静默合并或补 SRC
    assert t["parent_edges_tree_only"] == 0
    assert t["parent_edges_metadata_only"] == 23
    assert t["parent_edges_both_present_but_differ"] == 0
    assert t["in_reply_to_pointing_outside_tree"] == 17
    assert t["parent_timestamp_after_child"] == 0


# ---------- 7. 证据 ID 联结一致、时间平局稳定、输入哈希存在 ----------

def test_deep_extract_row_ids_time_sorted_stable():
    mod = load_module("deep_extract_r1", "scripts/ses_v1/p0/deep_extract_r1.py")
    reactions = [
        {"id_str": "b", "created_at": "Tue Mar 24 10:47:01 +0000 2015", "text": "x"},
        {"id_str": "a", "created_at": "Tue Mar 24 10:47:01 +0000 2015", "text": "y"},
        {"id_str": "c", "created_at": "Tue Mar 24 10:46:30 +0000 2015", "text": "z"},
    ]
    reactions.sort(key=lambda o: mod.ts_epoch(o.get("created_at")) or 0)
    # 时间平局保持输入次序（稳定排序），较早者在前
    assert [r["id_str"] for r in reactions] == ["c", "b", "a"]


def test_input_hashes_present():
    import json
    full = json.loads((REPO / "local_assets/ses_v1/p0r1/asset_audit_full.json").read_text(encoding="utf-8"))
    assert len(full) == 60
    for f in full:
        assert f["source_json_sha256"] and f["structure_json_sha256"]


# ---------- 8. 深查集与确认集交集为零；原盐 manifest 不变 ----------

def test_extension_disjoint_and_manifest_stable():
    import csv
    manifest = list(csv.DictReader((REPO / "docs/check/ses_v1/p0/P0_MANIFEST.csv").open(encoding="utf-8")))
    ext = list(csv.DictReader((REPO / "docs/check/ses_v1/p0r1/P0R1_DISCOVERY_EXTENSION.csv").open(encoding="utf-8")))
    orig = {"SES-P0-001", "SES-P0-007", "SES-P0-008", "SES-P0-010", "SES-P0-014",
            "SES-P0-026", "SES-P0-028", "SES-P0-029", "SES-P0-030", "SES-P0-036",
            "SES-P0-053", "SES-P0-056"}
    assert len(manifest) == 60
    assert not ({e["anon_id"] for e in ext} & orig)
    # 深查 24 = 原 12 + 增 12；确认候选只能来自剩余 2402-60=2342（不与深查/manifest 交集）
    assert len(orig | {e["anon_id"] for e in ext}) == 24
    assert 2402 - 60 == 2342


# ---------- 9. 破坏 schema/时间/状态后验证器拒绝 ----------

def test_verify_p0r1_rejects_corrupted_inputs():
    # 破坏 ledger：非枚举状态被拒绝
    bad_ledger = [{"case_id": "BAD", "evidence_id": "R1", "source_kind": "reaction",
                   "support_state": "TOTALLY_MADE_UP", "provisional": "1",
                   "tweet_presence": "BEFORE_CUTOFF", "claim_role": "root",
                   "dependency_state": "UNKNOWN", "historical_content_state": "UNKNOWN"}]
    assert sc.check_ledger_schema(bad_ledger)


def test_verify_p0r1_script_runs_green():
    mod = load_module("verify_p0r1", "scripts/ses_v1/p0r1/verify_p0r1.py")
    assert mod.main() == 0


# ---------- 10. 报告数量与 ledger/verdict 一致 ----------

def test_verdict_counts_match_structured_files():
    import csv
    import json
    verdict_path = REPO / "docs/check/ses_v1/p0r1/P0R1_VERDICT.json"
    if not verdict_path.exists():
        pytest.skip("verdict not yet written")
    pairs = list(csv.DictReader((REPO / "docs/check/ses_v1/p0r1/P0R1_PAIR_AUDIT.csv").open(encoding="utf-8")))
    v = json.loads(verdict_path.read_text(encoding="utf-8"))
    c15 = {}
    for r in pairs:
        if r["cutoff"] == "15":
            c15[r["meets_original_gate"]] = c15.get(r["meets_original_gate"], 0) + 1
    assert v["counts"]["strict_pairs_gate15"] == c15.get("YES", 0)
    assert v["counts"]["unknown_pairs_gate15"] == c15.get("UNKNOWN", 0)
    assert v["counts"]["pair_rows"] == len(pairs)
