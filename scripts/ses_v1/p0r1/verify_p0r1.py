# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 交付验证：重放、schema、语义一致性与预算记账。

输出 docs/check/ses_v1/p0r1/P0R1_VALIDATION.json。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET0 = REPO / "local_assets" / "ses_v1" / "p0"
P0 = REPO / "docs" / "check" / "ses_v1" / "p0"
R1 = REPO / "docs" / "check" / "ses_v1" / "p0r1"

SALT = "SES-v1-P0-20261006"
DATA_VERSION = "079f6ffdbc0b367399262f101774372e5d19dd8278c33d6c97a84461a9bc58dd"
TARGETS = {"true": 20, "false": 20, "unverified": 20}
ORIG_DEEP = {"SES-P0-001", "SES-P0-007", "SES-P0-008", "SES-P0-010", "SES-P0-014",
             "SES-P0-026", "SES-P0-028", "SES-P0-029", "SES-P0-030", "SES-P0-036",
             "SES-P0-053", "SES-P0-056"}
ENUM_CC = {"yes", "no", "unknown"}
ENUM_TEMPORAL = {"REPLY_TIME_VERIFIABLE_CONTENT_UNKNOWN", "AFTER_CUTOFF_ONLY",
                 "NO_CANDIDATE", "UNKNOWN"}
LEDGER_ENUMS = {
    "source_kind": {"reaction", "source", "historical_external", "annotation_audit_only"},
    "tweet_presence": {"BEFORE_CUTOFF", "AFTER_CUTOFF", "MISSING_TIME", "NOT_A_TWEET"},
    "claim_role": {"root", "side", "context"},
    "dependency_state": {"VERIFIED_COMMON_UPSTREAM", "VERIFIED_DISTINCT_UPSTREAM_CANDIDATE",
                         "UNKNOWN", "NOT_APPLICABLE"},
    "support_state": {"SUPPORTED_CORRECTION_CANDIDATE", "ASSERTION_ONLY", "QUESTION_ONLY",
                      "CONFIRMATION_ONLY", "SIDE_CLAIM_ONLY", "INSUFFICIENT", "UNKNOWN"},
    "historical_content_state": {"VERIFIED_AT_CUTOFF", "AFTER_CUTOFF_ONLY", "CURRENT_ONLY",
                                 "UNKNOWN", "NOT_APPLICABLE"},
    "gate": {"YES", "NO", "UNKNOWN"},
}


def replay_sample(rv_threads):
    rumour = [t for t in rv_threads if t["r"] == "rumour" and t["v"]]
    rows, seq = [], 0
    for v_class in ("true", "false", "unverified"):
        pool = [t for t in rumour if t["v"] == v_class]
        topics = sorted({t["topic"] for t in pool}, key=lambda tp: hashlib.sha256(tp.encode("utf-8")).hexdigest())
        by_topic = {tp: sorted([t for t in pool if t["topic"] == tp],
                               key=lambda t: hashlib.sha256(f"{SALT}|{DATA_VERSION}|{t['topic']}|{t['thread_id']}".encode()).hexdigest())
                    for tp in topics}
        picked, cursor = [], {tp: 0 for tp in topics}
        while len(picked) < TARGETS[v_class]:
            progressed = False
            for tp in topics:
                if len(picked) >= TARGETS[v_class]:
                    break
                lst = by_topic[tp]
                if cursor[tp] < len(lst):
                    picked.append(lst[cursor[tp]])
                    cursor[tp] += 1
                    progressed = True
            if not progressed:
                break
        for t in picked:
            seq += 1
            rows.append((f"SES-P0-{seq:03d}", v_class, t["topic"],
                         hashlib.sha256(f"{t['topic']}|{t['thread_id']}".encode()).hexdigest()[:16]))
    return rows


import hashlib  # noqa: E402


def main() -> int:
    checks = {}

    # R1-1 原 60 manifest 重放不变
    rv = json.loads((ASSET0 / "rv_map.json").read_text(encoding="utf-8"))
    replay = replay_sample(rv["threads"])
    manifest = list(csv.DictReader((P0 / "P0_MANIFEST.csv").open(encoding="utf-8")))
    checks["orig_manifest_replay_unchanged"] = (
        replay == [(m["anon_id"], m["v_class"], m["topic"], m["thread_hash"]) for m in manifest])

    # R1-2 增查集合：每类前 4（排除原 12），与 manifest 一致，与原 12 无交集
    ext = list(csv.DictReader((R1 / "P0R1_DISCOVERY_EXTENSION.csv").open(encoding="utf-8")))
    expect_ext = []
    for c in ("true", "false", "unverified"):
        cand = [m for m in manifest if m["v_class"] == c and m["anon_id"] not in ORIG_DEEP][:4]
        expect_ext += [m["anon_id"] for m in cand]
    checks["extension_set_correct"] = ([e["anon_id"] for e in ext] == expect_ext)
    checks["extension_no_overlap_with_orig_deep"] = (not (set(e["anon_id"] for e in ext) & ORIG_DEEP))
    checks["deep_total_24"] = (len(set(e["anon_id"] for e in ext) | ORIG_DEEP) == 24)

    # R1-3 ledger schema 与枚举
    ledger = list(csv.DictReader((R1 / "P0R1_EVIDENCE_LEDGER.csv").open(encoding="utf-8")))
    bad_enum = []
    for r in ledger:
        for k, allowed in LEDGER_ENUMS.items():
            if k != "gate" and r.get(k) not in allowed:
                bad_enum.append((r["case_id"], r["evidence_id"], k, r.get(k)))
    checks["ledger_enum_ok"] = (not bad_enum)
    checks["ledger_enum_violations"] = bad_enum[:5]
    checks["ledger_all_provisional"] = all(r["provisional"] == "1" for r in ledger)

    # R1-4 gate 语义：侧命题/提问/同源确认/无材料不得 YES；历史状态不可核不得 YES
    pairs = list(csv.DictReader((R1 / "P0R1_PAIR_AUDIT.csv").open(encoding="utf-8")))
    sup_by_case = {}
    for r in ledger:
        if r["source_kind"] == "annotation_audit_only" and r["support_state"] == "SUPPORTED_CORRECTION_CANDIDATE":
            bad_enum.append(("ANNOTATION_AS_CORRECTION", r["case_id"], "support_state", r["support_state"]))
        sup_by_case.setdefault(r["case_id"], set()).add(r["support_state"])
    gate_bad = []
    correction_like = {"SUPPORTED_CORRECTION_CANDIDATE", "ASSERTION_ONLY", "INSUFFICIENT"}
    for r in pairs:
        if r["meets_original_gate"] == "YES":
            gate_bad.append((r["case_id"], r["cutoff"], "gate YES present"))
            continue
        # UNKNOWN 的合法路径：该案例存在纠错尝试行（候选/断言/不足），且历史状态确实不可核
        if r["meets_original_gate"] == "UNKNOWN":
            sup = sup_by_case.get(r["case_id"], set())
            if not (sup & correction_like):
                gate_bad.append((r["case_id"], r["cutoff"], "UNKNOWN gate without any correction-attempt row"))
            if not r["historical_availability"].startswith(("UNKNOWN", "CURRENT_ONLY", "AFTER_CUTOFF_ONLY")):
                gate_bad.append((r["case_id"], r["cutoff"], "UNKNOWN gate with verified material"))
    checks["pair_gate_semantics_ok"] = (not gate_bad)
    checks["pair_gate_violations"] = gate_bad[:5]

    # R1-5 ledger cutoff 与 offset 一致（21.4min 不得进 15）
    bad_cut = []
    for r in ledger:
        off = r["reply_offset_min"]
        if not off:
            continue
        vals = []
        for part in str(off).split(","):
            part = part.strip()
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                vals += [float(a), float(b)]
            else:
                try:
                    vals.append(float(part))
                except ValueError:
                    pass
        vals = [v for v in vals if v >= 0]
        if not vals:
            continue
        mx = max(vals)
        if mx > 15 and "15" in r["cutoff"].split(";"):
            bad_cut.append((r["case_id"], r["evidence_id"]))
    checks["ledger_cutoff_consistent_with_offset"] = (not bad_cut)
    checks["cutoff_violations"] = bad_cut[:5]

    # R1-6 父边审计确定性处理（不默认补 SRC）
    cov = json.loads((R1 / "P0R1_ASSET_COVERAGE.json").read_text(encoding="utf-8"))["totals"]
    checks["parent_edges_recorded_not_defaulted"] = (
        cov["parent_edges_metadata_only"] == 23 and cov["in_reply_to_pointing_outside_tree"] == 17
        and cov["parent_timestamp_after_child"] == 0)

    # R1-7 输入哈希齐备（零 reactions 线程允许空合并哈希）
    full = json.loads((REPO / "local_assets/ses_v1/p0r1/asset_audit_full.json").read_text(encoding="utf-8"))
    need = ("source_json_sha256", "annotation_json_sha256", "structure_json_sha256", "reactions_merged_sha256")
    missing_hash = []
    for f in full:
        for k in need:
            if k == "reactions_merged_sha256":
                if not f.get(k) and (f.get("n_reaction_files") or 0) > 0:
                    missing_hash.append(f["anon_id"])
            elif not f.get(k):
                missing_hash.append(f["anon_id"])
    checks["input_hashes_present_60"] = (len(full) == 60 and not missing_hash)

    # R1-8 预算记账
    req = list(csv.DictReader((R1 / "P0R1_REQUEST_LEDGER.csv").open(encoding="utf-8")))
    checks["request_budget_ok"] = (len(set(r["object_no"] for r in req)) <= 18 and len(req) <= 48)

    # R1-9 隐私：无 17-19 位原始 ID
    pat = re.compile(r"\b\d{17,19}\b")
    leaks = []
    for f in sorted(list(R1.iterdir()) + [P0 / "P0_OBSERVABILITY.csv"]):
        if f.suffix in (".csv", ".json", ".md"):
            if pat.findall(f.read_text(encoding="utf-8", errors="ignore")):
                leaks.append(f.name)
    checks["no_raw_ids_in_r1_docs"] = (not leaks)
    checks["leak_files"] = leaks

    # R1-10 verdict 数量与 CSV 重算一致（verdict 存在时）
    verdict_path = R1 / "P0R1_VERDICT.json"
    if verdict_path.exists():
        v = json.loads(verdict_path.read_text(encoding="utf-8"))
        c15 = Counter(r["meets_original_gate"] for r in pairs if str(r["cutoff"]) == "15")
        checks["verdict_pair_counts_consistent"] = (
            v["counts"]["strict_pairs_gate15"] == c15.get("YES", 0)
            and v["counts"]["unknown_pairs_gate15"] == c15.get("UNKNOWN", 0)
            and v["counts"]["pair_rows"] == len(pairs))
    else:
        checks["verdict_pair_counts_consistent"] = "NOT_RUN"

    ok = all(vv for vv in checks.values() if isinstance(vv, bool))
    out = {"all_pass": ok, "checks": checks,
           "data_version_sha256": DATA_VERSION}
    (R1 / "P0R1_VALIDATION.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"all_pass": ok,
                      "failed": [k for k, vv in checks.items() if isinstance(vv, bool) and not vv]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
