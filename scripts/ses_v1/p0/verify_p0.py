# -*- coding: utf-8 -*-
"""SES-v1 P0 验证：重放固定选样、manifest/OBSERVABILITY schema 与语义一致性检查。

输出 docs/check/ses_v1/p0/P0_VALIDATION.json；全部检查项逐项记录，失败如实报告。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "local_assets" / "ses_v1" / "p0"
P0 = REPO / "docs" / "check" / "ses_v1" / "p0"

SALT = "SES-v1-P0-20261006"
DATA_VERSION = "079f6ffdbc0b367399262f101774372e5d19dd8278c33d6c97a84461a9bc58dd"
TARGETS = {"true": 20, "false": 20, "unverified": 20}

sys.path.insert(0, str(REPO / "scripts" / "ses_v1" / "p0"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay_sample(rv_threads):
    """重放计划 §4.1 选样规则（与 build_rv_map_and_sample.py 相同逻辑）。"""
    rumour = [t for t in rv_threads if t["r"] == "rumour" and t["v"]]
    rows = []
    seq = 0
    for v_class in ("true", "false", "unverified"):
        pool = [t for t in rumour if t["v"] == v_class]
        topics = sorted({t["topic"] for t in pool}, key=lambda tp: hashlib.sha256(tp.encode("utf-8")).hexdigest())
        by_topic = {tp: sorted([t for t in pool if t["topic"] == tp],
                               key=lambda t: sha256_hex(f"{SALT}|{DATA_VERSION}|{t['topic']}|{t['thread_id']}"))
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
            rows.append({"anon_id": f"SES-P0-{seq:03d}", "v_class": v_class, "topic": t["topic"],
                         "thread_hash": sha256_hex(f"{t['topic']}|{t['thread_id']}")[:16]})
    return rows


def main() -> int:
    checks = {}

    # 1. 重放选样
    rv = json.loads((ASSET / "rv_map.json").read_text(encoding="utf-8"))
    replay = replay_sample(rv["threads"])
    manifest = list(csv.DictReader((P0 / "P0_MANIFEST.csv").open(encoding="utf-8")))
    replay_keys = [(r["anon_id"], r["v_class"], r["topic"], r["thread_hash"]) for r in replay]
    manifest_keys = [(r["anon_id"], r["v_class"], r["topic"], r["thread_hash"]) for r in manifest]
    checks["sample_replay_matches_manifest"] = (replay_keys == manifest_keys)

    # 2. manifest 唯一性
    ids = [r["anon_id"] for r in manifest]
    hashes = [r["thread_hash"] for r in manifest]
    checks["manifest_anon_ids_unique"] = (len(set(ids)) == len(ids))
    checks["manifest_thread_hashes_unique"] = (len(set(hashes)) == len(hashes))

    # 3. R/V 语义
    checks["manifest_r_all_rumour"] = all(r["r"] == "rumour" for r in manifest)
    checks["manifest_v_matches_class"] = all(r["v"] == r["v_class"] for r in manifest)

    # 4. 样本数/缺额
    counts = {k: sum(1 for r in manifest if r["v_class"] == k) for k in TARGETS}
    checks["sample_counts"] = counts
    checks["sample_no_shortfall"] = all(counts[k] == TARGETS[k] for k in TARGETS)
    topics_covered = len({r["topic"] for r in manifest})
    checks["topics_covered"] = topics_covered

    # 5. OBSERVABILITY 与 manifest 一致
    obs = list(csv.DictReader((P0 / "P0_OBSERVABILITY.csv").open(encoding="utf-8")))
    obs_by_id = {r["anon_id"]: r for r in obs}
    mismatch = []
    for m in manifest:
        o = obs_by_id.get(m["anon_id"])
        if not o:
            mismatch.append(m["anon_id"] + ":missing")
            continue
        for k in ("v_class", "topic", "thread_hash"):
            if o[k] != m[k]:
                mismatch.append(f"{m['anon_id']}:{k}")
        for k in ("n_reactions", "in_15m", "in_60m", "in_360m"):
            if o[k] != m[k]:
                mismatch.append(f"{m['anon_id']}:count:{k}")
    checks["observability_rows_60"] = (len(obs) == 60)
    checks["observability_consistent_with_manifest"] = (not mismatch)
    checks["observability_mismatch_examples"] = mismatch[:10]

    # 6. provisional 标记全量
    checks["all_rows_marked_provisional"] = all(r.get("provisional") == "1" for r in obs)

    # 7. 截点语义：no_candidate_15m 与 in_15m 一致；after-cutoff 未混入截点前候选
    bad_nc = [r["anon_id"] for r in obs if (r["no_candidate_15m"] == "True") != (r["in_15m"] == "0")]
    checks["no_candidate_semantics_consistent"] = (not bad_nc)
    bad_temporal = [r["anon_id"] for r in obs
                    if "after_cutoff" in r["temporal_availability"] and "CORE PAIR" in r["correction_evidence"]]
    checks["no_after_cutoff_in_core_pairs"] = (not bad_temporal)

    # 8. 待提交文件无原始正文/原始长 ID 泄漏
    long_id_pat = re.compile(r"\b\d{17,19}\b")
    leaks = []
    for f in sorted(P0.iterdir()):
        if f.suffix in (".csv", ".json", ".md"):
            txt = f.read_text(encoding="utf-8", errors="ignore")
            found = long_id_pat.findall(txt)
            if found:
                leaks.append(f.name)
    checks["no_raw_twitter_ids_in_p0_docs"] = (not leaks)
    checks["leak_examples"] = leaks

    # 9. 关键文件存在
    required = ["P0_ASSET_AUDIT.json", "P0_MANIFEST.csv", "P0_OBSERVABILITY.csv",
                "P0_CASE_AUDIT.md", "P0_NEAREST_WORKS.md", "P0_VERDICT.json",
                "P0_NEXT_STAGE_DRAFT.md", "P0_REPORT.md"]
    missing = [n for n in required if not (P0 / n).exists()]
    checks["required_files_present"] = (not missing)
    checks["missing_files"] = missing

    # 10. 数据资产哈希
    archive = ASSET / "PHEME_veracity.tar.bz2"
    h = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    checks["archive_sha256_matches_data_version"] = (h == DATA_VERSION)

    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    validation = {
        "all_pass": ok,
        "data_version_sha256": DATA_VERSION,
        "checks": checks,
    }
    (P0 / "P0_VALIDATION.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": ok,
                      "failed": [k for k, v in checks.items() if isinstance(v, bool) and not v]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
