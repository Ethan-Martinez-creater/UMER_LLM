# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 修复 P0_OBSERVABILITY.csv（F4/F5）：

1. 修复字段错位行（014/026/028/030/036/053/056/007）：correction_candidate 只允许
   固定枚举；证据文本只出现在 correction_evidence/notes。
2. temporal_availability 拆分语义：回复出现时间（可核）与引用材料历史内容（多 UNKNOWN）
   分开；001 的 21.4-22.1min URL 簇纠正为"15 分钟截点之外、60 分钟之内"。
3. 全表用 csv.DictWriter 显式重写；thread_hash/n_* 数值列以 manifest 为权威同步。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P0 = REPO / "docs" / "check" / "ses_v1" / "p0"

# temporal_availability 新枚举
T_REPLY_OK = "REPLY_TIME_VERIFIABLE_CONTENT_UNKNOWN"
T_AFTER = "AFTER_CUTOFF_ONLY"
T_NONE = "NO_CANDIDATE"
T_UNK = "UNKNOWN"

# correction_candidate 固定枚举
C_ALLOWED = {"yes", "no", "unknown"}

MANIFEST = {m["anon_id"]: m for m in csv.DictReader((P0 / "P0_MANIFEST.csv").open(encoding="utf-8"))}
ROWS = list(csv.DictReader((P0 / "P0_OBSERVABILITY.csv").open(encoding="utf-8")))

# 逐行修正（显式覆盖）
FIX = {
    "SES-P0-001": {
        "temporal_availability": T_REPLY_OK,
        "notes": "same official upstream echoed; URL cluster at 21.4-22.1min is inside 60min but OUTSIDE the 15min cutoff; cited blogspot content not checked",
    },
    "SES-P0-007": {
        "correction_candidate": "no",
        "correction_evidence": "R008/R012 target SIDE claims (nationalities) or restate root from same BBC outlet; not independent root correction",
        "temporal_availability": T_REPLY_OK,
        "notes": "two same-outlet echo clusters; BBC short links unarchived, current redirect is a live page (content UNKNOWN); see P0R1 ledger",
    },
    "SES-P0-014": {
        "correction_candidate": "no",
        "correction_evidence": "R008/R010 attribution questions; R017 same-outlet confirmation; R024 chain retracing",
        "temporal_availability": T_REPLY_OK,
        "notes": "attribution challenge resolved within same CBC chain; no independent upstream; no material cited; see P0R1 ledger",
    },
    "SES-P0-026": {
        "correction_candidate": "unknown",
        "correction_evidence": "R001 assertion-only denial @0.4min without link; R007 source-check question; R015 side-claim question",
        "temporal_availability": T_REPLY_OK,
        "notes": "cited URL today resolves to a different story; cutoff-time content identity UNKNOWN; fails all-item-verifiable rule (see P0R1 pair audit)",
    },
    "SES-P0-028": {
        "correction_candidate": "unknown",
        "correction_evidence": "R003 prosecutor denial with bbc.in link @11.28min; R004 is in-tree propagation of R003, not new independent support",
        "temporal_availability": T_REPLY_OK,
        "notes": "strongest candidate; cited link unarchived and current redirect is a live page -> historical content UNKNOWN; fails all-item-verifiable rule",
    },
    "SES-P0-029": {
        "temporal_availability": T_AFTER,
        "notes": "no in-cutoff correction; annotation AGAINST debunk is audit-only and never cited in tree; R017 denial @521.6min",
    },
    "SES-P0-030": {
        "correction_candidate": "no",
        "correction_evidence": "assertion-only disbelief and a source-demand question; forged club-spokesman quote propagates (R009/R010/R020)",
        "temporal_availability": T_REPLY_OK,
        "notes": "no evidential correction in tree; annotation 4x AGAINST is audit-only; reverse case for pair design",
    },
    "SES-P0-036": {
        "correction_candidate": "unknown",
        "correction_evidence": "no in-tree correction; AGAINST debunks exist only in annotation (audit-only)",
        "temporal_availability": T_REPLY_OK,
        "notes": "long-horizon echo 2.05-232.18min; annotation carries 2 FOR + 1 observing + 2 AGAINST, never cited in tree",
    },
    "SES-P0-053": {
        "correction_candidate": "unknown",
        "correction_evidence": "R013 cites store manager prior denial without link; underlying statement not located",
        "temporal_availability": T_REPLY_OK,
        "notes": "usatoday FOR link has 2014-08-16 archive; correction itself remains unverified; fails all-item-verifiable rule",
    },
    "SES-P0-056": {
        "correction_candidate": "no",
        "correction_evidence": "R015 caught-vs-located question (side claim); R009 uncertain assertion note",
        "temporal_availability": T_REPLY_OK,
        "notes": "clarification questions rather than evidential correction; deep off-topic drift to depth 17",
    },
    "SES-P0-010": {"temporal_availability": T_AFTER},
    "SES-P0-022": {"temporal_availability": T_REPLY_OK},
    "SES-P0-023": {"temporal_availability": T_REPLY_OK},
    "SES-P0-024": {"temporal_availability": T_REPLY_OK},
    "SES-P0-025": {"temporal_availability": T_REPLY_OK},
    "SES-P0-031": {"temporal_availability": T_REPLY_OK},
    "SES-P0-033": {"temporal_availability": T_REPLY_OK},
    "SES-P0-037": {"temporal_availability": T_REPLY_OK},
    "SES-P0-038": {"temporal_availability": T_REPLY_OK},
    "SES-P0-039": {"temporal_availability": T_REPLY_OK},
    "SES-P0-045": {"temporal_availability": T_REPLY_OK},
    "SES-P0-046": {"temporal_availability": T_REPLY_OK},
    "SES-P0-048": {"temporal_availability": T_REPLY_OK},
    "SES-P0-049": {"temporal_availability": T_REPLY_OK},
    "SES-P0-050": {"temporal_availability": T_REPLY_OK},
    "SES-P0-058": {"temporal_availability": T_REPLY_OK},
    "SES-P0-060": {"temporal_availability": T_REPLY_OK},
}


def main() -> int:
    for r in ROWS:
        anon = r["anon_id"]
        m = MANIFEST[anon]
        for k in ("thread_hash", "n_reactions", "in_15m", "in_60m", "in_360m", "no_candidate_15m"):
            r[k] = m[k]
        if m["no_candidate_15m"] == "True" and m["in_360m"] == "0" and m["n_reactions"] == "0":
            r["temporal_availability"] = T_NONE
        else:
            r.setdefault("temporal_availability", T_UNK)
        if anon in FIX:
            r.update(FIX[anon])
        # correction_candidate 枚举清洗
        cc = (r.get("correction_candidate") or "").strip().lower()
        if cc.startswith("yes"):
            r["correction_candidate"] = "unknown" if r["correction_candidate"] != "yes" else "yes"
        elif cc.startswith("no"):
            r["correction_candidate"] = "no"
        else:
            r["correction_candidate"] = "unknown"
        r["provisional"] = "1"
    fields = list(ROWS[0].keys())
    with (P0 / "P0_OBSERVABILITY.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ROWS)
    print("rewritten rows:", len(ROWS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
