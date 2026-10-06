# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 有限历史证据核查（计划 §4-B）。

预算：≤18 个不同页面/快照对象，≤48 次 HTTP 尝试（失败也计入）。
仅核查 12+12 深查例引用的明确 URL/命题出处；不全网搜索。
原文不入库：只记录元数据与内容 SHA256。
输出 docs/check/ses_v1/p0r1/P0R1_REQUEST_LEDGER.csv。
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "check" / "ses_v1" / "p0r1"

PROXY = "http://127.0.0.1:7890"
TIMEOUT = 40

FIELDS = ["request_no", "requested_url", "final_url", "http_status", "utc_timestamp",
          "content_sha256", "related_case", "object_no", "result_class", "note"]

# 每对象最多尝试次数内逐条执行；全部请求记录（含失败）
PLAN = [
    # 对象 1：028 R003 引用的 BBC live 长链（短链 bbc.in/14ulyLt 当前重定向目标）历史存档
    ("1", "https://web.archive.org/web/20150109/https://www.bbc.co.uk/news/live/world-europe-30722098", "SES-P0-028", "wayback_snapshot_query"),
    # 对象 2：007 R002/R003/R012/R017 簇 bbc.in/1EMTxMz 的长链历史存档
    ("2", "https://web.archive.org/web/20150324/https://www.bbc.co.uk/news/live/32030778", "SES-P0-007", "wayback_snapshot_query"),
    # 对象 3：026 echo 簇 cbc.ca/1.2873068 的 2014-12-15 存档
    ("3", "https://web.archive.org/web/20141215/https://www.cbc.ca/news/world/sydney-caf%C3%A9-siege-victims-identified-as-tori-johnson-katrina-dawson-1.2873068", "SES-P0-026", "wayback_snapshot_query"),
    # 对象 4：053 annotation FOR 材料的既有存档内容确认（usatoday 2014-08-16 快照）
    ("4", "http://web.archive.org/web/20140816002324/http://www.usatoday.com/story/news/usanow/2014/08/15/ferguson-missouri-police-michael-brown-shooting/14098369/", "SES-P0-053", "archived_content_read"),
    # 对象 5：030 annotation AGAINST 材料的既有存档内容确认（bleacherreport 2014-12-06 快照）
    ("5", "http://web.archive.org/web/20141206002110/http://bleacherreport.com/articles/2229044-ghana-and-ac-milan-star-michael-essien-denies-rumours-his-has-ebola-virus", "SES-P0-030", "archived_content_read"),
]


def curl(url: str, attempts_left: list, ledger: list, obj_no: str, case: str, rclass: str):
    """执行一次请求并记账；返回 (status, content_sha256, final_url, text_head)。"""
    n = len(ledger) + 1
    try:
        r = subprocess.run(
            ["curl", "-s", "-x", PROXY, "-L", "--max-time", str(TIMEOUT),
             "-o", "-", "-w", "\n__META__%{http_code}|%{url_effective}", url],
            capture_output=True, timeout=TIMEOUT + 10)
        raw = r.stdout
        body, _, meta = raw.rpartition(b"__META__")
        status, _, final = meta.decode("utf-8", "replace").partition("|")
        sha = hashlib.sha256(body).hexdigest()
        head = body[:400].decode("utf-8", "replace")
        attempts_left[0] -= 1
        ledger.append({
            "request_no": n, "requested_url": url, "final_url": final.strip(),
            "http_status": status.strip(), "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "content_sha256": sha, "related_case": case, "object_no": obj_no,
            "result_class": rclass, "note": "",
        })
        return status.strip(), sha, final.strip(), head
    except Exception as e:  # noqa: BLE001
        attempts_left[0] -= 1
        ledger.append({
            "request_no": n, "requested_url": url, "final_url": "",
            "http_status": "ERROR", "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "content_sha256": "", "related_case": case, "object_no": obj_no,
            "result_class": rclass, "note": f"{type(e).__name__}",
        })
        return "ERROR", "", "", ""


def main() -> int:
    attempts = [48]
    ledger = []
    results = []
    for obj_no, url, case, rclass in PLAN:
        if attempts[0] <= 0:
            results.append({"object": obj_no, "outcome": "BUDGET_EXHAUSTED"})
            continue
        status, sha, final, head = curl(url, attempts, ledger, obj_no, case, rclass)
        outcome = "OK" if status.startswith("2") else f"HTTP_{status}"
        # wayback 快照查询：200 但可能是“快照不存在”页
        note = ""
        if rclass == "wayback_snapshot_query" and status.startswith("2"):
            if b"Wayback Machine has not archived" in head.encode("utf-8", "replace") or "not archived" in head.lower():
                outcome = "NO_SNAPSHOT"
                note = "archive.org reports no snapshot"
            elif "imp" not in head and len(head) < 250:
                note = "short body; see hash"
        results.append({"object": obj_no, "case": case, "url": url, "outcome": outcome,
                        "final_url": final, "content_sha256": sha})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "P0R1_REQUEST_LEDGER.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ledger)
    summary = {
        "budget_objects": {"used": len(PLAN), "cap": 18},
        "budget_http_attempts": {"used": len(ledger), "cap": 48},
        "results": results,
    }
    (OUT / "P0R1_REQUEST_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print(r.get("object"), r.get("outcome"), (r.get("final_url") or "")[:80])
    return 0


if __name__ == "__main__":
    sys.exit(main())
