# -*- coding: utf-8 -*-
"""SES-v1 P0 主审计脚本：R/V 映射、资产统计、固定盐 60 条选样与时间边界审计。

输入  : local_assets/ses_v1/p0/extracted/all-rnr-annotated-threads（figshare v1 包，CC BY 4.0）
输出  : local_assets/ses_v1/p0/rv_map.json            全量映射（含原始 ID，仅本地）
        local_assets/ses_v1/p0/review/selected_texts  选中线程审读文本（仅本地）
        docs/check/ses_v1/p0/P0_ASSET_AUDIT.json      资产统计（可推送，无原文）
        docs/check/ses_v1/p0/P0_MANIFEST.csv          固定选样 manifest（匿名 ID，可推送）

选样规则（计划 §4.1）：
  盐 = SES-v1-P0-20261006；数据版本标识 = figshare 包 SHA256；
  对 true/false/unverified 三个 V 类别，目标各 20 条：
  1) 话题按话题名 UTF-8 字节 SHA256 排序；
  2) 话题内线程按 SHA256(盐|版本|topic|source_id) 十六进制排序；
  3) 按话题顺序轮转，每轮每话题取下一条，直到该类别 20 条或池耗尽。

本脚本只读原始数据，不加载模型，不联网。
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "local_assets" / "ses_v1" / "p0"
DATA = ASSET / "extracted" / "all-rnr-annotated-threads"
OUT_DOC = REPO / "docs" / "check" / "ses_v1" / "p0"

SALT = "SES-v1-P0-20261006"
DATA_VERSION = "079f6ffdbc0b367399262f101774372e5d19dd8278c33d6c97a84461a9bc58dd"
TARGETS = {"true": 20, "false": 20, "unverified": 20}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def convert_veracity(annotation: dict):
    """官方 convert_veracity_annotations.py 逻辑的忠实移植；返回 (label, conflict_flag)。"""
    if "misinformation" in annotation and "true" in annotation:
        mis, tru = int(annotation["misinformation"]), int(annotation["true"])
        if mis == 0 and tru == 0:
            return "unverified", False
        if mis == 0 and tru == 1:
            return "true", False
        if mis == 1 and tru == 0:
            return "false", False
        return None, True  # OMG! They both are 1!
    if "misinformation" in annotation and "true" not in annotation:
        return ("false" if int(annotation["misinformation"]) == 1 else "unverified"), False
    return None, False


def scan_threads():
    """遍历全部事件与线程，构建 R/V 映射与字段覆盖统计。"""
    events = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    threads = []
    conflicts, v_missing = [], []
    for event in events:
        topic = event.replace("-all-rnr-threads", "")
        for r_kind in ("rumours", "non-rumours"):
            rdir = DATA / event / r_kind
            if not rdir.exists():
                continue
            for tdir in sorted(p for p in rdir.iterdir() if p.is_dir()):
                tid = tdir.name
                ann_path = tdir / "annotation.json"
                annotation = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
                # R 以目录结构为基准；annotation 的 is_rumour 原值仅作对照
                r_label = "rumour" if r_kind == "rumours" else "non-rumour"
                v_label, conflict = (None, False)
                if r_kind == "rumours":
                    v_label, conflict = convert_veracity(annotation)
                    if conflict:
                        conflicts.append(f"{topic}|{tid}")
                    elif v_label is None:
                        v_missing.append(f"{topic}|{tid}")
                threads.append({
                    "topic": topic,
                    "thread_id": tid,
                    "dir": str(tdir.relative_to(DATA)),
                    "r": r_label,
                    "is_rumour_raw": annotation.get("is_rumour"),
                    "v": v_label,
                    "annotation_keys": sorted(annotation.keys()),
                    "has_links": bool(annotation.get("links")),
                    "is_turnaround": annotation.get("is_turnaround"),
                })
    return events, threads, conflicts, v_missing


def load_thread_rows(tdir: Path, source_id: str):
    """读取一个线程的源帖与回复时间戳/正文信息（只读 JSON）。"""
    rows = {"source": None, "reactions": []}
    st_dir = tdir / "source-tweets"
    if st_dir.exists():
        for f in st_dir.glob("*.json"):
            obj = json.loads(f.read_text(encoding="utf-8"))
            rows["source"] = {
                "id": obj.get("id_str") or str(obj.get("id")),
                "created_at": obj.get("created_at"),
                "text": obj.get("text") or obj.get("full_text"),
                "user_id": (obj.get("user") or {}).get("id_str"),
                "user_followers": (obj.get("user") or {}).get("followers_count"),
                "urls": [u.get("expanded_url") or u.get("url") for u in (obj.get("entities") or {}).get("urls", [])],
            }
    r_dir = tdir / "reactions"
    if r_dir.exists():
        for f in sorted(r_dir.glob("*.json")):
            obj = json.loads(f.read_text(encoding="utf-8"))
            rows["reactions"].append({
                "id": obj.get("id_str") or str(obj.get("id")),
                "created_at": obj.get("created_at"),
                "text": obj.get("text") or obj.get("full_text"),
                "user_id": (obj.get("user") or {}).get("id_str"),
                "in_reply_to": obj.get("in_reply_to_status_id_str"),
                "urls": [u.get("expanded_url") or u.get("url") for u in (obj.get("entities") or {}).get("urls", [])],
            })
    return rows


def ts_epoch(s):
    if not s:
        return None
    from datetime import datetime, timezone
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S +0000 %Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def main() -> int:
    OUT_DOC.mkdir(parents=True, exist_ok=True)
    events, threads, conflicts, v_missing = scan_threads()

    # ---- 资产统计 ----
    topic_stats = defaultdict(lambda: {"rumour": 0, "non-rumour": 0,
                                       "true": 0, "false": 0, "unverified": 0,
                                       "v_conflict": 0, "v_missing": 0, "has_links": 0})
    for t in threads:
        s = topic_stats[t["topic"]]
        s[t["r"]] += 1
        if t["r"] == "rumour":
            if t["v"]:
                s[t["v"]] += 1
            elif t["annotation_keys"]:
                s["v_missing"] += 1
            else:
                s["v_missing"] += 1
            if t["has_links"]:
                s["has_links"] += 1
    asset_audit = {
        "generated_by": "scripts/ses_v1/p0/build_rv_map_and_sample.py",
        "data_source": {
            "provider": "figshare 6392078 v1 (Kochkina, Liakata, Zubiaga; 2018-06-10)",
            "license": "CC BY 4.0",
            "file": "PHEME_veracity.tar.bz2",
            "bytes": 46529729,
            "sha256": DATA_VERSION,
            "archive_safety": json.loads((ASSET / "archive_safety_check.json").read_text(encoding="utf-8"))["status"],
        },
        "events": events,
        "totals": {
            "threads_all": len(threads),
            "rumour": sum(1 for t in threads if t["r"] == "rumour"),
            "non_rumour": sum(1 for t in threads if t["r"] == "non-rumour"),
            "v_true": sum(1 for t in threads if t["v"] == "true"),
            "v_false": sum(1 for t in threads if t["v"] == "false"),
            "v_unverified": sum(1 for t in threads if t["v"] == "unverified"),
            "v_conflict": len(conflicts),
            "v_missing": len(v_missing),
        },
        "conflict_examples": conflicts[:10],
        "v_missing_examples": v_missing[:10],
        "topic_stats": {k: v for k, v in sorted(topic_stats.items())},
    }
    (OUT_DOC / "P0_ASSET_AUDIT.json").write_text(
        json.dumps(asset_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 全量映射（本地） ----
    rv_map = {"salt": SALT, "data_version_sha256": DATA_VERSION, "threads": threads}
    (ASSET / "rv_map.json").write_text(json.dumps(rv_map, ensure_ascii=False), encoding="utf-8")

    # ---- 固定盐选样 ----
    rumour_threads = [t for t in threads if t["r"] == "rumour" and t["v"]]
    manifest_rows, seq = [], 0
    for v_class in ("true", "false", "unverified"):
        pool = [t for t in rumour_threads if t["v"] == v_class]
        topics = sorted({t["topic"] for t in pool},
                        key=lambda tp: hashlib.sha256(tp.encode("utf-8")).hexdigest())
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
            manifest_rows.append({
                "anon_id": f"SES-P0-{seq:03d}",
                "v_class": v_class,
                "topic": t["topic"],
                "thread_hash": sha256_hex(f"{t['topic']}|{t['thread_id']}")[:16],
                "r": t["r"],
                "v": t["v"],
                "topic_order_hash": sha256_hex(t["topic"])[:16],
                "sort_key_hash": sha256_hex(f"{SALT}|{DATA_VERSION}|{t['topic']}|{t['thread_id']}")[:16],
            })

    # ---- 时间边界审计（60 条）+ 审读文本导出 ----
    id_map = []
    review_dir = ASSET / "review"
    review_dir.mkdir(exist_ok=True)
    cutoffs = (15 * 60, 60 * 60, 360 * 60)
    for row in manifest_rows:
        tdir = DATA / _dir_of(rv_map, row)
        src_obj = load_thread_rows(tdir, row["thread_hash"])
        src = src_obj["source"] or {}
        src_ts = ts_epoch(src.get("created_at"))
        counts = {f"in_{m}m": 0 for m in (15, 60, 360)}
        n_react, n_missing_ts, n_parent_conflict = 0, 0, 0
        reactions_out = []
        for rc in src_obj["reactions"]:
            n_react += 1
            rts = ts_epoch(rc.get("created_at"))
            if rts is None:
                n_missing_ts += 1
            elif src_ts is not None:
                delta = rts - src_ts
                for m in (15, 60, 360):
                    if 0 <= delta <= m * 60:
                        counts[f"in_{m}m"] += 1
            reactions_out.append({"id": rc["id"], "created_at": rc.get("created_at"),
                                  "text": rc.get("text"), "urls": rc.get("urls"),
                                  "user_id": rc.get("user_id")})
        row.update({
            "source_created_at": src.get("created_at"),
            "n_reactions": n_react,
            "reaction_ts_missing": n_missing_ts,
            **counts,
            "no_candidate_15m": counts["in_15m"] == 0,
        })
        id_map.append({"anon_id": row["anon_id"], "topic": row["topic"],
                       "thread_hash": row["thread_hash"],
                       "dir": str(tdir.relative_to(DATA)),
                       "source_created_at": src.get("created_at")})
        (review_dir / f"{row['anon_id']}.json").write_text(
            json.dumps({"source": {"created_at": src.get("created_at"), "text": src.get("text"),
                                   "urls": src.get("urls"), "user_id": src.get("user_id"),
                                   "followers": src.get("user_followers")},
                        "annotation_links": _annotation_links(tdir),
                        "reactions": reactions_out}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # ---- 写 manifest CSV ----
    fields = ["anon_id", "v_class", "topic", "thread_hash", "r", "v",
              "topic_order_hash", "sort_key_hash", "source_created_at",
              "n_reactions", "reaction_ts_missing", "in_15m", "in_60m", "in_360m", "no_candidate_15m"]
    with (OUT_DOC / "P0_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r[k] for k in fields})

    (ASSET / "manifest_id_map.json").write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")

    counts_by_class = defaultdict(int)
    for r in manifest_rows:
        counts_by_class[r["v_class"]] += 1
    print(json.dumps({
        "totals": asset_audit["totals"],
        "selected": dict(counts_by_class),
        "shortfalls": {k: TARGETS[k] - counts_by_class.get(k, 0) for k in TARGETS},
        "no_candidate_15m": sum(1 for r in manifest_rows if r["no_candidate_15m"]),
    }, ensure_ascii=False))
    return 0


def _dir_of(rv_map, row):
    for t in rv_map["threads"]:
        if t["topic"] == row["topic"] and hashlib.sha256(f"{t['topic']}|{t['thread_id']}".encode("utf-8")).hexdigest()[:16] == row["thread_hash"]:
            return DATA / t["dir"]
    raise KeyError(row["anon_id"])


def _annotation_links(tdir):
    ann = tdir / "annotation.json"
    if not ann.exists():
        return []
    a = json.loads(ann.read_text(encoding="utf-8"))
    return a.get("links") or []


if __name__ == "__main__":
    sys.exit(main())
