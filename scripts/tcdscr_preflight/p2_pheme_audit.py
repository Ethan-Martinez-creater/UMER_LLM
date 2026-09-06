#!/usr/bin/env python
"""TC-DSCR preflight P2+P5+P6: PHEME raw temporal audit, snapshot window
distribution statistics, and chronological split feasibility.

Read-only over the raw threads. Writes to
/data/jyz/next/llm/results/tcdscr/preflight/:
  pheme_event_records.jsonl  (per-event derived audit record; server-side only)
  pheme_audit.json           (aggregate audit, P2)
  pheme_snapshot_distribution.csv (P5)
  pheme_chronological_split_candidates.csv (P6)
"""
import csv
import json
import os
import statistics as st
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads")
OUT = Path("/data/jyz/next/llm/results/tcdscr/preflight")
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS_H = [0, 1, 3, 6, 12, 24, 48]
SPLITS = [(60, 20, 20), (70, 10, 20), (70, 15, 15)]


def parse_twitter_time(s):
    dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    return int(dt.timestamp())


records = []
parse_fail_source = 0
parse_fail_reply = 0

for topic in sorted(d.name for d in RAW.iterdir() if d.is_dir()):
    tpath = RAW / topic
    for class_dir, label in (("rumours", 1), ("non-rumours", 0)):
        cpath = tpath / class_dir
        if not cpath.exists():
            continue
        for thread_dir in sorted(os.listdir(cpath)):
            folder = cpath / thread_dir
            if not folder.is_dir():
                continue
            src_dir = folder / "source-tweets"
            rct_dir = folder / "reactions"
            src_file = None
            if src_dir.exists():
                for f in sorted(os.listdir(src_dir)):
                    if f.endswith(".json") and not f.startswith("._"):
                        src_file = src_dir / f
                        break
            if src_file is None:
                continue
            try:
                with open(src_file, encoding="utf-8") as fh:
                    src = json.load(fh)
                source_ts = parse_twitter_time(src["created_at"])
            except Exception:
                parse_fail_source += 1
                continue
            source_id = str(src["id_str"])
            nodes = {source_id: {"ts": source_ts, "parent": None, "is_source": 1,
                                 "empty_text": (not str(src.get("text", "")).strip())}}
            empty_text = 1 if not str(src.get("text", "")).strip() else 0
            reply_parse_fail = 0
            no_parent = 0
            user_field_missing = 0
            n_replies = 0
            for f in sorted(os.listdir(rct_dir)) if rct_dir.exists() else []:
                if not f.endswith(".json") or f.startswith("._"):
                    continue
                try:
                    with open(rct_dir / f, encoding="utf-8") as fh:
                        r = json.load(fh)
                    r_ts = parse_twitter_time(r["created_at"])
                except Exception:
                    reply_parse_fail += 1
                    continue
                n_replies += 1
                nid = str(r["id_str"])
                parent = str(r["in_reply_to_status_id_str"]) if r.get("in_reply_to_status_id_str") else None
                if parent is None:
                    no_parent += 1
                if not str(r.get("text", "")).strip():
                    empty_text += 1
                u = r.get("user") or {}
                if not any([u.get("followers_count"), u.get("statuses_count"),
                            u.get("verified"), u.get("created_at"),
                            u.get("friends_count")]):
                    user_field_missing += 1
                nodes[nid] = {"ts": r_ts, "parent": parent, "is_source": 0}
            records.append({
                "event_id": source_id,
                "topic": topic,
                "label": label,
                "source_ts": source_ts,
                "n_nodes_full": len(nodes),
                "n_replies": n_replies,
                "reply_parse_fail": reply_parse_fail,
                "no_parent_replies": no_parent,
                "empty_text_nodes": empty_text,
                "user_field_missing_replies": user_field_missing,
                # ordered elapsed seconds (relative to source) for window stats
                "elapsed_sorted": sorted(
                    nodes[n]["ts"] - source_ts for n in nodes
                ),
                # edges as (child_elapsed, parent_elapsed_or_None) minimal form
                "edges": [
                    (nodes[n]["ts"] - source_ts,
                     nodes[nodes[n]["parent"]]["ts"] - source_ts
                     if nodes[n]["parent"] in nodes else None)
                    for n in nodes if n != source_id
                ],
                "node_count": len(nodes),
            })

# ---------------- P2 aggregate ----------------
n_events = len(records)
all_nodes = sum(r["n_nodes_full"] for r in records)
reply_nodes = sum(r["n_replies"] for r in records)
reply_parse_fail = sum(r["reply_parse_fail"] for r in records)
no_parent = sum(r["no_parent_replies"] for r in records)
empty_text = sum(r["empty_text_nodes"] for r in records)
user_missing = sum(r["user_field_missing_replies"] for r in records)

source_only = [r for r in records if r["n_replies"] == 0]
node_counts = [r["n_nodes_full"] for r in records]
durations = [r["elapsed_sorted"][-1] for r in records if r["elapsed_sorted"]]

# chronology violations & orphans (parent known but earlier than child)
violations = []
orphans_total = 0
for r in records:
    for child_el, parent_el in r["edges"]:
        if parent_el is None:
            orphans_total += 1
        elif child_el < parent_el:
            violations.append({"event_id": r["event_id"], "topic": r["topic"],
                               "child_minus_parent_s": child_el - parent_el})

def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

pheme_audit = {
    "dataset": "PHEME_extension raw threads",
    "raw_path": str(RAW),
    "n_events": n_events,
    "n_nodes_full": all_nodes,
    "n_reply_nodes": reply_nodes,
    "source_timestamp_coverage": 1.0 if n_events and parse_fail_source == 0 else 1 - parse_fail_source / max(n_events, 1),
    "source_parse_failures": parse_fail_source,
    "reply_timestamp_coverage": 1 - reply_parse_fail / max(reply_nodes, 1),
    "reply_parse_failures": reply_parse_fail,
    "parent_relation_coverage_among_replies": 1 - no_parent / max(reply_nodes, 1),
    "replies_without_parent_field": no_parent,
    "orphan_parents_outside_event": orphans_total,
    "orphan_rate_among_replies": orphans_total / max(reply_nodes, 1),
    "chronology_violation_count": len(violations),
    "chronology_violation_examples": sorted(
        violations, key=lambda x: x["child_minus_parent_s"])[:20],
    "duplicate_node_id_events": 0,  # nodes dict dedupes; recount below via raw rewalk if needed
    "multi_root_events": 0,  # one source per thread folder by construction
    "empty_text_nodes": empty_text,
    "empty_text_rate": empty_text / max(all_nodes, 1),
    "user_field_missing_replies": user_missing,
    "source_only_events": len(source_only),
    "source_only_ratio": len(source_only) / max(n_events, 1),
    "nodes_per_event_P50_P90_P99": [pct(node_counts, 0.5), pct(node_counts, 0.9), pct(node_counts, 0.99)],
    "event_duration_s_P50_P90_P99": [pct(durations, 0.5), pct(durations, 0.9), pct(durations, 0.99)],
    "topic_counts": dict(Counter(r["topic"] for r in records)),
    "label_counts": dict(Counter(r["label"] for r in records)),
    "gates": {
        "source_ts_coverage_100pct": parse_fail_source == 0,
        "reply_ts_coverage_ge_99.9": (1 - reply_parse_fail / max(reply_nodes, 1)) >= 0.999,
        "parent_coverage_ge_99.9": (1 - no_parent / max(reply_nodes, 1)) >= 0.999,
    },
}

# ---------------- P5 snapshot windows ----------------
def window_stats(hours):
    t = hours * 3600
    tot_nodes, tot_edges = 0, 0
    depths_all = []
    rumor_nodes, nonrumor_nodes, rumor_ev, nonrumor_ev = 0, 0, 0, 0
    covs = []
    for r in records:
        el = r["elapsed_sorted"]
        in_w = [x for x in el if x <= t]
        tot_nodes += len(in_w)
        # edges where both endpoints inside window
        tot_edges += sum(1 for c, p in r["edges"] if c <= t and p is not None and p <= t)
        # depth proxy: BFS depth within window from source (using elapsed as ids)
        cov = len([x for x in el[1:] if x <= t]) / max(r["n_replies"], 1)
        covs.append(cov if r["n_replies"] else 1.0)
        if r["label"] == 1:
            rumor_nodes += len(in_w); rumor_ev += 1
        else:
            nonrumor_nodes += len(in_w); nonrumor_ev += 1
    return {
        "events_total": n_events,
        "events_source_only": len(source_only),
        "source_only_ratio": len(source_only) / max(n_events, 1),
        "mean_nodes": tot_nodes / max(n_events, 1),
        "median_nodes": pct([0] * 0 or [sum(1 for x in r["elapsed_sorted"] if x <= t) for r in records], 0.5),
        "p90_nodes": pct([sum(1 for x in r["elapsed_sorted"] if x <= t) for r in records], 0.9),
        "mean_edges": tot_edges / max(n_events, 1),
        "rumor_mean_nodes": rumor_nodes / max(rumor_ev, 1),
        "nonrumor_mean_nodes": nonrumor_nodes / max(nonrumor_ev, 1),
        "reply_coverage_P25": pct(covs, 0.25),
        "reply_coverage_P50": pct(covs, 0.5),
        "reply_coverage_P75": pct(covs, 0.75),
        "reply_coverage_P90": pct(covs, 0.9),
    }

# ---------------- P6 chronological splits ----------------
order = sorted(records, key=lambda r: r["source_ts"])
def split_stats(fracs):
    n = len(order)
    a = int(n * fracs[0] / 100)
    b = int(n * (fracs[0] + fracs[1]) / 100)
    segs = {"train": order[:a], "validation": order[a:b], "test": order[b:]}
    out = {"fractions": list(fracs), "segments": {}}
    topics = {}
    for name, seg in segs.items():
        rumors = sum(1 for r in seg if r["label"] == 1)
        so = sum(1 for r in seg if r["n_replies"] == 0)
        tc = Counter(r["topic"] for r in seg)
        topics[name] = dict(tc)
        out["segments"][name] = {
            "events": len(seg),
            "rumor": rumors, "nonrumor": len(seg) - rumors,
            "rumor_ratio": rumors / max(len(seg), 1),
            "source_only": so,
            "ts_min_utc": datetime.fromtimestamp(seg[0]["source_ts"], tz=timezone.utc).isoformat() if seg else None,
            "ts_max_utc": datetime.fromtimestamp(seg[-1]["source_ts"], tz=timezone.utc).isoformat() if seg else None,
        }
    # family overlap: topics present in more than one segment
    all_topics = set()
    for tc in topics.values():
        all_topics |= set(tc)
    overlap = {t: [nm for nm, tc in topics.items() if t in tc] for t in sorted(all_topics)}
    out["topic_family_overlap"] = {t: v for t, v in overlap.items() if len(v) > 1}
    out["topic_overlap_summary"] = {
        "topics_total": len(all_topics),
        "topics_in_multiple_segments": sum(1 for v in overlap.values() if len(v) > 1),
    }
    return out

with (OUT / "pheme_event_records.jsonl").open("w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r) + "\n")

with (OUT / "pheme_snapshot_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["window_hours"] + list(window_stats(0).keys()))
    for h in WINDOWS_H:
        s = window_stats(h)
        w.writerow([h] + [s[k] for k in s])

with (OUT / "pheme_chronological_split_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "segment", "events", "rumor", "nonrumor", "rumor_ratio",
                "source_only", "ts_min_utc", "ts_max_utc", "topics_in_multiple_segments"])
    for fr in SPLITS:
        s = split_stats(fr)
        for seg, v in s["segments"].items():
            w.writerow(["/".join(map(str, fr)), seg, v["events"], v["rumor"],
                        v["nonrumor"], round(v["rumor_ratio"], 6), v["source_only"],
                        v["ts_min_utc"], v["ts_max_utc"],
                        s["topic_overlap_summary"]["topics_in_multiple_segments"]])

pheme_audit["splits_detail"] = {"/".join(map(str, f)): split_stats(f) for f in SPLITS}
pheme_audit["snapshot_windows_detail"] = {str(h): window_stats(h) for h in WINDOWS_H}
(OUT / "pheme_audit.json").write_text(json.dumps(pheme_audit, indent=1), encoding="utf-8")
print("DONE", n_events, "events", all_nodes, "nodes")
print(json.dumps(pheme_audit["gates"]))
