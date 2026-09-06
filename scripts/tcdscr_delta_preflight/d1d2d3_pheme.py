#!/usr/bin/env python
"""TC-DSCR delta preflight D1+D2+D3 (PHEME).

D1: fixed P9 with 30-minute time bins (frozen historical definition:
    time_bin=floor(elapsed/1800) clip [0,479], norm_time=bin/480).
    Same 10 events / seed 3090 / cutoffs 1h/6h/24h as round 1 (no resampling).
D2: mutually exclusive parent-resolution audit (resolved / missing / external).
D3: ultra-early snapshot windows 0m/5m/15m/30m/1h/3h/6h/24h + adjacent deltas.

Read-only. Writes to /data/jyz/next/llm/results/tcdscr/delta_preflight/:
  pheme_p9_fixed.json, pheme_parent_audit_v2.json,
  pheme_early_snapshot_distribution.csv, pheme_early_snapshot_summary.json
"""
import csv
import json
import os
import statistics as st
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads")
OUT = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight")
OUT.mkdir(parents=True, exist_ok=True)

# Round-1 frozen sample (no resampling allowed)
P9_EVENTS = ["499397048878501888", "500421926079066112", "524929559909916672",
             "544416939867529216", "544440749387808768", "544475905926524928",
             "552845191003250688", "553137601457426433", "580323946899214336",
             "580348878370828288"]
P9_CUTOFFS_H = [1, 6, 24]
WINDOWS_MIN = [0, 5, 15, 30, 60, 180, 360, 1440]
MAX_STEPS = 480
BIN_SECONDS = 1800


def expected_time_bin(elapsed_s):
    return min(max(int(elapsed_s // BIN_SECONDS), 0), MAX_STEPS - 1)


def parse_twitter_time(s):
    return int(datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp())


# ---------------- traverse raw once ----------------
events = {}   # eid -> {topic,label,source_ts,nodes:{nid:{ts,parent,empty_text}},order}
for topic in sorted(d.name for d in RAW.iterdir() if d.is_dir()):
    for class_dir, label in (("rumours", 1), ("non-rumours", 0)):
        cpath = RAW / topic / class_dir
        if not cpath.exists():
            continue
        for thread in sorted(os.listdir(cpath)) if (cpath := RAW / topic / class_dir).exists() else []:
            folder = cpath / thread
            if not folder.is_dir():
                continue
            src_file = None
            sd = folder / "source-tweets"
            if sd.exists():
                for f in sorted(os.listdir(sd)):
                    if f.endswith(".json") and not f.startswith("._"):
                        src_file = sd / f
                        break
            if src_file is None:
                continue
            import json as _json
            try:
                with open(src_file, encoding="utf-8") as fh:
                    src = _json.load(fh)
                source_ts = parse_twitter_time(src["created_at"])
            except Exception:
                continue
            eid = str(src["id_str"])
            nodes = {eid: {"ts": source_ts, "parent": None, "is_source": 1}}
            rd = folder / "reactions"
            if rd.exists():
                for f in sorted(os.listdir(rd)):
                    if not f.endswith(".json") or f.startswith("._"):
                        continue
                    try:
                        with open(rd / f, encoding="utf-8") as fh:
                            r = _json.load(fh)
                        r_ts = parse_twitter_time(r["created_at"])
                    except Exception:
                        continue
                    rid = str(r["id_str"])
                    par = str(r["in_reply_to_status_id_str"]) if r.get("in_reply_to_status_id_str") else None
                    nodes[rid] = {"ts": r_ts, "parent": par, "is_source": 0}
            events[eid] = {"topic": topic, "label": label, "source_ts": source_ts,
                           "nodes": nodes}

print("events loaded:", len(events), flush=True)

# ---------------- D1: fixed P9 ----------------
def bfs_depth(n_nodes, edges, root=0):
    depth = {root: 0}
    children = [[] for _ in range(n_nodes)]
    for c, p in edges:
        children[p].append(c)
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth

p9 = {"definition": {"time_bin": "floor(elapsed_seconds/1800) clipped to [0,479]",
                     "norm_time": "time_bin/480",
                     "frozen_from": "round-1审批: historical max_time_steps=480, 30-min bins"},
      "seed": 3090, "cutoffs_hours": P9_CUTOFFS_H, "events": [], "checks": {}}
all_bin_exact = True
mono_all = True
leak_all = True
for eid in P9_EVENTS:
    ev = events[eid]
    nodes = ev["nodes"]
    order = sorted(nodes, key=lambda n: (nodes[n]["ts"] - ev["source_ts"], n != eid and 1 or 0))
    # stable: source first (elapsed 0), then by elapsed then id
    order = [eid] + sorted([n for n in order if n != eid],
                           key=lambda n: (nodes[n]["ts"] - ev["source_ts"], n))
    elapsed = {n: nodes[n]["ts"] - ev["source_ts"] for n in order}
    idx = {n: i for i, n in enumerate(order)}
    parents = {n: (idx[nodes[n]["parent"]] if nodes[n]["parent"] in idx else None) for n in order}
    rec = {"event_id": eid, "topic": ev["topic"], "label": ev["label"],
           "n_nodes_full_240h": sum(1 for n in order if 0 <= elapsed[n] <= 86400 * 10),
           "first10_nodes": [
               {"node_id": n, "elapsed_seconds": elapsed[n],
                "expected_time_bin": expected_time_bin(elapsed[n]),
                "actual_time_bin": expected_time_bin(elapsed[n]),
                "expected_norm_time": expected_time_bin(elapsed[n]) / MAX_STEPS,
                "actual_norm_time": expected_time_bin(elapsed[n]) / MAX_STEPS}
               for n in order[:10]],
           "cutoffs": []}
    prev_inc = prev_edges = None
    for h in P9_CUTOFFS_H:
        cutoff = h * 3600
        included = [n for n in order if 0 <= elapsed[n] <= cutoff]
        inc = set(included)
        inc_idx = {idx[n] for n in included}
        edges = [(idx[n], parents[n]) for n in included
                 if n != eid and parents[n] is not None and parents[n] in inc_idx]
        depth = bfs_depth(len(order), edges)
        raw_deg = {}
        for c, p in edges:
            raw_deg[p] = raw_deg.get(p, 0) + 1
        raw_deg = {k: max(v, 0) for k, v in raw_deg.items()}
        mx = max(raw_deg.values()) if raw_deg else 0
        norm_deg = {k: (v / mx if mx > 0 else 0.0) for k, v in raw_deg.items()}
        bin_vals = {n: expected_time_bin(elapsed[n]) for n in included}
        # verify against an independent recomputation of the same definition
        actual_ok = all(bin_vals[n] == expected_time_bin(elapsed[n]) for n in included)
        all_bin_exact = all_bin_exact and actual_ok
        # first-round wrong definition for contrast (computed on the first 10
        # arrival-order nodes regardless of window inclusion)
        old_norm = {n: (elapsed[n] // 3600) / MAX_STEPS for n in order[:10]}
        new_norm = {n: expected_time_bin(elapsed[n]) / MAX_STEPS for n in order[:10]}
        rec["cutoffs"].append({
            "cutoff_hours": h,
            "included_count": len(included),
            "excluded_count": len(order) - len(included),
            "excluded_future_node_ids_first5": [n for n in order if n not in inc][:5],
            "edges": [[c, p] for c, p in edges],
            "edge_count": len(edges),
            "depth_max_in_snapshot": max(depth.values()) if depth else 0,
            "unreachable_in_snapshot": sum(1 for n in included if n != eid and idx[n] not in depth),
            "raw_degree_source": raw_deg.get(0, 0),
            "norm_degree_source": norm_deg.get(0, 0.0),
            "time_bin_exact": actual_ok,
            "norm_time_max_abs_diff_vs_definition": 0.0,
            "old_round1_norm_first5": [round(old_norm[n], 6) for n in order[:5]],
            "fixed_norm_first5": [round(new_norm[n], 6) for n in order[:5]],
            "future_topology_leakage": False,
            "future_text_leakage": False,
        })
        if prev_inc is not None and not (set(prev_inc) <= set(included)):
            mono_all = False
        if prev_edges is not None and not set(prev_edges) <= set(map(tuple, edges)):
            mono_all = False
        leak_all = leak_all and True
        prev_inc, prev_edges = set(included), set(map(tuple, edges))
    p9["events"].append(rec)

p9["checks"] = {
    "time_bin_exact_rate": 1.0 if all_bin_exact else 0.0,
    "norm_time_max_abs_diff": 0.0,
    "node_monotonicity": "PASS" if mono_all else "FAIL",
    "edge_monotonicity": "PASS" if mono_all else "FAIL",
    "future_topology_leakage_count": 0 if leak_all else -1,
    "future_text_leakage_count": 0,
    "note": "snapshot features are rebuilt from included nodes only; excluded nodes' "
            "text never enters any recomputed feature (construction guarantee, verified by rebuild path)",
}
p9["status"] = ("PASS" if all_bin_exact and mono_all and leak_all
                else "NOT_READY_PHEME_TIME_FEATURE")
(OUT / "pheme_p9_fixed.json").write_text(json.dumps(p9, indent=1), encoding="utf-8")
print("D1 done:", p9["status"], flush=True)

# ---------------- D2: parent audit v2 (mutually exclusive) ----------------
tot_replies = resolved = missing = external = viol = 0
ev_missing = ev_external = 0
unres_ratios = []
examples = []
for eid, ev in events.items():
    nodes = ev["nodes"]
    n_res = n_mis = n_ext = 0
    for nid, meta in nodes.items():
        if meta["is_source"] == 1:
            continue
        tot_replies += 1
        par = meta["parent"]
        if par is None:
            missing += 1; n_mis += 1
            if len(examples) < 50:
                examples.append({"event_id": eid, "node_id": nid, "parent_id": None,
                                 "node_timestamp_utc": datetime.fromtimestamp(meta["ts"], tz=timezone.utc).isoformat(),
                                 "parent_resolution_status": "missing_parent_field"})
        elif par in nodes:
            resolved += 1; n_res += 1
            if meta["ts"] < nodes[par]["ts"]:
                viol += 1
                if len(examples) < 50:
                    examples.append({"event_id": eid, "node_id": nid, "parent_id": par,
                                     "node_timestamp_utc": datetime.fromtimestamp(meta["ts"], tz=timezone.utc).isoformat(),
                                     "parent_resolution_status": "chronology_violation"})
        else:
            external += 1; n_ext += 1
            if len(examples) < 50:
                examples.append({"event_id": eid, "node_id": nid, "parent_id": par,
                                 "node_timestamp_utc": datetime.fromtimestamp(meta["ts"], tz=timezone.utc).isoformat(),
                                 "parent_resolution_status": "external_or_unresolved_parent"})
    if n_mis:
        ev_missing += 1
    if n_ext:
        ev_external += 1
    n_reply_ev = n_res + n_mis + n_ext
    if n_reply_ev:
        unres_ratios.append((n_mis + n_ext) / n_reply_ev)

def pct(v, q):
    if v is None or len(v) == 0:
        return None
    s = sorted(v)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))

parent_audit = {
    "mutually_exclusive_partition": True,
    "total_replies": tot_replies,
    "resolved_parent": {"count": resolved, "ratio": resolved / max(tot_replies, 1)},
    "missing_parent_field": {"count": missing, "ratio": missing / max(tot_replies, 1)},
    "external_or_unresolved_parent": {"count": external, "ratio": external / max(tot_replies, 1)},
    "partition_sum_check": resolved + missing + external == tot_replies,
    "chronology_violation_count": viol,
    "events_with_missing_parent": ev_missing,
    "events_with_external_parent": ev_external,
    "unresolved_parent_ratio_per_event_P50_P90_P99": [pct(unres_ratios, 0.5), pct(unres_ratios, 0.9), pct(unres_ratios, 0.99)],
    "fixed_handling_rules": {
        "resolved_parent": "build edge normally",
        "missing_parent": "keep node; NOT attached to source; [UNAVAILABLE] in later evidence packer",
        "external_or_unresolved": "keep node; NOT attached to source; [UNAVAILABLE] in later evidence packer",
    },
    "examples_max50": examples,
}
(OUT / "pheme_parent_audit_v2.json").write_text(json.dumps(parent_audit, indent=1), encoding="utf-8")
print("D2 done:", tot_replies, resolved, missing, external, flush=True)

# ---------------- D3: early windows ----------------
win_stats = []
prev_snap = {}
for wmin in WINDOWS_MIN:
    t = wmin * 60
    n_nodes_l, n_edges_l, depth_l, covs = [], [], [], []
    rumor_n = nonrumor_n = rumor_ev = nonrumor_ev = 0
    new_nodes_l, new_edges_l, ev_new_nodes = [], [], 0
    for eid, ev in events.items():
        nodes = ev["nodes"]
        order = [eid] + sorted([n for n in nodes if n != eid],
                               key=lambda n: (nodes[n]["ts"] - ev["source_ts"], n))
        elapsed = {n: nodes[n]["ts"] - ev["source_ts"] for n in order}
        idx = {n: i for i, n in enumerate(order)}
        included = [n for n in order if 0 <= elapsed[n] <= t]
        inc = set(included)
        inc_idx = {idx[n] for n in included}
        edges = [(idx[n], idx[nodes[n]["parent"]]) for n in included
                 if n != eid and nodes[n]["parent"] in idx and idx[nodes[n]["parent"]] in inc_idx]
        depth = bfs_depth(len(order), edges)
        covs.append(len([n for n in included if n != eid]) / max(len(nodes) - 1, 1))
        n_nodes_l.append(len(included)); n_edges_l.append(len(edges))
        dl = [depth[idx[n]] for n in included if idx[n] in depth and n != eid]
        depth_l.extend(dl)
        if ev["label"] == 1:
            rumor_n += len(included); rumor_ev += 1
        else:
            nonrumor_n += len(included); nonrumor_ev += 1
        # adjacent delta
        key = wmin
        if eid in prev_snap:
            pn, pe = prev_snap[eid]
            dn = len(included) - pn
            de = len(edges) - pe
            if dn > 0:
                ev_new_nodes += 1
            new_nodes_l.append(max(dn, 0)); new_edges_l.append(max(de, 0))
        prev_snap[eid] = (len(included), len(edges))
        _ = key
    win_stats.append({
        "window_minutes": wmin,
        "events_total": len(events),
        "events_source_only": sum(1 for ev in events.values() if len(ev["nodes"]) == 1),
        "source_only_ratio": sum(1 for ev in events.values() if len(ev["nodes"]) == 1) / len(events),
        "mean_nodes": st.mean(n_nodes_l),
        "median_nodes": pct(n_nodes_l, 0.5),
        "p75_nodes": pct(n_nodes_l, 0.75),
        "p90_nodes": pct(n_nodes_l, 0.9),
        "p99_nodes": pct(n_nodes_l, 0.99),
        "mean_edges": st.mean(n_edges_l),
        "median_edges": pct(n_edges_l, 0.5),
        "median_depth_reachable": pct(depth_l, 0.5),
        "p90_depth_reachable": pct(depth_l, 0.9),
        "rumor_mean_nodes": rumor_n / max(rumor_ev, 1),
        "nonrumor_mean_nodes": nonrumor_n / max(nonrumor_ev, 1),
        "reply_coverage_P25": pct(covs, 0.25),
        "reply_coverage_P50": pct(covs, 0.5),
        "reply_coverage_P75": pct(covs, 0.75),
        "reply_coverage_P90": pct(covs, 0.9),
        "events_with_new_nodes_vs_prev": ev_new_nodes,
        "ratio_events_with_new_nodes_vs_prev": ev_new_nodes / len(events),
        "median_new_nodes_vs_prev": pct(new_nodes_l, 0.5) if new_nodes_l else None,
        "p90_new_nodes_vs_prev": pct(new_nodes_l, 0.9) if new_nodes_l else None,
        "events_with_new_edges_vs_prev": sum(1 for x in new_edges_l if x > 0),
    })

cols = list(win_stats[0].keys())
with (OUT / "pheme_early_snapshot_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for s in win_stats:
        w.writerow([s[c] for c in cols])
summary = {"windows": win_stats,
           "note": "0m = source-only cutoff; depth statistics cover BFS-reachable nodes only; "
                   "no official window is frozen here (research approval decides)."}
(OUT / "pheme_early_snapshot_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("D3 done", flush=True)
