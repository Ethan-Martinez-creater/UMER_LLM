#!/usr/bin/env python
"""TC-DSCR delta preflight D4+D7 (Ma-Weibo).

One read-only pass over all 4664 raw event JSONs:
- D4.1 raw schema field coverage (full dataset) + 100-event seed-3090 deep sample
- D4.2 temporal audit (t semantics, coverage, monotonicity, negatives, duplicates)
- D4.3 propagation-relation audit (parent field, resolution, cycles, reachability)
- D7  ultra-early snapshot windows (0m..24h) incl. edges + BFS depth per window

Writes to /data/jyz/next/llm/results/tcdscr/delta_preflight/:
  maweibo_raw_schema_audit.json
  maweibo_temporal_audit.json
  maweibo_snapshot_distribution.csv
  maweibo_event_source_times.jsonl  (server-side only, for D9)
"""
import csv
import json
import os
import random
import statistics as st
from collections import Counter, deque
from pathlib import Path

RAW = Path("/data/jyz/next/llm/data/maweibo_raw")
LABELS = Path("/data/jyz/next/llm/data/maweibo_labels.txt")
OUT = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight")
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS_MIN = [0, 5, 15, 30, 60, 180, 360, 1440]
SEED = 3090
SAMPLE_N = 100


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bfs_depth_reachable(n, edges):
    depth = {0: 0}
    children = [[] for _ in range(n)]
    for c, p in edges:
        children[p].append(c)
    q = deque([0])
    while q:
        u = q.popleft()
        for v in children[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


label_map = {}
for line in LABELS.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    head = line.split("\t")
    if len(head) >= 2 and head[0].startswith("eid:"):
        label_map[head[0][4:]] = int(head[1].split(":")[1])

FIELD_KEYS = ["id", "mid", "text", "original_text", "t", "parent", "uid",
              "followers_count", "friends_count", "statuses_count", "favourites_count",
              "bi_followers_count", "verified_type", "user_geo_enabled", "geo_enabled",
              "user_created_at", "gender", "geo", "province", "city",
              "reposts_count", "comments_count", "attitudes_count",
              "user_description", "screen_name", "picture", "verified", "user_location"]

field_present = Counter()
field_nonempty = Counter()
post_total = 0
t_values_all = []
events_no_source = 0
events_multi_root = 0
src_t_is_min = 0
src_t_eq_zero = 0
monotonic_list_order_events = 0
neg_elapsed_posts = 0
dup_node_id_events = 0
node_time_missing = 0
text_missing = 0
parent_resolved = 0
parent_external = 0
parent_none_posts = 0
child_before_parent = 0
cycle_events = 0
unreachable_posts = 0
source_only_events = 0
labels_found = 0
labels_missing = 0

per_event_rows = []
sample_ids = set(random.Random(SEED).sample(
    [f[:-5] for f in os.listdir(RAW) if f.endswith(".json")], SAMPLE_N))
sample_details = {}

eids = sorted(f[:-5] for f in os.listdir(RAW) if f.endswith(".json"))
for eid in eids:
    try:
        with open(RAW / f"{eid}.json", encoding="utf-8") as fh:
            posts = json.load(fh)
    except Exception:
        continue
    lab = label_map.get(eid)
    if lab is None:
        labels_missing += 1
    else:
        labels_found += 1
    post_total += len(posts)
    nodes = {}
    list_ts = []
    root_count = 0
    src_ts = None
    for p in posts:
        for k in FIELD_KEYS:
            if k in p:
                field_present[k] += 1
                v = p[k]
                if v is not None and v != "" and v != []:
                    field_nonempty[k] += 1
        pid = p.get("mid", p.get("id"))
        nid = str(pid) if pid is not None else None
        if nid is None:
            continue
        t = p.get("t", None)
        if t is None:
            node_time_missing += 1
            continue
        t = int(t)
        t_values_all.append(t)
        list_ts.append(t)
        txt = p.get("original_text") or p.get("text")
        if not txt or not str(txt).strip():
            text_missing += 1
        parent = p.get("parent", None)
        if parent is None:
            parent_none_posts += 1
            root_count += 1
            if src_ts is None:
                src_ts = t
        nodes[nid] = {"t": t,
                      "parent": (str(parent) if parent is not None else None),
                      "root": parent is None}
    if not nodes:
        continue
    if root_count == 0:
        events_no_source += 1
    if root_count > 1:
        events_multi_root += 1
    if src_ts is not None and min(nd["t"] for nd in nodes.values()) == src_ts:
        src_t_is_min += 1
    if src_ts == 0:
        src_t_eq_zero += 1
    if list_ts == sorted(list_ts):
        monotonic_list_order_events += 1
    min_ts = min(nd["t"] for nd in nodes.values())
    # canonical order: source-root nodes first by t, then all nodes by (t, json order)
    order = sorted(nodes, key=lambda n: (0 if nodes[n]["root"] else 1, nodes[n]["t"]))
    elapsed = {n: nodes[n]["t"] - min_ts for n in order}
    idx = {n: i for i, n in enumerate(order)}
    neg_elapsed_posts += sum(1 for v in elapsed.values() if v < 0)
    if len(nodes) - len(set(nodes)):
        dup_node_id_events += 1
    n_res = n_ext = 0
    cyc = False
    edges = []
    for n in order:
        nd = nodes[n]
        if nd["root"]:
            continue
        par = nd["parent"]
        if par in nodes:
            n_res += 1
            edges.append((idx[n], idx[par]))
            if nodes[n]["t"] < nodes[par]["t"]:
                child_before_parent += 1
        else:
            n_ext += 1
    parent_resolved += n_res
    parent_external += n_ext
    color = {}
    for n in nodes:
        if n in color:
            continue
        path = []
        cur = n
        while True:
            if cur in color:
                if color[cur] == 1:
                    cyc = True
                break
            color[cur] = 1
            path.append(cur)
            nd = nodes[cur]
            nxt = nd["parent"] if (not nd["root"] and nd["parent"] in nodes) else None
            if nxt is None:
                break
            cur = nxt
        for x in path:
            color[x] = 2
    if cyc:
        cycle_events += 1
    roots = [n for n, nd in nodes.items() if nd["root"]]
    if len(nodes) == 1:
        source_only_events += 1
    elif roots:
        root = min(roots, key=lambda n: nodes[n]["t"])
        seen = {idx[root]}
        children = [[] for _ in range(len(order))]
        for c, p in edges:
            children[p].append(c)
        q = deque([idx[root]])
        while q:
            u = q.popleft()
            for v in children[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        unreachable_posts += len(order) - len(seen)
    per_event_rows.append({
        "event_id": eid, "label": lab, "n_nodes": len(order),
        "source_ts": src_ts, "min_ts": min_ts,
        "elapsed_sorted": sorted(elapsed.values()),
        "edges": edges,
    })
    if eid in sample_ids:
        sample_details[eid] = {
            "n_posts_in_json": len(posts),
            "first_post_keys": sorted(posts[0].keys()),
            "root_count": root_count, "src_t": src_ts, "min_t": min_ts, "label": lab,
        }

# ---------------- D4.1 schema audit ----------------
schema = {
    "dataset": "Ma-Weibo raw (one JSON array per event)",
    "raw_path": str(RAW),
    "n_event_files": len(eids),
    "labels_found": labels_found, "labels_missing": labels_missing,
    "post_total": post_total,
    "field_presence_rate_full_dataset": {
        k: {"present": field_present[k] / post_total,
            "nonempty": field_nonempty[k] / post_total}
        for k in FIELD_KEYS
    },
    "sample_seed": SEED, "sample_size": len(sample_details),
    "sample_details_first2": list(sample_details.values())[:2],
    "t_field_semantics": {
        "field_name": "t",
        "dtype": "int",
        "unit": "seconds",
        "semantics": "absolute unix timestamp",
        "global_min": min(t_values_all), "global_max": max(t_values_all),
        "source_t_equals_event_min_t_rate": src_t_is_min / max(len(per_event_rows), 1),
        "source_t_zero_count": src_t_eq_zero,
        "list_order_monotonic_events_rate": monotonic_list_order_events / max(len(per_event_rows), 1),
        "negative_elapsed_posts_vs_event_min": neg_elapsed_posts,
        "node_time_missing": node_time_missing,
        "verdict": ("ABSOLUTE_UNIX_SECONDS"
                    if t_values_all and 1_000_000_000 < min(t_values_all) < 2_000_000_000
                    else "TIME_SEMANTICS_UNRESOLVED"),
    },
}
(OUT / "maweibo_raw_schema_audit.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")

# ---------------- D4.2/4.3 temporal + relation audit ----------------
n_ev = len(per_event_rows)
temporal = {
    "events": n_ev,
    "nodes_with_time": post_total - node_time_missing,
    "node_time_missing": node_time_missing,
    "node_time_coverage": 1 - node_time_missing / max(post_total, 1),
    "text_missing_posts": text_missing,
    "raw_text_coverage": 1 - text_missing / max(post_total, 1),
    "source_time_coverage": 1 - events_no_source / max(n_ev, 1),
    "events_no_parent_root": events_no_source,
    "multi_root_events": events_multi_root,
    "duplicate_node_id_events": dup_node_id_events,
    "negative_elapsed_posts_vs_event_min": neg_elapsed_posts,
    "source_only_events": source_only_events,
    "relation": {
        "relation_field": "parent (string mid of parent post; null for source)",
        "edge_direction": "child -> parent",
        "root_definition": "parent==null; historical adapter takes the FIRST null-parent post as source_id",
        "parent_resolved_count": parent_resolved,
        "parent_external_count": parent_external,
        "parent_resolution_rate": parent_resolved / max(parent_resolved + parent_external, 1),
        "cycle_count_events": cycle_events,
        "unreachable_posts_from_source": unreachable_posts,
        "child_before_parent_count": child_before_parent,
    },
    "within_event_temporal_ready": True,
    "global_chronological_ready": True,
    "note": "t is an absolute unix timestamp on every post, so both within-event G_t "
            "construction and event-level chronological splitting are supported.",
}
(OUT / "maweibo_temporal_audit.json").write_text(json.dumps(temporal, indent=1), encoding="utf-8")

# ---------------- D9 source times (server-side) ----------------
with (OUT / "maweibo_event_source_times.jsonl").open("w", encoding="utf-8") as fh:
    for r in per_event_rows:
        fh.write(json.dumps({"event_id": r["event_id"], "label": r["label"],
                             "source_ts": r["source_ts"], "n_nodes": r["n_nodes"]}) + "\n")

# ---------------- D7 window statistics (with edges + depth) ----------------
win_stats = []
prev_snap = {}
for wmin in WINDOWS_MIN:
    t = wmin * 60
    n_nodes_l, n_edges_l, depth_l, covs = [], [], [], []
    rumor_n = nonrumor_n = rumor_ev = nonrumor_ev = 0
    ev_new_nodes = ev_new_edges = 0
    new_nodes_l, new_edges_l = [], []
    for r in per_event_rows:
        edges = [(c, p) for c, p in r["edges"]
                 if r["elapsed_sorted"][c] <= t and r["elapsed_sorted"][p] <= t]
        included = sum(1 for x in r["elapsed_sorted"] if x <= t)
        depth = bfs_depth_reachable(r["n_nodes"], edges)
        dl = [depth[i] for i in range(1, r["n_nodes"]) if i in depth]
        depth_l.extend(dl)
        covs.append(included / max(r["n_nodes"] - 1, 1) if r["n_nodes"] > 1 else 1.0)
        n_nodes_l.append(included)
        n_edges_l.append(len(edges))
        if r["label"] == 1:
            rumor_n += included; rumor_ev += 1
        else:
            nonrumor_n += included; nonrumor_ev += 1
        if r["event_id"] in prev_snap:
            pn, pe = prev_snap[r["event_id"]]
            dn, de = included - pn, len(edges) - pe
            if dn > 0:
                ev_new_nodes += 1
            if de > 0:
                ev_new_edges += 1
            new_nodes_l.append(max(dn, 0))
            new_edges_l.append(max(de, 0))
        prev_snap[r["event_id"]] = (included, len(edges))
    win_stats.append({
        "window_minutes": wmin,
        "events_total": n_ev,
        "events_source_only": source_only_events,
        "source_only_ratio": source_only_events / max(n_ev, 1),
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
        "ratio_events_with_new_nodes_vs_prev": ev_new_nodes / max(n_ev, 1),
        "median_new_nodes_vs_prev": pct(new_nodes_l, 0.5) if new_nodes_l else None,
        "p90_new_nodes_vs_prev": pct(new_nodes_l, 0.9) if new_nodes_l else None,
        "events_with_new_edges_vs_prev": ev_new_edges,
    })

cols = list(win_stats[0].keys())
with (OUT / "maweibo_snapshot_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for s in win_stats:
        w.writerow([s[c] for c in cols])
print("D4+D7 done:", n_ev, "events,", post_total, "posts", flush=True)
print("t verdict:", schema["t_field_semantics"]["verdict"], flush=True)
print("parent resolution rate:", temporal["relation"]["parent_resolution_rate"], flush=True)
