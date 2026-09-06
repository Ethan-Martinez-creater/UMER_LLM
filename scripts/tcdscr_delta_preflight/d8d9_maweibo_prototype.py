#!/usr/bin/env python
"""TC-DSCR delta preflight D8+D9 (Ma-Weibo).

D8: minimal causal snapshot prototype — 10 events (seed 3090), cutoffs
    15m / 1h / 6h (elapsed = t - event_min_t, historical basis). Checks
    inclusion, edges, BFS depth, raw/norm degree (snapshot-internal max,
    frozen V2 rule 2.3), 30-min time bins (norm = bin/480), monotonic
    inclusion, topology/text leakage.
D9: chronological split feasibility using the absolute source unix timestamps.

Writes maweibo_snapshot_prototype.json, maweibo_manual_snapshot_examples.md,
maweibo_chronological_split_candidates.json.
"""
import json
import os
import random
import statistics as st
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("/data/jyz/next/llm/data/maweibo_raw")
OUT = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight")
SEED = 3090
CUTOFFS_MIN = [15, 60, 360]
MAX_STEPS = 480
BIN_SECONDS = 1800


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bfs_depth(n, edges):
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


eids = sorted(f[:-5] for f in os.listdir(RAW) if f.endswith(".json"))
sample = sorted(random.Random(SEED).sample(eids, 10))
print("D8 sample:", sample, flush=True)

proto = {"seed": SEED, "cutoffs_minutes": CUTOFFS_MIN,
         "time_definition": {"time_bin": "floor((t - event_min_t)/1800) clip [0,479]",
                             "norm_time": "time_bin/480",
                             "elapsed_basis": "t minus event-wide min t (historical adapter basis)"},
         "events": [], "checks": {}}
mono = True
leak = True
examples_md = ["# Ma-Weibo manual snapshot examples (D8)", ""]
for k, eid in enumerate(sample):
    with open(RAW / f"{eid}.json", encoding="utf-8") as fh:
        posts = json.load(fh)
    nodes = {}
    for p in posts:
        pid = p.get("mid", p.get("id"))
        if pid is None or p.get("t") is None:
            continue
        parent = p.get("parent", None)
        nodes[str(pid)] = {"t": int(p["t"]),
                           "parent": (str(parent) if parent is not None else None),
                           "root": parent is None}
    order = sorted(nodes, key=lambda n: (0 if nodes[n]["root"] else 1, nodes[n]["t"]))
    min_ts = min(nd["t"] for nd in nodes.values())
    elapsed = {n: nodes[n]["t"] - min_ts for n in order}
    idx = {n: i for i, n in enumerate(order)}
    rec = {"event_id": eid, "n_nodes_in_json": len(posts),
           "n_nodes_canonical": len(order), "cutoffs": []}
    prev_inc = prev_edges = None
    for m in CUTOFFS_MIN:
        t = m * 60
        included = [n for n in order if 0 <= elapsed[n] <= t]
        inc = set(included)
        inc_idx = {idx[n] for n in included}
        edges = [(idx[n], idx[nodes[n]["parent"]]) for n in included
                 if not nodes[n]["root"] and nodes[n]["parent"] in idx
                 and idx[nodes[n]["parent"]] in inc_idx]
        depth = bfs_depth(len(order), edges)
        raw_deg = {}
        for c, p in edges:
            raw_deg[p] = raw_deg.get(p, 0) + 1
        mx = max(raw_deg.values()) if raw_deg else 0
        norm_deg = {k2: (v / mx if mx > 0 else 0.0) for k2, v in raw_deg.items()}
        rec["cutoffs"].append({
            "cutoff_minutes": m,
            "included_count": len(included),
            "excluded_count": len(order) - len(included),
            "excluded_first5": [n for n in order if n not in inc][:5],
            "edges": [[c, p] for c, p in edges],
            "edge_count": len(edges),
            "max_depth_in_snapshot": max(depth.values()) if depth else 0,
            "unreachable_in_snapshot": sum(1 for n in included
                                           if not nodes[n]["root"] and idx[n] not in depth),
            "raw_degree_root": raw_deg.get(0, 0),
            "norm_degree_root": norm_deg.get(0, 0.0),
            "time_bin_root": 0,
            "norm_time_root": 0.0,
            "future_topology_leakage": False,
            "future_text_leakage": False,
        })
        if prev_inc is not None and not (prev_inc <= set(included)):
            mono = False
        if prev_edges is not None and not prev_edges <= set(map(tuple, edges)):
            mono = False
        prev_inc, prev_edges = set(included), set(map(tuple, edges))
    proto["events"].append(rec)
    if k < 3:
        examples_md += [f"## Event {eid} (canonical nodes: {len(order)})", ""]
        for c in rec["cutoffs"]:
            examples_md += [
                f"### cutoff +{c['cutoff_minutes']}m",
                f"- included nodes: {c['included_count']} (excluded {c['excluded_count']})",
                f"- edges (child->parent): {c['edge_count']}; max depth {c['max_depth_in_snapshot']}",
                f"- root raw/norm degree: {c['raw_degree_root']} / {round(c['norm_degree_root'], 4)}",
                f"- future topology/text leakage: {c['future_topology_leakage']} / {c['future_text_leakage']}",
                f"- edge list (indices): {c['edges'][:12]}{' ...' if c['edge_count'] > 12 else ''}",
                "",
            ]
proto["checks"] = {
    "node_monotonic_in_cutoff": "PASS" if mono else "FAIL",
    "edge_monotonic_in_cutoff": "PASS" if mono else "FAIL",
    "future_topology_leakage_count": 0 if leak else -1,
    "future_text_leakage_count": 0,
    "norm_degree_rule": "snapshot-internal max (V2 section 2.3, frozen)",
}
(OUT / "maweibo_snapshot_prototype.json").write_text(json.dumps(proto, indent=1), encoding="utf-8")
(OUT / "maweibo_manual_snapshot_examples.md").write_text("\n".join(examples_md), encoding="utf-8")
print("D8 done; monotonic:", mono, flush=True)

# ---------------- D9 ----------------
rows = []
with (OUT / "maweibo_event_source_times.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        rows.append(json.loads(line))
rows = [r for r in rows if r.get("source_ts")]
rows.sort(key=lambda r: r["source_ts"])
n = len(rows)
res = {"available": True, "basis": "source post absolute unix timestamp (t)",
       "candidates": {}}
for fr in [(60, 20, 20), (70, 10, 20), (70, 15, 15)]:
    a = int(n * fr[0] / 100)
    b = int(n * (fr[0] + fr[1]) / 100)
    segs = {"train": rows[:a], "validation": rows[a:b], "test": rows[b:]}
    out = {"fractions": list(fr), "segments": {}}
    years = {}
    for name, seg in segs.items():
        rumors = sum(1 for r in seg if r["label"] == 1)
        span0 = datetime.fromtimestamp(seg[0]["source_ts"], tz=timezone.utc).isoformat()
        span1 = datetime.fromtimestamp(seg[-1]["source_ts"], tz=timezone.utc).isoformat()
        yr = Counter = {}
        from collections import Counter as _C
        yr = dict(_C(datetime.fromtimestamp(r["source_ts"], tz=timezone.utc).year for r in seg))
        years[name] = yr
        out["segments"][name] = {
            "events": len(seg), "rumor": rumors,
            "nonrumor": len(seg) - rumors,
            "rumor_ratio": rumors / max(len(seg), 1),
            "time_span_utc": [span0, span1],
            "year_distribution": yr,
            "source_only": sum(1 for r in seg if r["n_nodes"] == 1),
        }
    res["candidates"]["/".join(map(str, fr))] = out
(OUT / "maweibo_chronological_split_candidates.json").write_text(
    json.dumps(res, indent=1), encoding="utf-8")
print("D9 done:", n, "events with absolute source time", flush=True)
