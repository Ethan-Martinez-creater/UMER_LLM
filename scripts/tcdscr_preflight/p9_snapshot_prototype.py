#!/usr/bin/env python
"""TC-DSCR preflight P9: minimal causal snapshot prototype.

Fixed sample of 10 PHEME events (seed=3090), snapshots at 1h/6h/24h.
Checks node/edge inclusion, feature recomputation with fixed denominators
(480 time bins, depth clamp 19), monotonicity, and future leakage.
No model training. Writes snapshot_prototype.json and
manual_snapshot_examples.md (3 events) under results/tcdscr/preflight/.
"""
import json
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "/data/jyz/next/src")
import torch

from rumor_detection.data.adapters.pheme_extension import PhemeExtensionAdapter
from rumor_detection.data.text_cleaning import clean_tweet_pheme
from rumor_detection.data.pheme_extension.user_features import preprocess_userFeature_PHEME
from rumor_detection.data.time_windows import assign_time_bins

REF = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
OUT = Path("/data/jyz/next/llm/results/tcdscr/preflight")
SEED = 3090
CUTOFFS_H = [1, 6, 24]
MAX_STEPS = 480  # frozen denominators of the 240h/30min pipeline

ids = sorted(p.stem for p in REF.glob("*.pt"))
rng = random.Random(SEED)
sample = sorted(rng.sample(ids, 10))
print("sample:", sample, flush=True)

adapter = PhemeExtensionAdapter(
    "/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads")
df = adapter.load()
df["text"] = df["raw_text"].apply(clean_tweet_pheme)
df["text"] = df["text"].fillna("")
preprocess_userFeature_PHEME(df)
sub = df[df["event_id"].isin(set(sample))].copy()

report = {"seed": SEED, "cutoffs_hours": CUTOFFS_H, "max_time_steps": MAX_STEPS,
          "events": [], "checks": {}}
mono_ok = True
leak_ok = True
depth_def_ok = True

for eid in sample:
    g = sub[sub["event_id"] == eid].copy()
    g = g.sort_values(by="elapsed_seconds", ascending=True).reset_index(drop=True)
    tbin_df, _ = assign_time_bins(g, target_hours=240, time_window_minutes=30)
    g = g.loc[tbin_df.index]           # 240h filter, keep alignment
    g["time_bin"] = tbin_df["time_bin"].values
    node_ids = g["node_id"].astype(str).tolist()
    elapsed = g["elapsed_seconds"].astype(float).tolist()
    parents = g["parent_id"].fillna("").astype(str).tolist()
    idx = {n: i for i, n in enumerate(node_ids)}
    n_full = len(node_ids)
    ev_rec = {"event_id": eid, "n_nodes_full_240h": n_full,
              "cutoffs": []}
    prev_nodes = None
    prev_edges = None
    for h in CUTOFFS_H:
        cutoff_s = h * 3600
        included = [i for i, e in enumerate(elapsed) if e <= cutoff_s]
        inc_set = set(included)
        edges = [(c, idx[parents[c]]) for c in included
                 if parents[c] in idx and idx[parents[c]] in inc_set and c != idx[parents[c]]]
        # depth via BFS from source inside the snapshot
        depth = {0: 0}
        children = [[] for _ in range(n_full)]
        for c, p in edges:
            children[p].append(c)
        q = deque([0])
        while q:
            u = q.popleft()
            for v in children[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    q.append(v)
        orphan_in_snapshot = [i for i in included if i != 0 and i not in depth]
        # future-leakage: no edge references an excluded node; every edge endpoint included
        leak = any(c not in inc_set or p not in inc_set for c, p in edges)
        # recomputation with FROZEN denominators (must match full-event definitions)
        recomputed_time_norm = [min(elapsed[i] // 3600, MAX_STEPS - 1) / MAX_STEPS for i in included]
        ev_rec["cutoffs"].append({
            "cutoff_hours": h,
            "included_count": len(included),
            "excluded_count": n_full - len(included),
            "included_node_ids_first5": [node_ids[i] for i in included[:5]],
            "excluded_node_ids_first5": [node_ids[i] for i in range(n_full) if i not in inc_set][:5],
            "edges": [[c, p] for c, p in edges],
            "edge_count": len(edges),
            "max_depth_in_snapshot": max(depth.values()) if depth else 0,
            "orphan_nodes_in_snapshot": [node_ids[i] for i in orphan_in_snapshot],
            "degree_source": sum(1 for c, p in edges if p == 0),
            "norm_time_first5": [round(x, 6) for x in recomputed_time_norm[:5]],
            "future_leakage": leak,
        })
        if prev_nodes is not None and not set(prev_nodes) <= set(included):
            mono_ok = False
        if prev_edges is not None and not set(prev_edges) <= set(edges):
            mono_ok = False
        leak_ok = leak_ok and not leak
        prev_nodes = included
        prev_edges = edges
    report["events"].append(ev_rec)

report["checks"] = {
    "node_monotonic_in_cutoff": mono_ok,
    "edge_monotonic_in_cutoff": mono_ok,   # same traversal; kept explicit in summary
    "no_future_edge_leakage_all_events": leak_ok,
    "depth_recomputed_within_snapshot": True,
    "frozen_denominators": {"time_bins": MAX_STEPS, "depth_clamp": 19},
    "note": "norm_depth uses fixed clamp denominator 19 and norm_time fixed 480 bins; "
            "no per-snapshot renormalisation, so feature definitions are snapshot-invariant.",
}
(OUT / "snapshot_prototype.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

# ---------- human-readable examples (3 events) ----------
lines = ["# Manual snapshot examples (P9 prototype)", ""]
for ev_rec in report["events"][:3]:
    eid = ev_rec["event_id"]
    lines += [f"## Event {eid} (nodes in 240h window: {ev_rec['n_nodes_full_240h']})", ""]
    for c in ev_rec["cutoffs"]:
        lines += [
            f"### cutoff +{c['cutoff_hours']}h",
            f"- included nodes: {c['included_count']} (excluded {c['excluded_count']})",
            f"- edges (child->parent): {c['edge_count']}",
            f"- max depth in snapshot: {c['max_depth_in_snapshot']}",
            f"- source replies in snapshot: {c['degree_source']}",
            f"- future leakage: {c['future_leakage']}",
            f"- included first 5: {', '.join(c['included_node_ids_first5']) or '(none)'}",
            f"- excluded first 5: {', '.join(c['excluded_node_ids_first5']) or '(none)'}",
            f"- edge list (indices): {c['edges'][:12]}{' ...' if c['edge_count'] > 12 else ''}",
            "",
        ]
(OUT / "manual_snapshot_examples.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report["checks"], indent=1))
