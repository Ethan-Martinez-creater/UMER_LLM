#!/usr/bin/env python
"""P4 diagnostic: locate which columns/blocks drive the residual diffs."""
import json
import sys
from pathlib import Path

import torch

REF = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
REB = Path("/data/jyz/next/llm/results/tcdscr/preflight/p4_rebuild/graph_final")
report = json.load(open("/data/jyz/next/llm/results/tcdscr/preflight/full_parity_report.json", encoding="utf-8"))
ev = [e for e in report["events"] if not e.get("missing") and not e.get("shape_mismatch")]

extra_bad = [e for e in ev if (e.get("extra_block_max_abs_diff") or 0) > 1e-5]
s3_bad = [e for e in ev if (e.get("summary3_max_abs_diff") or 0) > 1e-5]
print("extra diff events:", len(extra_bad), " s3 diff events:", len(s3_bad))

out = {"extra_column_diffs": {}, "s3_column_diffs": {}, "time_bin_mismatch_events": 0,
       "depth_examples": [], "examples": []}

# column-wise for a few worst events
extra_bad.sort(key=lambda e: -e["extra_block_max_abs_diff"])
for e in extra_bad[:10]:
    eid = e["event_id"]
    a = torch.load(REF / f"{eid}.pt", map_location="cpu", weights_only=True)
    b = torch.load(REB / f"{eid}.pt", map_location="cpu", weights_only=True)
    na, nb = a["node_feat"], b["node_feat"]
    col = (na[:, 384:] - nb[:, 384:]).abs().max(dim=0).values.tolist()
    out["extra_column_diffs"][eid] = {
        "labels": ["reposts", "comments", "attitudes", "followers", "statuses",
                   "favourites", "bi_followers", "verified", "geo", "created_norm",
                   "senti", "userFeatures"],
        "col_max_abs_diff": col,
    }

s3_bad.sort(key=lambda e: -e["summary3_max_abs_diff"])
for e in s3_bad[:10]:
    eid = e["event_id"]
    a = torch.load(REF / f"{eid}.pt", map_location="cpu", weights_only=True)
    b = torch.load(REB / f"{eid}.pt", map_location="cpu", weights_only=True)
    sa, sb = a["struct_feat"], b["struct_feat"]
    col = (sa[:, -3:] - sb[:, -3:]).abs().max(dim=0).values.tolist()
    tb_equal = torch.equal(a["time_bin"], b["time_bin"])
    if not tb_equal:
        out["time_bin_mismatch_events"] += 1
    out["s3_column_diffs"][eid] = {
        "labels": ["norm_degree", "norm_depth", "norm_timestep"],
        "col_max_abs_diff": col,
        "time_bin_equal": tb_equal,
    }
    # raw depth check not stored in final .pt; record norm_depth rows that differ
    dd = (sa[:, -2] - sb[:, -2]).abs()
    if dd.max() > 0:
        idx = torch.nonzero(dd > 1e-6).squeeze(1)[:5].tolist()
        out["depth_examples"].append({
            "event": eid,
            "ref_norm_depth": sa[idx, -2].tolist(),
            "reb_norm_depth": sb[idx, -2].tolist(),
        })

Path("/data/jyz/next/llm/results/tcdscr/preflight/p4_diagnostic.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1)[:3000])
