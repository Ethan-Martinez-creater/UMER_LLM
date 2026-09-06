#!/usr/bin/env python
"""TC-DSCR delta preflight D6: Ma-Weibo full-event feature parity.

Rebuilds the historical 240h/30-min pipeline from raw JSON for 100 events
(seed 3090, stratified) and compares against the frozen
/data/jyz/next/llm/data/maweibo_240h_data/graph_final/*.pt.

NOTE: reference .pt do NOT store node_ids (historical behaviour). Node-order
equivalence is therefore proven jointly by:
  - num_nodes exact,
  - time_bin row-exact (node-level temporal fingerprint),
  - text384 row-wise diffs (each row is one node's embedding),
  - adjacency signature (bit-level structure).

Per the V2 protocol, 11D extra features are NOT a pass condition; they are
reported as INFO only. Writes maweibo_full_parity_report.json.
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/data/jyz/next/src")
import torch

from rumor_detection.data.adapters.ma_weibo import MaWeiboAdapter
from rumor_detection.data.text_cleaning import clean_text_weibo
from rumor_detection.data.ma_weibo.user_features import preprocess_userFeature
from rumor_detection.data.sentiment import SentimentModel
from rumor_detection.data.embeddings import TextEmbedder
from rumor_detection.data.ma_weibo.graph_builder import save_graphData_Byeventid
from rumor_detection.data.preprocess_graph import preprocess_graph_data_final

RAW_DIR = "/data/jyz/next/llm/data/maweibo_raw"
LABELS = "/data/jyz/next/llm/data/maweibo_labels.txt"
REF_DIR = Path("/data/jyz/next/llm/data/maweibo_240h_data/graph_final")
OUT = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight")
REBUILD = OUT / "d6_rebuild"
SEED = 3090
SAMPLE_N = 100
OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- reference metadata ----------
ref = {}
for f in sorted(REF_DIR.glob("*.pt")):
    try:
        g = torch.load(f, map_location="cpu", weights_only=True)
    except Exception:
        continue
    ref[f.stem] = {"num_nodes": int(g["num_nodes"]),
                   "node_feat_dim": int(g["node_feat"].shape[-1]),
                   "max_time_steps": int(g["max_time_steps"])}

labels = {}
for line in open(LABELS, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    head = line.split("\t")
    if head[0].startswith("eid:"):
        labels[head[0][4:]] = int(head[1].split(":")[1])

# ---------- stratified sample ----------
buckets = {"source_only": [], "small": [], "medium": [], "large": []}
for eid, meta in ref.items():
    n = meta["num_nodes"]
    if n == 1:
        buckets["source_only"].append(eid)
    elif n <= 99:
        buckets["small"].append(eid)
    elif n <= 499:
        buckets["medium"].append(eid)
    else:
        buckets["large"].append(eid)
quota = {"source_only": len(buckets["source_only"]), "small": 20, "medium": 30, "large": 44}
sample = []
for name, ids in buckets.items():
    ids = sorted(ids)
    half = quota[name] // 2
    for want in (0, 1):
        pool = [e for e in ids if labels.get(e) == want]
        random.shuffle(pool)
        take = half if want == 0 else quota[name] - half
        sample.extend(pool[:take])
    got = len([e for e in sample if e in set(ids)])
    if got < quota[name]:
        rest = [e for e in ids if e not in sample]
        random.shuffle(rest)
        sample.extend(rest[: quota[name] - got])
sample = sample[:SAMPLE_N]
print(f"sampled {len(sample)}:", {k: len(v) for k, v in buckets.items()}, flush=True)

# ---------- full historical load (user features are frame-fitted) ----------
adapter = MaWeiboAdapter(LABELS, RAW_DIR)
df = adapter.load()
print("raw records:", len(df), flush=True)
# V2 fix: historical pipeline embedded clean_text_weibo(original_text); the
# current server pipeline.py lost this step (d6_diagnostic proved ref rows
# match cleaned text at cosine 1.0). Restore it for parity.
df["text"] = df["raw_text"].apply(clean_text_weibo)
df["text"] = df["text"].fillna("")
preprocess_userFeature(df)

sub = df[df["event_id"].isin(set(sample))].copy().reset_index(drop=True)
print("subset records:", len(sub), flush=True)

senti = SentimentModel(
    "/data/jyz/rumor_detection/model/Erlangshen-Roberta-110M-Sentiment",
    device, model_type="chinese")
sub["senti_feature"] = senti.predict(sub["text"].tolist(), batch_size=64, max_length=128)
del senti
import gc
gc.collect(); torch.cuda.empty_cache()
emb = TextEmbedder("/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2",
                   device=str(device))
embs = emb.encode(sub["text"].tolist(), batch_size=128)
sub["embedding"] = list(embs)
del emb
gc.collect(); torch.cuda.empty_cache()

# ---------- rebuild ----------
raw_dir = REBUILD / "graph_raw"
fin_dir = REBUILD / "graph_final"
raw_dir.mkdir(parents=True, exist_ok=True)
fails = []
for eid, grp in sub.groupby("event_id"):
    try:
        save_graphData_Byeventid(grp.copy(), str(eid), str(raw_dir),
                                 target_hours=240, time_window_minutes=30)
    except Exception as exc:  # noqa: BLE001
        fails.append((eid, f"graph_builder: {exc}"))
preprocess_graph_data_final(str(raw_dir), str(fin_dir), max_nodes=1021,
                            max_workers=8, time_steps=480)

# ---------- compare ----------
def rowwise_max_diff(a, b):
    return float((a - b).abs().max()) if a.shape == b.shape else None

report = {"seed": SEED, "text_cleaning_fixed": True, "sample_size": len(sample),
          "reference_root": str(REF_DIR), "rebuild_root": str(fin_dir),
          "time_window_minutes": 30, "time_steps": 480,
          "reference_stores_node_ids": False,
          "events": [], "build_failures": fails}
for eid in sorted(sample):
    rec = {"event_id": eid, "label": labels.get(eid),
           "ref_num_nodes": ref[eid]["num_nodes"] if eid in ref else None}
    rp, bp = REF_DIR / f"{eid}.pt", fin_dir / f"{eid}.pt"
    if not rp.exists() or not bp.exists():
        rec["missing"] = True
        report["events"].append(rec)
        continue
    a = torch.load(rp, map_location="cpu", weights_only=True)
    b = torch.load(bp, map_location="cpu", weights_only=True)
    rec["rebuild_num_nodes"] = int(b["num_nodes"])
    rec["num_nodes_exact"] = bool(a["num_nodes"] == b["num_nodes"])
    rec["time_bin_row_exact"] = bool(a["time_bin"].shape == b["time_bin"].shape
                                     and torch.equal(a["time_bin"], b["time_bin"]))
    na, nb = a["node_feat"], b["node_feat"]
    rec["node_feat_dim_ref"] = int(na.shape[-1])
    rec["node_feat_dim_rebuild"] = int(nb.shape[-1])
    if na.shape == nb.shape:
        rec["text384_rowwise_max_abs_diff"] = rowwise_max_diff(na[:, :384], nb[:, :384])
        rec["extra11_max_abs_diff_INFO"] = rowwise_max_diff(na[:, 384:395], nb[:, 384:395])
    sa, sb = a["struct_feat"], b["struct_feat"]
    if sa.shape == sb.shape:
        rec["adj_signature_max_abs_diff"] = float((sa[:, :1021] - sb[:, :1021]).abs().max())
        rec["summary3_max_abs_diff"] = float((sa[:, -3:] - sb[:, -3:]).abs().max())
    rec["max_time_steps_ref"] = int(a["max_time_steps"])
    rec["max_time_steps_rebuild"] = int(b["max_time_steps"])
    report["events"].append(rec)

ev = [e for e in report["events"] if not e.get("missing")]
def rate(key):
    vals = [e[key] for e in ev if e.get(key) is not None]
    return sum(1 for v in vals if v) / max(len(vals), 1)
def maxd(key):
    vals = [e[key] for e in ev if e.get(key) is not None]
    return max(vals) if vals else None
report["summary"] = {
    "events_compared": len(ev),
    "num_nodes_exact_rate": rate("num_nodes_exact"),
    "time_bin_row_exact_rate": rate("time_bin_row_exact"),
    "text384_rowwise_max_abs_diff_max": maxd("text384_rowwise_max_abs_diff"),
    "extra11_max_abs_diff_INFO_max": maxd("extra11_max_abs_diff_INFO"),
    "adj_signature_max_abs_diff_max": maxd("adj_signature_max_abs_diff"),
    "summary3_max_abs_diff_max": maxd("summary3_max_abs_diff"),
    "node_feat_dim_ref_distribution": sorted({e["node_feat_dim_ref"] for e in ev}),
    "node_feat_dim_rebuild_distribution": sorted({e["node_feat_dim_rebuild"] for e in ev}),
}
(OUT / "maweibo_full_parity_report.json").write_text(
    json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report["summary"], indent=1))
