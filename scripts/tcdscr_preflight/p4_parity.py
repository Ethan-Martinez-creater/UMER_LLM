#!/usr/bin/env python
"""TC-DSCR preflight P4: full-event feature parity.

Rebuilds the 240h pipeline from PHEME raw using the authoritative server
preprocessing code (/data/jyz/next/src) for a seed-3090 stratified sample of
100 events, then compares against the frozen
/data/jyz/next/llm/data/pheme_240h_data/graph_final/*.pt bit-for-bit /
within 1e-5.

Writes results/tcdscr/preflight/full_parity_report.json (server), plus
rebuild output kept under .../preflight/p4_rebuild/ (server-side only).
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/data/jyz/next/src")
import pandas as pd
import torch

from rumor_detection.data.adapters.pheme_extension import PhemeExtensionAdapter
from rumor_detection.data.text_cleaning import clean_tweet_pheme
from rumor_detection.data.pheme_extension.user_features import preprocess_userFeature_PHEME
from rumor_detection.data.sentiment import SentimentModel
from rumor_detection.data.embeddings import TextEmbedder
from rumor_detection.data.pheme_extension.graph_builder import save_graphData_pheme
from rumor_detection.data.preprocess_graph import preprocess_graph_data_final

RAW_DIR = "/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads"
REF_DIR = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
LABELS = Path("/data/jyz/next/llm/data/pheme_240h_data/labels.csv")
OUT = Path("/data/jyz/next/llm/results/tcdscr/preflight")
REBUILD = OUT / "p4_rebuild"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 3090
SAMPLE_N = 100

random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. reference metadata ----------
ref = {}
for f in sorted(REF_DIR.glob("*.pt")):
    try:
        g = torch.load(f, map_location="cpu", weights_only=True)
    except Exception:
        continue
    ref[f.stem] = {
        "num_nodes": int(g["num_nodes"]),
        "has_node_ids": "node_ids" in g,
    }
labels = {}
for line in LABELS.read_text(encoding="utf-8").splitlines()[1:]:
    if line.strip():
        eid, lab = line.rsplit(",", 1)
        labels[eid] = int(lab)

# ---------- 2. stratified sample (seed 3090) ----------
buckets = {"source_only": [], "small": [], "medium": [], "large": []}
for eid, meta in ref.items():
    n = meta["num_nodes"]
    if n == 1:
        buckets["source_only"].append(eid)
    elif n <= 9:
        buckets["small"].append(eid)
    elif n <= 49:
        buckets["medium"].append(eid)
    else:
        buckets["large"].append(eid)

quota = {"source_only": 20, "small": 20, "medium": 30, "large": 30}
sample = []
for name, ids in buckets.items():
    ids = sorted(ids)
    for want_label in (0, 1):  # balance labels within each bucket
        pool = [e for e in ids if labels.get(e) == want_label]
        random.shuffle(pool)
        take = quota[name] // 2 if want_label == 0 else quota[name] - quota[name] // 2
        sample.extend(pool[:take])
    if len([e for e in sample if e in set(ids)]) < quota[name]:
        rest = [e for e in ids if e not in sample]
        random.shuffle(rest)
        sample.extend(rest[: quota[name] - len([e for e in sample if e in set(ids)])])
sample = sample[:SAMPLE_N]
print(f"sampled {len(sample)} events; buckets:",
      {k: len(v) for k, v in buckets.items()}, flush=True)

# ---------- 3. full-load canonical frame (user features need full-dataset fit) ----------
adapter = PhemeExtensionAdapter(RAW_DIR)
df = adapter.load()
print("raw records:", len(df), flush=True)
df["text"] = df["raw_text"].apply(clean_tweet_pheme)
df["text"] = df["text"].fillna("")
preprocess_userFeature_PHEME(df)

# ---------- 4. sentiment + embedding for sampled events ----------
sub = df[df["event_id"].isin(set(sample))].copy().reset_index(drop=True)
print("subset records:", len(sub), flush=True)
senti = SentimentModel(
    "/data/jyz/rumor_detection/model/sentiment-roberta-large-english/model",
    device, model_type="english")
sub["senti_feature"] = senti.predict(sub["text"].tolist(), batch_size=64, max_length=128)
del senti
import gc; gc.collect(); torch.cuda.empty_cache()
emb = TextEmbedder("/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2",
                   device=str(device))
embs = emb.encode(sub["text"].tolist(), batch_size=128)
sub["embedding"] = list(embs)
del emb; gc.collect(); torch.cuda.empty_cache()

# ---------- 5. rebuild ----------
# Reference .pt carry max_time_steps=480 -> the frozen 240h data was built with
# 30-minute bins (240h * 60 / 30 = 480), not the current 60-minute default.
TIME_WINDOW_MINUTES = 30
TIME_STEPS = 480
raw_dir = REBUILD / "graph_raw"
fin_dir = REBUILD / "graph_final"
raw_dir.mkdir(parents=True, exist_ok=True)
fail = []
for eid, grp in sub.groupby("event_id"):
    try:
        save_graphData_pheme(grp.copy(), str(eid), str(raw_dir),
                             target_hours=240, time_window_minutes=TIME_WINDOW_MINUTES)
    except Exception as exc:  # noqa: BLE001
        fail.append((eid, f"graph_builder: {exc}"))
preprocess_graph_data_final(str(raw_dir), str(fin_dir), max_nodes=1021,
                            max_workers=8, time_steps=TIME_STEPS)

# ---------- 6. compare ----------
def cos(a, b):
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    return ((a * b).sum(-1) / (na * nb).clamp_min(1e-12)).mean().item()

report = {"seed": SEED, "sample_size": len(sample),
          "reference_root": str(REF_DIR),
          "rebuild_root": str(REBUILD / "graph_final"),
          "time_window_minutes": TIME_WINDOW_MINUTES,
          "time_steps": TIME_STEPS,
          "gates": {"node_ids_exact": [], "edge_index_exact": [], "num_nodes_exact": []},
          "events": [], "build_failures": fail}
for eid in sorted(sample):
    rec = {"event_id": eid, "label": labels.get(eid),
           "ref_num_nodes": ref[eid]["num_nodes"] if eid in ref else None}
    rp = REF_DIR / f"{eid}.pt"
    bp = fin_dir / f"{eid}.pt"
    if not rp.exists() or not bp.exists():
        rec["missing"] = True
        report["events"].append(rec)
        continue
    a = torch.load(rp, map_location="cpu", weights_only=True)
    b = torch.load(bp, map_location="cpu", weights_only=True)
    rec["rebuild_num_nodes"] = int(b["num_nodes"])
    rec["node_ids_exact"] = (a.get("node_ids") == b.get("node_ids"))
    rec["num_nodes_exact"] = bool(a["num_nodes"] == b["num_nodes"])
    ea, eb = a.get("edge_index"), b.get("edge_index")
    rec["edge_index_exact"] = bool(
        ea is not None and eb is not None and ea.shape == eb.shape
        and (ea.numel() == 0 or torch.equal(ea, eb)))
    rec["edge_count_ref"] = int(ea.shape[1]) if ea is not None else 0
    rec["edge_count_rebuild"] = int(eb.shape[1]) if eb is not None else 0
    na, nb = a["node_feat"], b["node_feat"]
    sa, sb = a["struct_feat"], b["struct_feat"]
    rec["node_feat_shape"] = list(na.shape)
    if na.shape == nb.shape:
        rec["text384_max_abs_diff"] = float((na[:, :384] - nb[:, :384]).abs().max())
        rec["extra_block_max_abs_diff"] = float((na[:, 384:] - nb[:, 384:]).abs().max())
        if rec["text384_max_abs_diff"] > 1e-5:
            rec["text384_cosine_mean"] = cos(na[:, :384], nb[:, :384])
    else:
        rec["text384_max_abs_diff"] = None
        rec["shape_mismatch"] = True
    if sa.shape == sb.shape:
        rec["adj_signature_max_abs_diff"] = float((sa[:, :1021] - sb[:, :1021]).abs().max())
        rec["summary3_max_abs_diff"] = float((sa[:, -3:] - sb[:, -3:]).abs().max())
    else:
        rec["adj_signature_max_abs_diff"] = None
        rec["struct_shape_mismatch"] = True
    report["gates"]["node_ids_exact"].append(rec.get("node_ids_exact", False))
    report["gates"]["edge_index_exact"].append(rec.get("edge_index_exact", False))
    report["gates"]["num_nodes_exact"].append(rec.get("num_nodes_exact", False))
    report["events"].append(rec)

ev = [e for e in report["events"] if not e.get("missing") and not e.get("shape_mismatch")]
def frac(key):
    vals = [e[key] for e in ev if e.get(key) is not None]
    return sum(1 for v in vals if v) / max(len(vals), 1)
def maxdiff(key):
    vals = [e[key] for e in ev if e.get(key) is not None]
    return max(vals) if vals else None
report["summary"] = {
    "events_compared": len(ev),
    "node_ids_exact_rate": frac("node_ids_exact"),
    "edge_index_exact_rate": frac("edge_index_exact"),
    "num_nodes_exact_rate": frac("num_nodes_exact"),
    "text384_max_abs_diff_max": maxdiff("text384_max_abs_diff"),
    "extra_block_max_abs_diff_max": maxdiff("extra_block_max_abs_diff"),
    "adj_signature_max_abs_diff_max": maxdiff("adj_signature_max_abs_diff"),
    "summary3_max_abs_diff_max": maxdiff("summary3_max_abs_diff"),
    "any_text384_over_1e-5": any(e.get("text384_max_abs_diff", 0) > 1e-5 for e in ev),
    "any_struct_over_1e-5": any(
        (e.get("adj_signature_max_abs_diff") or 0) > 1e-5
        or (e.get("summary3_max_abs_diff") or 0) > 1e-5 for e in ev),
}
(OUT / "full_parity_report.json").write_text(
    json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report["summary"], indent=1))

# ---------- 7. sentiment precision/batch control ----------
# The only residual extra-block diff lives in the senti column. Test whether it
# is fp16 batch-order noise: recompute the worst event with batch_size=1 and no
# autocast (fp32) and compare both against the reference column.
worst = max(
    (e for e in report["events"] if e.get("extra_block_max_abs_diff") is not None),
    key=lambda e: e["extra_block_max_abs_diff"], default=None)
if worst is not None:
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    eid = worst["event_id"]
    texts = sub[sub["event_id"] == eid]["text"].tolist()
    refg = torch.load(REF_DIR / f"{eid}.pt", map_location="cpu", weights_only=True)
    ref_senti_col = refg["node_feat"][:, 384 + 10]  # senti is the 11th extra col
    mp = "/data/jyz/rumor_detection/model/sentiment-roberta-large-english/model"
    tok = AutoTokenizer.from_pretrained(mp)
    mdl = AutoModelForSequenceClassification.from_pretrained(mp).to(device).eval()
    filled = [t if t and t.strip() else "missing content" for t in texts]
    enc = tok(filled, max_length=128, truncation=True, padding="max_length",
              return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = mdl(input_ids=enc["input_ids"],
                     attention_mask=enc["attention_mask"]).logits.float()
    senti32 = (torch.softmax(logits, -1)[:, 1] - torch.softmax(logits, -1)[:, 0])
    senti32 = senti32.cpu()
    empty = np.array([not (t and t.strip()) for t in texts])
    if empty.any():
        senti32[torch.tensor(empty)] = 0.0
    ref_valid = ref_senti_col[: len(senti32)]
    ctrl = {
        "event_id": eid,
        "n_nodes": int(len(senti32)),
        "ref_vs_rebuild_fp16_batch64_max_abs_diff": worst["extra_block_max_abs_diff"],
        "ref_vs_fp32_batch1_max_abs_diff": float((ref_valid - senti32).abs().max()),
        "rebuild_fp16_vs_fp32_batch1_max_abs_diff": None,
    }
    reb = torch.load(fin_dir / f"{eid}.pt", map_location="cpu", weights_only=True)
    reb_senti = reb["node_feat"][:, 384 + 10][: len(senti32)]
    ctrl["rebuild_fp16_vs_fp32_batch1_max_abs_diff"] = float(
        (reb_senti - senti32).abs().max())
    report["sentiment_control"] = ctrl
    (OUT / "full_parity_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print("sentiment_control:", json.dumps(ctrl))
