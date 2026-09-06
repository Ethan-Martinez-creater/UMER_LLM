#!/usr/bin/env python
"""P4b: corrected sentiment control — node-order-aligned fp32/batch1 recompute
vs reference and fp16/batch64 rebuild senti columns."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/jyz/next/src")
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REF = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
REB = Path("/data/jyz/next/llm/results/tcdscr/preflight/p4_rebuild/graph_final")
report = json.load(open("/data/jyz/next/llm/results/tcdscr/preflight/full_parity_report.json", encoding="utf-8"))
ev = [e for e in report["events"] if not e.get("missing") and not e.get("shape_mismatch")]
worst = max(ev, key=lambda e: e.get("extra_block_max_abs_diff") or 0)
eid = worst["event_id"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# rebuild the node-ordered text list exactly like the pipeline does
import pandas as pd
from rumor_detection.data.adapters.pheme_extension import PhemeExtensionAdapter
from rumor_detection.data.text_cleaning import clean_tweet_pheme
adapter = PhemeExtensionAdapter("/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads")
df = adapter.load()
df["text"] = df["raw_text"].apply(clean_tweet_pheme)
df["text"] = df["text"].fillna("")
g = df[df["event_id"] == eid].sort_values(by="elapsed_seconds", ascending=True).reset_index(drop=True)
texts = g["text"].tolist()
print("event", eid, "nodes:", len(texts))

mp = "/data/jyz/rumor_detection/model/sentiment-roberta-large-english/model"
tok = AutoTokenizer.from_pretrained(mp)
mdl = AutoModelForSequenceClassification.from_pretrained(mp).to(device).eval()
print("model id2label:", getattr(mdl.config, "id2label", None), flush=True)
filled = [t if t and t.strip() else "missing content" for t in texts]
out = []
with torch.inference_mode():
    for i in range(0, len(filled), 1):  # batch_size = 1, fp32, no autocast
        enc = tok(filled[i], max_length=128, truncation=True, padding="max_length",
                  return_tensors="pt").to(device)
        logits = mdl(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float()
        p = torch.softmax(logits, -1)[0]
        out.append(float(p[1] - p[0]))
senti32 = np.array(out)
for i, t in enumerate(texts):
    if not t or not t.strip():
        senti32[i] = 0.0
# Reference .pt store the graph-builder-normalised value (raw+1)/2 in [0,1];
# align the control to the same scale before comparing.
senti32_norm = (senti32 + 1.0) / 2.0

refg = torch.load(REF / f"{eid}.pt", map_location="cpu", weights_only=True)
rebg = torch.load(REB / f"{eid}.pt", map_location="cpu", weights_only=True)
ref_senti = refg["node_feat"][:, 384 + 10].numpy()
reb_senti = rebg["node_feat"][:, 384 + 10].numpy()
res = {
    "event_id": eid,
    "n_nodes": len(texts),
    "model_id2label": {str(k): str(v) for k, v in getattr(mdl.config, "id2label", {}).items()},
    "note": "reference/rebuild columns are graph-builder normalised (raw+1)/2; control aligned to same scale",
    "ref_vs_rebuild_fp16_batch64_max_abs_diff": float(np.abs(ref_senti - reb_senti).max()),
    "ref_vs_fp32_batch1_norm_max_abs_diff": float(np.abs(ref_senti - senti32_norm).max()),
    "ref_vs_fp32_batch1_norm_exact_rate": float(np.mean(np.abs(ref_senti - senti32_norm) <= 1e-6)),
}
print(json.dumps(res, indent=1))
Path("/data/jyz/next/llm/results/tcdscr/preflight/p4b_sentiment_control.json").write_text(
    json.dumps(res, indent=1), encoding="utf-8")
