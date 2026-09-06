#!/usr/bin/env python
"""D6 diagnostic: locate the source of text384 / extra11 diffs.

For the worst events:
 1) column-wise extra11 diff (senti vs user dims);
 2) per-node text-variant matching: encode original_text, text, and
    cleaned(original_text) with MiniLM, then match each reference .pt row
    against all variants to see which text the historical pipeline embedded.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/jyz/next/src")
import torch

from rumor_detection.data.adapters.ma_weibo import MaWeiboAdapter
from rumor_detection.data.embeddings import TextEmbedder

REF = Path("/data/jyz/next/llm/data/maweibo_240h_data/graph_final")
REB = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight/d6_rebuild/graph_final")
RAW = Path("/data/jyz/next/llm/data/maweibo_raw")
OUT = Path("/data/jyz/next/llm/results/tcdscr/delta_preflight")
report = json.load(open(OUT / "maweibo_full_parity_report.json", encoding="utf-8"))
ev = [e for e in report["events"] if not e.get("missing")]
bad = sorted(ev, key=lambda e: -(e.get("text384_rowwise_max_abs_diff") or 0))[:5]
print("worst:", [(e["event_id"], round(e["text384_rowwise_max_abs_diff"], 3)) for e in bad])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb = TextEmbedder("/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2",
                   device=str(device))

out = {"extra_column_diffs": {}, "text_variant_matching": []}
for e in bad[:2]:
    eid = e["event_id"]
    a = torch.load(REF / f"{eid}.pt", map_location="cpu", weights_only=True)
    b = torch.load(REB / f"{eid}.pt", map_location="cpu", weights_only=True)
    col = (a["node_feat"][:, 384:] - b["node_feat"][:, 384:]).abs().max(dim=0).values.tolist()
    out["extra_column_diffs"][eid] = {
        "labels": ["reposts", "comments", "attitudes", "followers", "statuses",
                   "favourites", "bi_fol_inv", "verified", "geo", "created_norm", "senti"],
        "col_max_abs_diff": col,
    }
    # text variants per node, canonical order used by graph_builder
    adapter = MaWeiboAdapter("/data/jyz/next/llm/data/maweibo_labels.txt", str(RAW))
    df_ev = adapter.load()
    # adapter.load is huge; instead read the single event file directly
    del df_ev
    with open(RAW / f"{eid}.json", encoding="utf-8") as fh:
        posts = json.load(fh)
    nodes = {}
    for p in posts:
        pid = p.get("mid", p.get("id"))
        if pid is None or p.get("t") is None:
            continue
        nodes[str(pid)] = p
    order = sorted(nodes, key=lambda n: (0 if nodes[n].get("parent") is None else 1,
                                         nodes[n]["t"]))
    texts_orig = [str(nodes[n].get("original_text") or nodes[n].get("text") or "") for n in order]
    texts_raw = [str(nodes[n].get("text") or "") for n in order]
    import re, html as _html
    def clean_wb(t):
        if not isinstance(t, str) or not t.strip():
            return ""
        t = _html.unescape(t)
        t = re.sub(r'<[^>]+>', '', t)
        t = re.sub(r'http\S*', '', t, flags=re.I)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    texts_clean = [clean_wb(t) for t in texts_orig]
    E_orig = emb.encode(texts_orig, batch_size=128, show_progress=False)
    E_raw = emb.encode(texts_raw, batch_size=128, show_progress=False)
    E_clean = emb.encode(texts_clean, batch_size=128, show_progress=False)
    ref = a["node_feat"][:, :384]
    reb = b["node_feat"][:, :384]
    def cosmat(A, B):
        An = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Bn = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return An @ Bn.T
    m_ref_orig = cosmat(ref, torch.tensor(E_orig))
    m_ref_raw = cosmat(ref, torch.tensor(E_raw))
    m_ref_clean = cosmat(ref, torch.tensor(E_clean))
    m_reb_orig = cosmat(reb, torch.tensor(E_orig))
    diag = lambda M: M.diagonal()
    out["text_variant_matching"].append({
        "event_id": eid,
        "n_nodes": ref.shape[0],
        "ref_vs_orig_diag_cos_min": float(diag(m_ref_orig).min()),
        "ref_vs_orig_diag_cos_mean": float(diag(m_ref_orig).mean()),
        "ref_vs_raw_diag_cos_mean": float(diag(m_ref_raw).mean()),
        "ref_vs_clean_diag_cos_mean": float(diag(m_ref_clean).mean()),
        "ref_vs_orig_best_match_rowwise_mean": float(m_ref_orig.max(dim=1).values.mean()),
        "reb_vs_orig_diag_cos_min": float(diag(m_reb_orig).min()),
        "reb_vs_orig_diag_cos_mean": float(diag(m_reb_orig).mean()),
        "ref_vs_reb_diag_cos_min": float(diag(cosmat(ref, reb)).min()),
        "ref_vs_reb_diag_cos_mean": float(diag(cosmat(ref, reb)).mean()),
        "rows_where_ref_matches_orig_below_0_99": int((diag(m_ref_orig) < 0.99).sum()),
        "rows_where_ref_matches_raw_above_orig": int((diag(m_ref_raw) > diag(m_ref_orig)).sum()),
        "rows_where_ref_matches_clean_above_orig": int((diag(m_ref_clean) > diag(m_ref_orig)).sum()),
    })
(OUT / "d6_diagnostic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1)[:2500])
