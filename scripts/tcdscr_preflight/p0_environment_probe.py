#!/usr/bin/env python
"""TC-DSCR preflight P0: environment freeze + processed-.pt structure probe.

Read-only. Writes environment_server.json and pt_probe.json under
/data/jyz/next/llm/results/tcdscr/preflight/.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

OUT = Path("/data/jyz/next/llm/results/tcdscr/preflight")
OUT.mkdir(parents=True, exist_ok=True)

info = {}

# --- host / python / libs ---
import torch
import transformers
info["python_version"] = sys.version.split()[0]
info["platform"] = platform.platform()
info["torch_version"] = torch.__version__
info["transformers_version"] = transformers.__version__
info["cuda_available"] = torch.cuda.is_available()
info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
try:
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20
    ).stdout.strip()
    info["gpu"] = gpu_name
except Exception as exc:  # noqa: BLE001
    info["gpu"] = f"error: {exc}"

# --- data / checkpoint paths (existence + size) ---
def probe_dir(path):
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    n = sum(1 for _ in p.rglob("*")) if p.is_dir() else 1
    return {"exists": True, "entries": int(n)}

paths = {
    "pheme_raw": "/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads",
    "pheme_processed_authoritative_source": "/data/jyz/next/modified/common/pheme_240h_data",
    "pheme_processed_working_copy": "/data/jyz/next/llm/data/pheme_240h_data",
    "maweibo_processed": "/data/jyz/next/llm/data/maweibo_240h_data",
    "maweibo_raw": "/data/jyz/next/llm/data/maweibo_raw",
    "deberta_v3_large": "/data/jyz/next/llm/model/deberta-v3-large",
    "pheme_checkpoints": "/data/jyz/next/llm/checkpoints/pheme",
    "maweibo_checkpoints": "/data/jyz/next/llm/checkpoints/maweibo",
    "minilm_model": "/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2",
    "sentiment_model": "/data/jyz/rumor_detection/model/sentiment-roberta-large-english",
    "preprocessing_project_src": "/data/jyz/next/src/rumor_detection/data",
}
info["paths"] = {k: probe_dir(v) for k, v in paths.items()}

# --- one .pt probe + dimension survey ---
import torch

gf = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
probe = {}
pts = sorted(gf.glob("*.pt"))
probe["n_pt_files"] = len(pts)
if pts:
    g = torch.load(pts[0], map_location="cpu", weights_only=True)
    probe["keys"] = sorted(g.keys())
    for k in ("node_feat", "struct_feat", "num_nodes", "time_bin", "max_time_steps"):
        if k in g:
            v = g[k]
            probe[k + "_shape"] = list(v.shape) if hasattr(v, "shape") else str(v)
    if "node_ids" in g:
        probe["node_ids_first3"] = [str(x) for x in g["node_ids"][:3]]
        probe["node_ids_len"] = len(g["node_ids"])
    # dimension survey over a deterministic sample of 50 files
    dims = {}
    struct_dims = {}
    has_node_ids = 0
    has_time_bin = 0
    sample = pts[:: max(1, len(pts) // 50)][:50]
    for f in sample:
        try:
            gg = torch.load(f, map_location="cpu", weights_only=True)
        except Exception:
            continue
        nf = gg.get("node_feat")
        sf = gg.get("struct_feat")
        if nf is not None:
            dims[str(nf.shape[-1])] = dims.get(str(nf.shape[-1]), 0) + 1
        if sf is not None:
            struct_dims[str(sf.shape[-1])] = struct_dims.get(str(sf.shape[-1]), 0) + 1
        if "node_ids" in gg:
            has_node_ids += 1
        if "time_bin" in gg:
            has_time_bin += 1
    probe["node_feat_dim_distribution_sample50"] = dims
    probe["struct_feat_dim_distribution_sample50"] = struct_dims
    probe["sample50_has_node_ids"] = has_node_ids
    probe["sample50_has_time_bin"] = has_time_bin

# --- labels.csv / splits / metadata shape ---
lp = Path("/data/jyz/next/llm/data/pheme_240h_data/labels.csv")
if lp.exists():
    head = lp.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    probe["labels_csv_head"] = head
    probe["labels_csv_lines"] = sum(1 for _ in lp.open(encoding="utf-8", errors="replace"))
sp = Path("/data/jyz/next/llm/data/pheme_240h_data/splits")
probe["splits_contents"] = sorted(x.name for x in sp.iterdir()) if sp.exists() else []
mp = Path("/data/jyz/next/llm/data/pheme_240h_data/metadata")
probe["metadata_contents"] = sorted(x.name for x in mp.iterdir()) if mp.exists() else []

# --- raw topic listing ---
raw = Path("/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads")
if raw.exists():
    topics = [d.name for d in raw.iterdir() if d.is_dir()]
    probe["pheme_raw_topics"] = sorted(topics)

# --- preprocessing project version markers ---
src_dir = Path("/data/jyz/next/src")
if src_dir.exists():
    probe["preprocess_src_files"] = sorted(
        str(p.relative_to(src_dir))
        for p in (src_dir / "rumor_detection" / "data").glob("*.py")
    )
    pe = src_dir / "rumor_detection" / "data" / "pheme_extension"
    if pe.exists():
        probe["preprocess_src_pheme_files"] = sorted(p.name for p in pe.glob("*.py"))

(OUT / "environment_server.json").write_text(
    json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
(OUT / "pt_probe.json").write_text(
    json.dumps(probe, indent=1, ensure_ascii=False), encoding="utf-8")
print("WROTE", OUT / "environment_server.json")
print(json.dumps({k: info[k] for k in ("python_version", "torch_version",
                                       "transformers_version", "cuda_version", "gpu")}, indent=1))
print(json.dumps(probe.get("keys", [])))
print("node_feat dims:", probe.get("node_feat_dim_distribution_sample50"))
print("struct dims:", probe.get("struct_feat_dim_distribution_sample50"))
