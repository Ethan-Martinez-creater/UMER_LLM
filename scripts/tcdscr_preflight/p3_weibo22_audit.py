#!/usr/bin/env python
"""TC-DSCR preflight P3: Weibo22 (KPG release v1) structural audit.

Read-only over the locally downloaded raw files. Writes
results/tcdscr/preflight/weibo22_audit.json (aggregate only; no raw data).
"""
import json
import statistics as st
from collections import Counter
from pathlib import Path

BASE = Path(r"E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER\data_raw\weibo22\extracted")
OUT = Path(r"E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER\results\tcdscr\preflight")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- labels ----------
label_rows = []
label_file = BASE / "label_all" / "Weibo_label_All.txt"
for line in label_file.read_text(encoding="utf-8").splitlines():
    if line.strip():
        label_rows.append(line.split("\t"))
label_counts = Counter(r[0] for r in label_rows)
topic_counts = Counter(r[1] for r in label_rows)
label_ids = [r[2] for r in label_rows]
dup_labels = len(label_ids) - len(set(label_ids))
label_col_counts = Counter(len(r) for r in label_rows)

# ---------- trees ----------
tree_file = BASE / "td_rvnn" / "data.TD_RvNN.vol_5000.txt"
rows = 0
bad_rows = 0
ids_in_trees = Counter()
roots = set()
max_index = 0
vocab_max = 0
tree_sizes = Counter()  # weibo_id -> node count
chain_depths = []
id_no_label = 0
label_id_set = set(label_ids)
example_lines = []
for line in tree_file.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
        continue
    rows += 1
    parts = line.split("\t")
    wid = parts[0]
    ids_in_trees[wid] += 1
    tree_sizes[wid] += 1
    # structural tokens between wid and first token containing ':'
    struct = [p for p in parts[1:] if ":" not in p]
    text_tokens = [p for p in parts[1:] if ":" in p]
    for tok in struct:
        if tok != "None":
            try:
                max_index = max(max_index, int(tok))
            except ValueError:
                bad_rows += 1
    if struct and struct[-1] == "None" or (len(parts) > 1 and parts[1] == "None"):
        roots.add(wid)
    for tok in text_tokens:
        try:
            idx = int(tok.split(":")[0])
            vocab_max = max(vocab_max, idx)
        except ValueError:
            bad_rows += 1
    if len(example_lines) < 3 and wid in label_id_set:
        example_lines.append({
            "weibo_id": wid,
            "structural_fields": struct[:8],
            "n_struct_fields": len(struct),
            "n_text_tokens": len(text_tokens),
        })

for wid in ids_in_trees:
    if wid not in label_id_set:
        id_no_label += 1

sizes = list(tree_sizes.values())
audit = {
    "dataset": "Weibo22 via KPG release v1",
    "source": {
        "repo_url": "https://github.com/kkkkk001/KPG",
        "release_tag": "v1",
        "release_commit": "8b1d16b3c0485336c6379e056c00b31581459687",
        "main_head_at_download": "7b41f6647fba7f23b8d461a85df33bdf4c699ce8",
        "download_date": "2026-09-06",
        "download_method": "raw.githubusercontent.com via local proxy (server has no GitHub access)",
        "files": {
            "Weibo_label_All.zip": {
                "bytes": 38203,
                "sha256": "451e51bc28a74ff26a19d212acab8796baf468928a908da5ac49b104b88e9ea0",
            },
            "data.TD_RvNN.vol_5000.zip": {
                "bytes": 7541566,
                "sha256": "f58ab42ee01b29a62987a3e6668af18c3dd3148d91d6c6d717a859bbbdcb550f",
            },
        },
    },
    "schema": {
        "label_file_format": "label(false|true) <TAB> topic <TAB> weibo_id — no timestamp, no text, no user fields",
        "tree_file_format": "weibo_id <TAB> [parent-index chain, 'None' marks root] <TAB> wordidx:count ... (vol_5000 vocabulary indices, not raw text)",
    },
    "labels": {
        "rows": len(label_rows),
        "column_count_distribution": dict(label_col_counts),
        "label_counts": dict(label_counts),
        "topic_counts": dict(topic_counts),
        "duplicate_ids": dup_labels,
    },
    "trees": {
        "lines": rows,
        "bad_rows": bad_rows,
        "distinct_weibo_ids": len(ids_in_trees),
        "roots_detected": len(roots),
        "max_structure_index": max_index,
        "max_word_index": vocab_max,
        "nodes_per_tree_mean": st.mean(sizes) if sizes else None,
        "nodes_per_tree_median": st.median(sizes) if sizes else None,
        "nodes_per_tree_max": max(sizes) if sizes else None,
        "ids_without_label_row": id_no_label,
    },
    "temporal_readiness": {
        "absolute_node_timestamp": False,
        "source_absolute_timestamp": False,
        "parent_child_relation": True,
        "repost_chain_semantics": "rooted tree (single parent per node, TD_RvNN format)",
        "raw_text_available": False,
        "user_fields_available": False,
    },
    "example_rows": example_lines,
    "verdict": "NOT_READY_WEIBO22_TEMPORAL",
    "verdict_reason": "Neither source nor node absolute timestamps exist anywhere in the released "
                      "files; text is vocabulary-index-only (vol_5000) and no user fields are "
                      "present, so UMER-style 11D/384D features and global chronological splits "
                      "cannot be constructed from this release.",
}
(OUT / "weibo22_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
print("labels:", dict(label_counts), "topics:", len(topic_counts))
print("trees rows:", rows, "ids:", len(ids_in_trees), "roots:", len(roots))
print("max word idx:", vocab_max, "bad rows:", bad_rows)
print("verdict:", audit["verdict"])
