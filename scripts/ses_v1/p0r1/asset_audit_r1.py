# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 资产补齐审计（F6）：对固定 60 条线程计算

1. 逐线程输入文件哈希（source/annotation/structure/reactions 合并流）
2. 字段覆盖分母（created_at / user.id / in_reply_to / entities.urls / structure 边）
3. 父边三态审计：tree 有边 / metadata(in_reply_to) 有边 / 两者都有且不同；
   缺 parent node（in_reply_to 指向树外）；父晚于子；缺/无效时间
4. thread 伪名哈希与内容哈希并列（伪名哈希不改称内容哈希）

输出：local_assets/ses_v1/p0r1/asset_audit_full.json（含哈希，本地）
      docs/check/ses_v1/p0r1/P0R1_ASSET_COVERAGE.json（聚合统计，可推送，无原始 ID）
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET0 = REPO / "local_assets" / "ses_v1" / "p0"
ASSET1 = REPO / "local_assets" / "ses_v1" / "p0r1"
DATA = ASSET0 / "extracted" / "all-rnr-annotated-threads"
OUT1 = REPO / "docs" / "check" / "ses_v1" / "p0r1"

MANIFEST = list(csv.DictReader((REPO / "docs/check/ses_v1/p0/P0_MANIFEST.csv").open(encoding="utf-8")))
IDMAP = {m["anon_id"]: m for m in json.loads((ASSET0 / "manifest_id_map.json").read_text(encoding="utf-8"))}


def ts_epoch(s):
    from datetime import datetime
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except (ValueError, TypeError):
        return None


def sha_file(p: Path):
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> int:
    ASSET1.mkdir(parents=True, exist_ok=True)
    OUT1.mkdir(parents=True, exist_ok=True)
    full = []
    agg = Counter()
    for m in MANIFEST:
        anon = m["anon_id"]
        tdir = DATA / IDMAP[anon]["dir"]
        thread_id = IDMAP[anon]["dir"].replace("\\", "/").split("/")[-1]
        rec = {"anon_id": anon, "topic": m["topic"], "thread_pseudonym_hash": m["thread_hash"],
               "thread_id_local_only": thread_id}
        # 文件哈希
        rec["source_json_sha256"] = sha_file(tdir / "source-tweets" / f"{thread_id}.json")
        rec["annotation_json_sha256"] = sha_file(tdir / "annotation.json")
        rec["structure_json_sha256"] = sha_file(tdir / "structure.json")
        rx = sorted((tdir / "reactions").glob("*.json")) if (tdir / "reactions").exists() else []
        h = hashlib.sha256()
        for f in rx:
            h.update(hashlib.sha256(f.name.encode("utf-8")).digest())
            h.update(f.read_bytes())
        rec["reactions_merged_sha256"] = h.hexdigest() if rx else None
        rec["n_reaction_files"] = len(rx)

        # 源帖与回复
        src = None
        try:
            src = json.loads((tdir / "source-tweets" / f"{thread_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            src = None
        reactions = []
        for f in rx:
            try:
                reactions.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                reactions.append(None)

        # structure 树边
        structure = {}
        try:
            structure = json.loads((tdir / "structure.json").read_text(encoding="utf-8")).get(thread_id, {})
        except (OSError, json.JSONDecodeError):
            pass
        tree_edges = {}

        def walk(node, parent):
            if not isinstance(node, dict):
                return
            for child, sub in node.items():
                tree_edges[child] = parent
                walk(sub, child)

        walk(structure, thread_id)

        rows = [("source", thread_id, src)] + [("reaction", f.stem, r) for f, r in zip(rx, reactions)]
        n_rows = 0
        n_missing_created, n_missing_user, n_meta_parent, n_urls = 0, 0, 0, 0
        n_tree_missing_edge, n_meta_missing_edge, n_both_differ = 0, 0, 0
        n_missing_parent_node, n_parent_after_child, n_bad_time = 0, 0, 0
        parent_state_counter = Counter()
        for kind, rid, obj in rows:
            if obj is None:
                n_bad_time += 1
                continue
            n_rows += 1
            created = obj.get("created_at")
            if not created or ts_epoch(created) is None:
                n_missing_created += 1
            if not (obj.get("user") or {}).get("id_str"):
                n_missing_user += 1
            if (obj.get("entities") or {}).get("urls"):
                n_urls += 1
            if kind == "source":
                continue
            meta_p = obj.get("in_reply_to_status_id_str")
            tree_p = tree_edges.get(rid)
            if meta_p:
                n_meta_parent += 1
            if meta_p and tree_p:
                if str(meta_p) != str(tree_p):
                    n_both_differ += 1
                    parent_state_counter["both_present_differ"] += 1
                else:
                    parent_state_counter["both_present_agree"] += 1
            elif meta_p and not tree_p:
                n_meta_missing_edge += 1
                parent_state_counter["metadata_only"] += 1
            elif tree_p and not meta_p:
                n_tree_missing_edge += 1
                parent_state_counter["tree_only"] += 1
            else:
                parent_state_counter["both_missing"] += 1
            if meta_p and meta_p != thread_id and meta_p not in tree_edges and meta_p != rid:
                # in_reply_to 指向树外（也非根）节点
                if not any(r2 and (r2.get("id_str") == meta_p) for r2 in reactions):
                    n_missing_parent_node += 1
            if meta_p == thread_id:
                parent_state_counter["meta_parent_is_root"] += 1
        # 父晚于子（用可配对边检查）
        id_ts = {}
        for kind, rid, obj in rows:
            if obj and obj.get("created_at"):
                e = ts_epoch(obj["created_at"])
                if e is not None:
                    id_ts[rid] = e
        for rid, parent in tree_edges.items():
            if rid in id_ts and parent in id_ts and id_ts[parent] > id_ts[rid]:
                n_parent_after_child += 1
        rec.update({
            "n_rows": n_rows, "n_missing_created": n_missing_created, "n_missing_user": n_missing_user,
            "n_rows_with_urls": n_urls, "n_tree_edges": len(tree_edges),
            "n_meta_parent_present": n_meta_parent, "n_both_differ": n_both_differ,
            "n_tree_only_edge": n_tree_missing_edge, "n_metadata_only_edge": n_meta_missing_edge,
            "n_missing_parent_node": n_missing_parent_node, "n_parent_after_child": n_parent_after_child,
            "parent_edge_state": dict(parent_state_counter),
        })
        agg["rows"] += n_rows
        agg["missing_created"] += n_missing_created
        agg["missing_user"] += n_missing_user
        agg["both_differ"] += n_both_differ
        agg["tree_only"] += n_tree_missing_edge
        agg["metadata_only"] += n_meta_missing_edge
        agg["missing_parent_node"] += n_missing_parent_node
        agg["parent_after_child"] += n_parent_after_child
        full.append(rec)

    (ASSET1 / "asset_audit_full.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    coverage = {
        "scope": "fixed 60 manifest threads; hashes of input files; no raw IDs in this public file",
        "totals": {
            "threads": 60,
            "rows_source_plus_reactions": agg["rows"],
            "rows_missing_or_invalid_created_at": agg["missing_created"],
            "rows_missing_user_id": agg["missing_user"],
            "parent_edges_both_present_but_differ": agg["both_differ"],
            "parent_edges_tree_only": agg["tree_only"],
            "parent_edges_metadata_only": agg["metadata_only"],
            "in_reply_to_pointing_outside_tree": agg["missing_parent_node"],
            "parent_timestamp_after_child": agg["parent_after_child"],
        },
        "hash_fields_per_thread": ["source_json_sha256", "annotation_json_sha256",
                                    "structure_json_sha256", "reactions_merged_sha256"],
        "note": "thread_hash in manifest is a pseudonym hash of topic|thread_id, NOT a content hash; content hashes live in local asset_audit_full.json keyed by anon_id",
    }
    (OUT1 / "P0R1_ASSET_COVERAGE.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
