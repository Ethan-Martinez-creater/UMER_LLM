# -*- coding: utf-8 -*-
"""SES-v1 P0 深查提取：对最多 12 条固定候选线程输出完整审计视图（本地）。

每条输出：源帖信息、annotation links、全部 reactions 的编号/分钟偏移/父节点/
展开 URL（去参 path）/文本前 220 字，以及 URL path 聚类与树深度。
输出 local_assets/ses_v1/p0/deep_audit.txt，仅供人工审读，不推送。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "local_assets" / "ses_v1" / "p0"
DATA = ASSET / "extracted" / "all-rnr-annotated-threads"
ID_MAP = {m["anon_id"]: m for m in json.loads((ASSET / "manifest_id_map.json").read_text(encoding="utf-8"))}

DEEP = ["SES-P0-001", "SES-P0-007", "SES-P0-008", "SES-P0-010", "SES-P0-014",
        "SES-P0-026", "SES-P0-028", "SES-P0-029", "SES-P0-030", "SES-P0-036",
        "SES-P0-053", "SES-P0-056"]


def ts_epoch(s):
    from datetime import datetime
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except (ValueError, TypeError):
        return None


def norm_path(u):
    if not u:
        return None
    return re.sub(r"[?#].*$", "", u).rstrip("/")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    out_lines = []
    for anon in DEEP:
        m = ID_MAP[anon]
        tdir = DATA / m["dir"]
        thread_id = m["dir"].replace("\\", "/").split("/")[-1]
        src_obj = load_json(tdir / "source-tweets" / f"{thread_id}.json") or {}
        src_user = (src_obj.get("user") or {}).get("screen_name")
        src_ts = ts_epoch(src_obj.get("created_at"))
        ann = load_json(tdir / "annotation.json") or {}
        structure = load_json(tdir / "structure.json") or {}

        # 父子映射：parent -> [child]
        parent_of = {}

        def walk(node, parent):
            for child, subtree in node.items():
                parent_of[child] = parent
                if isinstance(subtree, dict):
                    walk(subtree, child)

        walk(structure.get(thread_id, {}), thread_id)

        out_lines.append(f"\n{'='*80}\n## {anon} [{ann.get('is_rumour')}] topic={m['topic']}")
        out_lines.append(f"SRC @{src_user} {src_obj.get('created_at')} text: {(src_obj.get('text') or '')[:260]}")
        for l in (ann.get("links") or []):
            out_lines.append(f"ANN-LINK [{l.get('position')}/{l.get('mediatype')}]: {l.get('link')}")
        reactions = []
        rdir = tdir / "reactions"
        for f in sorted(rdir.glob("*.json")):
            obj = load_json(f)
            if obj:
                reactions.append(obj)
        reactions.sort(key=lambda o: ts_epoch(o.get("created_at")) or 0)
        path_cluster = defaultdict(list)
        rows = []
        for i, r in enumerate(reactions):
            rid = "R%03d" % (i + 1)
            mo = None
            if src_ts and r.get("created_at"):
                mo = round((ts_epoch(r.get("created_at")) - src_ts) / 60.0, 2)
            urls = [norm_path(u.get("expanded_url")) for u in (r.get("entities") or {}).get("urls", []) if u.get("expanded_url")]
            urls = [u for u in urls if u]
            for u in urls:
                path_cluster[u].append(rid)
            parent = parent_of.get(r.get("id_str"))
            parent_tag = "SRC" if parent == thread_id else (parent if parent else "UNKNOWN_PARENT")
            rows.append(f"{rid} @{str(mo).rjust(8)}min parent={str(parent_tag)[:20]} urls={urls} "
                        f"text: {(r.get('text') or '')[:220]}")
        out_lines.append(f"-- {len(rows)} reactions (time-sorted) --")
        out_lines.extend(rows)
        out_lines.append("-- URL clusters (>=2 sharers) --")
        for p, users in sorted(path_cluster.items(), key=lambda kv: -len(kv[1])):
            if len(users) >= 2:
                out_lines.append(f"  {len(users)}x {p[:110]} -> {users}")
        depth_histogram = defaultdict(int)
        for rid, parent in parent_of.items():
            d = 1
            cur = parent
            while cur != thread_id and cur in parent_of:
                d += 1
                cur = parent_of[cur]
            depth_histogram[d] += 1
        out_lines.append(f"-- reply depth histogram -- {dict(sorted(depth_histogram.items()))}")
    (ASSET / "deep_audit.txt").write_text("\n".join(out_lines), encoding="utf-8")
    print("written deep_audit.txt,", len(DEEP), "threads,", len("\n".join(out_lines)), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
