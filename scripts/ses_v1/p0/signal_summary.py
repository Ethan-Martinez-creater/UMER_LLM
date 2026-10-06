# -*- coding: utf-8 -*-
"""SES-v1 P0 审读辅助分析：为 60 条固定线程生成信号摘要（URL 重叠、n-gram 复述、
反驳/纠错信号词、时间偏移），输出到 local_assets（仅本地），供人工审读聚焦。

这不是观察编码本身；最终 provisional 观察记录由人工审读后写入
docs/check/ses_v1/p0/P0_OBSERVABILITY.csv。
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "local_assets" / "ses_v1" / "p0"
REVIEW = ASSET / "review"
ID_MAP = json.loads((ASSET / "manifest_id_map.json").read_text(encoding="utf-8"))
MANIFEST = list(csv.DictReader((REPO / "docs/check/ses_v1/p0/P0_MANIFEST.csv").open(encoding="utf-8")))

DENY_PAT = re.compile(
    r"\b(not true|false|fake|hoax|debunk|no evidence|denied?|denies|actually|"
    r"isn't true|is not true|wasn't true|rumor is false|rumour is false|"
    r"has been (confirmed|clarified)|correction|never happened|did not happen|"
    r"unfounded|baseless|misleading|wrong)\b", re.I)
SUPPORT_PAT = re.compile(
    r"\b(confirmed|official|police (say|said|confirm)|reported|according to|"
    r"per |breaking|witness|source:|http)", re.I)


def norm_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ngrams(text: str, n: int = 6):
    toks = norm_text(text).split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def ts_epoch(s):
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%a %b %d %H:%M:%S %z %Y",):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def minute_offset(src_ts, s):
    if src_ts is None or s is None:
        return None
    return round((ts_epoch(s) - src_ts) / 60.0, 2)


def main() -> int:
    out = {}
    for row, idm in zip(MANIFEST, ID_MAP):
        rid = row["anon_id"]
        obj = json.loads((REVIEW / f"{rid}.json").read_text(encoding="utf-8"))
        src = obj.get("source") or {}
        src_ts = ts_epoch(src.get("created_at"))
        src_urls = {u.split("/")[2] if u and "://" in u else u for u in (src.get("urls") or []) if u}
        reactions = obj.get("reactions") or []
        # 按 URL（去参数后的规范化 path）聚合
        url_owner = {}
        for i, r in enumerate(reactions):
            for u in (r.get("urls") or []):
                if not u:
                    continue
                path = re.sub(r"[?#].*$", "", u).rstrip("/")
                url_owner.setdefault(path, []).append(("R%03d" % (i + 1), minute_offset(src_ts, r.get("created_at"))))
        ann_links = [l.get("link") for l in (obj.get("annotation_links") or []) if l.get("link")]
        ann_paths = {re.sub(r"[?#].*$", "", u).rstrip("/") for u in ann_links}
        react_paths = set(url_owner)
        # reaction 间长 n-gram 复述对（与源帖或彼此共享 ≥2 个 6-gram）
        src_ng = ngrams(src.get("text") or "")
        pairs = []
        for i in range(len(reactions)):
            for j in range(i + 1, len(reactions)):
                shared = ngrams(reactions[i].get("text") or "") & ngrams(reactions[j].get("text") or "")
                if len(shared) >= 2:
                    pairs.append({"a": "R%03d" % (i + 1), "b": "R%03d" % (j + 1),
                                  "shared_ngrams": sorted(shared)[:3]})
        src_paraphrase = []
        for i, r in enumerate(reactions):
            shared = ngrams(r.get("text") or "") & src_ng
            if len(shared) >= 2:
                src_paraphrase.append({"r": "R%03d" % (i + 1), "shared_ngrams": sorted(shared)[:3]})
        deny_rows, support_rows = [], []
        for i, r in enumerate(reactions):
            txt = r.get("text") or ""
            mo = minute_offset(src_ts, r.get("created_at"))
            if DENY_PAT.search(txt):
                deny_rows.append({"r": "R%03d" % (i + 1), "min": mo, "text": txt[:160]})
            if SUPPORT_PAT.search(txt):
                support_rows.append({"r": "R%03d" % (i + 1), "min": mo, "text": txt[:160]})
        out[rid] = {
            "topic": row["topic"], "v_class": row["v_class"],
            "n_reactions": len(reactions),
            "src_urls": sorted(src_urls)[:5],
            "url_shared_by_reactions": {k: v for k, v in url_owner.items() if len(v) >= 2},
            "url_also_in_source": {k: v for k, v in url_owner.items() if re.sub(r"[?#].*$", "", k) in ann_paths or k in ann_paths},
            "annotation_links": ann_links,
            "reaction_pair_paraphrase": pairs[:8],
            "src_paraphrase": src_paraphrase[:8],
            "deny_signal": deny_rows[:10],
            "support_signal": support_rows[:10],
        }
    (ASSET / "signal_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 粗统计
    n_url_shared = sum(1 for v in out.values() if v["url_shared_by_reactions"])
    n_pair_para = sum(1 for v in out.values() if v["reaction_pair_paraphrase"])
    n_src_para = sum(1 for v in out.values() if v["src_paraphrase"])
    n_deny = sum(1 for v in out.values() if v["deny_signal"])
    print(json.dumps({"threads": len(out), "url_shared": n_url_shared,
                      "pair_paraphrase": n_pair_para, "src_paraphrase": n_src_para,
                      "deny_signal": n_deny}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
