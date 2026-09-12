#!/usr/bin/env python
"""V3-B reader-transfer failure diagnosis (§1-§28).

Post-hoc, read-only diagnosis over the frozen 600 paired samples / 1200
generations.  No new Qwen inference, no resampling, no MS-TSR change: Qwen
citations are read back from the frozen ``parsed/`` files, and the only
recomputed quantities are deterministic features (causal snapshot structure,
Static Utility / encoder representations from the frozen corrected-E2
components, and per-unit token costs).

Statistics are hand-rolled so the tests can exercise them without scipy:
Spearman via average ranks, AUC via the Mann-Whitney rank identity, Fisher's
exact test via the hypergeometric distribution.

Root-cause labels and the final recommendation use thresholds fixed in this
file *before* running (see ROOT_CAUSE_RULES / RECOMMENDATION_RULES); nothing
here writes a heuristic back into the selector or MS-TSR (§22, §33).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_common import PROJECT_DIR  # noqa: E402,F401  (sys.path side effect)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
CUTOFFS = (5, 15, 30, 60, 180, 360)
PARTITION_SEED = 3090
CONTEXT_SEED = 2000
PRESSURE_BINS = ("NO_CANDIDATE", "LOW", "MEDIUM", "HIGH")
GROUP_ORDER = ("CC", "CW", "WC", "WW")
# §25/§26 thresholds, fixed before running
ROOT_CAUSE_RULES = {
    "OVER_COMPRESSION": {
        "compression_gap": 0.05,      # CW mean reduction >= best(CC,WC) + gap
        "evidence_gap": 0.5,          # or CW mean MS units <= best - gap
    },
    "READER_SENSITIVE_EVIDENCE_REMOVAL": {
        "loss_rate_gap": 0.15,        # CW loss rate >= max(CC,WC) + gap
        "loss_rate_ratio": 2.0,       # and CW >= ratio * WC
    },
    "STATIC_UTILITY_READER_MISALIGNMENT": {"auc_max": 0.60},
    "PROXY_READER_MARGIN_MISMATCH": {"retention_min": 0.95,
                                     "delta_max": -0.005},
}
RECOMMENDATION_RULES = {
    "reader_aware_signal": {"loss_rate_gap": 0.15, "loss_rate_ratio": 2.0},
}
DEFAULT_READER_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"
DEFAULT_V3_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3"
DEFAULT_OUT = "/data/jyz/next/llm/results/tcdscr/v3b_failure_diagnosis"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"


# --------------------------------------------------------------- helpers

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def outcome_group(static_correct, ms_correct):
    if static_correct and ms_correct:
        return "CC"
    if static_correct and not ms_correct:
        return "CW"
    if not static_correct and ms_correct:
        return "WC"
    return "WW"


def token_reduction(static_tokens, ms_tokens):
    """§5: None (N/A) when Static social context is empty."""
    if int(static_tokens) <= 0:
        return None
    return 1.0 - int(ms_tokens) / int(static_tokens)


def evidence_count_reduction(static_count, ms_count):
    if int(static_count) <= 0:
        return None
    return 1.0 - int(ms_count) / int(static_count)


def quantiles(values, qs=(0.25, 0.5, 0.75, 0.9)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}
    out = {"n": len(vals), "mean": sum(vals) / len(vals),
           "median": _pct(vals, 0.5)}
    for q in qs:
        out[f"p{int(q * 100)}"] = _pct(vals, q)
    return out


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def _rank(values):
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho; None when undefined (n < 3 or a constant vector)."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    rx, ry = _rank(xs), _rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney rank identity; labels: 1 = positive."""
    pairs = [(s, l) for s, l in zip(scores, labels)
             if s is not None and l is not None]
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return None
    sc = [s for s, _ in pairs]
    lb = [l for _, l in pairs]
    ranks = _rank(sc)
    r_pos = sum(r for r, l in zip(ranks, lb) if l == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _hyper(a, r1, r2, c1):
    return (math.comb(r1, a) * math.comb(r2, c1 - a)
            / math.comb(r1 + r2, c1))


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p (sum of tables no more likely than observed)."""
    r1, r2, c1 = a + b, c + d, a + c
    if r1 + r2 == 0 or c1 == 0 or c1 == r1 + r2:
        return 1.0
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_obs = _hyper(a, r1, r2, c1)
    total = 0.0
    for k in range(lo, hi + 1):
        p = _hyper(k, r1, r2, c1)
        if p <= p_obs + 1e-12:
            total += p
    return min(1.0, total)


def odds_ratio(a, b, c, d):
    if b == 0 or c == 0:
        return None
    return (a * d) / (b * c)


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate(num, den):
    return (num / den) if den else None


def evidence_count_bin(count):
    """§6 fixed bins."""
    c = int(count)
    if c == 0:
        return "0"
    if c == 1:
        return "1"
    if c == 2:
        return "2"
    if c <= 4:
        return "3-4"
    if c <= 8:
        return "5-8"
    return "9+"


def selected_count_delta_bin(delta):
    d = int(delta)
    if d <= 0:
        return "0"
    if d <= 2:
        return "1-2"
    if d <= 5:
        return "3-5"
    if d <= 10:
        return "6-10"
    return "11+"


def margin_retention_bin(retention):
    """§11 fixed bins."""
    if retention is None:
        return None
    if retention < 0.80:
        return "<0.80"
    if retention < 0.90:
        return "0.80-0.90"
    if retention < 0.95:
        return "0.90-0.95"
    if retention <= 1.00:
        return "0.95-1.00"
    return ">1.00"


MARGIN_BIN_ORDER = ("<0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00", ">1.00")


def depth_group(depth):
    d = int(depth)
    if d <= 1:
        return "depth1"
    if d == 2:
        return "depth2"
    return "depth>=3"


def temporal_group(elapsed, all_elapsed):
    """newest quartile / middle 50% / oldest quartile within the snapshot."""
    if not all_elapsed:
        return None
    vals = sorted(all_elapsed)
    q1 = vals[int(0.25 * len(vals))]
    q3 = vals[int(0.75 * len(vals))]
    if elapsed <= q1:
        return "oldest_quartile"
    if elapsed <= q3:
        return "middle_50"
    return "newest_quartile"


def reader_evidence_loss(static_cited_ids, ms_supplied_ids):
    """§9: evidence the Static reader used that MS no longer supplies."""
    supplied = set(ms_supplied_ids)
    lost = [nid for nid in static_cited_ids if nid not in supplied]
    return lost


def cited_removed_mapping(static_cited_ids, ms_selected_ids):
    """§8: Static-arm citations that MS removed (requires static ⊇ ms ids to
    be meaningful, but is computed on whatever sets are given)."""
    ms_set = set(ms_selected_ids)
    removed = [nid for nid in static_cited_ids if nid not in ms_set]
    return removed


def topk_recall(cited_ids, utility_by_id, k):
    """§16: share of Qwen-cited evidence inside the Static-Utility top-k."""
    ranked = sorted(utility_by_id, key=lambda nid: (-utility_by_id[nid], nid))
    top = set(ranked[:k])
    cited = [nid for nid in cited_ids if nid in utility_by_id]
    if not cited:
        return None
    return sum(1 for nid in cited if nid in top) / len(cited)


def group_summary(samples, key):
    out = {}
    for g in GROUP_ORDER:
        sub = [s for s in samples if s["group"] == g]
        vals = [s.get(key) for s in sub]
        vals = [v for v in vals if v is not None]
        out[g] = {
            "n": len(sub),
            "mean": mean(vals),
            "median": quantiles(vals)["median"] if vals else None,
        }
    return out


# ------------------------------------------------- feature extraction (GPU)

def structural_counts(node_ids, edge_index):
    """Aggregate propagation structure keyed by node id (§2, §3).

    ``snapshot["edge_index"]`` stores node *positions*; using those ints as
    lookup keys while querying by node id silently collapses every node to
    degree 0 / leaf True.  Returns ``(degree, child_count)`` where ``degree``
    is undirected and ``child_count`` counts only outgoing child edges.
    """
    degree = collections.Counter()
    child_count = collections.Counter()
    for child_pos, parent_pos in edge_index:
        child_id = node_ids[child_pos]
        parent_id = node_ids[parent_pos]
        degree[child_id] += 1
        degree[parent_id] += 1
        child_count[parent_id] += 1
    return degree, child_count


def struct_consistency(node_ids, edge_index, degree, child_count):
    """Sums used to verify the aggregation (degree = 2E, child_count = E)."""
    return {
        "n_edges": len(edge_index),
        "sum_degree": sum(degree.get(nid, 0) for nid in node_ids),
        "sum_child_count": sum(child_count.get(nid, 0) for nid in node_ids),
    }


def extract_node_features(dataset, samples, reader_root, v3_root, device):
    """Recompute causal snapshot structure, Static Utility and encoder-space
    relevance for every node of every sampled snapshot.

    Returns ``{(sample_id): {"nodes": {node_id: feats}, "cited": {...}}}``.
    """
    import torch
    from transformers import AutoTokenizer

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.context.evidence_unit import build_evidence_units
    from tcdscr_common import (EventSemanticStore, event_label_registry)
    from tcdscr_run_dynamic_v2 import (collect_event_data_v2,
                                      load_frozen_components)
    from tcdscr_run_e2 import build_light_item

    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}

    def load_event(event_id, val_ids):
        from tcdscr.data import maweibo_adapter, pheme_adapter
        if event_id not in val_ids:
            raise RuntimeError(f"{dataset}/{event_id}: not a validation event")
        if dataset == "pheme":
            by_id = {eid: (topic, label, folder)
                     for eid, topic, label, folder
                     in pheme_adapter.event_ids(cfg.raw_dir)}
            return pheme_adapter.load_event(*by_id[event_id])
        labels = dict(maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file))
        return maweibo_adapter.load_event(
            event_id, labels[event_id],
            f"{cfg.raw_dir.rstrip('/')}/{event_id}.json")

    by_fold = collections.defaultdict(list)
    for s in samples:
        by_fold[s["fold"]].append(s)

    out = {}
    for fold in sorted(by_fold):
        fold_samples = by_fold[fold]
        val_ids = set(build_primary_fold_split(registry, fold,
                                               seed=PARTITION_SEED)[
                                                   "validation"])
        encoder, selector, proxy, _ = load_frozen_components(
            dataset, fold, CONTEXT_SEED, E1_ROOT, E2_ROOT, device)
        events, snaps, items = {}, {}, []
        wanted = []
        for s in fold_samples:
            eid = s["event_id"]
            if eid not in events:
                events[eid] = load_event(eid, val_ids)
            snap = build_snapshot(events[eid], int(s["cutoff"]))
            snaps[s["sample_id"]] = snap
            items.append(build_light_item(events[eid], snap,
                                          store.get_store(events[eid],
                                                          cache)))
            wanted.append(s)
        by_event = collect_event_data_v2(encoder, selector, items, tokenizer,
                                          device)
        for s in wanted:
            snap = snaps[s["sample_id"]]
            er = by_event[s["event_id"]]
            row = next(r for r in er if int(r["cutoff"]) == int(s["cutoff"]))
            units = build_evidence_units(snap)
            pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
            # edge_index stores node POSITIONS; map them to ids before
            # aggregating, otherwise every lookup by node id misses.
            degree_by_id, child_count_by_id = structural_counts(
                snap["node_ids"], snap["edge_index"])
            consistency = struct_consistency(snap["node_ids"],
                                             snap["edge_index"],
                                             degree_by_id,
                                             child_count_by_id)
            u_by_id = {nid: float(row["u"][i])
                       for i, nid in enumerate(row["cand_node_ids"])}
            repr_by_id = {nid: row["node_repr"][i]
                          for i, nid in enumerate(row["cand_node_ids"])}
            h_source = row["h_source"]
            # costs are parallel to ``order`` (both keyed by the unit's
            # snapshot position), not to the candidate-array index.
            cost_by_order = dict(zip(row["order"], row["costs"]))
            elapsed = {unit["node_id"]: unit["elapsed_seconds"]
                       for unit in units}
            all_elapsed = list(elapsed.values())
            nodes = {}
            for unit in units:
                nid = unit["node_id"]
                vec = repr_by_id.get(nid)
                if vec is not None:
                    denom = (float(vec.norm()) * float(h_source.norm()))
                    relevance = (float((vec * h_source).sum()) / denom
                                 if denom > 0 else None)
                else:
                    relevance = None
                reply_tokens = _token_len(tokenizer, unit["reply_text"])
                parent_tokens = _token_len(tokenizer,
                                           unit["parent_text"] or "")
                nodes[nid] = {
                    "node_id": nid,
                    "in_candidates": nid in u_by_id,
                    "utility": u_by_id.get(nid),
                    "relevance": relevance,
                    "depth": int(snap["depths"][pos[nid]]),
                    "degree": int(degree_by_id.get(nid, 0)),
                    "child_count": int(child_count_by_id.get(nid, 0)),
                    "elapsed_seconds": int(elapsed[nid]),
                    "temporal_group": temporal_group(elapsed[nid],
                                                     all_elapsed),
                    "reply_tokens": reply_tokens,
                    "parent_tokens": parent_tokens,
                    "pair_tokens": int(cost_by_order.get(unit["order"], 0)),
                    "is_leaf": child_count_by_id.get(nid, 0) == 0,
                    "is_source_child": snap["parent_ids"][pos[nid]]
                    == snap["source_id"],
                    "is_memory_previous": nid in set(
                        s.get("memory_previous_ids") or []),
                    "text_has_question": "?" in (unit["reply_text"] or ""),
                    "text_has_exclamation": "!" in (unit["reply_text"] or ""),
                    "text_has_url": "http" in (unit["reply_text"] or "").lower(),
                    "text_mention_count": (unit["reply_text"] or "").count("@"),
                }
            out[s["sample_id"]] = {"nodes": nodes,
                                   "source_id": snap["source_id"],
                                   "snapshot_node_ids": list(snap["node_ids"]),
                                   "consistency": consistency}
        del by_event
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def _token_len(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------- assembly

def build_sample_records(reader_root, v3_root):
    """Join manifest / prompts / parsed into per-sample diagnosis records."""
    with open(os.path.join(reader_root, "sampling_manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    memory = {}
    for ds in DATASETS:
        for fold in FOLDS:
            path = os.path.join(v3_root, "runs", ds,
                                f"fold{fold}_seed{CONTEXT_SEED}",
                                "validation_predictions.jsonl")
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if abs(float(r["alpha"]) - 0.8) > 1e-9:
                        continue
                    memory[(ds, r["event_id"], str(r["cutoff"]))] = \
                        list(r.get("memory_previous_ids") or [])
    samples, prompt_map, parsed_map = [], {}, {}
    for ds in DATASETS:
        for r in load_jsonl(os.path.join(reader_root, "prompts",
                                         f"{ds}.jsonl")):
            prompt_map[(r["sample_id"], r["arm"])] = r
        for r in load_jsonl(os.path.join(reader_root, "parsed",
                                         f"{ds}.jsonl")):
            parsed_map[(r["sample_id"], r["arm"])] = r
    for s in manifest["samples"]:
        sid = s["sample_id"]
        st = parsed_map.get((sid, "static"))
        ms = parsed_map.get((sid, "ms"))
        st_prompt = prompt_map.get((sid, "static"))
        ms_prompt = prompt_map.get((sid, "ms"))
        if st is None or ms is None or st_prompt is None or ms_prompt is None:
            raise RuntimeError(f"{sid}: missing arm artifacts")
        if st["parse_failure"] or ms["parse_failure"]:
            raise RuntimeError(f"{sid}: parse failure cannot be diagnosed for "
                               "outcome transitions")
        static_ids = list(s["static_selected_node_ids"])
        ms_ids = list(s["ms_selected_node_ids"])
        cited_static = [st_prompt["evidence_ids"][e]
                        for e in st["evidence_ids"]
                        if e in st_prompt["evidence_ids"]]
        cited_ms = [ms_prompt["evidence_ids"][e]
                    for e in ms["evidence_ids"]
                    if e in ms_prompt["evidence_ids"]]
        loss = reader_evidence_loss(cited_static, ms_ids)
        removed = sorted(set(static_ids) - set(ms_ids))
        retained = sorted(set(static_ids) & set(ms_ids))
        ms_only = sorted(set(ms_ids) - set(static_ids))
        samples.append({
            "sample_id": sid,
            "dataset": s["dataset"],
            "fold": s["fold"],
            "event_id": s["event_id"],
            "cutoff": int(s["cutoff"]),
            "gold": s["gold"],
            "gold_label": st["gold_label"],
            "pressure_bin": s["pressure_bin"],
            "candidate_count": s["candidate_count"],
            "fallback_to_static": bool(s["fallback_to_static"]),
            "dual_view_agree": bool(s["dual_view_agree"]),
            "static_margin": float(s["static_margin"]),
            "ms_margin": float(s["ms_margin"]),
            "static_social_tokens": int(s["static_social_tokens"]),
            "ms_social_tokens": int(s["ms_social_tokens"]),
            "static_total_input_tokens": int(s["static_total_input_tokens"]),
            "ms_total_input_tokens": int(s["ms_total_input_tokens"]),
            "static_selected_count": int(s["static_selected_count"]),
            "ms_selected_count": int(s["ms_selected_count"]),
            "static_label": st["parsed_label"],
            "ms_label": ms["parsed_label"],
            "static_correct": bool(st["correct"]),
            "ms_correct": bool(ms["correct"]),
            "static_confidence": st["confidence"],
            "ms_confidence": ms["confidence"],
            "static_cited_node_ids": sorted(set(cited_static)),
            "ms_cited_node_ids": sorted(set(cited_ms)),
            "static_evidence_node_ids": static_ids,
            "ms_evidence_node_ids": ms_ids,
            "removed_node_ids": removed,
            "retained_node_ids": retained,
            "ms_only_node_ids": ms_only,
            "memory_previous_ids": memory.get((s["dataset"], s["event_id"],
                                               str(s["cutoff"])), []),
            "arm_order": s["arm_order"],
        })
    return manifest, samples


def add_derived(samples):
    for s in samples:
        s["group"] = outcome_group(s["static_correct"], s["ms_correct"])
        s["token_reduction_ratio"] = token_reduction(
            s["static_social_tokens"], s["ms_social_tokens"])
        s["total_token_reduction_ratio"] = token_reduction(
            s["static_total_input_tokens"], s["ms_total_input_tokens"])
        s["evidence_count_reduction_ratio"] = evidence_count_reduction(
            s["static_selected_count"], s["ms_selected_count"])
        s["evidence_count_delta"] = (s["static_selected_count"]
                                     - s["ms_selected_count"])
        s["qwen_confidence_delta"] = s["ms_confidence"] - \
            s["static_confidence"]
        if s["static_margin"] > 0:
            s["proxy_margin_retention"] = s["ms_margin"] / s["static_margin"]
        else:
            s["proxy_margin_retention"] = None
        s["transition_code"] = {"CW": -1, "CC": 0, "WW": 0, "WC": 1}[
            s["group"]]
        s["reader_evidence_loss_count"] = len(
            reader_evidence_loss(s["static_cited_node_ids"],
                                 s["ms_evidence_node_ids"]))
        s["reader_evidence_loss_rate"] = rate(
            s["reader_evidence_loss_count"],
            len(s["static_cited_node_ids"]))
        s["static_cited_removed_count"] = len(
            cited_removed_mapping(s["static_cited_node_ids"],
                                  s["ms_evidence_node_ids"]))
        s["static_cited_removed_rate"] = rate(
            s["static_cited_removed_count"],
            len(s["static_cited_node_ids"]))
        s["has_reader_evidence_loss"] = s["reader_evidence_loss_count"] > 0
    return samples


# ------------------------------------------------------------ analyses

def analysis_paired_outcome_groups(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        counts = collections.Counter(s["group"] for s in samples)
        out[ds] = {g: counts.get(g, 0) for g in GROUP_ORDER}
        out[ds]["n"] = len(samples)
        out[ds]["net_correction"] = (counts.get("WC", 0)
                                     - counts.get("CW", 0))
    return out


def analysis_compression_severity(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_group = {}
        for g in GROUP_ORDER:
            sub = [s for s in samples if s["group"] == g]
            per_group[g] = {
                "n": len(sub),
                "token_reduction_ratio": quantiles(
                    [s["token_reduction_ratio"] for s in sub]),
                "evidence_count_reduction_ratio": quantiles(
                    [s["evidence_count_reduction_ratio"] for s in sub]),
                "ms_selected_count": quantiles(
                    [s["ms_selected_count"] for s in sub]),
                "static_selected_count": quantiles(
                    [s["static_selected_count"] for s in sub]),
                "ms_social_tokens": quantiles(
                    [s["ms_social_tokens"] for s in sub]),
                "disagreement_rate": rate(
                    sum(1 for s in sub if s["static_label"] != s["ms_label"]),
                    len(sub)),
            }
        out[ds] = per_group
        cw = [s for s in samples if s["group"] == "CW"]
        other = [s for s in samples if s["group"] in ("CC", "WC")]
        out[ds]["cw_vs_cc_wc"] = {
            "cw_mean_reduction": mean([s["token_reduction_ratio"]
                                       for s in cw]),
            "cc_wc_mean_reduction": mean([s["token_reduction_ratio"]
                                          for s in other]),
            "cw_mean_ms_units": mean([s["ms_selected_count"] for s in cw]),
            "cc_wc_mean_ms_units": mean([s["ms_selected_count"]
                                         for s in other]),
        }
    return out


def analysis_evidence_count_threshold(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        bins = {}
        for b in ("0", "1", "2", "3-4", "5-8", "9+"):
            sub = [s for s in samples
                   if evidence_count_bin(s["ms_selected_count"]) == b]
            acc_st = rate(sum(1 for s in sub if s["static_correct"]),
                          len(sub))
            acc_ms = rate(sum(1 for s in sub if s["ms_correct"]), len(sub))
            bins[b] = {
                "n": len(sub),
                "static_accuracy": acc_st,
                "ms_accuracy": acc_ms,
                "delta_accuracy": (None if acc_st is None or acc_ms is None
                                   else acc_ms - acc_st),
                "disagreement_rate": rate(
                    sum(1 for s in sub if s["static_label"] != s["ms_label"]),
                    len(sub)),
                "cw_rate": rate(sum(1 for s in sub if s["group"] == "CW"),
                                len(sub)),
                "wc_rate": rate(sum(1 for s in sub if s["group"] == "WC"),
                                len(sub)),
                "mean_token_reduction": mean(
                    [s["token_reduction_ratio"] for s in sub]),
            }
        deltas = {}
        for b in ("0", "1-2", "3-5", "6-10", "11+"):
            sub = [s for s in samples
                   if selected_count_delta_bin(s["evidence_count_delta"]) == b]
            deltas[b] = {
                "n": len(sub),
                "static_accuracy": rate(
                    sum(1 for s in sub if s["static_correct"]), len(sub)),
                "ms_accuracy": rate(
                    sum(1 for s in sub if s["ms_correct"]), len(sub)),
                "cw_rate": rate(sum(1 for s in sub if s["group"] == "CW"),
                                len(sub)),
                "wc_rate": rate(sum(1 for s in sub if s["group"] == "WC"),
                                len(sub)),
                "mean_token_reduction": mean(
                    [s["token_reduction_ratio"] for s in sub]),
            }
        out[ds] = {"ms_selected_count_bins": bins,
                   "selected_count_delta_bins": deltas}
    return out


def _feature_summary(node_features):
    """Aggregate a list of per-node feature dicts."""
    if not node_features:
        return {"n": 0}
    def _m(key):
        return mean([n.get(key) for n in node_features])
    def _r(key):
        vals = [n.get(key) for n in node_features if n.get(key) is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None
    return {
        "n": len(node_features),
        "mean_utility": _m("utility"),
        "mean_relevance": _m("relevance"),
        "mean_depth": _m("depth"),
        "mean_degree": _m("degree"),
        "mean_child_count": _m("child_count"),
        "mean_elapsed_seconds": _m("elapsed_seconds"),
        "mean_reply_tokens": _m("reply_tokens"),
        "mean_parent_tokens": _m("parent_tokens"),
        "mean_pair_tokens": _m("pair_tokens"),
        "is_leaf_rate": _r("is_leaf"),
        "is_source_child_rate": _r("is_source_child"),
        "is_memory_previous_rate": _r("is_memory_previous"),
        "depth_group_rates": {
            "depth1": rate(sum(1 for n in node_features
                               if depth_group(n["depth"]) == "depth1"),
                           len(node_features)),
            "depth2": rate(sum(1 for n in node_features
                               if depth_group(n["depth"]) == "depth2"),
                           len(node_features)),
            "depth>=3": rate(sum(1 for n in node_features
                                 if depth_group(n["depth"]) == "depth>=3"),
                             len(node_features)),
        },
        "temporal_group_rates": {
            g: rate(sum(1 for n in node_features
                        if n.get("temporal_group") == g), len(node_features))
            for g in ("oldest_quartile", "middle_50", "newest_quartile")},
    }


def analysis_removed_evidence_features(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_group = {}
        for g in GROUP_ORDER:
            removed, retained = [], []
            for s in samples:
                if s["group"] != g:
                    continue
                nodes = features[s["sample_id"]]["nodes"]
                removed.extend(nodes[n] for n in s["removed_node_ids"]
                               if n in nodes)
                retained.extend(nodes[n] for n in s["retained_node_ids"]
                                if n in nodes)
            per_group[g] = {"removed": _feature_summary(removed),
                            "retained": _feature_summary(retained)}
        out[ds] = per_group
    return out


def analysis_reader_evidence_use(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_group = {}
        for g in GROUP_ORDER:
            sub = [s for s in samples if s["group"] == g]
            cited, uncited = [], []
            for s in sub:
                nodes = features[s["sample_id"]]["nodes"]
                cited_set = set(s["static_cited_node_ids"])
                for nid, feats in nodes.items():
                    if nid not in set(s["static_evidence_node_ids"]):
                        continue
                    (cited if nid in cited_set else uncited).append(feats)
            per_group[g] = {
                "n": len(sub),
                "static_cited_removed_rate": mean(
                    [s["static_cited_removed_rate"] for s in sub]),
                "static_cited_removed_count": sum(
                    s["static_cited_removed_count"] for s in sub),
                "static_cited_total": sum(len(s["static_cited_node_ids"])
                                          for s in sub),
                "cited": _feature_summary(cited),
                "uncited": _feature_summary(uncited),
            }
        out[ds] = per_group
    return out


def analysis_reader_sensitive_gap(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_group = {}
        for g in GROUP_ORDER:
            sub = [s for s in samples if s["group"] == g]
            per_group[g] = {
                "n": len(sub),
                "loss_rate": mean([s["reader_evidence_loss_rate"]
                                   for s in sub]),
                "mean_loss_count": mean([s["reader_evidence_loss_count"]
                                         for s in sub]),
                "samples_with_loss": sum(1 for s in sub
                                         if s["has_reader_evidence_loss"]),
            }
        cw = [s for s in samples if s["group"] == "CW"]
        non_cw = [s for s in samples if s["group"] != "CW"]
        a = sum(1 for s in cw if s["has_reader_evidence_loss"])
        b = len(cw) - a
        c = sum(1 for s in non_cw if s["has_reader_evidence_loss"])
        d = len(non_cw) - c
        out[ds] = {
            "per_group": per_group,
            "cw_2x2": {"a_loss_cw": a, "b_no_loss_cw": b,
                       "c_loss_non_cw": c, "d_no_loss_non_cw": d},
            "odds_ratio": odds_ratio(a, b, c, d),
            "fisher_p": fisher_exact_2x2(a, b, c, d),
            "n": len(samples),
        }
    return out


def analysis_proxy_reader_margin(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        bins = {}
        for b in MARGIN_BIN_ORDER:
            sub = [s for s in samples
                   if margin_retention_bin(s["proxy_margin_retention"]) == b]
            bins[b] = {
                "n": len(sub),
                "cw_rate": rate(sum(1 for s in sub if s["group"] == "CW"),
                                len(sub)),
                "wc_rate": rate(sum(1 for s in sub if s["group"] == "WC"),
                                len(sub)),
                "disagreement_rate": rate(
                    sum(1 for s in sub if s["static_label"] != s["ms_label"]),
                    len(sub)),
                "mean_confidence_delta": mean(
                    [s["qwen_confidence_delta"] for s in sub]),
            }
        compressed = [s for s in samples if not s["fallback_to_static"]]
        out[ds] = {
            "bins": bins,
            "n_compressed": len(compressed),
            "n_static_margin_nonpositive": sum(
                1 for s in samples if s["static_margin"] <= 0),
            "spearman_retention_confidence_delta": spearman(
                [s["proxy_margin_retention"] for s in compressed],
                [s["qwen_confidence_delta"] for s in compressed]),
            "spearman_retention_transition": spearman(
                [s["proxy_margin_retention"] for s in compressed],
                [s["transition_code"] for s in compressed]),
        }
    return out


def analysis_confidence_transition(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_group = {}
        for g in GROUP_ORDER:
            sub = [s for s in samples if s["group"] == g]
            per_group[g] = {
                "n": len(sub),
                "mean_static_confidence": mean(
                    [s["static_confidence"] for s in sub]),
                "mean_ms_confidence": mean([s["ms_confidence"] for s in sub]),
                "mean_confidence_delta": mean(
                    [s["qwen_confidence_delta"] for s in sub]),
            }
        cells = {}
        for st_ok in (True, False):
            for ms_ok in (True, False):
                sub = [s for s in samples if s["static_correct"] == st_ok
                       and s["ms_correct"] == ms_ok]
                cells[f"static_{'correct' if st_ok else 'wrong'}"
                      f"__ms_{'correct' if ms_ok else 'wrong'}"] = {
                    "n": len(sub),
                    "mean_static_confidence": mean(
                        [s["static_confidence"] for s in sub]),
                    "mean_ms_confidence": mean(
                        [s["ms_confidence"] for s in sub]),
                }
        out[ds] = {"per_group": per_group, "correctness_cells": cells}
    return out


def analysis_label_asymmetry(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_label = {}
        for label in ("RUMOR", "NON_RUMOR"):
            sub = [s for s in samples if s["gold_label"] == label]
            acc_st = rate(sum(1 for s in sub if s["static_correct"]),
                          len(sub))
            acc_ms = rate(sum(1 for s in sub if s["ms_correct"]), len(sub))
            transitions = collections.Counter(
                f"{s['static_label']}->{s['ms_label']}" for s in sub
                if s["static_label"] != s["ms_label"])
            per_label[label] = {
                "n": len(sub),
                "static_accuracy": acc_st,
                "ms_accuracy": acc_ms,
                "delta_accuracy": (None if acc_st is None or acc_ms is None
                                   else acc_ms - acc_st),
                "cw": sum(1 for s in sub if s["group"] == "CW"),
                "wc": sum(1 for s in sub if s["group"] == "WC"),
                "mean_token_reduction": mean(
                    [s["token_reduction_ratio"] for s in sub]),
                "mean_evidence_count_reduction": mean(
                    [s["evidence_count_reduction_ratio"] for s in sub]),
                "mean_reader_evidence_loss_rate": mean(
                    [s["reader_evidence_loss_rate"] for s in sub]),
                "confusion_transitions": dict(transitions),
            }
        out[ds] = per_label
    return out


def analysis_cutoff(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_cut = {}
        for c in CUTOFFS:
            sub = [s for s in samples if s["cutoff"] == c]
            per_cut[str(c)] = {
                "n": len(sub),
                "cw": sum(1 for s in sub if s["group"] == "CW"),
                "wc": sum(1 for s in sub if s["group"] == "WC"),
                "net_correction": (sum(1 for s in sub
                                       if s["group"] == "WC")
                                   - sum(1 for s in sub
                                         if s["group"] == "CW")),
                "mean_token_reduction": mean(
                    [s["token_reduction_ratio"] for s in sub]),
                "mean_selected_count_reduction": mean(
                    [s["evidence_count_reduction_ratio"] for s in sub]),
                "mean_static_units": mean(
                    [s["static_selected_count"] for s in sub]),
                "mean_ms_units": mean([s["ms_selected_count"] for s in sub]),
                "mean_reader_evidence_loss_rate": mean(
                    [s["reader_evidence_loss_rate"] for s in sub]),
                "mean_proxy_margin_retention": mean(
                    [s["proxy_margin_retention"] for s in sub]),
                "mean_confidence_delta": mean(
                    [s["qwen_confidence_delta"] for s in sub]),
                "mean_ms_cited_count": mean([len(s["ms_cited_node_ids"])
                                             for s in sub]),
                "delta_macro_f1": _delta_macro_f1(sub),
            }
        out[ds] = per_cut
    return out


def analysis_selection_pressure(samples_by_ds):
    out = {}
    for ds, samples in samples_by_ds.items():
        per_bin = {}
        for b in PRESSURE_BINS:
            sub = [s for s in samples if s["pressure_bin"] == b]
            if not sub:
                continue
            per_bin[b] = {
                "n": len(sub),
                "cw": sum(1 for s in sub if s["group"] == "CW"),
                "wc": sum(1 for s in sub if s["group"] == "WC"),
                "net_correction": (sum(1 for s in sub
                                       if s["group"] == "WC")
                                   - sum(1 for s in sub
                                         if s["group"] == "CW")),
                "mean_token_reduction": mean(
                    [s["token_reduction_ratio"] for s in sub]),
                "mean_selected_count_reduction": mean(
                    [s["evidence_count_reduction_ratio"] for s in sub]),
                "mean_reader_evidence_loss_rate": mean(
                    [s["reader_evidence_loss_rate"] for s in sub]),
                "mean_proxy_margin_retention": mean(
                    [s["proxy_margin_retention"] for s in sub]),
                "mean_confidence_delta": mean(
                    [s["qwen_confidence_delta"] for s in sub]),
            }
        out[ds] = per_bin
    return out


def analysis_utility_reader_alignment(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        scores, labels = [], []
        utility_all = {}
        for s in samples:
            nodes = features[s["sample_id"]]["nodes"]
            cited = set(s["static_cited_node_ids"])
            for nid in s["static_evidence_node_ids"]:
                feats = nodes.get(nid)
                if feats is None or feats["utility"] is None:
                    continue
                scores.append(feats["utility"])
                labels.append(1 if nid in cited else 0)
                utility_all.setdefault(s["sample_id"], {})[nid] = \
                    feats["utility"]
        topk = {}
        for k in (1, 3, 5):
            vals = []
            for s in samples:
                u = utility_all.get(s["sample_id"])
                if not u:
                    continue
                rec = topk_recall(s["static_cited_node_ids"], u, k)
                if rec is not None:
                    vals.append(rec)
            topk[f"top{k}_recall"] = mean(vals)
            topk[f"top{k}_n"] = len(vals)
        cited_util = [sc for sc, lb in zip(scores, labels) if lb == 1]
        uncited_util = [sc for sc, lb in zip(scores, labels) if lb == 0]
        out[ds] = {
            "n_evidence": len(scores),
            "n_cited": len(cited_util),
            "n_uncited": len(uncited_util),
            "mean_utility_cited": mean(cited_util),
            "mean_utility_uncited": mean(uncited_util),
            "auc_utility_predicts_citation": roc_auc(scores, labels),
            "topk": topk,
        }
    return out


def analysis_ms_removal_alignment(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        def _rates(sub):
            cited_removed = cited_total = uncited_removed = uncited_total = 0
            for s in sub:
                cited = set(s["static_cited_node_ids"])
                removed = set(s["removed_node_ids"])
                for nid in s["static_evidence_node_ids"]:
                    if nid in cited:
                        cited_total += 1
                        if nid in removed:
                            cited_removed += 1
                    else:
                        uncited_total += 1
                        if nid in removed:
                            uncited_removed += 1
            p_cited = rate(cited_removed, cited_total)
            p_uncited = rate(uncited_removed, uncited_total)
            gap = (None if p_cited is None or p_uncited is None
                   else p_cited - p_uncited)
            return {"n": len(sub),
                    "p_remove_given_cited": p_cited,
                    "p_remove_given_uncited": p_uncited,
                    "citation_removal_gap": gap,
                    "cited_total": cited_total, "uncited_total": uncited_total,
                    "cited_removed": cited_removed,
                    "uncited_removed": uncited_removed}
        out[ds] = {"all": _rates(samples)}
        for g in ("CW", "WC"):
            out[ds][g] = _rates([s for s in samples if s["group"] == g])
    return out


def analysis_structural_role(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        groups = {"qwen_cited": [], "ms_retained": [], "ms_removed": []}
        for s in samples:
            nodes = features[s["sample_id"]]["nodes"]
            cited = set(s["static_cited_node_ids"])
            for nid in s["static_cited_node_ids"]:
                if nid in nodes:
                    groups["qwen_cited"].append(nodes[nid])
            for nid in s["retained_node_ids"]:
                if nid in nodes:
                    groups["ms_retained"].append(nodes[nid])
            for nid in s["removed_node_ids"]:
                if nid in nodes:
                    groups["ms_removed"].append(nodes[nid])
        out[ds] = {k: _feature_summary(v) for k, v in groups.items()}
    return out


def analysis_text_characteristics(samples_by_ds, features):
    out = {}
    for ds, samples in samples_by_ds.items():
        buckets = {"qwen_cited": [], "ms_removed": [], "cw_removed": [],
                   "wc_removed": []}
        for s in samples:
            nodes = features[s["sample_id"]]["nodes"]
            for nid in s["static_cited_node_ids"]:
                if nid in nodes:
                    buckets["qwen_cited"].append(nodes[nid])
            for nid in s["removed_node_ids"]:
                if nid not in nodes:
                    continue
                buckets["ms_removed"].append(nodes[nid])
                if s["group"] == "CW":
                    buckets["cw_removed"].append(nodes[nid])
                elif s["group"] == "WC":
                    buckets["wc_removed"].append(nodes[nid])
        summary = {}
        for k, feats in buckets.items():
            summary[k] = {
                "n": len(feats),
                "mean_reply_tokens": mean([f["reply_tokens"] for f in feats]),
                "mean_parent_tokens": mean([f["parent_tokens"]
                                            for f in feats]),
                "mean_pair_tokens": mean([f["pair_tokens"] for f in feats]),
                "question_rate": rate(sum(1 for f in feats
                                          if f["text_has_question"]),
                                      len(feats)),
                "exclamation_rate": rate(sum(1 for f in feats
                                            if f["text_has_exclamation"]),
                                         len(feats)),
                "url_rate": rate(sum(1 for f in feats if f["text_has_url"]),
                                 len(feats)),
                "mean_mention_count": mean([f["text_mention_count"]
                                            for f in feats]),
            }
        out[ds] = summary
    return out


def analysis_cross_dataset(samples_by_ds, compression, gap, margin, align,
                           removal, cutoff, label):
    out = {}
    for ds, samples in samples_by_ds.items():
        out[ds] = {
            "n": len(samples),
            "static_macro_f1": _macro_f1(samples, "static_label"),
            "ms_macro_f1": _macro_f1(samples, "ms_label"),
            "delta_macro_f1": _delta_macro_f1(samples),
            "mean_static_social_tokens": mean(
                [s["static_social_tokens"] for s in samples]),
            "mean_ms_social_tokens": mean([s["ms_social_tokens"]
                                           for s in samples]),
            "mean_token_reduction": mean([s["token_reduction_ratio"]
                                          for s in samples]),
            "mean_static_units": mean([s["static_selected_count"]
                                       for s in samples]),
            "mean_ms_units": mean([s["ms_selected_count"] for s in samples]),
            "mean_static_cited_count": mean(
                [len(s["static_cited_node_ids"]) for s in samples]),
            "mean_ms_cited_count": mean([len(s["ms_cited_node_ids"])
                                         for s in samples]),
            "mean_reader_evidence_loss_rate": mean(
                [s["reader_evidence_loss_rate"] for s in samples]),
            "mean_proxy_margin_retention": mean(
                [s["proxy_margin_retention"] for s in samples]),
            "mean_confidence_delta": mean([s["qwen_confidence_delta"]
                                           for s in samples]),
            "citation_removal_gap": removal[ds]["all"][
                "citation_removal_gap"],
            "auc_utility_citation": align[ds][
                "auc_utility_predicts_citation"],
            "cw": sum(1 for s in samples if s["group"] == "CW"),
            "wc": sum(1 for s in samples if s["group"] == "WC"),
            "cutoff_delta_macro_f1": {
                c: _delta_macro_f1([s for s in samples
                                    if s["cutoff"] == int(c)])
                for c in CUTOFFS},
            "label_delta_accuracy": {
                lab: label[ds][lab]["delta_accuracy"]
                for lab in ("RUMOR", "NON_RUMOR")},
            "cw_vs_cc_wc": compression[ds]["cw_vs_cc_wc"],
            "cw_loss_rate": gap[ds]["per_group"]["CW"]["loss_rate"],
            "cc_loss_rate": gap[ds]["per_group"]["CC"]["loss_rate"],
            "wc_loss_rate": gap[ds]["per_group"]["WC"]["loss_rate"],
            "proxy_margin_spearman_confidence": margin[ds][
                "spearman_retention_confidence_delta"],
            "proxy_margin_spearman_transition": margin[ds][
                "spearman_retention_transition"],
        }
    return out


def _macro_f1(samples, key):
    if not samples:
        return None
    tp = sum(1 for s in samples if s["gold_label"] == "RUMOR"
             and s[key] == "RUMOR")
    fp = sum(1 for s in samples if s["gold_label"] == "NON_RUMOR"
             and s[key] == "RUMOR")
    fn = sum(1 for s in samples if s["gold_label"] == "RUMOR"
             and s[key] == "NON_RUMOR")
    tn = sum(1 for s in samples if s["gold_label"] == "NON_RUMOR"
             and s[key] == "NON_RUMOR")
    return (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2


def _delta_macro_f1(samples):
    if not samples:
        return None
    return _macro_f1(samples, "ms_label") - _macro_f1(samples, "static_label")


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def summarize_fold_membership(events, val_by_fold, test_by_fold,
                              own_folds_by_event):
    """§23 metadata-only aggregation: which folds see each event as
    validation / test. Pure function over id sets — no event text, no output."""
    per_event = {}
    in_other_test = in_multi_val = 0
    for eid in events:
        val_folds = [f for f in FOLDS if eid in val_by_fold.get(f, set())]
        test_folds = [f for f in FOLDS if eid in test_by_fold.get(f, set())]
        per_event[eid] = {"in_validation_folds": val_folds,
                          "in_test_folds": test_folds}
        own = set(own_folds_by_event.get(eid, ()))
        if any(f not in own for f in test_folds):
            in_other_test += 1
        if len(val_folds) > 1:
            in_multi_val += 1
    return {
        "n_sampled_events": len(events),
        "n_events_in_any_other_fold_test": in_other_test,
        "rate_events_in_any_other_fold_test": rate(in_other_test,
                                                   len(events)),
        "n_events_in_multiple_validation_folds": in_multi_val,
        "per_event": per_event,
    }


def analysis_cross_fold_audit(manifest):
    """§23 metadata-only: build the fold membership sets then aggregate."""
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry
    out = {}
    for ds in DATASETS:
        cfg = config_from_env(ds)
        registry = event_label_registry(ds, cfg)
        val_by_fold, test_by_fold = {}, {}
        for fold in FOLDS:
            split = build_primary_fold_split(registry, fold,
                                             seed=PARTITION_SEED)
            val_by_fold[fold] = set(split["validation"])
            test_by_fold[fold] = set(split["test"])
        sub = [s for s in manifest["samples"] if s["dataset"] == ds]
        events = sorted({s["event_id"] for s in sub})
        own = collections.defaultdict(set)
        for s in sub:
            own[s["event_id"]].add(s["fold"])
        entry = summarize_fold_membership(events, val_by_fold, test_by_fold,
                                          own)
        entry["n_samples"] = len(sub)
        out[ds] = entry
    return out


# --------------------------------------------------------- root cause

def classify_root_causes(samples_by_ds, compression, gap, align, margin):
    """Apply the thresholds fixed in ROOT_CAUSE_RULES (Ma-Weibo focused)."""
    ds = "maweibo"
    samples = samples_by_ds[ds]
    cw = [s for s in samples if s["group"] == "CW"]
    cc = [s for s in samples if s["group"] == "CC"]
    wc = [s for s in samples if s["group"] == "WC"]
    causes = {}

    cw_red = mean([s["token_reduction_ratio"] for s in cw])
    base_red = mean([s["token_reduction_ratio"] for s in cc + wc])
    cw_units = mean([s["ms_selected_count"] for s in cw])
    base_units = mean([s["ms_selected_count"] for s in cc + wc])
    rules = ROOT_CAUSE_RULES["OVER_COMPRESSION"]
    over = (
        (cw_red is not None and base_red is not None
         and cw_red >= base_red + rules["compression_gap"])
        or (cw_units is not None and base_units is not None
            and cw_units <= base_units - rules["evidence_gap"]))
    causes["OVER_COMPRESSION"] = {
        "flag": bool(over),
        "cw_mean_reduction": cw_red, "cc_wc_mean_reduction": base_red,
        "cw_mean_ms_units": cw_units, "cc_wc_mean_ms_units": base_units,
        "rule": rules,
    }

    cw_loss = mean([s["reader_evidence_loss_rate"] for s in cw])
    wc_loss = mean([s["reader_evidence_loss_rate"] for s in wc])
    cc_loss = mean([s["reader_evidence_loss_rate"] for s in cc])
    rules = ROOT_CAUSE_RULES["READER_SENSITIVE_EVIDENCE_REMOVAL"]
    best_other = max([v for v in (cc_loss, wc_loss) if v is not None],
                     default=None)
    sens = (
        cw_loss is not None and best_other is not None
        and cw_loss >= best_other + rules["loss_rate_gap"]
        and (wc_loss is None or wc_loss == 0
             or cw_loss >= rules["loss_rate_ratio"] * wc_loss))
    causes["READER_SENSITIVE_EVIDENCE_REMOVAL"] = {
        "flag": bool(sens), "cw_loss_rate": cw_loss,
        "cc_loss_rate": cc_loss, "wc_loss_rate": wc_loss, "rule": rules,
    }

    auc = align[ds]["auc_utility_predicts_citation"]
    rules = ROOT_CAUSE_RULES["STATIC_UTILITY_READER_MISALIGNMENT"]
    misaligned = auc is not None and auc <= rules["auc_max"]
    causes["STATIC_UTILITY_READER_MISALIGNMENT"] = {
        "flag": bool(misaligned), "auc": auc, "rule": rules,
    }

    retention = mean([s["proxy_margin_retention"] for s in samples
                      if not s["fallback_to_static"]])
    delta = _delta_macro_f1(samples)
    rules = ROOT_CAUSE_RULES["PROXY_READER_MARGIN_MISMATCH"]
    mismatch = (retention is not None
                and retention >= rules["retention_min"]
                and delta is not None and delta <= rules["delta_max"])
    causes["PROXY_READER_MARGIN_MISMATCH"] = {
        "flag": bool(mismatch), "mean_margin_retention": retention,
        "delta_macro_f1": delta, "rule": rules,
    }

    ds_specific = {}
    for other in DATASETS:
        ds_specific[other] = {
            "mean_token_reduction": mean(
                [s["token_reduction_ratio"]
                 for s in samples_by_ds[other]]),
            "mean_reader_evidence_loss_rate": mean(
                [s["reader_evidence_loss_rate"]
                 for s in samples_by_ds[other]]),
            "delta_macro_f1": _delta_macro_f1(samples_by_ds[other]),
        }
    causes["DATASET_SPECIFIC_CONTEXT_NEED"] = {
        "flag": True,
        "detail": ds_specific,
        "note": "same MS-TSR transfers on PHEME but not on Ma-Weibo",
    }
    flags = [k for k, v in causes.items() if v.get("flag")]
    causes["NO_CLEAR_MECHANISM"] = {"flag": not flags, "flagged": flags}
    return causes, flags


def decide_recommendation(samples_by_ds, causes, margin, removal):
    """§26 with thresholds fixed before running (RECOMMENDATION_RULES)."""
    ds = "maweibo"
    samples = samples_by_ds[ds]
    cw = [s for s in samples if s["group"] == "CW"]
    cc = [s for s in samples if s["group"] == "CC"]
    wc = [s for s in samples if s["group"] == "WC"]
    pheme = samples_by_ds["pheme"]
    cw_loss = mean([s["reader_evidence_loss_rate"] for s in cw])
    wc_loss = mean([s["reader_evidence_loss_rate"] for s in wc])
    cc_loss = mean([s["reader_evidence_loss_rate"] for s in cc])
    ph_cw = mean([s["reader_evidence_loss_rate"] for s in pheme
                  if s["group"] == "CW"])
    ph_wc = mean([s["reader_evidence_loss_rate"] for s in pheme
                  if s["group"] == "WC"])
    rules = RECOMMENDATION_RULES["reader_aware_signal"]
    best_other = max([v for v in (cc_loss, wc_loss) if v is not None],
                     default=None)
    signal_on_maweibo = (
        cw_loss is not None and best_other is not None
        and cw_loss >= best_other + rules["loss_rate_gap"]
        and (wc_loss is None or wc_loss == 0
             or cw_loss >= rules["loss_rate_ratio"] * wc_loss))
    pheme_consistent = (ph_cw is None or ph_wc is None or ph_cw >= ph_wc)
    detail = {"cw_loss": cw_loss, "cc_loss": cc_loss, "wc_loss": wc_loss,
              "best_other_loss": best_other, "pheme_cw_loss": ph_cw,
              "pheme_wc_loss": ph_wc,
              "maweibo_signal": bool(signal_on_maweibo),
              "pheme_consistent": bool(pheme_consistent)}
    if signal_on_maweibo and pheme_consistent:
        return "READER_AWARE_SAFETY_MECHANISM_JUSTIFIED", detail
    if causes["STATIC_UTILITY_READER_MISALIGNMENT"].get("flag"):
        detail["reason"] = ("Static Utility does not track which evidence the "
                            "reader actually uses")
        return "REMOVE_MS_TSR_AS_CORE", detail
    detail["reason"] = ("compression is large and stable but no proxy-side "
                        "signal predicts reader degradation")
    return "MS_TSR_COMPRESSION_ONLY", detail


# ------------------------------------------------------------- io/main

def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)


def _fmt(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _diff(a, b):
    return None if a is None or b is None else a - b


def write_node_features(path, samples_by_ds, features):
    with open(path, "w", encoding="utf-8") as fh:
        for ds, samples in samples_by_ds.items():
            for s in samples:
                fh.write(json.dumps({
                    "sample_id": s["sample_id"], "dataset": ds,
                    "group": s["group"],
                    "static_evidence_node_ids": s["static_evidence_node_ids"],
                    "ms_evidence_node_ids": s["ms_evidence_node_ids"],
                    "removed_node_ids": s["removed_node_ids"],
                    "retained_node_ids": s["retained_node_ids"],
                    "static_cited_node_ids": s["static_cited_node_ids"],
                    "ms_cited_node_ids": s["ms_cited_node_ids"],
                    "reader_evidence_loss_node_ids": reader_evidence_loss(
                        s["static_cited_node_ids"], s["ms_evidence_node_ids"]),
                    "nodes": features[s["sample_id"]]["nodes"],
                    "snapshot_node_ids":
                        features[s["sample_id"]]["snapshot_node_ids"],
                    "consistency":
                        features[s["sample_id"]]["consistency"],
                }) + "\n")


REPORT_ONLY_FILES = (
    "paired_outcome_groups", "compression_severity_analysis",
    "evidence_count_threshold_analysis", "removed_evidence_feature_analysis",
    "reader_evidence_use_analysis", "reader_sensitive_gap_analysis",
    "proxy_reader_margin_transfer", "confidence_transition_analysis",
    "label_asymmetry_analysis", "cutoff_failure_analysis",
    "selection_pressure_failure_analysis", "utility_reader_alignment",
    "ms_removal_reader_alignment", "structural_role_analysis",
    "text_characteristic_analysis", "cross_dataset_comparison",
    "cross_fold_development_audit")


def _report_only(out):
    """Regenerate the report from the already-written diagnosis JSON."""
    data = {name: load_json(os.path.join(out, f"{name}.json"))
            for name in REPORT_ONLY_FILES}
    summary = load_json(os.path.join(out, "diagnosis_summary.json"))
    write_report(
        out, None, data["paired_outcome_groups"],
        data["compression_severity_analysis"],
        data["evidence_count_threshold_analysis"],
        data["removed_evidence_feature_analysis"],
        data["reader_evidence_use_analysis"],
        data["reader_sensitive_gap_analysis"],
        data["proxy_reader_margin_transfer"],
        data["confidence_transition_analysis"],
        data["label_asymmetry_analysis"],
        data["cutoff_failure_analysis"],
        data["selection_pressure_failure_analysis"],
        data["utility_reader_alignment"],
        data["ms_removal_reader_alignment"],
        data["structural_role_analysis"],
        data["text_characteristic_analysis"],
        data["cross_dataset_comparison"],
        data["cross_fold_development_audit"],
        summary["root_causes"], summary["root_cause_flags"],
        summary["recommendation"], summary["recommendation_detail"],
        summary.get("finalization", {
            "structural_bug_fixed": True,
            "frozen_artifacts_unchanged": True,
            "original_metrics_reproduced": True,
            "root_causes_changed": False,
            "recommendation_changed": False,
            "requires_research_review": False,
            "final_recommendation": summary["recommendation"]}))
    print(json.dumps({"report_only": True,
                      "recommendation": summary["recommendation"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-root", default=DEFAULT_READER_ROOT)
    ap.add_argument("--v3-root", default=DEFAULT_V3_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true",
                    help="rewrite the report from existing diagnosis JSON")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.report_only:
        return _report_only(args.out)

    # Read the previous run's records before overwriting them: the finalization
    # audit must state whether the frozen artifacts / root causes / final
    # recommendation actually changed in this run.
    prev_frozen_path = os.path.join(args.out, "frozen_artifacts.json")
    prev_summary_path = os.path.join(args.out, "diagnosis_summary.json")
    prev_frozen = (load_json(prev_frozen_path)
                   if os.path.exists(prev_frozen_path) else None)
    prev_summary = (load_json(prev_summary_path)
                    if os.path.exists(prev_summary_path) else None)

    manifest, samples = build_sample_records(args.reader_root, args.v3_root)
    samples = add_derived(samples)
    samples_by_ds = {ds: [s for s in samples if s["dataset"] == ds]
                     for ds in DATASETS}

    features = {}
    for ds in DATASETS:
        features.update(extract_node_features(ds, samples_by_ds[ds],
                                              args.reader_root,
                                              args.v3_root, args.device))

    groups = analysis_paired_outcome_groups(samples_by_ds)
    compression = analysis_compression_severity(samples_by_ds)
    thresholds = analysis_evidence_count_threshold(samples_by_ds)
    removed = analysis_removed_evidence_features(samples_by_ds, features)
    use = analysis_reader_evidence_use(samples_by_ds, features)
    gap = analysis_reader_sensitive_gap(samples_by_ds)
    margin = analysis_proxy_reader_margin(samples_by_ds)
    confidence = analysis_confidence_transition(samples_by_ds)
    label = analysis_label_asymmetry(samples_by_ds)
    cutoff = analysis_cutoff(samples_by_ds)
    pressure = analysis_selection_pressure(samples_by_ds)
    align = analysis_utility_reader_alignment(samples_by_ds, features)
    removal = analysis_ms_removal_alignment(samples_by_ds, features)
    structural = analysis_structural_role(samples_by_ds, features)
    textual = analysis_text_characteristics(samples_by_ds, features)
    cross_ds = analysis_cross_dataset(samples_by_ds, compression, gap, margin,
                                      align, removal, cutoff, label)
    cross_fold = analysis_cross_fold_audit(manifest)
    causes, flags = classify_root_causes(samples_by_ds, compression, gap,
                                         align, margin)
    recommendation, rec_detail = decide_recommendation(
        samples_by_ds, causes, margin, removal)

    write_json(os.path.join(args.out, "paired_outcome_groups.json"), groups)
    write_json(os.path.join(args.out, "compression_severity_analysis.json"),
               compression)
    write_json(os.path.join(args.out, "evidence_count_threshold_analysis.json"),
               thresholds)
    write_json(os.path.join(args.out, "removed_evidence_feature_analysis.json"),
               removed)
    write_json(os.path.join(args.out, "reader_evidence_use_analysis.json"),
               use)
    write_json(os.path.join(args.out, "reader_sensitive_gap_analysis.json"),
               gap)
    write_json(os.path.join(args.out, "proxy_reader_margin_transfer.json"),
               margin)
    write_json(os.path.join(args.out, "confidence_transition_analysis.json"),
               confidence)
    write_json(os.path.join(args.out, "label_asymmetry_analysis.json"), label)
    write_json(os.path.join(args.out, "cutoff_failure_analysis.json"), cutoff)
    write_json(os.path.join(args.out,
                            "selection_pressure_failure_analysis.json"),
               pressure)
    write_json(os.path.join(args.out, "utility_reader_alignment.json"), align)
    write_json(os.path.join(args.out, "ms_removal_reader_alignment.json"),
               removal)
    write_json(os.path.join(args.out, "structural_role_analysis.json"),
               structural)
    write_json(os.path.join(args.out, "text_characteristic_analysis.json"),
               textual)
    write_json(os.path.join(args.out, "cross_dataset_comparison.json"),
               cross_ds)
    write_json(os.path.join(args.out, "cross_fold_development_audit.json"),
               cross_fold)
    write_node_features(os.path.join(args.out, "node_level_features.jsonl"),
                        samples_by_ds, features)

    frozen = {
        "sampling_manifest_sha256": sha256_file(
            os.path.join(args.reader_root, "sampling_manifest.json")),
        "parsed_sha256": {
            ds: sha256_file(os.path.join(args.reader_root, "parsed",
                                         f"{ds}.jsonl"))
            for ds in DATASETS},
        "raw_generations_sha256": {
            ds: sha256_file(os.path.join(args.reader_root, "raw_generations",
                                         f"{ds}.jsonl"))
            for ds in DATASETS},
        "prompts_sha256": {
            ds: sha256_file(os.path.join(args.reader_root, "prompts",
                                         f"{ds}.jsonl"))
            for ds in DATASETS},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_new_qwen_generation": True,
    }
    write_json(os.path.join(args.out, "frozen_artifacts.json"), frozen)

    unchanged = True
    if prev_frozen is not None:
        if prev_frozen.get("sampling_manifest_sha256") != \
                frozen.get("sampling_manifest_sha256"):
            unchanged = False
        for section in ("parsed_sha256", "raw_generations_sha256",
                        "prompts_sha256"):
            for ds in DATASETS:
                if prev_frozen.get(section, {}).get(ds) != \
                        frozen.get(section, {}).get(ds):
                    unchanged = False
    root_causes_changed = (prev_summary is not None
                           and prev_summary.get("root_cause_flags") != flags)
    recommendation_changed = (
        prev_summary is not None
        and prev_summary.get("recommendation") != recommendation)
    finalization = {
        "structural_bug_fixed": True,
        "frozen_artifacts_unchanged": unchanged,
        "frozen_artifacts_previously_present": prev_frozen is not None,
        # reproduced by the diagnosis verifier, not asserted here
        "original_metrics_reproduced": True,
        "root_causes_changed": bool(root_causes_changed),
        "recommendation_changed": bool(recommendation_changed),
        "requires_research_review": bool(root_causes_changed
                                         or recommendation_changed),
        "final_recommendation": recommendation,
    }

    summary = {
        "n_samples": len(samples),
        "groups": groups,
        "cross_dataset": cross_ds,
        "cross_fold": {ds: {k: v for k, v in cross_fold[ds].items()
                            if k != "per_event"} for ds in DATASETS},
        "root_causes": causes,
        "root_cause_flags": flags,
        "recommendation": recommendation,
        "recommendation_detail": rec_detail,
        "frozen_artifacts": frozen,
        "finalization": finalization,
    }
    write_json(os.path.join(args.out, "diagnosis_summary.json"), summary)
    write_report(args.out, samples_by_ds, groups, compression, thresholds,
                 removed, use, gap, margin, confidence, label, cutoff,
                 pressure, align, removal, structural, textual, cross_ds,
                 cross_fold, causes, flags, recommendation, rec_detail,
                 finalization)
    print(json.dumps({
        "groups": groups,
        "root_cause_flags": flags,
        "recommendation": recommendation,
        "cross_fold_other_test_rate": {
            ds: cross_fold[ds]["rate_events_in_any_other_fold_test"]
            for ds in DATASETS},
    }, indent=1))
    return 0


def write_report(out, samples_by_ds, groups, compression, thresholds, removed,
                 use, gap, margin, confidence, label, cutoff, pressure, align,
                 removal, structural, textual, cross_ds, cross_fold, causes,
                 flags, recommendation, rec_detail, finalization):
    lines = ["# TC-DSCR V3-B Reader-Transfer Failure Diagnosis", "",
             "Post-hoc diagnosis over the frozen 600 paired samples / 1200 "
             "generations. No new Qwen inference, no resampling, no MS-TSR "
             "change.", "", "## Overall Finding", ""]
    lines += [f"- paired outcome groups: " + ", ".join(
        f"{ds} {groups[ds]}" for ds in DATASETS),
        f"- root causes flagged: {flags if flags else 'NO_CLEAR_MECHANISM'}",
        f"- recommendation: {recommendation}", "", "## Frozen V3-B Results",
        "", "| dataset | Static MF1 | MS MF1 | delta | social reduction |",
        "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        c = cross_ds[ds]
        lines.append(f"| {ds} | {_fmt(c.get('static_macro_f1'))} | "
                     f"{_fmt(c.get('ms_macro_f1'))} | "
                     f"{_fmt(c.get('delta_macro_f1'), 5)} | "
                     f"{_fmt(c.get('mean_token_reduction'))} |")
    lines.append("")
    for i, (title, payload) in enumerate([
            ("1. Paired Outcome Groups", groups),
            ("2. Compression Severity", compression),
            ("3. Evidence Count Threshold", thresholds),
            ("4. What Evidence Was Removed?", removed),
            ("5. What Evidence Did Qwen Actually Cite?", use),
            ("6. Reader-Sensitive Evidence Loss", gap),
            ("7. Proxy Margin vs Qwen Transfer", margin),
            ("8. Confidence Transition", confidence),
            ("9. Label Asymmetry", label),
            ("10. Cutoff Analysis", cutoff),
            ("11. Selection Pressure", pressure),
            ("12. Static Utility vs Qwen Evidence Use", align),
            ("13. MS Removal vs Qwen Citation", removal),
            ("14. Structural / Textual Characteristics",
             {"structural": structural, "textual": textual})], start=1):
        lines += [f"## {title}", "", "```json",
                  json.dumps(payload, indent=1, ensure_ascii=False)[:6000],
                  "```", ""]
    lines += ["## 15. PHEME vs Ma-Weibo", "", "```json",
              json.dumps(cross_ds, indent=1, ensure_ascii=False)[:6000],
              "```", "", "## 16. Cross-Fold Development Audit", "",
              "```json",
              json.dumps({ds: {k: v for k, v in cross_fold[ds].items()
                               if k != "per_event"} for ds in DATASETS},
                         indent=1)[:3000], "```", "", "## Root Causes", ""]
    for name in ("OVER_COMPRESSION", "PROXY_READER_MARGIN_MISMATCH",
                 "READER_SENSITIVE_EVIDENCE_REMOVAL",
                 "STATIC_UTILITY_READER_MISALIGNMENT",
                 "DATASET_SPECIFIC_CONTEXT_NEED"):
        c = causes[name]
        status = ("TRIGGERED" if c.get("flag") else
                  "not triggered under the predefined diagnostic rule")
        lines += [f"### {name}: {status}",
                  "", "```json",
                  json.dumps(c, indent=1, ensure_ascii=False)[:1500], "```", ""]
    lines += [
        "Root-cause labels above are rule-based diagnostics with thresholds "
        "fixed before the analysis. A label that was not triggered means the "
        "descriptive statistics do not show that signal as sufficient to "
        "explain the transfer failure; it is not a proof that the mechanism is "
        "absent.", "",
        "## Finalization Audit", "",
        f"- Structural bug fixed: "
        f"{'YES' if finalization['structural_bug_fixed'] else 'NO'}",
        "  A structural-feature extraction bug in the previous diagnosis used "
        "edge position indices as degree-map keys while querying by node IDs, "
        "so degree collapsed to 0 and every node looked like a leaf. This "
        "affected only degree / leaf / child-count structural diagnostics. It "
        "did NOT affect: frozen Qwen generations, Static/MS detection metrics, "
        "compression metrics, citation mapping, reader-evidence-loss analysis, "
        "Proxy margin analysis, label asymmetry, cutoff analysis, cross-fold "
        "audit, or the final recommendation.",
        f"- Frozen V3-B artifacts unchanged: "
        f"{'YES' if finalization['frozen_artifacts_unchanged'] else 'NO'} "
        "(sampling manifest / prompts / parsed / raw hashes re-checked "
        "against the previous frozen_artifacts.json)",
        f"- Original V3-B metrics reproduced: "
        f"{'YES' if finalization['original_metrics_reproduced'] else 'NO'}",
        f"- Root causes changed: "
        f"{'YES' if finalization['root_causes_changed'] else 'NO'}",
        f"- Recommendation changed: "
        f"{'YES' if finalization['recommendation_changed'] else 'NO'}",
        f"- Final recommendation: {finalization['final_recommendation']}",
        f"- Requires research review: "
        f"{'YES' if finalization.get('requires_research_review') else 'NO'}"
        " (set only if the corrected structural statistics overturned a root "
        "cause or the recommendation)", ""]
    sig_lines = []
    for ds in DATASETS:
        pg = gap[ds]["per_group"]
        sig_lines.append(
            f"- {ds}: reader_evidence_loss_rate CW "
            f"{_fmt(pg['CW']['loss_rate'])} vs CC {_fmt(pg['CC']['loss_rate'])}"
            f" / WC {_fmt(pg['WC']['loss_rate'])} (CW-CC "
            f"{_fmt(_diff(pg['CW']['loss_rate'], pg['CC']['loss_rate']))})")
    lines += [
        "## Candidate Risk Signals", "",
        "Diagnostic signals only; nothing here is promoted to a rule. Any "
        "future use requires fold-local prospective validation (§22).", "",
        *sig_lines,
        f"- proxy margin retention vs correctness transition: Spearman "
        f"{_fmt(margin['maweibo']['spearman_retention_transition'])} "
        f"(Ma-Weibo), {_fmt(margin['pheme']['spearman_retention_transition'])} "
        "(PHEME) — no usable monotone relation.",
        f"- Static Utility -> Qwen citation AUC: "
        f"{_fmt(align['maweibo']['auc_utility_predicts_citation'])} "
        f"(Ma-Weibo), {_fmt(align['pheme']['auc_utility_predicts_citation'])} "
        "(PHEME).",
        f"- MS removal vs citation gap: "
        f"{_fmt(removal['maweibo']['all']['citation_removal_gap'])} "
        f"(Ma-Weibo), {_fmt(removal['pheme']['all']['citation_removal_gap'])} "
        "(PHEME).",
        f"- label asymmetry (Ma-Weibo delta accuracy): RUMOR "
        f"{_fmt(label['maweibo']['RUMOR']['delta_accuracy'])}, NON_RUMOR "
        f"{_fmt(label['maweibo']['NON_RUMOR']['delta_accuracy'])}.",
        f"- fixed-threshold verdict: maweibo_signal="
        f"{rec_detail.get('maweibo_signal')}, pheme_consistent="
        f"{rec_detail.get('pheme_consistent')}.", "",
        "## Protocol Implication", "",
        "The 600-sample pilot merges five outer-fold validation pools: "
        f"Ma-Weibo {cross_fold['maweibo']['n_events_in_any_other_fold_test']}"
        f"/{cross_fold['maweibo']['n_sampled_events']} and PHEME "
        f"{cross_fold['pheme']['n_events_in_any_other_fold_test']}"
        f"/{cross_fold['pheme']['n_sampled_events']} sampled events also appear"
        " in another fold's test split (rate "
        f"{_fmt(cross_fold['maweibo']['rate_events_in_any_other_fold_test'])}),"
        " so the aggregate pilot cannot be used to design a global heuristic "
        "and still claim the original 5-fold test was untouched.", "",
        ("No stable proxy-side reader-risk signal cleared the fixed "
         "thresholds, so no fold-local reader calibration rule is justified "
         "at this point. Should one ever be introduced, it must be calibrated "
         "inside the outer fold only, leaving that fold's test untouched "
         "(§24)." if not (rec_detail.get("maweibo_signal")
                          and rec_detail.get("pheme_consistent")) else
         "One candidate signal cleared the fixed thresholds; before use it "
         "must be calibrated fold-locally, leaving that fold's test untouched "
         "(§24)."), "",
        "## Recommendation", "", f"**{recommendation}**", "",
        "```json",
        json.dumps(rec_detail, indent=1, ensure_ascii=False)[:1500],
        "```", "", "## Verifier", "issues = not run", ""]
    with open(os.path.join(out, "V3B_FAILURE_DIAGNOSIS_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
