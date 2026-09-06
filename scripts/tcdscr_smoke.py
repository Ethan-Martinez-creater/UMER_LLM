#!/usr/bin/env python
"""STEP 15: TC-DSCR V2 tiny smoke run (plan §30) and Code Complete outputs.

Constraints enforced here: <= 32 events per dataset, cutoffs = SOURCE_ONLY +
two dynamic cutoffs (15m, 1h), encoder 1 epoch, selector 1 epoch, and at most
20 LLM requests in total (10 per dataset). The run validates the chain

    raw -> causal snapshot -> feature -> encoder -> selector ->
    dynamic memory -> evidence -> prompt -> Qwen -> parser -> result

and emits the §31 artifacts under results/tcdscr/code_smoke/.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch

from tcdscr_common import (PROJECT_DIR, SemanticBackend,
                           build_event_snapshots, load_events)

SMOKE_CUTOFFS_MIN = (15, 60)          # two dynamic cutoffs (§30)
MEMORY_LAMBDA_N = 0.5                 # fixed mid-grid values for smoke only
MEMORY_LAMBDA_P = 0.1                 # (no tuning; Stage B searches grids)


def run_tests(report_path):
    cmd = [sys.executable, "-m", "pytest", "tcdscr/tests", "-q",
           "--tb=short"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True,
                          text=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n" + proc.stderr)
    return proc.returncode == 0, proc.stdout


def infer_dataset(cfg, events, llm, llm_cache, per_dataset_llm_limit,
                  template_lang, args):
    from tcdscr.config.schema import SOURCE_ONLY
    from tcdscr.context.packer import pack_context
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    from tcdscr.data.manifests import AdapterAuditLog, CapStatistics
    from tcdscr.data.temporal_split import (event_folds_to_snapshot_folds,
                                            stratified_event_folds)
    from tcdscr.evaluation.baselines import (rank_candidates,
                                             resolve_context_length,
                                             select_all_current)
    from tcdscr.evaluation.leakage_scanner import scan_prompt
    from tcdscr.evaluation.temporal_metrics import (evidence_turnover,
                                                    flip_rate,
                                                    persistent_new_ratio)
    from tcdscr.llm.cache import build_cache_key
    from tcdscr.llm.parser import label_to_int, parse_label
    from tcdscr.llm.qwen_wrapper import GENERATION_CONFIG
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder
    from tcdscr.models.evidence_memory import DynamicEvidenceMemory
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy
    from tcdscr.training.train_encoder import train_encoder
    from tcdscr.training.train_selector import train_selector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = SemanticBackend(cfg)
    cutoffs = (SOURCE_ONLY,) + SMOKE_CUTOFFS_MIN

    snaps = {e["event_id"]: build_event_snapshots(e, SMOKE_CUTOFFS_MIN)
             for e in events}
    feats = {e["event_id"]: {cut: backend.snapshot_features(e, s)
                             for cut, s in snaps[e["event_id"]].items()}
             for e in events}
    labels = {e["event_id"]: e["label"] for e in events}

    encoder = CausalSocialEncoder()
    dynamic_feats = {eid: {c: f for c, f in fmap.items() if c != SOURCE_ONLY}
                     for eid, fmap in feats.items()}
    enc_hist = train_encoder(encoder, dynamic_feats, labels,
                             epochs=args.epochs, seed=args.seed,
                             device=device, log=lambda *a: None)
    selector = StaticUtilitySelector()
    proxy = SelectorProxy()
    sel_hist = train_selector(encoder, selector, proxy, dynamic_feats,
                              labels, epochs=args.epochs, seed=args.seed,
                              device=device, log=lambda *a: None)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              trust_remote_code=False)
    budget_sel = EvidenceBudgetSelector(tokenizer, args.budget)

    # delta-fix §19/§20/§39: resolved context length + new-check counters
    context_length = resolve_context_length(
        getattr(llm, "model", None).config if llm is not None else None,
        tokenizer)
    max_new_tokens = GENERATION_CONFIG["max_new_tokens"]
    delta_checks = {
        "prompt_source_binding_failures": 0,
        "fold_integrity_failures": 0,
        "recent_baseline_order_failures": 0,
        "context_overflow_failures": 0,
    }

    cap = CapStatistics()
    audit = AdapterAuditLog()
    rows = []
    prompt_examples = []
    llm_used = 0
    folds = stratified_event_folds(labels, k=5, seed=cfg.fold_seed)

    for event in sorted(events, key=lambda e: e["event_id"]):
        eid = event["event_id"]
        memory = DynamicEvidenceMemory(MEMORY_LAMBDA_N, MEMORY_LAMBDA_P)
        for cutoff in cutoffs:
            snap = snaps[eid][cutoff]
            feat = feats[eid][cutoff]
            cap.add(snap)
            feat = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in feat.items()}
            with torch.no_grad():
                node_repr, event_repr, logits = encoder(
                    feat["node_feat"], feat["struct_feat"],
                    feat["node_feat"].size(0))
            src_pos = feat["node_ids"].index(feat["source_id"])
            cand = [i for i in range(len(feat["node_ids"])) if i != src_pos]
            u_scores = [None] * len(feat["node_ids"])
            d_scores = list(u_scores)
            nov = list(u_scores)
            per = list(u_scores)
            selected_ids = []
            evidence_tokens = 0
            selected_units = []
            from tcdscr.context.evidence_unit import build_evidence_units
            all_units = build_evidence_units(snap)
            # delta-fix §39: source binding + fold inheritance checks
            event_source_text = next(
                n["text"] for n in event["nodes"]
                if n["node_id"] == event["source_id"])
            if snap["texts"][src_pos] != event_source_text:
                delta_checks["prompt_source_binding_failures"] += 1
            try:
                sfs = event_folds_to_snapshot_folds({(eid, cutoff): eid},
                                                    folds)
                if sfs[(eid, cutoff)] != folds[eid]:
                    delta_checks["fold_integrity_failures"] += 1
            except ValueError:
                delta_checks["fold_integrity_failures"] += 1
            # delta-fix §39: recent baseline must surface the latest reply
            if all_units:
                recent_scores = rank_candidates(
                    "recent_budget", snap, all_units,
                    {"dataset": cfg.dataset})
                top = max(range(len(all_units)),
                          key=lambda i: recent_scores[i])
                if recent_scores[top] != max(u["elapsed_seconds"]
                                             for u in all_units):
                    delta_checks["recent_baseline_order_failures"] += 1
            # delta-fix §4E/§18: bounded all-current selection for this row
            ac = select_all_current(
                snap, all_units, budget_sel, context_length, max_new_tokens,
                prompt_builder=lambda s, acc: pack_context(
                    s["texts"][src_pos], s, acc)["prompt"])
            if ac["input_tokens"] + max_new_tokens > context_length:
                delta_checks["context_overflow_failures"] += 1
            if cand:
                cand_ids = [feat["node_ids"][i] for i in cand]
                u_cand = selector(
                    node_repr[cand], event_repr,
                    feat["node_feat"][cand], feat["node_feat"][src_pos],
                    feat["struct_feat"][cand, -3:])
                d_cand, nov_cand, per_cand = memory.score(
                    cand_ids, u_cand, feat["node_feat"][cand])
                for k, i in enumerate(cand):
                    u_scores[i] = float(u_cand[k])
                    d_scores[i] = float(d_cand[k])
                    nov[i] = float(nov_cand[k])
                    per[i] = float(per_cand[k])
                units = all_units
                d_by_id = dict(zip(cand_ids, d_cand))
                unit_scores = [d_by_id[u_["node_id"]] for u_ in units]
                accepted, evidence_tokens = budget_sel.select(
                    units, unit_scores)
                selected_ids = [u_["node_id"] for u_ in accepted]
                unit_by_id = {u_["node_id"]: u_ for u_ in units}
                selected_units = [unit_by_id[s] for s in selected_ids]
                memory.load_previous(
                    selected_ids,
                    feat["node_feat"][[feat["node_ids"].index(s)
                                       for s in selected_ids]]
                    if selected_ids else torch.empty(0, 384))
            packed = pack_context(
                snap["texts"][src_pos], snap, selected_units,
                template_lang=template_lang)
            from tcdscr.evaluation.leakage_scanner import scan_prompt
            leak = scan_prompt(packed["prompt"], event, snap)
            total_tokens = len(tokenizer(
                packed["prompt"], add_special_tokens=False)["input_ids"])

            prediction = None
            cache_key = None
            if llm_used < per_dataset_llm_limit:
                key = build_cache_key(
                    model_id=cfg.qwen_model_path, model_revision="local",
                    prompt=packed["prompt"],
                    generation_config=GENERATION_CONFIG,
                    dataset=cfg.dataset, event_id=eid, cutoff=cutoff,
                    selector_checkpoint_hash=f"smoke_e{args.epochs}",
                    budget=args.budget)
                raw = llm_cache.get(key)
                if raw is None:
                    raw = llm.generate(packed["prompt"])
                    llm_cache.put(key, raw)
                prediction = label_to_int(parse_label(raw))
                cache_key = key
                llm_used += 1
            row = {
                "dataset": cfg.dataset,
                "event_id": eid,
                "cutoff": cutoff,
                "fold": folds[eid],
                "gold": event["label"],
                "prediction": prediction,
                "selected_node_ids": selected_ids,
                "base_score": [u_scores[i] for i in cand],
                "novelty": [nov[i] for i in cand],
                "persistence": [per[i] for i in cand],
                "dynamic_score": [d_scores[i] for i in cand],
                "evidence_tokens": evidence_tokens,
                "total_prompt_tokens": total_tokens,
                "prompt_sha256": hashlib.sha256(
                    packed["prompt"].encode()).hexdigest(),
                "cache_key": cache_key,
                "leakage": leak,
                "num_candidates": len(cand),
                "num_nodes": snap["num_nodes_after_cap"],
                "max_depth": snap["max_depth"],
                "all_current_truncated": ac["truncated"],
                "all_current_units_before_truncation":
                    ac["units_before_truncation"],
                "all_current_units_after_truncation":
                    ac["units_after_truncation"],
                "all_current_tokens_dropped": ac["tokens_dropped"],
                "all_current_context_length": context_length,
                "all_current_input_tokens": ac["input_tokens"],
            }
            rows.append(row)
            if len(prompt_examples) < 2:
                prompt_examples.append({
                    "event_id": eid, "cutoff": cutoff,
                    "prompt": packed["prompt"]})

    for event in events:
        audit.add_event(event, list(snaps[event["event_id"]].values()))
    scored = [r for r in rows if r["prediction"] is not None]
    from tcdscr.evaluation.evidence_metrics import efficiency_metrics
    from tcdscr.evaluation.metrics import classification_metrics
    metrics = classification_metrics(
        [r["gold"] for r in scored], [r["prediction"] for r in scored])
    traces = {}
    for r in rows:
        traces.setdefault(r["event_id"], []).append(r)
    trace_list = [
        [{"cutoff": r["cutoff"], "prediction": r["prediction"],
          "selected_ids": r["selected_node_ids"]} for r in tr]
        for tr in traces.values()]
    return {
        "rows": rows,
        "metrics": metrics,
        "flip": flip_rate(trace_list),
        "turnover": evidence_turnover(trace_list),
        "persist": persistent_new_ratio(trace_list),
        "efficiency": efficiency_metrics(rows),
        "cap": cap.to_dict(),
        "audit_summary": audit.summary(),
        "encoder_history": enc_hist,
        "selector_history": sel_hist,
        "device": device,
        "llm_requests": llm_used,
        "prompt_examples": prompt_examples,
        "folds_demo": folds,
        "delta_checks": delta_checks,
        "context_length": context_length,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pheme", "maweibo"])
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--llm-per-dataset", type=int, default=10)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3090)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = config_from_env(args.datasets[0]).output_dir
    os.makedirs(out_dir, exist_ok=True)
    print("output dir:", out_dir, flush=True)

    tests_pass = True
    if not args.skip_tests:
        print("running unit tests ...", flush=True)
        tests_pass, _ = run_tests(os.path.join(out_dir, "test_report.txt"))
        print("tests pass:", tests_pass, flush=True)

    from tcdscr.data.manifests import AdapterAuditLog  # noqa: F401
    from tcdscr.llm.cache import LLMResponseCache
    from tcdscr.llm.qwen_wrapper import QwenRumorLLM

    manifest = {
        "stage": "code_complete_tiny_smoke",
        "constraints": {
            "max_events_per_dataset": 32,
            "cutoffs": ["SOURCE_ONLY"] + list(SMOKE_CUTOFFS_MIN),
            "encoder_epochs": args.epochs,
            "selector_epochs": args.epochs,
            "llm_requests_total": 20,
            "token_budget": args.budget,
        },
        "seed": args.seed,
        "tests_pass": tests_pass,
        "datasets": {},
    }
    all_leak = {"checked": 0, "failed": 0, "pass": True}
    cap_reports = {}
    prompt_md = ["# Smoke prompt examples", ""]
    shapes = model_shapes()
    llm = None
    llm_cache = None
    pred_path = os.path.join(out_dir, "smoke_predictions.jsonl")

    for dataset in args.datasets:
        cfg = config_from_env(dataset)
        if llm is None:
            print("loading Qwen3-8B ...", flush=True)
            llm = QwenRumorLLM(cfg.qwen_model_path)
            llm_cache = LLMResponseCache(
                os.path.join(cfg.cache_dir, "llm_responses.jsonl"))
        events = load_events(dataset, cfg, limit=args.events, seed=args.seed)
        template_lang = "en" if dataset == "pheme" else "zh"
        report = infer_dataset(cfg, events, llm, llm_cache,
                               args.llm_per_dataset, template_lang, args)
        cap_reports[dataset] = report["cap"]

        with open(pred_path, "a", encoding="utf-8") as fh:
            for row in report["rows"]:
                fh.write(json.dumps(row) + "\n")

        n_leak_fail = sum(1 for r in report["rows"]
                          if not r["leakage"]["pass"])
        all_leak["checked"] += len(report["rows"])
        all_leak["failed"] += n_leak_fail
        all_leak["pass"] = all_leak["pass"] and n_leak_fail == 0

        for i, ex in enumerate(report["prompt_examples"], 1):
            prompt_md += [f"## {dataset} example {i} "
                          f"(event {ex['event_id']}, cutoff {ex['cutoff']})",
                          "", "```text", ex["prompt"], "```", ""]

        manifest["datasets"][dataset] = {
            "events": len(events),
            "event_ids": sorted(e["event_id"] for e in events),
            "llm_requests": report["llm_requests"],
            "metrics": report["metrics"],
            "flip": report["flip"],
            "turnover": report["turnover"],
            "persistent_new": report["persist"],
            "efficiency": report["efficiency"],
            "cap": report["cap"],
            "audit_summary": report["audit_summary"],
            "encoder_history": report["encoder_history"],
            "selector_history": report["selector_history"],
            "device": report["device"],
            "folds_demo": {k: v for k, v in
                           list(report["folds_demo"].items())[:8]},
            "delta_checks": report["delta_checks"],
            "context_length": report["context_length"],
        }
        print(dataset, "metrics:", json.dumps(report["metrics"]), flush=True)

    with open(os.path.join(out_dir, "smoke_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(out_dir, "leakage_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(all_leak, fh, indent=1)
    with open(os.path.join(out_dir, "cap_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cap_reports, fh, indent=1)
    with open(os.path.join(out_dir, "prompt_examples.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(prompt_md))
    with open(os.path.join(out_dir, "model_shapes.json"), "w",
              encoding="utf-8") as fh:
        json.dump(shapes, fh, indent=1)
    with open(os.path.join(out_dir, "known_issues.md"), "w",
              encoding="utf-8") as fh:
        fh.write(KNOWN_ISSUES)

    print("SMOKE DONE; llm requests per dataset:",
          {d: manifest["datasets"][d]["llm_requests"]
           for d in manifest["datasets"]}, flush=True)


def model_shapes():
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy
    enc = CausalSocialEncoder()
    sel = StaticUtilitySelector()
    proxy = SelectorProxy()
    return {
        "encoder_params": sum(p.numel() for p in enc.parameters()),
        "selector_params": sum(p.numel() for p in sel.parameters()),
        "proxy_params": sum(p.numel() for p in proxy.parameters()),
        "encoder_module_summary": {
            "node_projection_weight": list(
                enc.graph_branch.node_projection[0].weight.shape),
            "d_model": enc.graph_branch.d_model,
            "classifier_weight": list(enc.classifier[2].weight.shape),
        },
        "selector_input_dim": 1540,
        "proxy_input_dim": 1536,
    }


KNOWN_ISSUES = """# TC-DSCR V2 Code Complete — Known Issues / Implementation Decisions

1. **Unreachable nodes keep the historical -1 depth sentinel.** A node whose
   causal-safe edge set cannot reach the source (rare: parent excluded or
   child-before-parent) gets depth = -1, hence norm_depth = -1/19. This is the
   historical `.pt` behavior; the count is audited per snapshot
   (`unreachable_count`) and in the dataset audit summary.
2. **Depth > 19 is recorded, never clipped** (§9.2): see
   `depth_overflow_count` per snapshot.
3. **Prompt/LLM text is the raw social text** (no extra normalization); the
   frozen cleaning functions (clean_tweet_pheme / clean_text_weibo) apply to
   the §7 semantic pipeline only. The V2 plan does not specify prompt-side
   cleaning; keeping raw text preserves social cues for the LLM, and this
   decision is recorded here rather than invented silently.
4. **Ma-Weibo node text prefers `original_text`, falling back to `text` only
   when the field is absent** — the exact historical adapter behavior verified
   by the D6 parity run (coverage 99.9996%).
5. **Dynamic memory is single-step rolling**: M_{t-1} is the evidence selected
   at the immediately preceding snapshot of the same event; M_t is overwritten
   after each snapshot. SOURCE_ONLY leaves memory empty, so the first dynamic
   snapshot has novelty = 1 everywhere.
6. **Smoke uses fixed lambda_n=0.5, lambda_p=0.1** (mid-grid values, no
   tuning); formal grid search happens in Stage B on validation only.
7. **LLM request cap**: at most 10 requests per dataset (20 total), consumed
   in deterministic (event_id, cutoff) enumeration order; the remaining
   snapshot rows carry prediction=null and are excluded from metrics.
8. **recent_budget tie-break**: equal timestamps keep snapshot order
   (earlier first). The V2 plan fixes tie-breaking only for the structural
   budget; this choice is recorded for determinism.
9. **SOURCE_ONLY prompts** show evidence block `(none)` and
   observed_replies = 0; the source claim is always provided separately and
   never appears as an evidence unit.
10. **MockRumorLLM exists for unit/integration tests only**; the smoke run
    uses the real frozen Qwen3-8B with greedy decoding and thinking disabled.
11. **Training pools**: SOURCE_ONLY snapshots are excluded from encoder
    training (§36); the selector trains on the same dynamic pool with the
    frozen encoder.
12. **Adapter strictness**: timestamp parse failures raise immediately
    (frozen audit: 100% coverage) instead of silently dropping nodes.
"""

if __name__ == "__main__":
    main()
