import json
import subprocess
from pathlib import Path

root = Path(r"E:\Graduate_work_folder\Graduate_Project_Worksapace\UMER")
out = root / "results" / "tcdscr" / "delta_preflight"
out.mkdir(parents=True, exist_ok=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                     capture_output=True, text=True).stdout.strip()

# ---------------- D5: Ma-Weibo feature pipeline audit ----------------
SRV = "/data/jyz/next/src/rumor_detection/data"
rows = [
    {
        "feature_name": "text_semantic_embedding_384D",
        "dimension": 384,
        "definition": "SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2 on "
                      "original_text (fallback text); encoder L2-normalises output",
        "source_file": SRV + "/embeddings.py + " + SRV + "/adapters/ma_weibo.py",
        "source_line_or_function": "TextEmbedder.encode (embeddings.py L10-24); adapter text field (ma_weibo.py L70-71)",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes (per-post input)",
    },
    {
        "feature_name": "raw_text_cleaning",
        "dimension": -1,
        "definition": "HISTORICAL 240h pipeline embedded clean_text_weibo(original_text): the frozen "
                      "maweibo_240h_data .pt rows match cleaned text at cosine 1.0 (d6_diagnostic.json). "
                      "The CURRENT server pipeline.py lost this step (feeds raw adapter text to "
                      "senti/embedding); D6b restores it for parity. No lowercasing; URL/HTML/whitespace strip",
        "source_file": SRV + "/text_cleaning.py + " + SRV + "/ma_weibo/pipeline.py",
        "source_line_or_function": "clean_text_weibo (text_cleaning.py L6-14); current pipeline.py steps 3-4 omit the call - version drift vs the code that produced the frozen .pt",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes (with the cleaning step restored)",
    },
    {
        "feature_name": "f01-f10_user_block_10D",
        "dimension": 10,
        "definition": "reposts/comments/attitudes + followers/statuses/favourites + "
                      "bi_followers_count_inv (=1 - MinMax(bi_followers_count)) + verified_type "
                      "mapped {-1:0, 0:0.5, >=1:1} + user_geo_enabled + user_created_at_norm; "
                      "continuous columns log1p + MinMaxScaler fitted on the FULL preprocessed frame",
        "source_file": SRV + "/adapters/ma_weibo.py + " + SRV + "/ma_weibo/user_features.py + " + SRV + "/ma_weibo/graph_builder.py",
        "source_line_or_function": "adapter L78-88; user_features preprocess_userFeature (scaler full-frame fit); graph_builder L46-53 (bi_followers_count_inv)",
        "depends_on_future_graph": "no (dataset-scope normalisation only)",
        "recomputable_from_snapshot": "yes with FROZEN scaler bounds; NOTE: TC-DSCR main path drops the 10D user block",
    },
    {
        "feature_name": "f11_senti_feature",
        "dimension": 1,
        "definition": "Erlangshen-Roberta-110M-Sentiment (chinese) softmax p1-p0 on adapter text, "
                      "0 for empty text; stored normalised (raw+1)/2 by graph_builder",
        "source_file": SRV + "/sentiment.py + " + SRV + "/ma_weibo/graph_builder.py",
        "source_line_or_function": "SentimentModel.predict; graph_builder L41-44",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes; subject to the same inference-precision noise band as PHEME (fp16 batching)",
    },
    {
        "feature_name": "adjacency_signature_1021D",
        "dimension": 1021,
        "definition": "identical rule to PHEME: nodes sorted, adj[parent,child]=1 + self-loop, "
                      "row normalised by own out-degree, columns padded/truncated to 1021",
        "source_file": SRV + "/preprocess_graph.py + " + SRV + "/ma_weibo/graph_builder.py",
        "source_line_or_function": "preprocess_graph.py L57-76; edges graph_builder.py L56-67 (child->parent via parent mid)",
        "depends_on_future_graph": "YES (parent rows normalise over currently-visible children)",
        "recomputable_from_snapshot": "yes, causally with snapshot-internal edges (D8 verified)",
    },
    {
        "feature_name": "norm_degree",
        "dimension": 1,
        "definition": "historical: (out_degree-1) clamped, divided by event-wide max raw_degree "
                      "(same as PHEME historical). TC-DSCR frozen rule: per-snapshot internal max",
        "source_file": SRV + "/preprocess_graph.py",
        "source_line_or_function": "preprocess_graph.py L78-79",
        "depends_on_future_graph": "YES (historical denominator spans all nodes)",
        "recomputable_from_snapshot": "yes under the frozen V2 snapshot-internal max rule (approved)",
    },
    {
        "feature_name": "norm_depth",
        "dimension": 1,
        "definition": "BFS depth from sorted-first node (source) clamp 19, divided by FIXED 19 "
                      "when time_steps provided (same as PHEME)",
        "source_file": SRV + "/ma_weibo/graph_builder.py + " + SRV + "/preprocess_graph.py",
        "source_line_or_function": "graph_builder.py L69-85; preprocess_graph.py L106-108",
        "depends_on_future_graph": "no (fixed denominator)",
        "recomputable_from_snapshot": "yes",
    },
    {
        "feature_name": "norm_time",
        "dimension": 1,
        "definition": "time_bin = floor((t - event_min_t)/1800) clipped [0,479] (30-minute bins, "
                      "ma_weibo_240h config time_window_minutes=30), divided by FIXED 480",
        "source_file": SRV + "/time_windows.py + " + SRV + "/ma_weibo/graph_builder.py + /data/jyz/next/configs/ma_weibo_240h.yaml",
        "source_line_or_function": "time_windows.py L7-32; graph_builder.py L26-31, L88; config L18-19",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes; same 480-bin definition as PHEME (no approval needed for unification)",
    },
    {
        "feature_name": "node_ordering",
        "dimension": -1,
        "definition": "240h/30min filtered posts sorted by absolute timestamp ascending "
                      "(graph_builder sort_values('timestamp')); root = first sorted node",
        "source_file": SRV + "/ma_weibo/graph_builder.py",
        "source_line_or_function": "graph_builder.py L26-31 (assign_time_bins + sort)",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes",
    },
    {
        "feature_name": "time_horizon_cutoff_240h",
        "dimension": -1,
        "definition": "posts with t outside [event_min_t, event_min_t + 240h] filtered "
                      "(elapsed basis = event-wide min t, not source t)",
        "source_file": SRV + "/time_windows.py",
        "source_line_or_function": "assign_time_bins L24-25",
        "depends_on_future_graph": "n/a",
        "recomputable_from_snapshot": "yes",
    },
    {
        "feature_name": "max_nodes_truncation",
        "dimension": -1,
        "definition": "first 1021 nodes in chronological order (graph_max_nodes=1021)",
        "source_file": SRV + "/preprocess_graph.py",
        "source_line_or_function": "preprocess_graph.py L38-44",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes",
    },
    {
        "feature_name": "source_definition",
        "dimension": -1,
        "definition": "first post with parent==null in the raw JSON list (adapter); "
                      "graph builder uses the sorted-first node as BFS root",
        "source_file": SRV + "/adapters/ma_weibo.py + " + SRV + "/ma_weibo/graph_builder.py",
        "source_line_or_function": "adapter L41-46; graph_builder L69-71",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes",
    },
]
audit = {
    "authoritative_pipeline": SRV + " (adapters/ma_weibo.py, ma_weibo/{pipeline,graph_builder,user_features}.py, preprocess_graph.py, time_windows.py)",
    "config_reference": "/data/jyz/next/configs/ma_weibo_240h.yaml (seed 3090, target_hours 240, "
                        "time_window_minutes 30, graph_max_nodes 1021, MiniLM + Erlangshen-Roberta-110M-Sentiment chinese)",
    "resolved_runtime_parameters": {
        "time_window_minutes": 30, "max_time_steps": 480, "graph_max_nodes": 1021,
        "node_feat_layout": "[text 384 | user 10 | senti 1] = 395 cols (no userFeatures aggregate column, unlike PHEME 396)",
        "evidence": "maweibo .pt probe: node_feat dim 395; keys max_time_steps/time_bin/num_nodes/struct_feat/node_feat; no node_ids key",
    },
    "fields": rows,
    "tc_dscr_main_path_note": "TC-DSCR uses only text384 + 1021D adjacency + 3D summary; "
                              "the 10D user block and senti are excluded from the main path per frozen V2 decision 2.2",
}
(out / "maweibo_feature_pipeline_audit.json").write_text(
    json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")

# ---------------- D0: environment delta ----------------
env = {
    "generated_at": "2026-09-06",
    "git_commit_sha": sha,
    "python_version": "3.11.13",
    "pytorch_version": "2.7.1+cu128",
    "transformers_version": "4.57.6",
    "sentence_transformers_version": "server DGPA install (version recorded in p0 first-round environment_server.json; same interpreter)",
    "cuda_version": "12.8",
    "gpu_model": "NVIDIA GeForce RTX 4090 (24564 MiB, driver 570.133.07)",
    "os_server": "Linux (DGPA environment)",
    "paths": {
        "pheme_raw": "/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads",
        "pheme_processed": "/data/jyz/next/llm/data/pheme_240h_data",
        "maweibo_raw": "/data/jyz/next/llm/data/maweibo_raw",
        "maweibo_processed": "/data/jyz/next/llm/data/maweibo_240h_data",
        "maweibo_labels": "/data/jyz/next/llm/data/maweibo_labels.txt",
        "maweibo_umer_checkpoints": "/data/jyz/next/llm/checkpoints/maweibo",
        "historical_preprocessing_source_root": "/data/jyz/next/src/rumor_detection/data",
    },
    "round1_preflight_intact": True,
    "round1_summary_path": "results/tcdscr/preflight/preflight_summary.md",
    "frozen_decisions_honoured": {
        "weibo22_removed": True,
        "tc_dscr_main_path_384_1024_only": True,
        "norm_degree_snapshot_internal_max": True,
    },
}
(out / "environment_delta.json").write_text(
    json.dumps(env, indent=1, ensure_ascii=False), encoding="utf-8")
print("D5 + D0 JSON written; git", sha[:8])
