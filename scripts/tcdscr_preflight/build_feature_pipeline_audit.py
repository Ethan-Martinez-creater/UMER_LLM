import json
from pathlib import Path

SRV = "/data/jyz/next/src/rumor_detection/data"
rows = [
    {
        "feature_name": "text_semantic_embedding",
        "dimension": 384,
        "definition": "SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2 on clean_tweet_pheme(raw text); L2-normalised by the encoder",
        "source_file": SRV + "/embeddings.py + " + SRV + "/text_cleaning.py + " + SRV + "/pheme_extension/pipeline.py",
        "source_line_or_function": "TextEmbedder.encode (embeddings.py L10-24); clean_tweet_pheme (text_cleaning.py L17-40); pipeline steps 2/4",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes (per-post input; P4 parity max_abs_diff 9.5e-7)",
    },
]
user_cols = [
    ("f01_reposts_or_retweets_count", "log1p+MinMax of retweet_count (tweet)"),
    ("f02_comments_count", "no PHEME source field; adapter hardcodes 0.0"),
    ("f03_attitudes_or_favorites_count", "log1p+MinMax of favorite_count (tweet)"),
    ("f04_followers_count", "log1p+MinMax of user.followers_count"),
    ("f05_statuses_count", "log1p+MinMax of user.statuses_count"),
    ("f06_favourites_count", "log1p+MinMax of user.favourites_count"),
    ("f07_bi_followers_count", "no PHEME source field; adapter hardcodes 0.0"),
    ("f08_user_verified", "user.verified binarised 0/1"),
    ("f09_user_geo_enabled", "user.geo_enabled binarised 0/1"),
    ("f10_user_created_at_norm", "1 - (created_at - min)/(max - min) over the FULL preprocessed frame"),
]
for name, d in user_cols:
    rows.append({
        "feature_name": name,
        "dimension": 1,
        "definition": d,
        "source_file": SRV + "/adapters/pheme_extension.py + " + SRV + "/pheme_extension/user_features.py",
        "source_line_or_function": "adapter _extract_record L104-136; user_features.py L13-163 (MinMaxScaler fit L86-87 on the full frame; created_at_norm L92-96)",
        "depends_on_future_graph": "no (f10 uses dataset-scope min/max, not same-event future nodes)",
        "recomputable_from_snapshot": "yes with FROZEN scaler bounds from the full preprocessing frame",
    })
rows += [
    {
        "feature_name": "f11_senti_feature",
        "dimension": 1,
        "definition": "sentiment-roberta-large-english softmax p_POS-p_NEG on cleaned text, 0 for empty text; stored as graph-builder-normalised (raw+1)/2 in [0,1]",
        "source_file": SRV + "/sentiment.py + " + SRV + "/pheme_extension/graph_builder.py",
        "source_line_or_function": "SentimentModel.predict (sentiment.py L20-56); senti_norm (graph_builder.py L76-79)",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes; value depends on inference precision/batching (fp16 batch-order noise up to 0.068 in [0,1] scale; fp32 control 0.030 vs reference)",
    },
    {
        "feature_name": "f12_userFeatures_aggregate",
        "dimension": 1,
        "definition": "0.5*influence+0.3*credibility+0.2*activity with per-dimension 95th-percentile robust normalisation over the FULL frame; concatenated as node_feat col 395 but NOT consumed by UMER (model slices [:, :395])",
        "source_file": SRV + "/pheme_extension/user_features.py + " + SRV + "/preprocess_graph.py",
        "source_line_or_function": "user_features.py L100-155; preprocess_graph.py L46-55",
        "depends_on_future_graph": "dataset-scope normalisation only",
        "recomputable_from_snapshot": "yes with frozen frame statistics; irrelevant to UMER forward",
    },
    {
        "feature_name": "adjacency_signature",
        "dimension": 1021,
        "definition": "rows = nodes sorted by elapsed_seconds; adj[parent, child]=1 with self-loop on the diagonal; each row divided by its own out-degree (+1e-8); columns padded/truncated to 1021",
        "source_file": SRV + "/preprocess_graph.py + " + SRV + "/pheme_extension/graph_builder.py",
        "source_line_or_function": "preprocess_graph.py L57-76 (adj, row-norm, pad); edge construction graph_builder.py L111-125 (child->parent)",
        "depends_on_future_graph": "YES - a parent row's normaliser equals its currently-visible children count, so values change as future replies arrive",
        "recomputable_from_snapshot": "yes, causally: rebuild with snapshot-internal edges only (P9 verified no future-edge inclusion)",
    },
    {
        "feature_name": "norm_degree",
        "dimension": 1,
        "definition": "(out_degree - 1) clamped at 0, divided by the event-wide max raw_degree",
        "source_file": SRV + "/preprocess_graph.py",
        "source_line_or_function": "preprocess_graph.py L78-79",
        "depends_on_future_graph": "YES - denominator is the event-wide max over ALL nodes incl. future hubs",
        "recomputable_from_snapshot": "rule-recomputable with snapshot-internal max (causal but value differs from full-event); freezing a dataset-wide denominator requires research approval",
    },
    {
        "feature_name": "norm_depth",
        "dimension": 1,
        "definition": "BFS depth from source (root=0, children from child->parent edges, clamp 19) divided by the FIXED constant 19 when time_steps is provided",
        "source_file": SRV + "/preprocess_graph.py + " + SRV + "/pheme_extension/graph_builder.py",
        "source_line_or_function": "graph_builder.py L130-154 (BFS, clamp 19); preprocess_graph.py L106-108 (depth/19)",
        "depends_on_future_graph": "no (fixed denominator; depth recomputed within the visible graph)",
        "recomputable_from_snapshot": "yes (P4 parity summary3 bit-exact; P9 recomputation verified)",
    },
    {
        "feature_name": "norm_timestep",
        "dimension": 1,
        "definition": "time_bin = floor(elapsed_seconds/1800) clipped to [0,479] (30-minute bins over a 240h horizon), divided by FIXED 480",
        "source_file": SRV + "/time_windows.py + " + SRV + "/preprocess_graph.py",
        "source_line_or_function": "time_windows.py L7-32 (assign_time_bins, 30-min bins); graph_builder.py L64-68; preprocess_graph.py L108-110 (t/480)",
        "depends_on_future_graph": "no (fixed 480-bin denominator; evidence: reference .pt max_time_steps=480)",
        "recomputable_from_snapshot": "yes (bit-exact in P4)",
    },
    {
        "feature_name": "time_horizon_cutoff_240h",
        "dimension": -1,
        "definition": "nodes with elapsed_seconds outside [0, 240h] are FILTERED (not clamped) before graph building; edge set filtered accordingly",
        "source_file": SRV + "/time_windows.py + " + SRV + "/preprocess_graph.py",
        "source_line_or_function": "time_windows.py L24-25 (in_horizon); preprocess_graph.py L42-61 (timestep < time_steps filter)",
        "depends_on_future_graph": "n/a",
        "recomputable_from_snapshot": "yes (deterministic filter)",
    },
    {
        "feature_name": "max_nodes_truncation",
        "dimension": -1,
        "definition": "chronologically sorted nodes truncated to the first 1021 (graph_max_nodes)",
        "source_file": SRV + "/preprocess_graph.py",
        "source_line_or_function": "preprocess_graph.py L38-44 (keep_n = min(total_n, max_nodes))",
        "depends_on_future_graph": "no (truncation by arrival order)",
        "recomputable_from_snapshot": "yes",
    },
    {
        "feature_name": "source_only_events",
        "dimension": -1,
        "definition": "events with a single node: empty edge_index, depth=[0], retained in the dataset",
        "source_file": SRV + "/pheme_extension/graph_builder.py",
        "source_line_or_function": "graph_builder.py L115-125 (empty edge case); L133-137 (root fallback)",
        "depends_on_future_graph": "no",
        "recomputable_from_snapshot": "yes",
    },
]
audit = {
    "pipeline_identity": "authoritative 240h pipeline located at /data/jyz/next/src/rumor_detection/data "
                         "(local mirror Algorithm/modified/ablation: 4/7 core files byte-identical; "
                         "preprocess_graph.py, pheme_extension/graph_builder.py, sentiment.py differ - server version used for parity)",
    "config_reference": "/data/jyz/next/configs/pheme_extension_with_mask.yaml (seed 3090, graph_max_nodes 1021, MiniLM + sentiment-roberta-large-english)",
    "resolved_runtime_parameters": {
        "time_window_minutes": 30, "max_time_steps": 480, "graph_max_nodes": 1021,
        "node_feat_layout": "[text 384 | user 10 | senti 1 | userFeatures 1] = 396 cols; UMER consumes [:, :395]",
        "evidence": "reference .pt carry max_time_steps=480 and node_feat dim 396 (pt_probe.json)",
    },
    "fields": rows,
    "stop_p1_triggered": False,
}
Path("results/tcdscr/preflight/feature_pipeline_audit.json").write_text(
    json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
print("feature_pipeline_audit.json:", len(rows), "field rows")
