#!/usr/bin/env python
"""TC-DSCR preflight P7+P8: 384D Chinese-semantic compatibility and Weibo22
11D user-feature compatibility. Static audit (no model calls beyond version
probe of the stored encoder directory). Writes text_encoder_11d_audit.json.
"""
import json
from pathlib import Path

OUT = Path("/data/jyz/next/llm/results/tcdscr/preflight")

# ---------- P7: 384D encoder ----------
minilm = Path("/data/jyz/rumor_detection/model/paraphrase-multilingual-MiniLM-L12-v2")
p7 = {
    "encoder_used_by_historical_preprocessing": "paraphrase-multilingual-MiniLM-L12-v2",
    "same_model_for_pheme_and_maweibo": True,
    "evidence": [
        "server configs: /data/jyz/next/configs/pheme_extension_with_mask.yaml line 68 and "
        "ma_weibo_240h config use the same text_embedding_model path",
        "local mirror configs (ablation) identical: model/paraphrase-multilingual-MiniLM-L12-v2",
    ],
    "multilingual": True,
    "weights_present": minilm.exists(),
    "weight_files": sorted(p.name for p in minilm.glob("*")) if minilm.exists() else [],
    "tokenizer_freezable": True,
    "reproducibility_evidence": (
        "P4 parity: rebuilt 384D embeddings vs frozen 240h .pt give "
        "max_abs_diff 9.5e-7 over 100 events with the current DGPA environment"
    ),
    "weibo22_directly_reusable": False,
    "weibo22_blocker": "Weibo22 release contains no raw text (vol_5000 word indices only), "
                       "so no text exists to encode; the encoder itself is fine",
    "verdict": "ENCODER_OK; no NEED_TEXT_ENCODER_DECISION for the encoder itself",
}

# ---------- P8: 11D compatibility ----------
# 11D extra block (server .pt layout: node_feat[:,384:396], model consumes [:,384:395]
# = 10 user dims + senti; the trailing userFeatures aggregate column is unused by UMER)
dims = [
    ("f01_reposts_or_retweets_count", "retweet_count (tweet object)"),
    ("f02_comments_count", "no PHEME source field; hardcoded 0.0 by adapter"),
    ("f03_attitudes_or_favorites_count", "favorite_count (tweet object)"),
    ("f04_followers_count", "user.followers_count"),
    ("f05_statuses_count", "user.statuses_count"),
    ("f06_favourites_count", "user.favourites_count"),
    ("f07_bi_followers_count", "no PHEME source field; hardcoded 0.0 by adapter"),
    ("f08_user_verified", "user.verified (bool -> 0/1)"),
    ("f09_user_geo_enabled", "user.geo_enabled (bool -> 0/1)"),
    ("f10_user_created_at_norm", "user.created_at parsed to unix, then 1-(x-min)/(max-min) over the full preprocessed frame"),
    ("f11_senti_feature", "sentiment-roberta-large-english logits p_pos-p_neg on cleaned text, 0 for empty text"),
]
rows = []
for name, definition in dims:
    rows.append({
        "feature": name,
        "definition": definition,
        "pheme_availability": "available" if "hardcoded" not in definition else "constant_0",
        "weibo22_availability": "absent",
        "missing_rate_weibo22": 1.0,
        "can_compute_exactly_weibo22": False,
        "note": "Weibo22 ships neither tweet/user JSON fields nor raw text; per protocol, "
                "filling 0, substituting proxies, or redefining the 11D schema is forbidden.",
    })
p8 = {
    "schema_source": "server /data/jyz/next/src/rumor_detection/data/pheme_extension/graph_builder.py user_cols + senti_norm; "
                     "preprocess_graph.py concatenates text384 + user10 + senti + userFeatures -> node_feat 396 cols",
    "umer_consumed_block": "node_feat[:,384:395] (10 user + senti); trailing userFeatures column unused by the model",
    "weibo22_rows": rows,
    "verdict": "NEED_FEATURE_SCHEMA_DECISION",
}
(OUT / "text_encoder_11d_audit.json").write_text(
    json.dumps({"p7_384d_encoder": p7, "p8_11d_weibo22": p8}, indent=1, ensure_ascii=False),
    encoding="utf-8")
print("text_encoder_11d_audit.json written")
print("P7 verdict:", p7["verdict"])
print("P8 verdict:", p8["verdict"])
