from __future__ import annotations

import json
from pathlib import Path


EXPECTED_SCHEMA_ID = "umer_cognitive_training_view_v1"


def load_aligned_cognitive_views(event_ids, view_path, schema_path):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if schema.get("schema_id") != EXPECTED_SCHEMA_ID:
        raise ValueError("unexpected cognitive training-view schema")
    event_ids = [str(event_id) for event_id in event_ids]
    valid_ids = set(event_ids)
    mapping = {}
    for number, line in enumerate(
        Path(view_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = str(row["event_id"])
        if event_id in mapping:
            raise ValueError(f"duplicate cognitive view at line {number}")
        if event_id not in valid_ids:
            raise ValueError(f"unknown cognitive-view event at line {number}")
        view = " ".join(str(row["view"]).split())
        if not view:
            raise ValueError(f"empty cognitive view at line {number}")
        mapping[event_id] = view
    aligned = [mapping.get(event_id) for event_id in event_ids]
    report = {
        "schema_id": EXPECTED_SCHEMA_ID,
        "dataset_event_count": len(event_ids),
        "available_view_count": len(mapping),
        "coverage": len(mapping) / max(len(event_ids), 1),
    }
    return aligned, report
