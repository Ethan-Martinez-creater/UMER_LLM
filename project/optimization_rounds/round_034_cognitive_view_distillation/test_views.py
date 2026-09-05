import json

import pytest

from optimization_rounds.round_034_cognitive_view_distillation.views import (
    load_aligned_cognitive_views,
)


def test_views_align_and_report_missing(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps({"schema_id": "umer_cognitive_training_view_v1"}),
        encoding="utf-8",
    )
    views = tmp_path / "views.jsonl"
    views.write_text(
        json.dumps({"event_id": "b", "view": " teacher  view "}) + "\n",
        encoding="utf-8",
    )
    aligned, report = load_aligned_cognitive_views(["a", "b"], views, schema)
    assert aligned == [None, "teacher view"]
    assert report["coverage"] == pytest.approx(0.5)
