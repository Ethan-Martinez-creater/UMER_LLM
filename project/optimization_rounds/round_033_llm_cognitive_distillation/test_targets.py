import json

import pytest
import torch

from optimization_rounds.round_033_llm_cognitive_distillation.targets import (
    active_event_coverage,
    load_aligned_cognitive_targets,
)


def test_targets_align_by_event_id_and_mask_missing(tmp_path):
    schema = {
        "schema_id": "umer_cognitive_event_target_v1",
        "target_dimension": 2,
        "target_names": ["a", "b"],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    target_path = tmp_path / "targets.jsonl"
    target_path.write_text(
        json.dumps({"event_id": "e2", "target": [0.2, 0.8], "target_mask": [1, 1]})
        + "\n",
        encoding="utf-8",
    )
    targets, masks, report = load_aligned_cognitive_targets(
        ["e1", "e2"], target_path, schema_path
    )
    assert torch.equal(targets[0], torch.zeros(2))
    assert torch.allclose(targets[1], torch.tensor([0.2, 0.8]))
    assert torch.equal(masks[0], torch.zeros(2))
    assert active_event_coverage(masks, [0, 1]) == pytest.approx(0.5)
    assert report["events_with_any_active_target"] == 1


def test_targets_reject_unknown_event(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({
            "schema_id": "umer_cognitive_event_target_v1",
            "target_dimension": 1,
            "target_names": ["a"],
        }),
        encoding="utf-8",
    )
    target_path = tmp_path / "targets.jsonl"
    target_path.write_text(
        json.dumps({"event_id": "other", "target": [0.5], "target_mask": [1]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown event IDs"):
        load_aligned_cognitive_targets(["e1"], target_path, schema_path)
