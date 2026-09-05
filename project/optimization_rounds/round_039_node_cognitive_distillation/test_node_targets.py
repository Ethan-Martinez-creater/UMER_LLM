import json

import torch

from optimization_rounds.round_039_node_cognitive_distillation.node_targets import (
    ROLE_NAMES,
    STANCE_NAMES,
    active_event_coverage,
    load_aligned_node_targets,
    pack_node_targets,
)


def test_load_pack_and_coverage_preserve_original_positions(tmp_path):
    schema = {
        "schema_id": "umer_node_cognitive_target_v1",
        "target_dimension": 14,
        "stance_names": STANCE_NAMES,
        "role_names": ROLE_NAMES,
    }
    row = {
        "event_id": "event-a",
        "nodes": [
            {
                "original_position": 0,
                "stance": 0,
                "stance_mask": True,
                "role": 0,
                "role_mask": True,
                "salience": 0.9,
                "salience_mask": True,
            },
            {
                "original_position": 7,
                "stance": 3,
                "stance_mask": True,
                "role": 5,
                "role_mask": False,
                "salience": 0.4,
                "salience_mask": True,
            },
        ],
    }
    target_path = tmp_path / "targets.jsonl"
    schema_path = tmp_path / "schema.json"
    target_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    aligned, report = load_aligned_node_targets(
        ["event-a", "event-b"], target_path, schema_path
    )
    assert report["events_with_nodes"] == 1
    assert active_event_coverage(aligned, [0, 1]) == 0.5
    packed = pack_node_targets(aligned, [0, 1], 8, torch.device("cpu"))
    assert packed["stance"][0, 7].item() == 3
    assert packed["stance_mask"][0, 7]
    assert not packed["role_mask"][0, 7]
    assert torch.isclose(packed["salience"][0, 7], torch.tensor(0.4))
    assert not packed["salience_mask"][1].any()

