import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_node_cognitive_targets import make_node_target  # noqa: E402


def test_builder_maps_visible_indices_to_original_graph_positions():
    record = {
        "event_id": "x",
        "event_key": "key",
        "output_hash": "hash",
        "selected_original_positions": [0, 8, 19],
        "validation": {"errors": []},
        "parsed": {
            "conversation_analysis": [
                {"node_index": 0, "stance": "source", "role": "claim", "salience": 0.9},
                {"node_index": 1, "stance": "query", "role": "comment", "salience": 0.5},
                {"node_index": 2, "stance": "deny", "role": "correction", "salience": 0.7},
            ]
        },
    }
    target, errors = make_node_target(record)
    assert errors == []
    assert [node["original_position"] for node in target["nodes"]] == [0, 8, 19]
    assert [node["stance"] for node in target["nodes"]] == [0, 3, 2]


def test_builder_masks_only_invalid_ontology_group():
    record = {
        "event_id": "x",
        "event_key": "key",
        "output_hash": "hash",
        "selected_original_positions": [0],
        "validation": {"errors": ["invalid_role"]},
        "parsed": {
            "conversation_analysis": [
                {"node_index": 0, "stance": "source", "role": "unknown", "salience": 0.9}
            ]
        },
    }
    target, errors = make_node_target(record)
    assert errors == []
    assert target["nodes"][0]["stance_mask"]
    assert not target["nodes"][0]["role_mask"]
    assert target["nodes"][0]["salience_mask"]

