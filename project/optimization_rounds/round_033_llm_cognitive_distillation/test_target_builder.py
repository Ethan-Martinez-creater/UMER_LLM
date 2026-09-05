import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_cognitive_targets import make_target  # noqa: E402


def example():
    return {
        "conversation_analysis": [
            {"node_index": 0, "stance": "source", "role": "claim", "salience": 0.9},
            {"node_index": 1, "stance": "support", "role": "evidence", "salience": 0.5},
        ],
        "event_signals": {
            "commonsense_conflict": 0.1,
            "logical_inconsistency": 0.2,
            "emotional_manipulation": 0.3,
            "uncertainty": 0.4,
        },
    }


def test_source_misuse_masks_only_stance_group():
    parsed = example()
    parsed["conversation_analysis"][1]["stance"] = "source"
    _, mask, reasons = make_target(parsed)
    assert mask[:4] == [1.0] * 4
    assert mask[4:9] == [0.0] * 5
    assert mask[9:] == [1.0] * 9
    assert "reply_uses_source" in reasons


def test_invalid_role_flag_masks_only_role_group():
    _, mask, reasons = make_target(example(), mask_role=True)
    assert mask[:9] == [1.0] * 9
    assert mask[9:16] == [0.0] * 7
    assert mask[16:] == [1.0] * 2
    assert reasons == ["invalid_role_group"]
