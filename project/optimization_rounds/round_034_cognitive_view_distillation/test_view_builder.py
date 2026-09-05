import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_cognitive_training_views import build_view  # noqa: E402


def test_cognitive_view_is_deterministic_and_compact():
    parsed = {
        "claim_summary": "A concise claim",
        "atomic_claims": [{"text": "Claim A", "checkability": "high"}],
        "conversation_analysis": [
            {"node_index": 0, "stance": "source", "role": "claim", "salience": 0.9},
            {"node_index": 1, "stance": "deny", "role": "correction", "salience": 0.5},
        ],
        "event_signals": {
            "commonsense_conflict": 0.1,
            "logical_inconsistency": 0.2,
            "emotional_manipulation": 0.7,
            "uncertainty": 0.4,
        },
    }
    first = build_view(parsed)
    second = build_view(parsed)
    assert first == second
    assert "deny=1" in first
    assert "correction=1" in first
    assert "emotional_manipulation=high(0.70)" in first
    assert len(first) < 1200
