"""Unit tests: exact LLM parser (plan §28 — test_exact_parser,
test_invalid_parser)."""
from ..llm.parser import INVALID_OUTPUT, NON_RUMOR, RUMOR, label_to_int, parse_label


def test_exact_parser():
    assert parse_label("RUMOR") == RUMOR
    assert parse_label("NON_RUMOR") == NON_RUMOR
    assert parse_label("  RUMOR\n") == RUMOR      # surrounding whitespace only
    assert parse_label("\tNON_RUMOR ") == NON_RUMOR


def test_invalid_parser():
    # no fuzzy matching is allowed (§23): anything but the exact strings,
    # after stripping, is INVALID_OUTPUT
    assert parse_label("rumor") == INVALID_OUTPUT
    assert parse_label("The answer is RUMOR") == INVALID_OUTPUT
    assert parse_label("RUMOR.") == INVALID_OUTPUT
    assert parse_label("**RUMOR**") == INVALID_OUTPUT
    assert parse_label("") == INVALID_OUTPUT
    assert parse_label("RUMOR OR NON_RUMOR") == INVALID_OUTPUT
    assert parse_label(None) == INVALID_OUTPUT
    assert label_to_int(RUMOR) == 1
    assert label_to_int(NON_RUMOR) == 0
    assert label_to_int(INVALID_OUTPUT) is None
