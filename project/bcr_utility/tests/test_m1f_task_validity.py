"""M1-F Task A tests: the task-validity audit reads the frozen sources."""
from __future__ import annotations

from pathlib import Path

import pytest

from ..m1f import task_validity as tv

REPO_DIR = Path(__file__).resolve().parents[3]


def test_task_validity_records_the_frozen_boundaries():
    payload = tv.task_validity(str(REPO_DIR))
    assert payload["relabelled_historical_data"] is False
    assert payload["changed_threshold_or_split"] is False
    assert payload["utility_meaning"]["is_verification_utility"] is False
    assert payload["utility_meaning"]["claim_allowed"] == \
        "reader-specific rumour-label evidence utility"
    pheme = payload["datasets"]["pheme"]
    assert pheme["is_truth_verification"] is False
    assert "rumour != false" in pheme["boundary"]
    assert "non-rumour != true" in pheme["boundary"]
    assert payload["cross_dataset_boundary"]["is_language_only_shift"] is False
    assert payload["conclusion"]["validity_blocked"] is False
    assert payload["conclusion"]["supports_intended_claim"] is True


def test_task_validity_utility_formula_is_the_inherited_one():
    payload = tv.task_validity(str(REPO_DIR))
    definition = payload["utility_definition"]
    assert definition["formula"] == \
        "u_r(e) = p_r(gold | SRC) - p_r(gold | SRC without e)"
    assert definition["threshold"] == 0.05
    assert definition["supervision"].startswith("I1_atomic only")


def test_task_validity_reports_the_reader_prompt_target():
    payload = tv.task_validity(str(REPO_DIR))
    assert "A = RUMOR" in payload["reader_prompt"]["target"]
    assert "B = NON_RUMOR" in payload["reader_prompt"]["target"]


def test_task_validity_markdown_covers_the_report_fields():
    md = tv.task_validity_markdown(tv.task_validity(str(REPO_DIR)))
    assert "Task A: Task-Validity Audit" in md
    assert "is verification utility" in md
    assert "Claim forbidden" in md
    assert "rumour != false" in md


def test_verify_sources_fails_closed_without_the_frozen_sources(tmp_path):
    with pytest.raises(tv.ValidityRefused):
        tv.verify_sources(str(tmp_path))


def test_verify_sources_detects_prompt_drift(tmp_path):
    root = Path(tmp_path)
    for rel in (tv.LABEL_TOKEN_SOURCE, tv.PROMPT_SOURCE, tv.SCORER_SOURCE):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tv._read_text(str(REPO_DIR), rel), encoding="utf-8")
    drifted = root / tv.PROMPT_SOURCE
    drifted.write_text(drifted.read_text(encoding="utf-8").replace(
        "A = RUMOR", "A = TRUE"), encoding="utf-8")
    with pytest.raises(tv.ValidityRefused):
        tv.verify_sources(str(root))
