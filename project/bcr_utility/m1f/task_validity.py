"""M1-F Task A — task-validity audit.

Documents, from primary sources only (never from a summary written by another
stage), what each dataset's label means, what the frozen reader prompt actually
asks, and therefore what ``utility`` can and cannot be claimed to represent.

The audit is re-derived by reading the frozen sources:

* ``project/cr_tser/config/pilot_config.py`` — ``LABEL_TOKEN`` (A/B naming);
* ``project/cr_tser/readers/base_reader.py`` — the single reader prompt;
* ``project/cr_tser/readers/sequence_scorer.py`` — the scoring contract;
* ``project/bcr_utility/config/protocol.py`` — the inherited utility
  definition (threshold ±0.05 over ``p_r(gold|SRC) - p_r(gold|SRC without e)``).

It relabels nothing and rewrites no historical artifact. The key boundary it
must preserve: on PHEME ``rumour``/``non-rumour`` is **not** ``false``/``true``.
"""
from __future__ import annotations

import json
import os

from ..config import protocol as P

LABEL_TOKEN_SOURCE = "project/cr_tser/config/pilot_config.py"
PROMPT_SOURCE = "project/cr_tser/readers/base_reader.py"
SCORER_SOURCE = "project/cr_tser/readers/sequence_scorer.py"

_EXPECTED_LABEL_TOKEN_LINE = 'LABEL_TOKEN = {"A": "RUMOR", "B": "NON_RUMOR"}'
_EXPECTED_PROMPT_MARKERS = ("A = RUMOR", "B = NON_RUMOR",
                            "Answer with a single letter: A or B.")
_EXPECTED_SCORER_MARKERS = ("gold 1=RUMOR(A) / 0=NON_RUMOR(B)",)


class ValidityRefused(RuntimeError):
    """Raised when the task-semantics sources are not the frozen ones."""


def _read_text(repo_root, rel_path: str) -> str:
    full = os.path.join(str(repo_root), rel_path.replace("/", os.sep))
    if not os.path.exists(full):
        raise ValidityRefused(f"missing task-semantics source: {rel_path}")
    with open(full, encoding="utf-8") as fh:
        return fh.read()


def verify_sources(repo_root) -> dict:
    """Re-derive the prompt/label/scoring contract from the frozen sources."""
    label_text = _read_text(repo_root, LABEL_TOKEN_SOURCE)
    prompt_text = _read_text(repo_root, PROMPT_SOURCE)
    scorer_text = _read_text(repo_root, SCORER_SOURCE)
    problems = []
    if _EXPECTED_LABEL_TOKEN_LINE not in label_text:
        problems.append(f"{LABEL_TOKEN_SOURCE}: LABEL_TOKEN drifted")
    for marker in _EXPECTED_PROMPT_MARKERS:
        if marker not in prompt_text:
            problems.append(f"{PROMPT_SOURCE}: missing {marker!r}")
    for marker in _EXPECTED_SCORER_MARKERS:
        if marker not in scorer_text:
            problems.append(f"{SCORER_SOURCE}: missing {marker!r}")
    if problems:
        raise ValidityRefused("; ".join(problems[:6]))
    return {
        "label_token": {"A": "RUMOR", "B": "NON_RUMOR"},
        "prompt_markers": list(_EXPECTED_PROMPT_MARKERS),
        "scoring_contract": "teacher-forced candidate log probability "
                            "p_r(y|C), y in {A=RUMOR, B=NON_RUMOR}",
        "sources": [LABEL_TOKEN_SOURCE, PROMPT_SOURCE, SCORER_SOURCE],
    }


def _dataset_semantics() -> dict:
    return {
        "maweibo": {
            "language": "Chinese",
            "platform": "Sina Weibo",
            "task": "event-level rumour vs non-rumour detection",
            "label_semantics":
                "the event's rumour / non-rumour annotation carried by the "
                "frozen Ma-Weibo source (Ma et al., IJCAI 2016)",
            "is_truth_verification": False,
            "boundary":
                "the label is a rumour annotation, not a verified "
                "true/false verdict on the underlying claim",
        },
        "pheme": {
            "language": "English",
            "platform": "Twitter",
            "task": "event-level rumour vs non-rumour detection",
            "label_semantics":
                "the event's rumour / non-rumour annotation carried by the "
                "frozen PHEME source (Zubiaga et al., PLOS ONE 2016)",
            "is_truth_verification": False,
            "boundary":
                "rumour != false and non-rumour != true: PHEME annotates "
                "whether a claim circulated as an unverified rumour, not "
                "whether it is factually true or false",
        },
    }


def task_validity(repo_root) -> dict:
    """The Task A payload: semantics, prompt target, and the utility meaning."""
    sources = verify_sources(repo_root)
    return {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f",
        "task": "task_validity_audit",
        "relabelled_historical_data": False,
        "changed_threshold_or_split": False,
        "sources": sources["sources"],
        "reader_prompt": {
            "target": "A = RUMOR vs B = NON_RUMOR (source post plus the "
                      "observed social evidence visible up to the cutoff)",
            "scoring": sources["scoring_contract"],
            "single_prompt_builder": PROMPT_SOURCE,
        },
        "utility_definition": {
            "formula": "u_r(e) = p_r(gold | SRC) - p_r(gold | SRC without e)",
            "gold": "the dataset's frozen rumour/non-rumour label of the "
                    "event the evidence belongs to",
            "threshold": P.UTILITY_THRESHOLD,
            "sign_classes": list(P.SIGN_CLASSES),
            "supervision": "I1_atomic only; I2-I5 are non-supervision",
        },
        "utility_meaning": {
            "positive": "the evidence unit makes this reader more likely to "
                        "answer with the dataset's gold rumour/non-rumour "
                        "label for that event",
            "negative": "the evidence unit makes this reader less likely to "
                        "answer with that gold label",
            "is_verification_utility": False,
            "claim_allowed":
                "reader-specific rumour-label evidence utility",
            "claim_forbidden":
                "fact-checking / truth-verification utility, or any claim "
                "that the evidence was shown to be true or false",
        },
        "datasets": _dataset_semantics(),
        "cross_dataset_boundary": {
            "is_language_only_shift": False,
            "note": "Ma-Weibo and PHEME differ in platform, language, "
                    "annotation process and claim distribution; M1-E measured "
                    "a systematic E3 distribution shift, so the M1-F contrast "
                    "must not be reported as a language-only shift",
        },
        "conclusion": {
            "supports_intended_claim": True,
            "reason":
                "utility is defined over the reader's rumour/non-rumour "
                "answer agreement, so an evidence-level claim is supported "
                "provided it is stated as rumour-label utility and never as "
                "truth verification",
            "validity_blocked": False,
        },
    }


def task_validity_markdown(payload: dict) -> str:
    """Human-readable Task A report (mirrors the JSON one-to-one)."""
    lines = [
        "# BCR-Utility M1-F — Task A: Task-Validity Audit",
        "",
        "```text",
        f"protocol                    = {payload['protocol']}",
        f"relabelled_historical_data  = {payload['relabelled_historical_data']}",
        f"changed_threshold_or_split  = {payload['changed_threshold_or_split']}",
        "```",
        "",
        "## 1. What the frozen reader actually predicts",
        "",
        f"* prompt builder: `{payload['reader_prompt']['single_prompt_builder']}`",
        f"* target: {payload['reader_prompt']['target']}",
        f"* scoring: {payload['reader_prompt']['scoring']}",
        "",
        "The prompt is fixed text: the source post, the observed social "
        "evidence up to the cutoff, and the instruction to answer with a "
        "single letter where `A = RUMOR` and `B = NON_RUMOR`. The reader is "
        "never asked whether the claim is factually true.",
        "",
        "## 2. What `utility` therefore means",
        "",
        "```text",
        f"{payload['utility_definition']['formula']}",
        f"gold      = {payload['utility_definition']['gold']}",
        f"threshold = +-{payload['utility_definition']['threshold']}",
        f"supervision = {payload['utility_definition']['supervision']}",
        "```",
        "",
        f"* positive utility: {payload['utility_meaning']['positive']}",
        f"* negative utility: {payload['utility_meaning']['negative']}",
        f"* is verification utility: **{payload['utility_meaning']['is_verification_utility']}**",
        "",
        f"**Claim allowed:** {payload['utility_meaning']['claim_allowed']}.",
        "",
        f"**Claim forbidden:** {payload['utility_meaning']['claim_forbidden']}.",
        "",
        "## 3. Dataset label semantics",
        "",
    ]
    for dataset, spec in payload["datasets"].items():
        lines += [
            f"### {dataset}",
            "",
            "```text",
            f"platform            = {spec['platform']}",
            f"language            = {spec['language']}",
            f"task                = {spec['task']}",
            f"label_semantics     = {spec['label_semantics']}",
            f"is_truth_verification = {spec['is_truth_verification']}",
            "```",
            "",
            f"{spec['boundary']}.",
            "",
        ]
    lines += [
        "## 4. Cross-dataset boundary",
        "",
        f"* language-only shift: **{payload['cross_dataset_boundary']['is_language_only_shift']}**",
        f"* {payload['cross_dataset_boundary']['note']}",
        "",
        "## 5. Conclusion",
        "",
        "```text",
        f"supports_intended_claim = {payload['conclusion']['supports_intended_claim']}",
        f"validity_blocked        = {payload['conclusion']['validity_blocked']}",
        "```",
        "",
        payload["conclusion"]["reason"] + ".",
        "",
        "Sources re-read for this audit: "
        + ", ".join(f"`{s}`" for s in payload.get("sources", [])),
        "",
    ]
    return "\n".join(lines)


def dumps(payload: dict) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False)
