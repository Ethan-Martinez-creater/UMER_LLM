"""Amendment A1 — v2r1 autoregressive boundary audit.

The legacy diagnostic asks whether ``tokenize(prompt + candidate)`` equals
``tokenize(prompt) + tokenize(candidate)``. A SentencePiece tokenizer may merge
the token straddling that seam even though autoregressive inference never
re-tokenizes an already-tokenized prompt prefix, so amendment A1 replaces the
v2r1 gate with the tokenizer-native prompt identity while keeping the legacy
check as a visible informational field.

These tests use synthetic tokenizers only — no model, no GPU — and cover the
A1 §7 list: historical V1/V2 semantics unchanged, the amended gate, native/
scorer prompt-id equality, candidate decode validity, special-token and
empty-candidate rejection, the visible Mistral ``text_retokenization_stable``
result, synthetic mismatch failing closed, and the Qwen/InternLM regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


# --------------------------------------------------------------------------
# synthetic tokenizers
# --------------------------------------------------------------------------
class _CharTokenizer:
    """Word/char-level double whose concatenation identity holds exactly.

    Stands in for Qwen3 and InternLM3, whose chat templates end in a text token
    that does not merge with a following ``A``/``B``.
    """

    def __init__(self, suffix: str = "<|im_end|>\n"):
        self.suffix = suffix
        self.all_special_ids = [1, 2]

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in str(text)]}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        rendered = "\n".join(m["content"] for m in messages) + self.suffix
        if tokenize:
            return [ord(c) for c in rendered]
        return rendered

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


class _MergeTokenizer(_CharTokenizer):
    """Mistral shape: the token at the ``[/INST]`` + candidate seam merges.

    ``apply_chat_template(tokenize=True)`` still encodes the rendered text
    exactly as the scorer does, so the amended gate holds; only the legacy
    re-tokenization of ``prompt + candidate`` fails.
    """

    def __init__(self):
        super().__init__(suffix="[/INST]")

    def __call__(self, text, add_special_tokens=True):
        text = str(text)
        ids, i = [], 0
        while i < len(text):
            if text[i] == "]" and i + 1 < len(text) and text[i + 1] in "AB":
                ids.append(900 + ord(text[i + 1]))
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        # a merged seam token decodes to the candidate label it swallowed
        return "".join(chr(int(i) - 900) if int(i) >= 900 else chr(int(i))
                       for i in ids)


class _OffsetTokenizer(_CharTokenizer):
    """A tokenizer whose native ids differ from the scorer's text path."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        rendered = super().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(c) for c in rendered] + [7]
        return rendered


class _RecordingTokenizer(_CharTokenizer):
    def __init__(self):
        super().__init__()
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.calls.append({"tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "kwargs": dict(kwargs)})
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt, **kwargs)


# --------------------------------------------------------------------------
# 1. the amended gate, and the legacy diagnostic staying visible
# --------------------------------------------------------------------------
def test_native_prompt_ids_match_scorer_prompt_ids_for_both_shapes():
    from ..readers.sequence_scorer import (ab_boundary_audit, apply_chat,
                                           native_prompt_ids, tokenize_prompt)
    from ..readers.base_reader import build_messages

    for tok in (_CharTokenizer(), _MergeTokenizer()):
        messages = build_messages("source post body")
        chat = apply_chat(tok, messages)
        assert native_prompt_ids(tok, messages) == tokenize_prompt(tok, chat)
        audit = ab_boundary_audit(tok, "source post body")
        assert audit["native_prompt_matches_scorer"] is True
        assert audit["native_prompt_ids_hash"] == audit["scorer_prompt_ids_hash"]
        assert audit["autoregressive_boundary_ok"] is True


def test_mistral_merge_stays_visible_and_does_not_block_the_amended_gate():
    """`[/INST]` seam merge: legacy false, amended gate true — both recorded."""
    from ..readers.sequence_scorer import ab_boundary_audit

    audit = ab_boundary_audit(_MergeTokenizer(), "source post body")
    assert audit["text_retokenization_stable"] is False
    assert audit["legacy_all_boundaries_ok"] is False
    assert audit["autoregressive_boundary_ok"] is True
    for candidate in ("A", "B"):
        info = audit["candidates"][candidate]
        assert info["text_retokenization_stable"] is False
        # the evidence itself is never hidden: the seam merged, so the joined
        # tail is not the candidate's own ids (here it is even shorter than the
        # prompt, because `]` and the label became one token)
        assert info["joined_len"] > 0
        assert info["joined_tail"] != info["candidate_ids"]


def test_qwen_internlm_regression_legacy_check_still_passes():
    from ..readers.sequence_scorer import ab_boundary_audit

    audit = ab_boundary_audit(_CharTokenizer(), "source post body")
    assert audit["text_retokenization_stable"] is True
    assert audit["autoregressive_boundary_ok"] is True
    assert audit["candidate_ids_valid"] is True
    # when nothing merges, the legacy tail is exactly the candidate ids
    for candidate in ("A", "B"):
        info = audit["candidates"][candidate]
        assert info["joined_tail"] == info["candidate_ids"]
        assert info["joined_len"] == audit["prompt_tokens"] + info["n_tokens"]


# --------------------------------------------------------------------------
# 2. candidate validation (A1 §2.4)
# --------------------------------------------------------------------------
def test_candidate_decode_matches_the_intended_label():
    from ..readers.sequence_scorer import (ab_boundary_audit,
                                           candidate_token_audit)

    audit = ab_boundary_audit(_MergeTokenizer(), "source post body")
    for candidate in ("A", "B"):
        info = audit["candidates"][candidate]
        assert info["candidate_ids"]
        assert info["decoded"] == candidate
        assert info["decode_matches_label"] is True
        assert info["valid"] is True
    assert candidate_token_audit(_CharTokenizer(), "A")["valid"] is True


def test_candidate_normalization_is_explicit_and_tested():
    from ..readers.sequence_scorer import normalize_candidate_text

    assert normalize_candidate_text("A") == "A"
    assert normalize_candidate_text(" a ") == "A"
    assert normalize_candidate_text("a\n") == "A"
    assert normalize_candidate_text("") == ""
    assert normalize_candidate_text("ab") == "AB"


def test_special_token_in_candidate_is_rejected():
    from ..readers.sequence_scorer import candidate_token_audit

    class _SpecialTokenizer(_CharTokenizer):
        def __init__(self):
            super().__init__()
            self.all_special_ids = [ord("A")]

    info = candidate_token_audit(_SpecialTokenizer(), "A")
    assert info["has_unexpected_special_token"] is True
    assert info["special_ids_present"] == [ord("A")]
    assert info["valid"] is False


class _EmptyCandidateTokenizer(_CharTokenizer):
    """Tokenizes the prompt normally but has no id for a bare A/B candidate."""

    def __call__(self, text, add_special_tokens=True):
        if len(str(text)) == 1:
            return {"input_ids": []}
        return {"input_ids": [ord(c) for c in str(text)]}


def test_empty_candidate_is_rejected():
    from ..readers.sequence_scorer import candidate_token_audit

    info = candidate_token_audit(_EmptyCandidateTokenizer(), "A")
    assert info["non_empty"] is False
    assert info["candidate_ids"] == []
    assert info["valid"] is False


def test_candidate_that_does_not_decode_back_is_rejected():
    from ..readers.sequence_scorer import candidate_token_audit

    class _WrongDecodeTokenizer(_CharTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "Z"

    info = candidate_token_audit(_WrongDecodeTokenizer(), "A")
    assert info["decoded"] == "Z"
    assert info["decode_matches_label"] is False
    assert info["valid"] is False


# --------------------------------------------------------------------------
# 3. fail-closed behaviour
# --------------------------------------------------------------------------
def test_native_prompt_mismatch_fails_closed_without_blaming_candidates():
    from ..readers.sequence_scorer import ab_boundary_audit

    audit = ab_boundary_audit(_OffsetTokenizer(), "source post body")
    assert audit["native_prompt_matches_scorer"] is False
    assert audit["native_prompt_ids_hash"] != audit["scorer_prompt_ids_hash"]
    assert audit["autoregressive_boundary_ok"] is False
    assert audit["candidate_ids_valid"] is True


def test_gate_requires_both_prompt_identity_and_valid_candidates():
    from ..readers.sequence_scorer import ab_boundary_audit

    audit = ab_boundary_audit(_EmptyCandidateTokenizer(), "source post body")
    assert audit["native_prompt_matches_scorer"] is True
    assert audit["candidate_ids_valid"] is False
    assert audit["no_empty_continuation"] is False
    assert audit["autoregressive_boundary_ok"] is False


def test_audit_uses_the_frozen_scorer_chat_formatter():
    """The native path and the text path share one template call site."""
    from ..readers.sequence_scorer import (ab_boundary_audit, apply_chat,
                                           tokenize_prompt)
    from ..readers.base_reader import build_messages

    tok = _RecordingTokenizer()
    apply_chat(tok, build_messages("source post body"))
    assert tok.calls[-1]["tokenize"] is False
    assert tok.calls[-1]["add_generation_prompt"] is True
    assert tok.calls[-1]["kwargs"] == {"enable_thinking": False}
    ab_boundary_audit(tok, "source post body")
    assert tok.calls[-1]["tokenize"] is True
    assert tok.calls[-1]["kwargs"] == {"enable_thinking": False}
    # the scorer path is still the rendered-text path
    assert tok.calls[-2]["tokenize"] is False
    assert len(tokenize_prompt(tok, apply_chat(
        tok, build_messages("source post body")))) > 0


def test_thinking_kwarg_fallback_is_shared_by_both_paths():
    """A tokenizer without ``enable_thinking`` still gets identical ids."""
    from ..readers.sequence_scorer import (ab_boundary_audit, apply_chat,
                                           native_prompt_ids, tokenize_prompt)
    from ..readers.base_reader import build_messages

    class _NoThinkingKwarg(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            return super().apply_chat_template(
                messages, tokenize=tokenize,
                add_generation_prompt=add_generation_prompt)

    tok = _NoThinkingKwarg()
    messages = build_messages("source post body")
    assert native_prompt_ids(tok, messages) == \
        tokenize_prompt(tok, apply_chat(tok, messages))
    assert ab_boundary_audit(tok, "source post body")[
        "autoregressive_boundary_ok"] is True


# --------------------------------------------------------------------------
# 4. historical V1/V2 semantics unchanged
# --------------------------------------------------------------------------
def test_legacy_boundary_api_is_unchanged():
    from ..readers.sequence_scorer import (ab_token_report,
                                           continuation_boundary,
                                           tokenize_continuation,
                                           tokenize_prompt)

    tok = _CharTokenizer()
    info = continuation_boundary(tok, "<s>hello", "A")
    assert set(info) >= {"prompt_tokens", "candidate_ids", "joined_tail",
                         "joined_len", "boundary_ok"}
    assert info["boundary_ok"] is True
    assert info["prompt_tokens"] == len(tokenize_prompt(tok, "<s>hello"))
    assert info["candidate_ids"] == tokenize_continuation(tok, "A")
    report = ab_token_report(tok, "source post body")
    assert report["all_boundaries_ok"] is True
    # and it is still the legacy check under the merge tokenizer
    assert ab_token_report(_MergeTokenizer(),
                           "source post body")["all_boundaries_ok"] is False


def test_historical_gate_selector_keeps_v1_v2_on_the_legacy_field():
    import cr_tser_p0_audit as p0

    assert p0.boundary_gate_key("v1") == "boundaries_ok"
    assert p0.boundary_gate_key("v2") == "boundaries_ok"
    assert p0.boundary_gate_key("v2r1") == "autoregressive_boundary_ok"


def test_v2_readiness_still_gates_on_boundaries_ok():
    import cr_tser_p0_audit as p0

    merged = {"qwen": {"identical_predictions": True,
                       "boundaries_ok": False,
                       "autoregressive_boundary_ok": True}}
    v2 = p0.evaluate_readiness_v2({}, {}, {}, merged, False, True,
                                  protocol="v2")
    assert v2["ab_boundary_gate"] == "boundaries_ok"
    assert v2["ab_boundaries_ok"] is False


def test_v2r1_readiness_gates_on_the_autoregressive_audit():
    import cr_tser_p0_audit as p0

    merged = {"qwen": {"identical_predictions": True,
                       "boundaries_ok": False,
                       "autoregressive_boundary_ok": True}}
    v2r1 = p0.evaluate_readiness_v2({}, {}, {}, merged, False, True,
                                    protocol="v2r1")
    assert v2r1["ab_boundary_gate"] == "autoregressive_boundary_ok"
    assert v2r1["ab_boundaries_ok"] is True
    assert v2r1["ab_autoregressive_boundary_ok"] is True
    assert v2r1["ab_text_retokenization_stable"] is False


def test_v2r1_readiness_still_fails_on_a_failed_autoregressive_audit():
    import cr_tser_p0_audit as p0

    bad = {"mistral": {"identical_predictions": True,
                       "boundaries_ok": True,
                       "autoregressive_boundary_ok": False}}
    v2r1 = p0.evaluate_readiness_v2({}, {}, {}, bad, False, True,
                                    protocol="v2r1")
    assert v2r1["ab_boundaries_ok"] is False


def test_v2r1_readiness_still_requires_prediction_identity():
    import cr_tser_p0_audit as p0

    nonidentical = {"qwen": {"identical_predictions": False,
                             "boundaries_ok": True,
                             "autoregressive_boundary_ok": True}}
    v2r1 = p0.evaluate_readiness_v2({}, {}, {}, nonidentical, False, True,
                                    protocol="v2r1")
    # the boundary gate itself holds; the overall sanity verdict does not
    assert v2r1["ab_boundaries_ok"] is True
    assert v2r1["ab_sanity_ok"] is False


# --------------------------------------------------------------------------
# 5. the audit script records the amended fields for v2r1 only
# --------------------------------------------------------------------------
def test_sanity_records_amended_fields_under_v2r1(monkeypatch):
    import cr_tser_p0_audit as p0
    from cr_tser.readers import base_reader
    from ..config.pilot_config import PilotPaths

    class _FakeReader:
        def __init__(self, key):
            self.spec = base_reader.ReaderSpec(
                key, "/fake/model", dtype="bfloat16", device="cpu")
            self.tokenizer = _MergeTokenizer()

        def score_ab(self, prompt):
            from ..readers.sequence_scorer import ab_scores
            return ab_scores({"A": -0.1, "B": -2.0})

        def ab_token_report(self, prompt):
            from ..readers.sequence_scorer import ab_token_report
            return ab_token_report(self.tokenizer, prompt)

        def ab_boundary_audit(self, prompt):
            from ..readers.sequence_scorer import ab_boundary_audit
            return ab_boundary_audit(self.tokenizer, prompt)

        def identity(self):
            return {"model_id": "fake"}

        def unload(self):
            pass

    monkeypatch.setattr(base_reader, "build_reader",
                        lambda key, spec, mock=False: _FakeReader(key))
    monkeypatch.setattr(p0, "build_reader",
                        lambda key, spec, mock=False: _FakeReader(key))
    monkeypatch.setattr(p0.os.path, "isdir", lambda p: True)

    paths = PilotPaths(qwen_model="/fake/model")
    amended = p0.label_scoring_sanity(paths, n_examples=3, device="cpu",
                                      readers=("qwen",), protocol="v2r1")
    entry = amended["qwen"]
    assert entry["boundary_gate"] == "autoregressive_boundary_ok"
    assert entry["autoregressive_boundary_ok"] is True
    assert entry["native_prompt_matches_scorer"] is True
    assert entry["text_retokenization_stable"] is False   # visible, not hidden
    assert entry["boundaries_ok"] is False                # legacy kept too
    assert entry["candidate_ids_valid"] is True
    assert entry["candidate_decode"]["A"]["decoded"] == "A"
    assert entry["candidate_decode"]["B"]["text_retokenization_stable"] is False

    legacy = p0.label_scoring_sanity(paths, n_examples=3, device="cpu",
                                     readers=("qwen",), protocol="v2")
    assert "autoregressive_boundary_ok" not in legacy["qwen"]
    assert legacy["qwen"]["boundaries_ok"] is False


def test_apply_chat_failure_never_silently_passes(monkeypatch):
    """A reader that cannot produce a native prompt fails the gate closed."""
    from ..readers.sequence_scorer import ab_boundary_audit

    class _ExplodingTokenizer(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True, **kwargs):
            if tokenize:
                raise RuntimeError("native tokenization unavailable")
            return super().apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt)

    with pytest.raises(RuntimeError):
        ab_boundary_audit(_ExplodingTokenizer(), "source post body")
