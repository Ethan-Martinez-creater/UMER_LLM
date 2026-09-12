"""Reader output parsing, schema validation and citation accounting (V3-B §17-§18, §28-§31).

Deterministic, no LLM judge: the parser accepts a strict JSON object (or the
first balanced ``{...}`` block found in the text), validates the frozen schema,
and the citation helpers compare the ids the reader cited against the ids that
were actually in front of it.
"""
from __future__ import annotations

import json
import re

LABELS = ("RUMOR", "NON_RUMOR")

_REF_RE = re.compile(r"\[?(E\d+)\]?")


def _first_json_object(text: str):
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i + 1]
    return None


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped


def validate_reader_object(obj):
    """Return ``(parsed, errors)``; ``parsed`` is None when the schema fails."""
    if not isinstance(obj, dict):
        return None, ["not_an_object"]
    errors = []
    label = obj.get("label")
    if not isinstance(label, str) or label.strip().upper() not in LABELS:
        errors.append("bad_label")
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errors.append("bad_confidence")
    elif not 0.0 <= float(conf) <= 1.0:
        errors.append("bad_confidence")
    ids = obj.get("evidence_ids")
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        errors.append("bad_evidence_ids")
    reason = obj.get("reason")
    if not isinstance(reason, str):
        errors.append("bad_reason")
    if errors:
        return None, errors
    return {
        "label": label.strip().upper(),
        "confidence": float(conf),
        "evidence_ids": [x.strip() for x in ids],
        "reason": reason,
    }, []


def parse_reader_output(raw):
    """``(parsed, errors)`` for one raw generation; never raises."""
    if raw is None or not str(raw).strip():
        return None, ["empty_output"]
    text = _strip_fence(str(raw))
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        blob = _first_json_object(text)
        if blob is not None:
            try:
                obj = json.loads(blob)
            except Exception:
                obj = None
    if obj is None:
        return None, ["no_json_object"]
    return validate_reader_object(obj)


def citation_stats(cited_ids, allowed_ids):
    """Split citations into those in the arm prompt and those that are not."""
    allowed = set(allowed_ids)
    valid = [c for c in cited_ids if c in allowed]
    invalid = [c for c in cited_ids if c not in allowed]
    return {"n_cited": len(cited_ids), "valid": valid, "invalid": invalid}


def reason_invalid_refs(reason, allowed_ids):
    """``E<i>`` tokens written in ``reason`` that are not in the arm prompt."""
    allowed = set(allowed_ids)
    refs = _REF_RE.findall(reason or "")
    return [r for r in refs if r not in allowed]
