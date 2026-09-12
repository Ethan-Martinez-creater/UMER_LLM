"""Frozen reader-transfer prompt (V3-B §14-§16).

One implementation of the evidence rendering, the user prompt, the chat
formatting and the token accounting, so the sampling manifest, the inference
runner and the verifier can never disagree about what the reader saw.

Frozen by design (V3-B §15): the templates in this module are written once and
must not be iterated on the basis of Qwen results.

Evidence unit = Reply-Parent Pair (§13). Every selected node is rendered in
snapshot order, which is timestamp ascending with ties broken by the adapter's
original order (§12), and receives a stable ``E<i>`` id (§14).
"""
from __future__ import annotations

UNAVAILABLE = "[UNAVAILABLE]"

PROMPT_VERSION = "v3b-2"

SYSTEM_PROMPT = (
    "You are evaluating whether a social-media source post is a rumor.\n"
    "\n"
    "You must base the judgment only on:\n"
    "1. the source post\n"
    "2. the supplied social evidence\n"
    "\n"
    "Do not use external knowledge.\n"
    "Do not invent facts that are not supported by the supplied evidence.\n"
    "\n"
    "If the supplied evidence is weak, incomplete, or conflicting,\n"
    "reflect that uncertainty in confidence.\n"
    "\n"
    "Return only the required JSON object."
)

USER_PROMPT = """Source Post:
{source_text}

Observed Social Evidence up to {cutoff_label}:
{evidence_block}

Task:
Determine whether the Source Post should be classified as:

RUMOR
or
NON_RUMOR

Return JSON only, with exactly these fields:
{{"label": "RUMOR" or "NON_RUMOR", "confidence": 0.0, "evidence_ids": ["E1"], "reason": "short reason"}}

Use only evidence ids that appear above; use [] when none applies."""

CUTOFF_LABELS = {5: "5m", 15: "15m", 30: "30m", 60: "1h", 180: "3h",
                 360: "6h"}

RETRY_SUFFIX = "Return valid JSON matching the required schema."

MAX_NEW_TOKENS = 256


def cutoff_label(cutoff) -> str:
    """Human label for a cutoff in minutes; unknown cutoffs stay in minutes."""
    return CUTOFF_LABELS.get(int(cutoff), f"{int(cutoff)}m")


def render_evidence(unit: dict, index: int) -> str:
    """One Reply-Parent Pair as ``[Ei]`` with its relative timestamp."""
    parent = unit["parent_text"]
    if parent is None:
        parent = UNAVAILABLE
    return "\n".join([
        f"[E{index}]",
        f"Reply: {unit['reply_text']}",
        f"Parent: {parent}",
        f"Observed: +{int(unit['elapsed_seconds'])}s",
    ])


def render_evidence_block(units) -> str:
    if not units:
        return "(none)"
    return "\n\n".join(render_evidence(u, i)
                       for i, u in enumerate(units, start=1))


def evidence_id_map(units) -> dict:
    """``{"E1": node_id, ...}`` for one arm's prompt (§14)."""
    return {f"E{i}": u["node_id"] for i, u in enumerate(units, start=1)}


def order_units_by_snapshot(units, selected_node_ids):
    """Selected units in snapshot order (= timestamp asc, tie by original).

    ``units`` comes from ``build_evidence_units`` and therefore already carries
    snapshot order; this filters it, never re-sorts it (§12, §38).
    """
    wanted = set(selected_node_ids)
    return [u for u in units if u["node_id"] in wanted]


def build_user_prompt(source_text: str, cutoff, evidence_block: str) -> str:
    return USER_PROMPT.format(source_text=source_text,
                              cutoff_label=cutoff_label(cutoff),
                              evidence_block=evidence_block)


def build_messages(user_prompt: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]


def format_reader_chat(tokenizer, user_prompt: str) -> str:
    """Chat text for system+user with the thinking block disabled.

    Mirrors ``qwen_wrapper.format_chat_prompt`` but carries the system message
    required by §11/§15 and is the single formatting used for token counting
    and for inference.
    """
    messages = build_messages(user_prompt)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def count_chat_tokens(tokenizer, user_prompt: str) -> int:
    return count_tokens(tokenizer, format_reader_chat(tokenizer, user_prompt))


def build_arm_prompt(tokenizer, source_text, cutoff, units):
    """Full paired-arm bundle: prompt text, token accounting and E# mapping.

    ``social_tokens`` counts the evidence units only and is 0 when the arm has
    no evidence at all (§9, §20): the ``(none)`` placeholder is rendered for
    the reader but must not enter the compression mean, and §20/§23 treat a
    zero Static social context as N/A rather than as a 100%-savings sample.
    """
    block = render_evidence_block(units)
    user_prompt = build_user_prompt(source_text, cutoff, block)
    return {
        "prompt": user_prompt,
        "system_prompt": SYSTEM_PROMPT,
        "evidence_block": block,
        "evidence_ids": evidence_id_map(units),
        "social_tokens": count_tokens(tokenizer, block) if units else 0,
        "total_input_tokens": count_chat_tokens(tokenizer, user_prompt),
        "n_evidence": len(units),
    }
