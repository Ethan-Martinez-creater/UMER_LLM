"""Frozen prompt templates (plan §22).

Forbidden content (checked by tests and the leakage scanner): gold labels,
future node counts, final full depth, external evidence, retrieval labels,
rationale requests, JSON output instructions.
"""

PROMPT_TEMPLATE_EN = """SOURCE CLAIM
{source_text}

CURRENT SNAPSHOT
elapsed_time = {elapsed_time}
observed_replies = {observed_replies}
max_depth = {max_depth}

SELECTED SOCIAL EVIDENCE
{evidence_block}

TASK
Using only the source claim and the social evidence shown above,
classify the source claim as RUMOR or NON_RUMOR.
Return exactly one label."""

PROMPT_TEMPLATE_ZH = """源帖
{source_text}

当前传播状态
elapsed_time = {elapsed_time}
observed_replies = {observed_replies}
max_depth = {max_depth}

选中的社会证据
{evidence_block}

任务
仅依据上面的源帖和社会证据，
判断该源帖为 RUMOR 或 NON_RUMOR。
只返回一个标签。"""

TEMPLATES = {"en": PROMPT_TEMPLATE_EN, "zh": PROMPT_TEMPLATE_ZH}


def build_prompt(template_lang: str, source_text: str, elapsed_label: str,
                 observed_replies: int, max_depth: int,
                 evidence_block: str) -> str:
    try:
        template = TEMPLATES[template_lang]
    except KeyError:
        raise ValueError(
            f"template language must be en/zh, got {template_lang!r}")
    return template.format(
        source_text=source_text,
        elapsed_time=elapsed_label,
        observed_replies=int(observed_replies),
        max_depth=int(max_depth),
        evidence_block=evidence_block if evidence_block else "(none)",
    )
