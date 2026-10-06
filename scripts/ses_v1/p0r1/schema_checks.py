# -*- coding: utf-8 -*-
"""SES-v1 P0/P0-R1 共享 schema 与语义校验（供验证脚本与回归测试复用）。"""
from __future__ import annotations

from typing import Any

ENUM_CC = {"yes", "no", "unknown"}
ENUM_TEMPORAL = {"REPLY_TIME_VERIFIABLE_CONTENT_UNKNOWN", "AFTER_CUTOFF_ONLY",
                 "NO_CANDIDATE", "UNKNOWN"}
ENUM_SOURCE_KIND = {"reaction", "source", "historical_external", "annotation_audit_only"}
ENUM_PRESENCE = {"BEFORE_CUTOFF", "AFTER_CUTOFF", "MISSING_TIME", "NOT_A_TWEET"}
ENUM_CLAIM_ROLE = {"root", "side", "context"}
ENUM_DEP = {"VERIFIED_COMMON_UPSTREAM", "VERIFIED_DISTINCT_UPSTREAM_CANDIDATE",
            "UNKNOWN", "NOT_APPLICABLE"}
ENUM_SUPPORT = {"SUPPORTED_CORRECTION_CANDIDATE", "ASSERTION_ONLY", "QUESTION_ONLY",
                "CONFIRMATION_ONLY", "SIDE_CLAIM_ONLY", "INSUFFICIENT", "UNKNOWN"}
ENUM_HIST = {"VERIFIED_AT_CUTOFF", "AFTER_CUTOFF_ONLY", "CURRENT_ONLY",
             "UNKNOWN", "NOT_APPLICABLE"}
ENUM_GATE = {"YES", "NO", "UNKNOWN"}
CORRECTION_LIKE = {"SUPPORTED_CORRECTION_CANDIDATE", "ASSERTION_ONLY", "INSUFFICIENT"}


def check_observability_schema(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """F4：枚举、provisional、状态/证据位置检查。返回各类违规样本。"""
    bad_cc, bad_temporal, misplaced, not_provisional = [], [], [], []
    for r in rows:
        if r.get("correction_candidate") not in ENUM_CC:
            bad_cc.append(r.get("anon_id", "?"))
        if r.get("temporal_availability") not in ENUM_TEMPORAL:
            bad_temporal.append(r.get("anon_id", "?"))
        ce = (r.get("correction_evidence") or "").lower()
        ta = (r.get("temporal_availability") or "").lower()
        if "before_cutoff" in ce or "after_cutoff" in ce or "core pair" in ta:
            misplaced.append(r.get("anon_id", "?"))
        if r.get("provisional") != "1":
            not_provisional.append(r.get("anon_id", "?"))
    return {"bad_cc": bad_cc, "bad_temporal": bad_temporal,
            "misplaced": misplaced, "not_provisional": not_provisional}


def check_ledger_schema(rows: list[dict[str, Any]]) -> list[tuple]:
    bad = []
    for r in rows:
        for k, allowed in (("source_kind", ENUM_SOURCE_KIND), ("tweet_presence", ENUM_PRESENCE),
                           ("claim_role", ENUM_CLAIM_ROLE), ("dependency_state", ENUM_DEP),
                           ("support_state", ENUM_SUPPORT), ("historical_content_state", ENUM_HIST)):
            if r.get(k) not in allowed:
                bad.append((r.get("case_id"), r.get("evidence_id"), k, r.get(k)))
        if r.get("provisional") != "1":
            bad.append((r.get("case_id"), r.get("evidence_id"), "provisional", r.get("provisional")))
        # annotation 永不作为有效纠错（F3）
        if r.get("source_kind") == "annotation_audit_only" and r.get("support_state") == "SUPPORTED_CORRECTION_CANDIDATE":
            bad.append((r.get("case_id"), r.get("evidence_id"), "annotation_as_correction", r.get("support_state")))
    return bad


def check_cutoff_consistency(rows: list[dict[str, Any]]) -> list[tuple]:
    """F5：按 offset 推导的截点集合必须一致（21.4min 材料不得声明在 15min 内）。"""
    bad = []
    for r in rows:
        off = r.get("reply_offset_min") or ""
        if not off:
            continue
        vals = []
        for part in str(off).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                try:
                    vals += [float(a), float(b)]
                except ValueError:
                    pass
            else:
                try:
                    vals.append(float(part))
                except ValueError:
                    pass
        vals = [v for v in vals if v >= 0]
        if not vals:
            continue
        mx = max(vals)
        declared = set(str(r.get("cutoff", "")).split(";"))
        expected = {str(c) for c in (15, 60, 360) if mx <= c}
        if "15" in declared and mx > 15:
            bad.append((r.get("case_id"), r.get("evidence_id"), "offset>15 but 15 declared"))
        if not (declared & expected or declared == {"NONE"}):
            bad.append((r.get("case_id"), r.get("evidence_id"), "cutoff set mismatch"))
    return bad


def check_pair_semantics(pairs: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[tuple]:
    """gate 语义：YES 仅当全项可核；UNKNOWN 必须有纠错尝试且历史不可核。"""
    sup_by_case: dict[str, set[str]] = {}
    for r in ledger:
        sup_by_case.setdefault(r.get("case_id", ""), set()).add(r.get("support_state"))
    bad = []
    for r in pairs:
        gate = r.get("meets_original_gate")
        if gate not in ENUM_GATE:
            bad.append((r.get("case_id"), r.get("cutoff"), "bad gate value", gate))
            continue
        if gate == "YES":
            # 全项可核：必须存在被支持的纠错且历史可核
            sup = sup_by_case.get(r.get("case_id", ""), set())
            if "SUPPORTED_CORRECTION_CANDIDATE" not in sup:
                bad.append((r.get("case_id"), r.get("cutoff"), "YES without supported correction"))
            if not str(r.get("historical_availability", "")).startswith("VERIFIED_AT_CUTOFF"):
                bad.append((r.get("case_id"), r.get("cutoff"), "YES without verified material"))
            continue
        if gate == "UNKNOWN":
            sup = sup_by_case.get(r.get("case_id", ""), set())
            if not (sup & CORRECTION_LIKE):
                bad.append((r.get("case_id"), r.get("cutoff"), "UNKNOWN without correction-attempt row"))
            if not str(r.get("historical_availability", "")).startswith(("UNKNOWN", "CURRENT_ONLY", "AFTER_CUTOFF_ONLY")):
                bad.append((r.get("case_id"), r.get("cutoff"), "UNKNOWN with verified material"))
    return bad
