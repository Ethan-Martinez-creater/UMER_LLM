# -*- coding: utf-8 -*-
"""SES-v1 P0-R1 结构化证据台账与配对资格重算（F1/F5）。

覆盖：原 12 深查例 + R1 新增 12 深查例 = 24 例。
- P0R1_EVIDENCE_LEDGER.csv：材料/回复行级证据（显式 DictWriter，固定枚举）
- P0R1_PAIR_AUDIT.csv：case x cutoff 配对资格（全项可核才计 YES）

状态枚举（计划 §5）：
tweet_presence: BEFORE_CUTOFF / AFTER_CUTOFF / MISSING_TIME / NOT_A_TWEET
historical_content_state: VERIFIED_AT_CUTOFF / AFTER_CUTOFF_ONLY / CURRENT_ONLY /
                          UNKNOWN / NOT_APPLICABLE
dependency_state: VERIFIED_COMMON_UPSTREAM / VERIFIED_DISTINCT_UPSTREAM_CANDIDATE /
                  UNKNOWN / NOT_APPLICABLE
support_state: SUPPORTED_CORRECTION_CANDIDATE / ASSERTION_ONLY / QUESTION_ONLY /
               CONFIRMATION_ONLY / SIDE_CLAIM_ONLY / INSUFFICIENT / UNKNOWN

全部内容 provisional，不做人工 gold。数据为执行智能体审读记录，
数值行号 R### 以 deep_audit*.txt 的时间排序为准（与 signal_summary 的文件序不同，已统一）。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "check" / "ses_v1" / "p0r1"

LEDGER_FIELDS = ["case_id", "evidence_id", "target_claim_id", "claim_role", "provisional",
                 "source_kind", "reply_offset_min", "cutoff", "tweet_presence",
                 "historical_content_state", "attribution_state", "dependency_state",
                 "support_state", "evidence_ref", "unknown_reason"]
PAIR_FIELDS = ["case_id", "v_class", "topic", "cutoff", "root_claim", "common_upstream_evidence",
               "correction_target_claim", "correction_support_evidence", "upstream_relation",
               "historical_availability", "meets_original_gate", "reason", "provisional"]

CLAIMS = {
    "001": ("G320-crash", "Germanwings A320 4U9525 lost at 6800ft near Digne", "true"),
    "002": ("OTT-statement", "Harper to make statement after gunman shot dead; PM safe", "true"),
    "003": ("GURL-accept", "Swiss museum accepts Gurlitt art bequest", "true"),
    "004": ("FER-stop", "Police chief: stopped for walking/blocking traffic", "true"),
    "005": ("SYD-activity", "Heavy police activity at Sydney cafe; many shots fired", "true"),
    "007": ("G320-toll", "144 passengers + 6 crew on board crashed 4U9525", "true"),
    "008": ("OTT-soldier", "Canadian soldier shot at War Memorial has died", "true"),
    "010": ("FER-stop2", "Chief: stopped for walking in road, not cigar suspicion", "true"),
    "014": ("OTT-photo", "Photo shows gunman Michael Zehaf-Bibeau", "true"),
    "021": ("G320-toll2", "A320 carried 142 passengers, 2 pilots, 4 crew", "false"),
    "022": ("ESS-denial", "Milan states Essien-Ebola reports are false (thread V refers to rumour 'Essien has Ebola')", "false"),
    "023": ("OTT-3scenes", "Police investigating 3 shooting scenes", "false"),
    "024": ("PRIN-show", "Prince surprise show at Massey Hall tonight", "false"),
    "026": ("SYD-airspace", "Sydney airspace closed (side claims: Opera House evacuated; ISIS-style flag)", "false"),
    "028": ("CH-fatality", "At least one person killed in Dammartin shootout", "false"),
    "029": ("G320-convert", "Co-pilot was Muslim convert", "false"),
    "030": ("ESS-ebola", "Essien contracted Ebola, club confirmed", "false"),
    "036": ("CH-banksy", "Drawing circulating is by Banksy", "false"),
    "041": ("G320-depr", "German media report co-pilot had serious depressive episode", "unverified"),
    "042": ("OTT-lockdown", "Parliament locked down; at least one guard injured (side: guard died)", "unverified"),
    "044": ("GURL-will", "Gurlitt not mentally fit to write will", "unverified"),
    "043": ("PRIN-stage", "Picture shows Prince stage being set up at Massey Hall", "unverified"),
    "053": ("FER-robbery", "Mike Brown connected to earlier robbery per documents", "unverified"),
    "056": ("CH-located", "Two suspects located in northern France (side: caught vs located)", "unverified"),
}

# 每例证据行（R### 为 deep_audit 时间排序行号；@min 为分钟偏移）
E = [
    # ---- 001 true ----
    ("001", "SRC", "G320-crash", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", "annotation links 4x for/news-media"),
    ("001", "R002,R010", "G320-crash", "root", "reaction", "2.22,8.48", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "verbatim RT of @flightradar24 source"),
    ("001", "R024-R028", "G320-crash", "context", "reaction", "21.42-22.10", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_COMMON_UPSTREAM", "ASSERTION_ONLY", "5 accounts share same blogspot URL at 21.4-22.1min; content unrelated to root claim; historical content not checked"),
    # ---- 002 true ----
    ("002", "SRC", "OTT-statement", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", None),
    ("002", "R001-R005", "OTT-statement", "context", "reaction", "1.08-412.62", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "UNKNOWN", "all direct replies; no paraphrase/URL/correction signal; R003 notes NBC gives more detail (cross-media comparison, not correction)"),
    # ---- 003 true ----
    ("003", "SRC", "GURL-accept", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", None),
    ("003", "ANN", "GURL-accept", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "8 annotation links: 7 for + 1 against (museum media statement 21-11); audit-only, never deployable input; zero reactions"),
    # ---- 004 true ----
    ("004", "R002,R003", "FER-stop", "root", "reaction", "6.25,6.50", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "chained RT of R001 echo"),
    ("004", "R004", "FER-stop", "context", "reaction", "99.57", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "stance reversal about shooting; not a fact correction of root"),
    # ---- 005 true ----
    ("005", "R002", "SYD-activity", "root", "reaction", "8.02", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "verbatim RT @BuzzFeedNews; source itself is @BuzzFeedNews citing @abcnews"),
    # ---- 007 true ----
    ("007", "R004-R006,R010,R011", "G320-toll", "context", "reaction", "7.13-27.52", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "5 users share bbc.in/1LSgxjV (Hollande nationalities item = SIDE claim, not root toll)"),
    ("007", "R002,R003,R012,R017", "G320-toll", "root", "reaction", "2.97-67.03", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "4 users share bbc.in/1EMTxMz restate same 144+6 toll from same BBC outlet"),
    ("007", "R008", "G320-toll", "side", "reaction", "14.38", "BEFORE_CUTOFF", "UNKNOWN", "UNKNOWN", "SIDE_CLAIM_ONLY", "requests correction of victim nationalities (side claim); root toll untouched; link content not archived"),
    ("007", "R012", "G320-toll", "root", "reaction", "28.37", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "Japanese 'kousei' note restates same 144+6 from same BBC item; restatement not independent correction"),
    # ---- 008 true ----
    ("008", "R001-R019", "OTT-soldier", "root", "reaction", "0.82-179.55", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "large same-text RT cluster of one @CNN tweet; pure echo, no correction"),
    # ---- 010 true ----
    ("010", "R001", "FER-stop2", "root", "reaction", "0.0", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "verbatim duplicate of source text"),
    ("010", "R076,R078", "FER-stop2", "context", "reaction", "370.08,380.75", "AFTER_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "disagreement/video-fake question after 360min; outside all cutoffs"),
    # ---- 014 true ----
    ("014", "R008,R010", "OTT-photo", "context", "reaction", "6.27,7.22", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "QUESTION_ONLY", "attribution/source verification questions; no material cited"),
    ("014", "R017", "OTT-photo", "root", "reaction", "10.75", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "answers R010 citing CBC TV/police = same outlet chain as the echo; confirmation not independent correction"),
    ("014", "R024", "OTT-photo", "context", "reaction", "53.0", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "retraces earlier claim chain (photo claimed from extremist account)"),
    # ---- 021 false ----
    ("021", "SRC", "G320-toll2", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", "toll 142+2+4 differs from later 144+6; no reactions at any cutoff so no in-tree correction"),
    # ---- 022 false ----
    ("022", "SRC", "ESS-denial", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", "source post itself is the club denial; thread V=false refers to the rumour claim, label-content mismatch recorded"),
    ("022", "R002,R003", "ESS-denial", "root", "reaction", "7.10,13.98", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "R003 cites acmilan.com link = same club chain as source; no independent new upstream"),
    ("022", "ANN", "ESS-denial", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "4 AGAINST links (player Instagram + 3 media); audit-only"),
    # ---- 023 false ----
    ("023", "R001", "OTT-3scenes", "root", "reaction", "3.40", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "restates @ErinAnderssen source text"),
    # ---- 024 false ----
    ("024", "ANN", "PRIN-show", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "annotation carries AGAINST (LiveNation Facebook) but zero reactions; audit-only"),
    # ---- 026 false ----
    ("026", "R002,R004,R008,R014", "SYD-airspace", "root", "reaction", "1.22-78.62", "BEFORE_CUTOFF", "CURRENT_ONLY", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "4 users share cbc.ca/1.2873068; that URL TODAY resolves to a different story (victims identified) -> content identity at cutoff UNKNOWN; short link not archived"),
    ("026", "R001", "SYD-airspace", "root", "reaction", "0.40", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "asserts airspace not closed; no link/material; statement itself now consistent with record but historical material absent"),
    ("026", "R007", "SYD-airspace", "root", "reaction", "10.20", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "QUESTION_ONLY", "asks to check sources and offers TFR alternative; no cited material"),
    ("026", "R015", "SYD-airspace", "side", "reaction", "151.93", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "QUESTION_ONLY", "flags unsubstantiated ISIS-style-flag side claim"),
    # ---- 028 false ----
    ("028", "R001", "CH-fatality", "root", "reaction", "3.65", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "quote of @BBCBreaking source text"),
    ("028", "R003", "CH-fatality", "root", "reaction", "11.28", "BEFORE_CUTOFF", "UNKNOWN", "VERIFIED_DISTINCT_UPSTREAM_CANDIDATE", "SUPPORTED_CORRECTION_CANDIDATE", "prosecutor-denial item with bbc.in/14ulyLt link; short link not archived, current redirect is a live page -> historical content UNKNOWN"),
    ("028", "R004", "CH-fatality", "root", "reaction", "14.42", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "CONFIRMATION_ONLY", "in-tree reply to R003 restating same correction; propagation, NOT a new independent support"),
    ("028", "R006", "CH-fatality", "context", "reaction", "22.10", "BEFORE_CUTOFF", "UNKNOWN", "UNKNOWN", "ASSERTION_ONLY", "awd-news conspiracy link; unreliable material"),
    # ---- 029 false ----
    ("029", "ANN", "G320-convert", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "annotation carries AGAINST blog debunk; in-tree rows never cite it inside cutoffs; R017 denial at 521.6min is AFTER_CUTOFF"),
    # ---- 030 false ----
    ("030", "R001-R019", "ESS-ebola", "root", "reaction", "5.75-85.82", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "large verbatim quote cluster of rumour origin @YuryAlkaev"),
    ("030", "R009,R010,R020", "ESS-ebola", "root", "reaction", "35.18-85.82", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "forged club-spokesman quote propagating the rumour; misinformation amplification inside tree"),
    ("030", "R016", "ESS-ebola", "root", "reaction", "78.48", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "QUESTION_ONLY", "demands source; no material"),
    ("030", "R004,R017", "ESS-ebola", "root", "reaction", "14.03,84.35", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "assertion-only disbelief (no cited material)"),
    # ---- 036 false ----
    ("036", "R002-R019", "CH-banksy", "root", "reaction", "2.05-232.18", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "long same-text echo of @HuffPostCanada across 4h"),
    ("036", "ANN", "CH-banksy", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "annotation: 2 FOR + 1 observing + 2 AGAINST (fake-attribution debunks); audit-only, never cited in tree"),
    # ---- 041 unverified ----
    ("041", "R004,R005,R007,R008,R009,R010,R012", "G320-depr", "root", "reaction", "7.05-245.60", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "7 users share usat.ly/1bAjjKo quoting same @USATODAY item; clean echo cluster, no correction"),
    # ---- 042 unverified ----
    ("042", "R002", "OTT-lockdown", "root", "reaction", "8.87", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "RT @WSJLive echo"),
    ("042", "R004", "OTT-lockdown", "side", "reaction", "11.62", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "SIDE_CLAIM_ONLY", "corrects SIDE claim injured->died; root lockdown claim untouched; no material"),
    # ---- 053 unverified ----
    ("053", "R001", "FER-robbery", "root", "reaction", "0.0", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "verbatim duplicate of @TheAnonMessage2 source text"),
    ("053", "R005,R011,R012,R017", "FER-robbery", "root", "reaction", "1.37-10.62", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "denial/question rows without cited material"),
    ("053", "R013", "FER-robbery", "root", "reaction", "9.03", "BEFORE_CUTOFF", "UNKNOWN", "UNKNOWN", "INSUFFICIENT", "cites store manager prior denial but no link; underlying statement not located"),
    ("053", "ANN", "FER-robbery", "root", "annotation_audit_only", None, "NOT_A_TWEET", "UNKNOWN", "NOT_APPLICABLE", "UNKNOWN", "2 FOR news-media (usatoday archived 2014-08-16) + 1 AGAINST youtube; audit-only"),
    # ---- 056 unverified ----
    ("056", "R001", "CH-located", "root", "reaction", "0.50", "BEFORE_CUTOFF", "NOT_APPLICABLE", "VERIFIED_COMMON_UPSTREAM", "CONFIRMATION_ONLY", "quote of @SkyNewsBreak source text"),
    ("056", "R015", "CH-located", "side", "reaction", "16.70", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "QUESTION_ONLY", "asks caught-vs-located distinction on side claim"),
    ("056", "R009", "CH-located", "root", "reaction", "5.62", "BEFORE_CUTOFF", "NOT_APPLICABLE", "UNKNOWN", "ASSERTION_ONLY", "uncertain prior-surrender note, self-flagged unsure, no material"),
    # ---- 043/044 ----
    ("044", "SRC", "GURL-will", "root", "source", None, "BEFORE_CUTOFF", "NOT_APPLICABLE", "NOT_APPLICABLE", "UNKNOWN", "zero reactions; annotation 1 FOR link (DW) audit-only"),
]


def build_ledger():
    rows = []
    for case, eid, claim, role, kind, off, presence, hist, dep, sup, ref in E:
        # F5：按 offset 推导该行适用截点；21.4min 材料不得声明在 15min 内
        if kind in ("source", "annotation_audit_only") or off is None:
            cutoffs = "15;60;360"
        else:
            vals = []
            for part in str(off).split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part.lstrip("-"):
                    a, b = part.split("-", 1)
                    try:
                        vals.extend([float(a), float(b)])
                    except ValueError:
                        pass
                else:
                    try:
                        vals.append(float(part))
                    except ValueError:
                        pass
            vals = [v for v in vals if v >= 0]
            mx = max(vals) if vals else 0.0
            applicable = [c for c in (15, 60, 360) if mx <= c]
            cutoffs = ";".join(str(c) for c in applicable) if applicable else "NONE"
        rows.append({
            "case_id": f"SES-P0-{case}", "evidence_id": eid, "target_claim_id": claim,
            "claim_role": role, "provisional": "1", "source_kind": kind,
            "reply_offset_min": off or "", "cutoff": cutoffs, "tweet_presence": presence,
            "historical_content_state": hist, "attribution_state": "UNKNOWN" if kind in ("reaction", "source") else "NOT_APPLICABLE",
            "dependency_state": dep, "support_state": sup,
            "evidence_ref": ref or "", "unknown_reason": "" if hist != "UNKNOWN" else "historical content of cited material not verifiable (no archive / live-page drift / login wall)",
        })
    return rows


def build_pairs():
    rows = []
    for case, (cid, claim, vcls) in sorted(CLAIMS.items()):
        topic_map = {"001": "germanwings-crash", "002": "ottawashooting", "003": "gurlitt",
                     "004": "ferguson", "005": "sydneysiege", "007": "germanwings-crash",
                     "008": "ottawashooting", "010": "ferguson", "011": "sydneysiege",
                     "012": "charliehebdo", "013": "germanwings-crash", "014": "ottawashooting",
                     "016": "ferguson", "017": "sydneysiege", "019": "germanwings-crash",
                     "021": "germanwings-crash", "022": "ebola-essien", "023": "ottawashooting",
                     "024": "prince-toronto", "026": "sydneysiege", "028": "charliehebdo",
                     "029": "germanwings-crash", "030": "ebola-essien", "036": "charliehebdo",
                     "041": "germanwings-crash", "042": "ottawashooting",
                     "043": "prince-toronto", "044": "gurlitt",
                     "053": "ferguson", "056": "charliehebdo"}
        # 案例级共同上游证据与最强纠错行
        info = {
            "001": ("RT of @flightradar24 + blogspot URL cluster 21.4-22.1min (outside 15min)", "none observed", "NO", "no correction row at any cutoff", "VERIFIED_COMMON_UPSTREAM", "UNKNOWN"),
            "002": ("none", "none", "NO", "replies are reactions; no paraphrase/URL/correction"),
            "003": ("none (zero reactions)", "none", "NO", "no_candidate all cutoffs; annotation AGAINST exists audit-only"),
            "004": ("chained RT echo", "none (R004 stance, not fact)", "NO", "no fact correction"),
            "005": ("verbatim RT @BuzzFeedNews", "none", "NO", "echo only"),
            "007": ("two BBC URL clusters 4x+5x", "side-claim nationality correction (R008); restatement kousei (R012)", "NO", "corrections target SIDE claims or restate root from same outlet; not independent root correction", "VERIFIED_COMMON_UPSTREAM", "AFTER_CUTOFF_ONLY (same-day 17:35 snapshot of live page, later than 13:07)"),
            "008": ("same-text RT cluster of one @CNN tweet", "none", "NO", "pure echo"),
            "010": ("verbatim duplicate at 0.0min", "challenges only AFTER_CUTOFF", "NO", "inside cutoffs echo only"),
            "011": ("none", "false-flag conspiracy assertion", "NO", "challenge is itself false; ineffective"),
            "012": ("RT @SkyNews echo", "none", "NO", "echo only"),
            "013": ("unknown (1 row)", "none", "NO", "no signal"),
            "014": ("quote cluster of @CBCOttawa", "attribution questions answered by same-outlet confirmation", "NO", "questions + confirmation from same chain; no independent upstream correction", "VERIFIED_COMMON_UPSTREAM", "UNKNOWN"),
            "016": ("RT echo", "none", "NO", "echo only"),
            "017": ("quote cluster @CBCNews", "none", "NO", "echo only"),
            "019": ("verbatim RT @FoxNews", "none", "NO", "echo only"),
            "021": ("none (zero reactions)", "none", "NO", "no_candidate all cutoffs"),
            "022": ("quote of @Milanello denial", "none (source itself is the denial; R003 cites same club chain)", "NO", "no in-tree correction of the underlying rumour; label-content mismatch", "VERIFIED_COMMON_UPSTREAM", "UNKNOWN"),
            "023": ("restatement of source", "none", "NO", "echo only"),
            "024": ("none (joke replies)", "none", "NO", "annotation AGAINST audit-only; zero in-tree correction"),
            "026": ("cbc.ca URL shared by 4 users (content identity at cutoff UNKNOWN)", "R001 assertion 'airspace isn't closed' @0.4min; R007 source-check question; R015 side-claim flag question", "UNKNOWN", "correction is assertion-only without verifiable material; URL content at cutoff UNKNOWN; fails all-item-verifiable rule", "VERIFIED_COMMON_UPSTREAM", "AFTER_CUTOFF_ONLY (next-day snapshot; page title already changed)"),
            "028": ("quote of @BBCBreaking source text", "R003 prosecutor-denial with bbc.in link @11.28min (distinct-upstream candidate); R004 in-tree propagation", "UNKNOWN", "strongest case; but cited link resolves to a live page whose same-day 22:17 snapshot postdates the 09:26 correction row; historical material UNKNOWN; fails all-item-verifiable rule", "VERIFIED_DISTINCT_UPSTREAM_CANDIDATE", "AFTER_CUTOFF_ONLY (same-day 22:17 snapshot of live page, later than 09:26)"),
            "029": ("RT @daxtonbrown echo", "denial only AFTER_CUTOFF (R017 @521.6min)", "NO", "no in-cutoff correction; AGAINST debunk exists only in annotation"),
            "030": ("large verbatim echo cluster of rumour origin", "assertion-only disbelief + source-demand; forged quote propagates", "NO", "no evidential correction inside tree; annotation AGAINST never cited", "VERIFIED_COMMON_UPSTREAM", "UNKNOWN"),
            "036": ("long same-text echo @HuffPostCanada", "none in tree (AGAINST debunks annotation-only)", "NO", "echo only"),
            "041": ("usat.ly URL shared by 7 users", "none", "NO", "clean echo cluster", "VERIFIED_COMMON_UPSTREAM", "CURRENT_ONLY"),
            "042": ("RT @WSJLive echo", "R004 side-claim correction injured->died @11.62min", "NO", "side-claim only; root untouched; no material"),
            "044": ("none (zero reactions)", "none", "NO", "no_candidate"),
            "043": ("none (zero reactions)", "none", "NO", "no_candidate all cutoffs"),
            "053": ("verbatim duplicate at 0.0min", "R013 cites store manager prior denial @9.03min without link", "UNKNOWN", "correction cites unlinked prior statement; underlying manager denial not located; fails all-item-verifiable rule", "UNKNOWN", "UNKNOWN"),
            "056": ("quote @SkyNewsBreak echo", "R015 caught-vs-located distinction question; R009 uncertain correction note", "NO", "questions/uncertain notes; no evidential material"),
        }
        entry = info[case]
        cu, corr, gate, reason = entry[0], entry[1], entry[2], entry[3]
        upstream = entry[4] if len(entry) > 4 else "UNKNOWN"
        hist = entry[5] if len(entry) > 5 else "UNKNOWN"
        # 028 的根命题是 false 且纠错针对 root；053 同理（unverified）
        for cutoff in (15, 60, 360):
            meets = gate
            if corr == "none" or corr.startswith("challenges only AFTER") or corr.startswith("denial only AFTER"):
                meets = "NO"
            rows.append({
                "case_id": f"SES-P0-{case}", "v_class": vcls, "topic": topic_map[case],
                "cutoff": cutoff, "root_claim": claim,
                "common_upstream_evidence": cu, "correction_target_claim": corr,
                "correction_support_evidence": "",
                "upstream_relation": upstream, "historical_availability": hist,
                "meets_original_gate": meets, "reason": reason, "provisional": "1",
            })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger()
    with (OUT / "P0R1_EVIDENCE_LEDGER.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(ledger)
    pairs = build_pairs()
    with (OUT / "P0R1_PAIR_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        w.writeheader()
        w.writerows(pairs)
    from collections import Counter
    gate_counts = Counter(r["meets_original_gate"] for r in pairs if r["cutoff"] == 15)
    print("ledger rows:", len(ledger), "| pair rows:", len(pairs), "| gate@15min:", dict(gate_counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
