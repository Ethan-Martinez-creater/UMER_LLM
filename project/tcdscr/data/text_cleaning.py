"""Text cleaning for TC-DSCR.

Verbatim re-use of the historically verified preprocessing: PHEME text goes
through ``clean_tweet_pheme`` (extracted from the authoritative 240h pipeline)
and Ma-Weibo text goes through ``clean_text_weibo``. The Ma-Weibo step is the
one lost by the server-side version drift uncovered in the D6 parity check;
TC-DSCR must restore it explicitly (V2 plan §3.2/§7).
"""
from __future__ import annotations

import html
import re


def clean_text_weibo(text: str) -> str:
    """Clean Chinese Weibo text: strip HTML, URLs, normalize whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S*', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_tweet_pheme(text: str, lower: bool = True,
                      preserve_hashtag: bool = False,
                      preserve_mention: bool = False) -> str:
    """Clean English Twitter text — verbatim historical preprocessing."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S*', '', text, flags=re.I)

    if not preserve_mention:
        text = re.sub(r'@\w+', '', text)
    if not preserve_hashtag:
        text = re.sub(r'#\w+', '', text)

    text = re.sub(r'^(RT|via)\s+@\w+:\s*', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[​‌‍﻿]', '', text)

    if lower:
        text = text.lower()

    return text
