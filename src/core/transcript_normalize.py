"""Normalize STT transcripts before routing."""

import re

_WAKE_ECHO_RE = re.compile(
    r"^\s*(yes\??|yeah|yep|yup|ok(?:ay)?|sure|what\??|huh\??)\s*\.?$",
    re.IGNORECASE,
)
_PLAY_IT_SUFFIX_RE = re.compile(
    r"^(?P<body>.+?)[,.\s]+(?:please\s+)?play\s+it\s*\.?$",
    re.IGNORECASE,
)
_PLAY_IT_PREFIX_RE = re.compile(
    r"^play\s+it[,.\s]+(?P<body>.+?)\s*\.?$",
    re.IGNORECASE,
)
_BUY_BY_RE = re.compile(r"\bbuy\b", re.IGNORECASE)
_FILLER_PREFIX_RE = re.compile(
    r"^\s*(?:hey\s+jarvis[,.\s]+|jarvis[,.\s]+|"
    r"um[,.\s]+|uh[,.\s]+|maybe[,.\s]+|"
    r"i\s+don'?t\s+know[,.\s]+|can\s+you[,.\s]+|"
    r"i\s+want\s+to\s+hear[,.\s]+|"
    r"could\s+you[,.\s]+|please\s+)\s*",
    re.IGNORECASE,
)


def is_wake_echo(transcript: str) -> bool:
    """True when the user likely echoed Jarvis's 'Yes?' ack."""
    return bool(_WAKE_ECHO_RE.match(transcript.strip()))


def normalize_transcript(transcript: str) -> str:
    """Fix common STT shapes before routing."""
    text = transcript.strip()
    if not text:
        return text

    m = _PLAY_IT_SUFFIX_RE.match(text)
    if m and len(m.group("body").strip()) >= 3:
        return f"play {m.group('body').strip()}"

    m = _PLAY_IT_PREFIX_RE.match(text)
    if m and len(m.group("body").strip()) >= 3:
        return f"play {m.group('body').strip()}"

    if _BUY_BY_RE.search(text) and "videoclub" in text.lower():
        return _BUY_BY_RE.sub("by", text)

    prev = None
    while prev != text:
        prev = text
        text = _FILLER_PREFIX_RE.sub("", text).strip()

    return text
