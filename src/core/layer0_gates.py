"""Layer 0 pre-checks — when to skip web search and quick verb hints."""

import re
from typing import Optional

from src.core.fast_router import BY_ARTIST_RE, OFFICIAL_VIDEO_RE
from src.music.candidate_ranker import clean_query
from src.tools.apps import resolve_app_key

_SKIP_WEB_VERBS = frozenset({
    "pause", "skip", "volume", "lock", "shutdown", "restart", "read", "navigate",
})

_MEDIA_WORDS_RE = re.compile(
    r"^\s*(pause|resume|next|previous|forward|backward|mute|unmute|"
    r"volume\s+up|volume\s+down|lock|shutdown|restart|reboot)\s*\.?$",
    re.IGNORECASE,
)

_PLAY_HINT_RE = re.compile(
    r"\b(song|music|youtube|spotify|album|track|artist|band|play|listen)\b",
    re.IGNORECASE,
)


def quick_verb_hint(transcript: str, settings: dict | None = None) -> Optional[str]:
    """Fast heuristic verb before any LLM or web call."""
    raw = transcript.strip()
    lower = raw.lower()

    if _MEDIA_WORDS_RE.match(lower):
        if re.search(r"\b(pause|resume)\b", lower):
            return "pause"
        if re.search(r"\b(next|forward|skip)\b", lower):
            return "skip"
        if re.search(r"\b(previous|backward)\b", lower):
            return "skip"
        if "volume" in lower or "mute" in lower:
            return "volume"
        if "lock" in lower:
            return "lock"
        if "shutdown" in lower or "shut down" in lower:
            return "shutdown"
        if "restart" in lower or "reboot" in lower:
            return "restart"

    cleaned = clean_query(raw)
    if cleaned and resolve_app_key(cleaned, settings):
        return "open"
    if lower.startswith(("open ", "launch ", "start ")):
        tail = clean_query(raw)
        if tail and resolve_app_key(tail, settings):
            return "open"

    if BY_ARTIST_RE.match(cleaned or raw) or OFFICIAL_VIDEO_RE.match(raw):
        return "play"
    if _PLAY_HINT_RE.search(raw):
        return "play"
    if lower.endswith("?"):
        return "answer"

    return None


def should_skip_web_search(transcript: str, verb_hint: Optional[str], settings: dict | None = None) -> bool:
    """True when Layer 0 web search is unlikely to help."""
    if verb_hint in _SKIP_WEB_VERBS:
        return True

    cleaned = clean_query(transcript) or transcript.strip()
    if verb_hint == "open" and resolve_app_key(cleaned, settings):
        return True

    if verb_hint == "workflow":
        return True

    return False


def build_search_query(transcript: str, verb_hint: Optional[str]) -> str:
    """Craft a smarter search query from transcript + verb hint."""
    q = transcript.strip()
    if verb_hint == "play":
        if "song" not in q.lower() and "youtube" not in q.lower():
            return f"{q} song artist"
    if verb_hint == "answer":
        return q.rstrip("?")
    if verb_hint == "open" and not q.lower().startswith("open"):
        return f"{q} app download"
    return q
