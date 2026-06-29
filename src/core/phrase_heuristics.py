"""Heuristic routes for music-shaped phrases without 'play' or 'by'."""

import re
from typing import Any, Optional

_QUESTION_START = re.compile(
    r"^\s*(what|who|how|when|where|why|which|is|are|do|does|did|can|could|would)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"\b[\w.-]+\.(com|org|net|io|dev|app|tv|co|uk|github)\b",
    re.IGNORECASE,
)
_SEARCH_PREFIX_RE = re.compile(r"^\s*(search|google|look up|find)\s+", re.IGNORECASE)
_GO_PREFIX_RE = re.compile(r"^\s*go\s+to\s+", re.IGNORECASE)

_BLOCK_WORDS = frozenset({
    "open", "launch", "start", "run", "shutdown", "restart", "lock",
    "search", "google", "task", "manager", "explorer", "settings",
    "chrome", "vscode", "notepad", "calculator", "jarvis",
})

_WEB_ALIASES = {
    "reddit": "https://reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "facebook": "https://facebook.com",
    "instagram": "https://instagram.com",
    "linkedin": "https://linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "wikipedia": "https://wikipedia.org",
    "amazon": "https://amazon.com",
    "netflix": "https://netflix.com",
    "spotify": "https://open.spotify.com",
    "twitch": "https://twitch.tv",
    "discord": "https://discord.com",
}

_MEDIA_WORDS = frozenset({
    "stop", "pause", "resume", "next", "previous", "skip", "mute", "unmute",
    "volume", "louder", "quieter", "silence", "quiet", "fullscreen",
})


def bare_music_route(transcript: str) -> Optional[dict[str, Any]]:
    """Bare 'title artist' or short song-like phrase → search_youtube."""
    text = transcript.strip().rstrip(".")
    if len(text) < 3:
        return None
    if _QUESTION_START.match(text):
        return None
    if _URL_RE.search(text):
        return None
    if _SEARCH_PREFIX_RE.match(text) or _GO_PREFIX_RE.match(text):
        return None

    lower = text.lower()
    words = lower.split()
    if len(words) > 14:
        return None
    if len(words) == 1:
        if len(words[0]) < 4 or words[0] in _BLOCK_WORDS or words[0] in _MEDIA_WORDS:
            return None
    elif len(words) < 2:
        return None
    if words[0] in _MEDIA_WORDS:
        return None
    if words[0] in _BLOCK_WORDS and words[0] not in ("play",):
        return None
    if lower in _MEDIA_WORDS:
        return None

    # Song-like: multi-word, not a pure app/command phrase
    if any(w in _BLOCK_WORDS for w in words):
        blocked = [w for w in words if w in _BLOCK_WORDS]
        if len(blocked) >= 2 or (blocked and blocked[0] == words[0]):
            return None

    return {
        "tool_name": "search_youtube",
        "arguments": {"query": text},
        "confidence": 75.0,
        "needs_confirmation": False,
        "tier": "bare_music",
    }


def question_route(transcript: str) -> Optional[dict[str, Any]]:
    """Factual questions → speak (Layer 2 can refine) or search."""
    text = transcript.strip().rstrip("?.").strip()
    if not _QUESTION_START.match(text):
        return None
    return {
        "tool_name": "browser_search",
        "arguments": {"query": text},
        "confidence": 70.0,
        "needs_confirmation": False,
        "tier": "question",
    }


def web_alias_route(transcript: str) -> Optional[dict[str, Any]]:
    """open <site> when not a registered app → browser."""
    text = transcript.strip().lower()
    for prefix in ("open ", "launch ", "start "):
        if text.startswith(prefix) and len(text) > len(prefix):
            target = text[len(prefix):].strip()
            url = _WEB_ALIASES.get(target)
            if url:
                return {
                    "tool_name": "browser_go_to_website",
                    "arguments": {"url": url},
                    "confidence": 90.0,
                    "needs_confirmation": False,
                    "tier": "web_alias",
                }
    return None


def url_route(transcript: str) -> Optional[dict[str, Any]]:
    text = transcript.strip()
    m = _URL_RE.search(text)
    if not m:
        return None
    url = m.group(0)
    if not url.startswith("http"):
        url = "https://" + url
    return {
        "tool_name": "browser_go_to_website",
        "arguments": {"url": url},
        "confidence": 90.0,
        "needs_confirmation": False,
        "tier": "url",
    }


def search_prefix_route(transcript: str) -> Optional[dict[str, Any]]:
    text = transcript.strip()
    m = _SEARCH_PREFIX_RE.match(text)
    if m:
        query = text[m.end():].strip()
        if query:
            return {
                "tool_name": "browser_search",
                "arguments": {"query": query},
                "confidence": 85.0,
                "needs_confirmation": False,
                "tier": "search",
            }
    m = _GO_PREFIX_RE.match(text)
    if m:
        target = text[m.end():].strip()
        if target:
            url = target if _URL_RE.search(target) else f"https://{target}"
            return {
                "tool_name": "browser_go_to_website",
                "arguments": {"url": url},
                "confidence": 85.0,
                "needs_confirmation": False,
                "tier": "url",
            }
    if text.lower().startswith("open ") and _URL_RE.search(text):
        return url_route(text.replace("open ", "", 1))
    return None
