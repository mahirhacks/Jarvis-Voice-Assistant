"""Local YouTube candidate ranking without LLM."""

import logging
import math
import re

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

OFFICIAL_KEYWORDS = {"official", "official music video", "official audio", "vevo", "topic"}
HARD_PENALTY_KEYWORDS = {"karaoke", "cover", "instrumental", "remix", "slowed", "reverb"}
SOFT_PENALTY_KEYWORDS = {"lyrics", "lyric video"}
REJECT_KEYWORDS = {
    "mujra", "stage", "drama", "episode", "trailer", "reaction", "prank",
    "thoughts", "review", "recap", "clip", "arc", "theory", "amv",
    "documentary", "interview", "movie scene",
}
LONG_ALLOW_KEYWORDS = {"mix", "playlist", "live", "podcast", "lecture", "full"}
MUSIC_CONTEXT_PRESERVE = {"from", "movie", "anime", "opening", "ending", "ost", "soundtrack", "nasheed"}
FILLER_WORDS = {"me", "the", "song", "a", "some", "that", "this"}
YOUTUBE_SUFFIXES = (" in youtube", " on youtube", " from youtube")
_COMMAND_PREFIXES = (
    "play me the song ",
    "play the song ",
    "play me ",
    "play song ",
    "listen to ",
    "put on ",
)


def _strip_command_prefix(text: str) -> str:
    """Strip voice-command prefixes once; preserve song titles like 'Play Date'."""
    result = text.strip()
    lower = result.lower()
    for prefix in _COMMAND_PREFIXES:
        if lower.startswith(prefix):
            return result[len(prefix) :].strip()
    if lower.startswith("play "):
        remainder = result[5:].strip()
        rem_lower = remainder.lower()
        if rem_lower.startswith("play date") or rem_lower.startswith("playdate"):
            return remainder
        return remainder
    return result


def clean_query(raw: str) -> str:
    result = _strip_command_prefix(raw.strip().rstrip("."))
    lower = result.lower()
    for suffix in YOUTUBE_SUFFIXES:
        if lower.endswith(suffix):
            result = result[: -len(suffix)].strip()
            lower = result.lower()
    return result if result else raw.strip()


def parse_song_artist(query: str) -> tuple[str, str]:
    lower = query.lower()
    for sep in (" by ", " from "):
        if sep in lower:
            idx = lower.index(sep)
            return query[:idx].strip(), query[idx + len(sep):].strip()
    return query, ""


def _should_reject(entry: dict, query_lower: str) -> str | None:
    title = entry.get("title", "").lower()
    duration = entry.get("duration") or 0
    for kw in REJECT_KEYWORDS:
        if kw in title and kw not in query_lower:
            return f"reject keyword '{kw}'"
    if duration > 900 and not any(kw in query_lower for kw in LONG_ALLOW_KEYWORDS):
        if not any(kw in query_lower for kw in MUSIC_CONTEXT_PRESERVE):
            return f"duration {duration}s too long"
    return None


def score_entry(entry: dict, query: str, song_title: str, artist: str) -> tuple[float, dict]:
    title = entry.get("title", "").lower()
    channel = entry.get("uploader", entry.get("channel", "")).lower()
    duration = entry.get("duration") or 0
    views = entry.get("view_count") or 0
    q = query.lower()

    song_score = fuzz.WRatio(song_title.lower(), title) if song_title else 0
    artist_score = max(
        fuzz.WRatio(artist.lower(), title),
        fuzz.WRatio(artist.lower(), channel),
    ) if artist else 0
    full_score = fuzz.WRatio(q, f"{title} {channel}")
    composite = song_score * 0.50 + artist_score * 0.25 + full_score * 0.15

    official_bonus = 0
    combined = f"{title} {channel}"
    for kw in OFFICIAL_KEYWORDS:
        if kw in combined:
            official_bonus = 10
            break

    view_bonus = min(math.log10(views), 5) if views > 0 else 0
    penalty = 0
    for kw in HARD_PENALTY_KEYWORDS:
        if kw in title and kw not in q:
            penalty += 20
    for kw in SOFT_PENALTY_KEYWORDS:
        if kw in title and kw not in q:
            penalty += 5
    if duration > 0 and not (60 <= duration <= 600):
        penalty += 5

    final = composite + official_bonus + view_bonus - penalty
    return final, {
        "song_score": song_score, "artist_score": artist_score,
        "full_score": full_score, "final": round(final, 1),
    }


def rank_candidates(entries: list[dict], query: str) -> list[tuple[dict, float]]:
    q = query.lower()
    song_title, artist = parse_song_artist(query)
    results = []
    for entry in entries:
        reason = _should_reject(entry, q)
        if reason:
            continue
        score, details = score_entry(entry, query, song_title, artist)
        if song_title and details["song_score"] < 70:
            continue
        results.append((entry, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def entry_to_candidate_dict(entry: dict) -> dict:
    return {
        "title": entry.get("title", ""),
        "video_id": entry.get("id", entry.get("video_id", "")),
        "uploader": entry.get("uploader", entry.get("channel", "")),
        "duration": int(entry.get("duration") or 0),
        "view_count": int(entry.get("view_count") or 0),
        "webpage_url": entry.get("webpage_url", ""),
    }


def normalize_score_to_confidence(score: float, auto_thresh: float = 82) -> float:
    return min(1.0, max(0.0, score / max(auto_thresh, 1)))
