"""Music routing via vocabulary — no LLM."""

import logging
import re
from typing import Any, Optional

from rapidfuzz import fuzz

from src.core.models import VocabularyMatch
from src.music.vocabulary_matcher import load_vocabulary, match_vocabulary

logger = logging.getLogger(__name__)

_VOCAB_FUZZY_THRESHOLD = 0.72
_VOCAB_FUZZY_STRICT = 0.88
_QUESTION_START = re.compile(
    r"^\s*(what|who|how|when|where|why|which)\b",
    re.IGNORECASE,
)


def vocabulary_route(
    transcript: str,
    settings: dict | None = None,
) -> Optional[dict[str, Any]]:
    """Route to search_youtube when transcript matches music vocabulary."""
    text = transcript.strip()
    if _QUESTION_START.match(text):
        return None

    path = "config/music_vocabulary.json"
    if settings:
        path = settings.get("music_vocabulary_path", path)
    vocabulary = load_vocabulary(path)
    if not vocabulary:
        return None

    matches = list(match_vocabulary(transcript, vocabulary))
    if not matches:
        normalised = transcript.lower().strip().rstrip(".")
        word_count = len(normalised.split())
        threshold = _VOCAB_FUZZY_THRESHOLD
        if word_count > 6:
            threshold = _VOCAB_FUZZY_STRICT
        for key, entry in vocabulary.items():
            if not isinstance(entry, dict):
                continue
            canonical = str(entry.get("canonical_form", ""))
            if not canonical:
                continue
            ratio = fuzz.partial_ratio(key.lower(), normalised) / 100.0
            if ratio >= threshold:
                matches.append(VocabularyMatch(
                    matched_term=key,
                    canonical_form=canonical,
                    score=ratio,
                ))

    if not matches:
        return None

    lower = transcript.lower()
    if " by " in lower or lower.startswith("by "):
        by_matches = [m for m in matches if " by " in m.canonical_form.lower()]
        if by_matches:
            best = max(by_matches, key=lambda m: m.score)
        else:
            best = max(matches, key=lambda m: m.score)
    else:
        best = max(matches, key=lambda m: m.score)
    query = best.canonical_form
    logger.info(
        "[VocabRoute] '%s' → '%s' (%.2f)",
        transcript[:60], query, best.score,
    )
    return {
        "tool_name": "search_youtube",
        "arguments": {"query": query},
        "confidence": min(100.0, best.score * 100),
        "needs_confirmation": False,
        "tier": "vocab",
    }
