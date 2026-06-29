"""Music vocabulary loading and matching."""

import json
import logging
from typing import Any

from rapidfuzz import fuzz

from src.core.models import VocabularyMatch

logger = logging.getLogger(__name__)
_FUZZY_THRESHOLD = 0.7


def load_vocabulary(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load vocabulary from %s: %s", path, exc)
        return {}


def match_vocabulary(transcript: str, vocabulary: dict[str, Any]) -> list[VocabularyMatch]:
    normalised = transcript.lower()
    matches: list[VocabularyMatch] = []
    for key, entry in vocabulary.items():
        if not isinstance(entry, dict) or "canonical_form" not in entry:
            continue
        key_lower = key.lower()
        if key_lower in normalised:
            matches.append(VocabularyMatch(
                matched_term=key, canonical_form=entry["canonical_form"],
                match_type=entry.get("type", "correction"), score=1.0,
            ))
            continue
        ratio = fuzz.ratio(key_lower, normalised) / 100.0
        if ratio > _FUZZY_THRESHOLD:
            matches.append(VocabularyMatch(
                matched_term=key, canonical_form=entry["canonical_form"],
                match_type=entry.get("type", "correction"), score=ratio,
            ))
    return matches
