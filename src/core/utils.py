"""Confidence gating and confirmation phrase matching."""

from typing import FrozenSet, Literal


def confidence_gate(
    confidence: float,
    auto_play_threshold: float = 0.85,
    confirm_threshold: float = 0.65,
) -> Literal["auto_play", "confirm", "clarify"]:
    if confidence >= auto_play_threshold:
        return "auto_play"
    if confidence >= confirm_threshold:
        return "confirm"
    return "clarify"


AFFIRMATIVE_PHRASES: FrozenSet[str] = frozenset({
    "yes", "yeah", "yep", "yup", "confirm", "okay", "ok", "okey",
    "alright", "sure", "do it", "play it", "go ahead", "correct", "y",
})

NEGATIVE_PHRASES: FrozenSet[str] = frozenset({
    "no", "nope", "cancel", "stop", "don't", "not that one",
    "wrong", "never mind", "n",
})


def match_confirmation_phrase(transcript: str) -> Literal["affirmative", "negative", "unknown"]:
    normalized = transcript.lower().strip()
    if normalized in AFFIRMATIVE_PHRASES:
        return "affirmative"
    if normalized in NEGATIVE_PHRASES:
        return "negative"
    return "unknown"
