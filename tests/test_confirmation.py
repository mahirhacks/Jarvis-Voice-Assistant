"""Tests for confirmation phrase matching."""

from src.core.utils import match_confirmation_phrase, confidence_gate


def test_affirmative():
    assert match_confirmation_phrase("yes") == "affirmative"
    assert match_confirmation_phrase("yeah") == "affirmative"


def test_negative():
    assert match_confirmation_phrase("no") == "negative"
    assert match_confirmation_phrase("cancel") == "negative"


def test_unknown():
    assert match_confirmation_phrase("maybe later") == "unknown"


def test_confidence_gate():
    assert confidence_gate(0.9) == "auto_play"
    assert confidence_gate(0.7) == "confirm"
    assert confidence_gate(0.5) == "clarify"
