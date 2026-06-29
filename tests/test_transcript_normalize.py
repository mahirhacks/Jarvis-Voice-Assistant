"""Tests for transcript normalization."""

from src.core.transcript_normalize import is_wake_echo, normalize_transcript


def test_wake_echo_yes():
    assert is_wake_echo("Yes.")
    assert is_wake_echo("yeah")


def test_normalize_play_it_suffix():
    assert normalize_transcript("They call this love, play it.") == (
        "play They call this love"
    )


def test_normalize_buy_videoclub():
    assert "by" in normalize_transcript("Buy VideoClub.").lower()
    assert "videoclub" in normalize_transcript("Buy VideoClub.").lower()
