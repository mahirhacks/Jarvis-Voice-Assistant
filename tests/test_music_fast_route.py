"""Tests for vocabulary fast route."""

from src.core.music_fast_route import vocabulary_route


def test_vocab_en_nuit():
    route = vocabulary_route("En nuit.", {})
    assert route is not None
    assert route["tool_name"] == "search_youtube"
    assert "En nuit" in route["arguments"]["query"]


def test_vocab_they_call_this_love():
    route = vocabulary_route("They call this love.", {})
    assert route is not None
    assert "Call This Love" in route["arguments"]["query"]
