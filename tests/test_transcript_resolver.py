"""Tests for full transcript resolution (no tool execution)."""

from src.core.transcript_resolver import resolve_transcript


def test_resolve_play_by_artist():
    r = resolve_transcript("Play En nuit by VideoClub.", {})
    assert r.path == "fast"
    assert r.route["tool_name"] == "search_youtube"


def test_resolve_bare_en_nuit_vocab():
    r = resolve_transcript("En nuit.", {})
    assert r.path in ("vocab", "fast", "heuristic")
    assert r.route["tool_name"] == "search_youtube"


def test_resolve_play_it_suffix():
    r = resolve_transcript("I think they call this love, play it.", {})
    assert r.normalized.startswith("play ")
    assert r.route["tool_name"] == "search_youtube"


def test_resolve_ignores_yes():
    r = resolve_transcript("Yes.", {})
    assert r.ignored is True


def test_resolve_open_task_manager():
    r = resolve_transcript("Open Task Manager.", {})
    assert r.path == "fast"
    assert r.route["tool_name"] == "open_app"


def test_resolve_ask_manager_not_task_manager():
    r = resolve_transcript("Ask Manager.", {})
    assert r.path != "fast" or r.route.get("tool_name") != "open_app"
