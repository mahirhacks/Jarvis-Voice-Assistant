"""Tests for fast router."""

import pytest

from src.core.fast_router import route


def test_pause_fast_path():
    result = route("pause")
    assert result is not None
    assert result["tool_name"] == "media_play_pause"
    assert result["tier"] == 0


def test_play_song_fast_path():
    result = route("play bohemian rhapsody")
    assert result is not None
    assert result["tool_name"] == "search_youtube"
    assert result["arguments"]["query"] == "bohemian rhapsody"


def test_title_by_artist_fast_path():
    result = route("En nuit by VideoClub")
    assert result is not None
    assert result["tool_name"] == "search_youtube"
    assert result["arguments"]["query"] == "En nuit by VideoClub"


def test_official_video_fast_path():
    result = route("Murtaza Hum, Official Video")
    assert result is not None
    assert result["tool_name"] == "search_youtube"


def test_study_mode_workflow():
    result = route("study mode")
    assert result is not None
    assert result["tool_name"] == "run_workflow"
    assert result["arguments"]["name"] == "study_mode"


def test_shutdown_needs_confirmation():
    result = route("shutdown")
    assert result is not None
    assert result["needs_confirmation"] is True


def test_restart_laptop_fast_path():
    result = route("restart laptop")
    assert result is not None
    assert result["tool_name"] == "restart"
    assert result["needs_confirmation"] is True


def test_open_task_manager_fast_path():
    result = route("open task manager")
    assert result is not None
    assert result["tool_name"] == "open_app"
    assert result["arguments"]["app_key"] == "task_manager"


def test_open_launch_prefix():
    result = route("launch calculator")
    assert result is not None
    assert result["tool_name"] == "open_app"
    assert result["arguments"]["app_key"] == "calculator"


def test_unknown_returns_none():
    assert route("xyzzy completely unknown phrase") is None


def test_long_phrase_play_it_not_pause():
    result = route("i think they call this love, play it.")
    assert result is None or result["tool_name"] != "media_play_pause"


def test_ask_manager_not_task_manager():
    assert route("ask manager.") is None
