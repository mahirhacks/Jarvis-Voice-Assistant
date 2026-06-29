"""Tests for candidate ranker."""

from src.music.candidate_ranker import clean_query, rank_candidates, score_entry


def test_clean_query_strips_play():
    assert clean_query("play me the song hello") == "hello"


def test_clean_query_strips_youtube_suffix():
    assert clean_query("Played It by Melania Martinez in YouTube") == (
        "Played It by Melania Martinez"
    )


def test_clean_query_preserves_play_date():
    assert clean_query("play play date by melanie martinez") == (
        "play date by melanie martinez"
    )
    assert clean_query("play playdate by malenie martizen") == (
        "playdate by malenie martizen"
    )


def test_rank_prefers_matching_title():
    entries = [
        {"title": "Hello Official Audio", "uploader": "Artist Topic", "id": "a",
         "duration": 200, "view_count": 1000000},
        {"title": "Random Karaoke Hello", "uploader": "Covers", "id": "b",
         "duration": 200, "view_count": 100},
    ]
    ranked = rank_candidates(entries, "hello")
    assert ranked
    assert ranked[0][0]["id"] == "a"


def test_normalize_score():
    from src.music.candidate_ranker import normalize_score_to_confidence
    assert normalize_score_to_confidence(82) == 1.0
    assert normalize_score_to_confidence(41) == 0.5
