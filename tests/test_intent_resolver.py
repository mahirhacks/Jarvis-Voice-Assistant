"""Tests for intent resolution parsing and routing."""

import json

from src.core.intent_resolver import (
    extract_bracket_answer,
    infer_intent_heuristic,
    intent_from_bracket_answer,
    intent_to_tool_route,
    meets_confidence,
    parse_structured_intent,
)


def test_parse_json_intent_play():
    raw = json.dumps({
        "action": "play",
        "objective": "En nuit by VideoClub",
        "confidence": "high",
    })
    intent = parse_structured_intent(raw)
    assert intent is not None
    assert intent.action == "play"
    assert intent.objective == "En nuit by VideoClub"
    assert intent.confidence == "high"


def test_parse_action_objective_lines():
    raw = "Action: Play\nObjective: En nuit by VideoClub\nConfidence: high"
    intent = parse_structured_intent(raw)
    assert intent is not None
    assert intent.action == "play"
    assert intent.objective == "En nuit by VideoClub"


def test_heuristic_title_by_artist():
    intent = infer_intent_heuristic("En nuit by VideoClub", "VideoClub song on YouTube")
    assert intent is not None
    assert intent.action == "play"
    assert "En nuit" in intent.objective
    assert "VideoClub" in intent.objective


def test_intent_to_search_youtube():
    from src.core.models import ResolvedIntent

    intent = ResolvedIntent(action="play", objective="Play Date by Melanie Martinez")
    route = intent_to_tool_route(intent, {})
    assert route is not None
    assert route["tool_name"] == "search_youtube"
    assert route["arguments"]["query"] == "Play Date by Melanie Martinez"


def test_meets_confidence():
    from src.core.models import ResolvedIntent

    assert meets_confidence(ResolvedIntent(action="play", confidence="high"), "medium")
    assert not meets_confidence(ResolvedIntent(action="play", confidence="low"), "medium")


def test_extract_bracket_answer():
    text = "Some intro [start] play|En nuit by VideoClub [end] more text"
    assert extract_bracket_answer(text) == "play|En nuit by VideoClub"


def test_intent_from_bracket_play_pipe():
    intent = intent_from_bracket_answer("play|En nuit by VideoClub")
    assert intent is not None
    assert intent.action == "play"
    assert intent.objective == "En nuit by VideoClub"
    assert intent.confidence == "high"


def test_intent_from_bracket_plain_title():
    intent = intent_from_bracket_answer("Relaxing Jazz Music on YouTube")
    assert intent is not None
    assert intent.action == "play"
