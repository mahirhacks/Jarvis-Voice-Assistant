"""Tests for layered pipeline (L0/L1/L2 helpers)."""

from src.core.layer0_gates import build_search_query, quick_verb_hint, should_skip_web_search
from src.core.layer1_intent import (
    ACTION_VERBS,
    extract_one_word,
    extract_phrase,
    parse_json_field,
    verb_to_tool,
)
from src.core.layer_pipeline import (
    build_execution_brief,
    needs_layer2_llm,
    plan_to_direct_route,
    resolve_tool_for_plan,
)
from src.core.models import LayerPlan
from src.core.tool_schemas import narrow_tool_schemas


def test_extract_one_word_verb():
    assert extract_one_word("play", ACTION_VERBS) == "play"
    assert extract_one_word("The verb is: play", ACTION_VERBS) == "play"


def test_parse_json_verb():
    assert parse_json_field('{"verb":"play"}', "verb", ACTION_VERBS) == "play"


def test_parse_json_phrase_and_confidence():
    raw = '{"phrase":"En nuit by VideoClub","confidence":"high"}'
    assert parse_json_field(raw, "phrase") == "En nuit by VideoClub"
    assert parse_json_field(raw, "confidence", ("low", "medium", "high")) == "high"


def test_verb_to_tool():
    assert verb_to_tool("play") == "search_youtube"
    assert verb_to_tool("answer") == "speak"


def test_quick_verb_hint_play_by_artist():
    assert quick_verb_hint("En nuit by VideoClub") == "play"


def test_should_skip_web_for_pause():
    assert should_skip_web_search("pause", "pause") is True


def test_build_search_query_play():
    q = build_search_query("en newt video club", "play")
    assert "song" in q.lower()


def test_build_execution_brief():
    plan = LayerPlan(
        transcript="en newt video club",
        verb="play",
        phrase="En nuit by VideoClub",
        tool="search_youtube",
        search_context="1. En nuit - VideoClub",
        confidence="high",
    )
    brief = build_execution_brief(plan, {})
    assert "En nuit by VideoClub" in brief
    assert "search_youtube" in brief


def test_plan_to_direct_route_youtube():
    plan = LayerPlan(
        transcript="en newt",
        verb="play",
        phrase="En nuit by VideoClub",
        tool="search_youtube",
        confidence="high",
    )
    route = plan_to_direct_route(plan, {})
    assert route is not None
    assert route["tool_name"] == "search_youtube"


def test_plan_to_direct_route_answer():
    plan = LayerPlan(
        transcript="what is the capital of france",
        verb="answer",
        phrase="The capital of France is Paris.",
        tool="speak",
        confidence="high",
    )
    route = plan_to_direct_route(plan, {})
    assert route["tool_name"] == "speak"
    assert "Paris" in route["arguments"]["text"]


def test_needs_layer2_high_confidence_play():
    plan = LayerPlan(
        transcript="en newt",
        verb="play",
        phrase="En nuit by VideoClub",
        tool="search_youtube",
        confidence="high",
    )
    assert needs_layer2_llm(plan, {}) is False


def test_needs_layer2_ambiguous_volume():
    plan = LayerPlan(
        transcript="volume",
        verb="volume",
        phrase="",
        tool="media_volume_up",
        confidence="medium",
    )
    assert needs_layer2_llm(plan, {}) is True


def test_resolve_tool_volume_down():
    plan = LayerPlan(
        transcript="volume down",
        verb="volume",
        phrase="volume down",
        tool="media_volume_up",
        confidence="high",
    )
    assert resolve_tool_for_plan(plan, {}) == "media_volume_down"


def test_narrow_tool_schemas():
    schemas = narrow_tool_schemas("search_youtube", {})
    names = [s["function"]["name"] for s in schemas]
    assert names[0] == "search_youtube"
    assert len(names) <= 4


def test_plan_none_returns_none():
    plan = LayerPlan(transcript="hmm", verb="none", phrase="", tool="none")
    assert plan_to_direct_route(plan, {}) is None
