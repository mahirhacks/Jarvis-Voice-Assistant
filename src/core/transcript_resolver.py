"""Resolve a transcript to a tool route without voice capture (dry-run friendly)."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.core.fast_router import route as fast_route
from src.core.intent_resolver import infer_intent_heuristic, intent_to_tool_route, meets_confidence
from src.core.layer_pipeline import LayerPipeline, needs_layer2_llm, plan_to_direct_route
from src.core.music_fast_route import vocabulary_route
from src.core.phrase_heuristics import (
    bare_music_route,
    question_route,
    search_prefix_route,
    url_route,
    web_alias_route,
)
from src.core.transcript_normalize import is_wake_echo, normalize_transcript

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    transcript: str
    normalized: str
    path: str
    route: Optional[dict[str, Any]] = None
    plan_verb: str = ""
    plan_phrase: str = ""
    plan_confidence: str = ""
    ignored: bool = False
    message: str = ""


def resolve_transcript(
    transcript: str,
    settings: dict,
    layer_pipeline: LayerPipeline | None = None,
    search_context: str = "",
) -> ResolveResult:
    """Classify transcript through tier0 → vocab → pre-heuristic → layers."""
    raw = transcript.strip()
    if not raw:
        return ResolveResult(transcript=raw, normalized=raw, path="empty", message="empty")

    if is_wake_echo(raw):
        return ResolveResult(
            transcript=raw, normalized=raw, path="ignored",
            ignored=True, message="wake echo",
        )

    normalized = normalize_transcript(raw)

    fast = fast_route(
        normalized,
        settings.get("fuzzy_medium_confidence", 65),
        settings.get("fuzzy_high_confidence", 92),
    )
    if fast:
        return ResolveResult(
            transcript=raw, normalized=normalized, path="fast",
            route=fast, message=fast["tool_name"],
        )

    vocab = vocabulary_route(normalized, settings)
    if vocab:
        return ResolveResult(
            transcript=raw, normalized=normalized, path="vocab",
            route=vocab, message=vocab["tool_name"],
        )

    for route_fn, path_name in (
        (url_route, "url"),
        (web_alias_route, "web_alias"),
        (search_prefix_route, "search"),
        (question_route, "question"),
        (bare_music_route, "bare_music"),
    ):
        hit = route_fn(normalized)
        if hit:
            return ResolveResult(
                transcript=raw, normalized=normalized, path=path_name,
                route=hit, message=hit["tool_name"],
            )

    hint = infer_intent_heuristic(normalized, search_context)
    if hint and meets_confidence(hint, "medium"):
        route = intent_to_tool_route(hint, settings)
        if route:
            return ResolveResult(
                transcript=raw, normalized=normalized, path="heuristic",
                route=route, message=route["tool_name"],
            )

    if layer_pipeline and settings.get("layer_pipeline_enabled", True):
        gather = layer_pipeline.gather(normalized)
        plan = layer_pipeline.find_intent(gather)
        if plan and plan.verb != "none" and plan.tool != "none":
            route = plan_to_direct_route(plan, settings)
            if route and not needs_layer2_llm(plan, settings):
                return ResolveResult(
                    transcript=raw, normalized=normalized, path="layer1-direct",
                    route=route,
                    plan_verb=plan.verb,
                    plan_phrase=plan.phrase,
                    plan_confidence=plan.confidence,
                    message=route["tool_name"],
                )
            if route:
                return ResolveResult(
                    transcript=raw, normalized=normalized, path="layer1-needs-llm",
                    route=route,
                    plan_verb=plan.verb,
                    plan_phrase=plan.phrase,
                    plan_confidence=plan.confidence,
                    message=f"{plan.tool} (needs layer2)",
                )

        hint2 = infer_intent_heuristic(normalized, gather.context)
        if hint2 and meets_confidence(hint2, "medium"):
            route = intent_to_tool_route(hint2, settings)
            if route:
                return ResolveResult(
                    transcript=raw, normalized=normalized, path="heuristic+context",
                    route=route, message=route["tool_name"],
                )

    return ResolveResult(
        transcript=raw, normalized=normalized, path="unresolved",
        message="unresolved",
    )
