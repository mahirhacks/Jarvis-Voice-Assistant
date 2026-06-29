"""Resolve unclear voice commands via web search + structured LLM output."""

import json
import logging
import re
from typing import Any, Optional

from src.core.models import ResolvedIntent
from src.music.candidate_ranker import clean_query
from src.tools.apps import resolve_app_key

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({
    "play", "open_url", "search_web", "open_app", "run_workflow", "answer", "none",
})

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


_BY_ARTIST_RE = re.compile(r"^(.+?)\s+by\s+(.+)$")
_MUSIC_HINT_RE = re.compile(
    r"\b(song|music|youtube|spotify|album|single|lyrics|official|track|artist|band|feat\.?)\b",
    re.IGNORECASE,
)


_START_END_RE = re.compile(r"\[start\]\s*(.*?)\s*\[end\]", re.DOTALL | re.IGNORECASE)


def build_google_ai_prompt(transcript: str, settings: dict) -> str:
    """Wrap user speech in a Google-prompt that asks for [start]...[end] structured output."""
    template = settings.get(
        "google_ai_query_template",
        (
            'reply me only the exact answer in this format to my question '
            '[start] answer [end]. '
            'User said: "{query}". '
            'If this is a song/music request reply: play|Song Title by Artist. '
            'If opening a website reply: open_url|https://.... '
            'If a web search reply: search_web|search terms. '
            'If opening an app reply: open_app|app name. '
            'If a factual question reply: answer|short answer. '
            'If unclear reply: none|'
        ),
    )
    return template.format(query=transcript.strip())


def extract_bracket_answer(text: str) -> Optional[str]:
    """Pull answer from Google AI [start]...[end] format."""
    if not text:
        return None
    match = _START_END_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def intent_from_bracket_answer(answer: str) -> Optional[ResolvedIntent]:
    """Convert [start] payload into a ResolvedIntent."""
    if not answer:
        return None

    cleaned = answer.strip().strip('"').strip("'")

    if "|" in cleaned:
        action, objective = cleaned.split("|", 1)
        action = action.strip().lower()
        objective = objective.strip()
        if action in _VALID_ACTIONS and objective:
            return ResolvedIntent(
                action=action,
                objective=objective,
                confidence="high",
                raw_response=f"google_ai:{cleaned}",
            )
        if action in _VALID_ACTIONS and action == "none":
            return ResolvedIntent(
                action="none",
                objective="",
                confidence="high",
                raw_response=f"google_ai:{cleaned}",
            )

    lower = cleaned.lower()
    if "youtube.com" in lower or "youtu.be" in lower:
        return ResolvedIntent(
            action="open_url",
            objective=cleaned,
            confidence="high",
            raw_response=f"google_ai:{cleaned}",
        )

    if cleaned:
        return ResolvedIntent(
            action="play",
            objective=cleaned,
            confidence="high",
            raw_response=f"google_ai:{cleaned}",
        )

    return None


def infer_intent_heuristic(transcript: str, search_context: str = "") -> Optional[ResolvedIntent]:
    """Fallback when LLM returns empty — pattern + search + vocabulary signals."""
    from src.music.vocabulary_matcher import load_vocabulary, match_vocabulary

    query = clean_query(transcript).strip().rstrip(".")
    if len(query) < 2:
        return None

    music_in_search = bool(_MUSIC_HINT_RE.search(search_context))
    by_match = _BY_ARTIST_RE.match(query)
    if by_match:
        title, artist = by_match.group(1).strip(), by_match.group(2).strip()
        if title and artist and (music_in_search or len(title.split()) <= 8):
            objective = f"{title} by {artist}"
            conf = "high" if music_in_search else "medium"
            return ResolvedIntent(
                action="play",
                objective=objective,
                confidence=conf,
                raw_response="heuristic:title_by_artist",
            )

    if query.lower().startswith(("play ", "listen to ", "put on ")):
        objective = clean_query(query)
        if objective:
            return ResolvedIntent(
                action="play",
                objective=objective,
                confidence="medium",
                raw_response="heuristic:play_prefix",
            )

    if re.search(r"\b(official video|official audio|music video|lyrics)\b", query, re.IGNORECASE):
        return ResolvedIntent(
            action="play",
            objective=query.rstrip("."),
            confidence="medium",
            raw_response="heuristic:official_video",
        )

    vocabulary = load_vocabulary("config/music_vocabulary.json")
    vocab_matches = match_vocabulary(transcript, vocabulary)
    if vocab_matches:
        best = max(vocab_matches, key=lambda m: m.score)
        if best.score >= 0.7:
            return ResolvedIntent(
                action="play",
                objective=best.canonical_form,
                confidence="high" if best.score >= 0.85 else "medium",
                raw_response="heuristic:vocabulary",
            )

    if music_in_search and 2 <= len(query.split()) <= 10:
        return ResolvedIntent(
            action="play",
            objective=query,
            confidence="medium",
            raw_response="heuristic:music_search_context",
        )

    if len(query.split()) <= 5 and not any(
        w in query.lower() for w in ("open", "shutdown", "restart", "lock", "search")
    ):
        if _MUSIC_HINT_RE.search(search_context) or "song" in search_context.lower():
            return ResolvedIntent(
                action="play",
                objective=query,
                confidence="low",
                raw_response="heuristic:short_phrase_music",
            )

    return None


def _strip_thinking_tags(text: str) -> str:
    """Remove model thinking wrappers; keep remainder for JSON extraction."""
    cleaned = text
    for pat in (
        r"<think\b[^>]*>.*?</think>",
        r"<think>.*?</think>",
    ):
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def parse_structured_intent(text: str) -> Optional[ResolvedIntent]:
    """Parse JSON or 'Action:/Objective:' formatted LLM output."""
    if not text or not text.strip():
        return None

    raw = _strip_thinking_tags(text.strip())
    if not raw:
        raw = text.strip()

    # JSON block (possibly inside markdown fences)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    elif "{" in raw or "{" in text:
        blob = raw if "{" in raw else text
        start = blob.find("{")
        end = blob.rfind("}")
        if end > start:
            raw = blob[start : end + 1]

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return _intent_from_dict(data)
        except json.JSONDecodeError:
            pass

    # Action: / Objective: lines
    action_m = re.search(r"(?im)^\s*action\s*:\s*(.+?)\s*$", text)
    objective_m = re.search(r"(?im)^\s*objective\s*:\s*(.+?)\s*$", text)
    confidence_m = re.search(r"(?im)^\s*confidence\s*:\s*(.+?)\s*$", text)
    if action_m:
        return ResolvedIntent(
            action=action_m.group(1).strip().lower(),
            objective=(objective_m.group(1).strip() if objective_m else "").strip(),
            confidence=(confidence_m.group(1).strip().lower() if confidence_m else "medium"),
            raw_response=text,
        )

    return None


def _intent_from_dict(data: dict[str, Any]) -> Optional[ResolvedIntent]:
    action = str(data.get("action", "none")).strip().lower()
    objective = str(data.get("objective", "")).strip()
    confidence = str(data.get("confidence", "medium")).strip().lower()
    if action not in _VALID_ACTIONS:
        action = "none"
    if confidence not in _CONFIDENCE_RANK:
        confidence = "medium"
    return ResolvedIntent(
        action=action,
        objective=objective,
        confidence=confidence,
        raw_response=json.dumps(data),
    )


def meets_confidence(intent: ResolvedIntent, minimum: str) -> bool:
    return _CONFIDENCE_RANK.get(intent.confidence, 0) >= _CONFIDENCE_RANK.get(minimum, 1)


def intent_to_tool_route(intent: ResolvedIntent, settings: dict) -> Optional[dict[str, Any]]:
    """Map structured intent to tool_name + arguments for the registry."""
    action = intent.action
    objective = intent.objective.strip()
    if action == "none" or not objective:
        return None

    if action == "play":
        return {
            "tool_name": "search_youtube",
            "arguments": {"query": objective},
            "tier": 2,
        }

    if action == "open_url":
        return {
            "tool_name": "browser_go_to_website",
            "arguments": {"url": objective},
            "tier": 2,
        }

    if action == "search_web":
        return {
            "tool_name": "browser_search",
            "arguments": {"query": objective},
            "tier": 2,
        }

    if action == "open_app":
        app_key = resolve_app_key(objective, settings) or objective.lower().replace(" ", "_")
        return {
            "tool_name": "open_app",
            "arguments": {"app_key": app_key},
            "tier": 2,
        }

    if action == "run_workflow":
        wf = objective.lower().replace(" ", "_")
        return {
            "tool_name": "run_workflow",
            "arguments": {"name": wf},
            "tier": 2,
        }

    if action == "answer":
        return {
            "tool_name": "speak",
            "arguments": {"text": objective},
            "tier": 2,
        }

    return None


class IntentResolver:
    def __init__(self, settings: dict, ollama, browser):
        self._settings = settings
        self._ollama = ollama
        self._browser = browser

    def resolve(self, transcript: str, session_context: dict) -> Optional[ResolvedIntent]:
        if not self._settings.get("intent_resolution_enabled", True):
            return None
        if not self._settings.get("browse_enabled", True):
            return None

        query = clean_query(transcript) or transcript.strip()
        if len(query) < 2:
            return None

        # Tier 2a: Google Search AI with [start]...[end] structured prompt
        if self._settings.get("google_ai_search_enabled", True):
            try:
                prompt = build_google_ai_prompt(transcript, self._settings)
                google_text = self._browser.google_structured_search(prompt)
                bracket = extract_bracket_answer(google_text)
                if bracket:
                    logger.info("[Intent] Google AI bracket answer: %s", bracket[:120])
                    intent = intent_from_bracket_answer(bracket)
                    if intent and intent.action != "none":
                        logger.info(
                            "[Intent] Google AI action=%s objective='%s'",
                            intent.action, intent.objective,
                        )
                        return intent
                else:
                    logger.info("[Intent] No [start]...[end] in Google response — fallback")
            except Exception as exc:
                logger.warning("[Intent] Google AI search failed: %s", exc)

        # Tier 2b: DuckDuckGo + local LLM JSON
        try:
            search_context = self._browser.search(query)
        except Exception as exc:
            logger.warning("[Intent] Web search failed: %s", exc)
            search_context = f"(search unavailable: {exc})"

        logger.info("[Intent] Search context length=%d chars", len(search_context))

        llm_result = self._ollama.parse_intent(
            transcript=transcript,
            search_context=search_context,
            session_context={},  # avoid prior song context confusing classification
        )

        if "error" in llm_result:
            logger.warning("[Intent] LLM parse failed: %s", llm_result.get("error"))
            return None

        content = (llm_result.get("content") or "").strip()
        intent = parse_structured_intent(content)
        if not intent:
            logger.warning(
                "[Intent] Could not parse LLM output (%r) — trying heuristic",
                content[:300],
            )
            intent = infer_intent_heuristic(transcript, search_context)
        if not intent:
            return None

        logger.info(
            "[Intent] action=%s objective='%s' confidence=%s",
            intent.action, intent.objective, intent.confidence,
        )
        return intent
