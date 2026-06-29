"""Layer 1 — two-loop intent finding (verb JSON → phrase JSON)."""

import json
import logging
import re
from typing import Any, Optional

from src.core.models import GatherResult, LayerPlan
from src.music.vocabulary_matcher import match_vocabulary

logger = logging.getLogger(__name__)

ACTION_VERBS = (
    "play",
    "pause",
    "open",
    "search",
    "skip",
    "volume",
    "lock",
    "shutdown",
    "restart",
    "workflow",
    "answer",
    "navigate",
    "read",
    "none",
)

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

_THINKING_RE = re.compile(
    r"<think\b[^>]*>.*?</think>|"
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

VERB_TO_TOOL = {
    "play": "search_youtube",
    "pause": "media_play_pause",
    "open": "open_app",
    "search": "browser_search",
    "skip": "media_next",
    "volume": "media_volume_up",
    "lock": "lock",
    "shutdown": "shutdown",
    "restart": "restart",
    "workflow": "run_workflow",
    "answer": "speak",
    "navigate": "browser_go_to_website",
    "read": "browse_read",
}


def _strip_model_noise(text: str) -> str:
    cleaned = _THINKING_RE.sub("", text or "")
    return cleaned.strip()


def parse_json_field(
    content: str,
    key: str,
    allowed: Optional[tuple[str, ...]] = None,
) -> Optional[str]:
    """Extract a field from JSON LLM output."""
    raw = _strip_model_noise(content)
    if not raw:
        return None

    blob = raw
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            blob = raw[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    value = str(data.get(key, "")).strip()
    if key == "verb" or key == "confidence":
        value = value.lower()
    if allowed and value.lower() not in allowed:
        for item in allowed:
            if item in value.lower():
                return item
        return None
    return value or None


def extract_one_word(text: str, allowed: tuple[str, ...]) -> Optional[str]:
    """Fallback: take the first token that matches an allowed value."""
    cleaned = _strip_model_noise(text)
    if not cleaned:
        return None
    first_line = cleaned.split("\n", 1)[0].strip()
    token = re.split(r"[\s,.:;!?]+", first_line.lower())[0]
    if token in allowed:
        return token
    for word in allowed:
        if re.search(rf"\b{re.escape(word)}\b", first_line.lower()):
            return word
    return None


def extract_phrase(text: str) -> str:
    """Fallback phrase extraction from prose."""
    field = parse_json_field(text, "phrase")
    if field:
        return field
    cleaned = _strip_model_noise(text)
    if not cleaned:
        return ""
    line = cleaned.split("\n", 1)[0].strip()
    return line.strip('"').strip("'")


def verb_to_tool(verb: str) -> str:
    return VERB_TO_TOOL.get(verb, "browse_search")


def apply_vocabulary_phrase(transcript: str, phrase: str, vocabulary: dict[str, Any]) -> str:
    """Boost confidence when vocabulary corrects the phrase."""
    matches = match_vocabulary(transcript, vocabulary)
    if matches:
        best = max(matches, key=lambda m: m.score)
        if best.score >= 0.85:
            return best.canonical_form
    return phrase


class Layer1Intent:
    """Two JSON loops: verb → corrected phrase + confidence. Tool from verb map."""

    def __init__(self, ollama, settings: dict, vocabulary: dict | None = None):
        self._ollama = ollama
        self._settings = settings
        self._vocabulary = vocabulary or {}

    def find(self, gather: GatherResult) -> Optional[LayerPlan]:
        if not self._settings.get("layer1_intent_enabled", True):
            return None

        transcript = gather.transcript
        verbs = tuple(self._settings.get("layer1_action_verbs", ACTION_VERBS))
        ctx_block = gather.context or "(no web results — use transcript only)"
        max_ctx = self._settings.get("layer1_max_context_chars", 4000)
        if len(ctx_block) > max_ctx:
            ctx_block = ctx_block[:max_ctx] + "\n...[truncated]"

        hint_line = f"Quick verb hint: {gather.verb_hint}\n" if gather.verb_hint else ""
        base = (
            f'User transcript: "{transcript}"\n'
            f"{hint_line}\n"
            f"Context:\n{ctx_block}\n"
        )

        verb = self._loop_verb(base, verbs, gather.verb_hint)
        if not verb:
            verb = self._fallback_verb(gather, transcript, verbs)
        if not verb or verb == "none":
            return LayerPlan(
                transcript=transcript,
                verb="none",
                phrase="",
                tool="none",
                search_context=gather.context,
                confidence="high",
                source="layer1",
            )

        phrase, confidence = self._loop_phrase(base, verb)
        if not phrase:
            phrase = transcript.strip()
            confidence = "low"

        phrase = apply_vocabulary_phrase(transcript, phrase, self._vocabulary)
        if gather.verb_hint == verb and phrase != transcript.strip():
            confidence = _bump_confidence(confidence)

        tool = verb_to_tool(verb)
        source = "layer1"
        if gather.verb_hint == verb:
            source = "layer1+hint"

        plan = LayerPlan(
            transcript=transcript,
            verb=verb,
            phrase=phrase,
            tool=tool,
            search_context=gather.context,
            confidence=confidence,
            source=source,
        )
        logger.info(
            "[Layer1] verb=%s phrase='%s' tool=%s confidence=%s",
            plan.verb, plan.phrase[:80], plan.tool, plan.confidence,
        )
        return plan

    def _loop_verb(
        self,
        base: str,
        verbs: tuple[str, ...],
        hint: Optional[str],
    ) -> Optional[str]:
        verb_list = ", ".join(verbs)
        system = (
            "You classify voice commands for Jarvis on Windows. "
            'Output ONLY JSON: {"verb":"<one>"} where verb is from the allowed list.'
        )
        hint_note = f" Hint suggests: {hint}." if hint else ""
        user = (
            f"{base}\n"
            f"What is the user likely asking me to do?{hint_note} "
            f"Allowed verbs: {verb_list}."
        )
        result = self._ollama.layer_ask_json(system, user, max_tokens=32)
        if "error" in result:
            if hint and hint in verbs:
                return hint
            logger.warning("[Layer1] loop1 error: %s", result.get("error"))
            return None

        content = result.get("content", "")
        verb = parse_json_field(content, "verb", verbs) or extract_one_word(content, verbs)
        if not verb and hint and hint in verbs:
            verb = hint
        logger.info("[Layer1] loop1 raw=%r → %s", content[:60], verb)
        return verb

    def _loop_phrase(self, base: str, verb: str) -> tuple[str, str]:
        system = (
            "You correct misheard voice commands for Jarvis. "
            'Output ONLY JSON: {"phrase":"<corrected text>","confidence":"high|medium|low"}. '
            "phrase is what the user meant; confidence reflects how sure you are."
        )
        user = (
            f"{base}\n"
            f"What is the user asking me to {verb}? "
            "Write the proper phrase they meant (fix spelling/names from context)."
        )
        result = self._ollama.layer_ask_json(system, user, max_tokens=96)
        if "error" in result:
            logger.warning("[Layer1] loop2 error: %s", result.get("error"))
            return "", "low"

        content = result.get("content", "")
        phrase = parse_json_field(content, "phrase") or extract_phrase(content)
        confidence = parse_json_field(content, "confidence", ("low", "medium", "high")) or "medium"
        logger.info("[Layer1] loop2 raw=%r → phrase=%r conf=%s", content[:80], phrase[:60], confidence)
        return phrase, confidence

    def _fallback_verb(
        self,
        gather: GatherResult,
        transcript: str,
        verbs: tuple[str, ...],
    ) -> Optional[str]:
        if gather.verb_hint and gather.verb_hint in verbs:
            logger.info("[Layer1] verb fallback from hint → %s", gather.verb_hint)
            return gather.verb_hint

        vocab_matches = match_vocabulary(transcript, self._vocabulary)
        if vocab_matches:
            logger.info("[Layer1] verb fallback from vocabulary → play")
            return "play"

        ctx = (gather.context or "").lower()
        if any(k in ctx for k in ("musicbrainz", "vocabulary", "song", "youtube", "artist")):
            logger.info("[Layer1] verb fallback from music context → play")
            return "play"

        return None


def _bump_confidence(confidence: str) -> str:
    if confidence == "low":
        return "medium"
    if confidence == "medium":
        return "high"
    return confidence
