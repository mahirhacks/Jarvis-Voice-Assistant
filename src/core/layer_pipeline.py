"""Layered command pipeline: gather → intent → execute."""

import logging
import re
from typing import Any, Optional

from src.core.layer0_search import Layer0Search
from src.core.layer1_intent import Layer1Intent, verb_to_tool
from src.core.models import GatherResult, LayerPlan
from src.core.tool_schemas import narrow_tool_schemas
from src.tools.apps import resolve_app_key

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

_NO_ARG_TOOLS = frozenset({
    "media_play_pause", "media_next", "media_previous", "media_forward",
    "media_backward", "media_volume_up", "media_volume_down", "media_mute",
    "lock", "shutdown", "restart", "cancel_shutdown", "browse_read",
    "browser_new_tab", "browser_close_tab", "browser_next_tab",
    "browser_previous_tab", "browser_refresh", "browser_back", "browser_forward",
})


def resolve_tool_for_plan(plan: LayerPlan, settings: dict) -> str:
    """Adjust tool from verb + phrase (volume direction, skip prev/next)."""
    combined = f"{plan.phrase} {plan.transcript}".lower()

    if plan.verb == "volume":
        if any(w in combined for w in ("mute", "unmute", "silence")):
            return "media_mute"
        if any(w in combined for w in ("down", "quieter", "lower", "decrease")):
            return "media_volume_down"
        return "media_volume_up"

    if plan.verb == "skip":
        if any(w in combined for w in ("previous", "back", "last")):
            return "media_previous"
        if "forward" in combined:
            return "media_forward"
        return "media_next"

    if plan.verb == "pause" and "resume" in combined:
        return "media_play_pause"

    return plan.tool


def needs_layer2_llm(plan: LayerPlan, settings: dict) -> bool:
    """True when Layer 2 LLM is needed instead of direct execution."""
    if plan.verb == "none" or plan.tool == "none":
        return False

    if plan.verb == "answer":
        return plan.confidence != "high" or not plan.phrase.strip()

    tool = resolve_tool_for_plan(plan, settings)

    if plan.verb == "volume":
        combined = f"{plan.phrase} {plan.transcript}".lower()
        if any(w in combined for w in ("up", "down", "mute", "unmute", "louder", "quieter", "lower")):
            return False
        return True

    if plan.confidence == "low":
        return True

    if not plan.phrase.strip() and tool not in _NO_ARG_TOOLS:
        return True

    if tool in _NO_ARG_TOOLS:
        return False

    if tool == "search_youtube" and plan.phrase.strip():
        return plan.confidence == "low"

    if tool == "open_app":
        return resolve_app_key(plan.phrase, settings) is None

    if tool == "run_workflow":
        return plan.confidence != "high"

    if tool == "browser_go_to_website":
        return not re.search(r"[\w.-]+\.(com|org|net|io)\b", plan.phrase, re.I)

    if plan.confidence == "high":
        return False

    return plan.confidence == "medium" and tool in ("browse_search", "browser_search")


def build_execution_brief(plan: LayerPlan, settings: dict) -> str:
    """Format Layer 1 output for Layer 2 tool execution."""
    ctx = plan.search_context or "(none)"
    if len(ctx) > 3500:
        ctx = ctx[:3500] + "\n...[truncated]"
    tool = resolve_tool_for_plan(plan, settings)
    return (
        f'Original transcript: "{plan.transcript}"\n'
        f"Verb: {plan.verb}\n"
        f"Corrected intent: {plan.phrase}\n"
        f"Confidence: {plan.confidence}\n"
        f"Pre-selected tool: {tool}\n\n"
        f"Web context:\n{ctx}\n\n"
        "Call the pre-selected tool with correct arguments. "
        "If arguments are obvious from the corrected intent, use them directly."
    )


def plan_to_direct_route(plan: LayerPlan, settings: dict) -> Optional[dict[str, Any]]:
    """Map a LayerPlan to a direct tool route (no Layer 2 LLM)."""
    if plan.verb == "none" or plan.tool == "none":
        return None

    tool = resolve_tool_for_plan(plan, settings)
    phrase = plan.phrase.strip()

    if plan.verb == "answer" and phrase and plan.confidence == "high":
        return {"tool_name": "speak", "arguments": {"text": phrase}, "tier": "layer"}

    if tool == "search_youtube" and phrase:
        return {"tool_name": tool, "arguments": {"query": phrase}, "tier": "layer"}

    if tool == "browser_go_to_website" and phrase:
        url = phrase if phrase.startswith("http") else f"https://{phrase}"
        return {"tool_name": tool, "arguments": {"url": url}, "tier": "layer"}

    if tool == "browser_search" and phrase:
        return {"tool_name": tool, "arguments": {"query": phrase}, "tier": "layer"}

    if tool == "open_app" and phrase:
        app_key = resolve_app_key(phrase, settings) or phrase.lower().replace(" ", "_")
        return {"tool_name": tool, "arguments": {"app_key": app_key}, "tier": "layer"}

    if tool == "run_workflow" and phrase:
        wf = phrase.lower().replace(" ", "_")
        return {"tool_name": tool, "arguments": {"name": wf}, "tier": "layer"}

    if tool == "browse_search" and phrase:
        return {"tool_name": tool, "arguments": {"query": phrase}, "tier": "layer"}

    if tool == "speak" and phrase:
        return {"tool_name": "speak", "arguments": {"text": phrase}, "tier": "layer"}

    if tool in _NO_ARG_TOOLS:
        return {"tool_name": tool, "arguments": {}, "tier": "layer"}

    return None


class LayerPipeline:
    """Orchestrates Layer 0 → 1 → 2 for unclear voice commands."""

    def __init__(self, settings: dict, ollama, browser, entity_lookup=None, vocabulary=None):
        self._settings = settings
        self._ollama = ollama
        self._layer0 = Layer0Search(browser, settings, entity_lookup, vocabulary)
        self._layer1 = Layer1Intent(ollama, settings, vocabulary)

    def gather(self, transcript: str) -> GatherResult:
        return self._layer0.gather(transcript)

    def find_intent(self, gather: GatherResult) -> Optional[LayerPlan]:
        return self._layer1.find(gather)

    def execute(self, plan: LayerPlan, tool_schemas: list) -> dict[str, Any]:
        """Layer 2 — fresh LLM context, narrowed tool calling."""
        if plan.verb == "none" or plan.tool == "none":
            return {"error": "no_intent", "error_type": "intent"}

        tool = resolve_tool_for_plan(plan, self._settings)
        brief = build_execution_brief(plan, self._settings)
        return self._ollama.layer_execute(brief, tool, tool_schemas)
