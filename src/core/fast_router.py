"""Tier 0 fast routing — no LLM."""

import logging
import re
from typing import Any, Optional

from rapidfuzz import fuzz, process

from src.tools.apps import resolve_app_key

logger = logging.getLogger(__name__)

PLAY_PREFIXES = ["play the song ", "play me ", "play song ", "put on ", "listen to ", "play "]
OPEN_PREFIXES = ["open ", "launch ", "start ", "run "]
MEDIA_PLAY_WORDS = {"play", "resume", "play resume", "pause"}
BY_ARTIST_RE = re.compile(r"^(.+?)\s+by\s+(.+)$")
OFFICIAL_VIDEO_RE = re.compile(
    r"^(.+?),\s*(official video|official audio|music video|lyrics)\s*\.?$",
    re.IGNORECASE,
)
# Long sentences must not fuzzy-match short media commands
_FUZZY_MEDIA_KEYS = frozenset({
    "pause", "resume", "play resume", "next", "previous", "forward", "backward",
    "mute", "unmute", "volume up", "volume down",
})
_MIN_WORDS_FOR_MEDIA_FUZZY = 3

FAST_PATH_TOOLS = {
    "pause": ("media_play_pause", {}),
    "resume": ("media_play_pause", {}),
    "play resume": ("media_play_pause", {}),
    "next": ("media_next", {}),
    "previous": ("media_previous", {}),
    "forward": ("media_forward", {}),
    "backward": ("media_backward", {}),
    "volume up": ("media_volume_up", {}),
    "volume down": ("media_volume_down", {}),
    "mute": ("media_mute", {}),
    "unmute": ("media_mute", {}),
    "open new tab": ("browser_new_tab", {}),
    "new tab": ("browser_new_tab", {}),
    "close tab": ("browser_close_tab", {}),
    "next tab": ("browser_next_tab", {}),
    "previous tab": ("browser_previous_tab", {}),
    "cancel shutdown": ("cancel_shutdown", {}),
    "fullscreen": ("media_fullscreen", {}),
    "full screen": ("media_fullscreen", {}),
    "exit fullscreen": ("media_exit_fullscreen", {}),
    "exit full screen": ("media_exit_fullscreen", {}),
    "loop on": ("media_loop_toggle", {}),
    "loop off": ("media_loop_toggle", {}),
    "stop": ("media_play_pause", {}),
    "stop music": ("media_play_pause", {}),
    "stop playing": ("media_play_pause", {}),
    "next track": ("media_next", {}),
    "next song": ("media_next", {}),
    "previous song": ("media_previous", {}),
    "skip": ("media_next", {}),
    "skip this": ("media_next", {}),
    "louder": ("media_volume_up", {}),
    "quieter": ("media_volume_down", {}),
    "turn it up": ("media_volume_up", {}),
    "turn it down": ("media_volume_down", {}),
    "silence": ("media_mute", {}),
    "quiet": ("media_mute", {}),
    "go back": ("browser_back", {}),
    "replay": ("replay_last", {}),
    "replay that": ("replay_last", {}),
    "play that again": ("replay_last", {}),
    "play it again": ("replay_last", {}),
    "one more time": ("replay_last", {}),
    "same song": ("replay_last", {}),
}

BUILTIN_TOOLS = {
    **FAST_PATH_TOOLS,
    "shutdown": ("shutdown", {}),
    "shutdown laptop": ("shutdown", {}),
    "shut down": ("shutdown", {}),
    "shut down laptop": ("shutdown", {}),
    "turn off laptop": ("shutdown", {}),
    "turn off the laptop": ("shutdown", {}),
    "restart": ("restart", {}),
    "restart laptop": ("restart", {}),
    "restart the laptop": ("restart", {}),
    "reboot": ("restart", {}),
    "reboot laptop": ("restart", {}),
    "reboot the laptop": ("restart", {}),
    "lock": ("lock", {}),
    "lock laptop": ("lock", {}),
    "lock the laptop": ("lock", {}),
    "lock computer": ("lock", {}),
    "restart ollama": ("restart_ollama", {}),
    "open task manager": ("open_app", {"app_key": "task_manager"}),
    "task manager": ("open_app", {"app_key": "task_manager"}),
    "open file explorer": ("open_app", {"app_key": "explorer"}),
    "file explorer": ("open_app", {"app_key": "explorer"}),
    "open settings": ("open_app", {"app_key": "settings"}),
    "open calculator": ("open_app", {"app_key": "calculator"}),
    "open notepad": ("open_app", {"app_key": "notepad"}),
    "open chrome": ("open_app", {"app_key": "chrome"}),
    "open vscode": ("open_app", {"app_key": "vscode"}),
    "open control panel": ("open_app", {"app_key": "control_panel"}),
    "open device manager": ("open_app", {"app_key": "device_manager"}),
    "device manager": ("open_app", {"app_key": "device_manager"}),
    "google chrome": ("open_app", {"app_key": "chrome"}),
    "command prompt": ("open_app", {"app_key": "cmd"}),
    "windows settings": ("open_app", {"app_key": "settings"}),
    "explorer": ("open_app", {"app_key": "explorer"}),
}

WORKFLOW_ALIASES = {
    "study mode": "study_mode",
    "assignment mode": "assignment_mode",
    "bug bounty mode": "bug_bounty_mode",
    "play music": "play_music",
}

RISKY_TOOLS = frozenset({"shutdown", "restart", "lock", "restart_ollama"})

RISKY_MESSAGES = {
    "shutdown": "Are you sure you want to shut down?",
    "restart": "Are you sure you want to restart?",
    "lock": "Do you want to lock the laptop?",
    "restart_ollama": "Restart Ollama?",
}


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip()


def _try_open_app_route(text: str) -> Optional[dict[str, Any]]:
    from src.core.phrase_heuristics import _WEB_ALIASES

    for prefix in OPEN_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            target = text[len(prefix):].strip()
            if not target:
                continue
            app_key = resolve_app_key(target)
            if app_key:
                return _make_route("open_app", {"app_key": app_key}, 100.0)
            url = _WEB_ALIASES.get(target.lower())
            if url:
                return _make_route("browser_go_to_website", {"url": url}, 92.0)
    return None


def _try_fuzzy_open_app(text: str, cutoff: float) -> Optional[dict[str, Any]]:
    """Fuzzy app match only when transcript looks like an open-app request."""
    cleaned = _strip_punctuation(text)
    word_count = len(cleaned.split())
    if word_count > 6 and not any(text.startswith(p) for p in OPEN_PREFIXES):
        return None
    if "ask " in text and "task" not in text:
        return None
    if not any(kw in text for kw in ("open", "launch", "start", "task manager", "file explorer")):
        return None
    app_keys = list(BUILTIN_TOOLS.keys())
    open_keys = [k for k in app_keys if k.startswith("open ") or k in ("task manager", "file explorer")]
    cleaned_map = {_strip_punctuation(k): k for k in open_keys}
    result = process.extractOne(
        cleaned, list(cleaned_map.keys()),
        scorer=fuzz.WRatio, score_cutoff=cutoff,
    )
    if not result:
        return None
    matched, score, _ = result
    tool_name, args = BUILTIN_TOOLS[cleaned_map[matched]]
    if tool_name != "open_app":
        return None
    return _make_route(tool_name, args, score)


def route(
    transcript: str,
    medium_confidence: float = 65,
    high_confidence: float = 92,
) -> Optional[dict[str, Any]]:
    """Return fast-route dict with tool_name, arguments, confidence, needs_confirmation."""
    original = transcript.strip()
    text = original.lower()
    if not text:
        return None

    if text in BUILTIN_TOOLS:
        tool_name, args = BUILTIN_TOOLS[text]
        return _make_route(tool_name, args, 100.0)

    if text in WORKFLOW_ALIASES:
        return _make_route("run_workflow", {"name": WORKFLOW_ALIASES[text]}, 100.0)

    if text in MEDIA_PLAY_WORDS:
        return _make_route("media_play_pause", {}, 100.0)

    for prefix in PLAY_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            query = text[len(prefix):].strip()
            if query:
                return _make_route("search_youtube", {"query": query}, 100.0)

    by_match = BY_ARTIST_RE.match(text)
    if by_match and len(by_match.group(1).strip()) >= 2:
        return _make_route(
            "search_youtube",
            {"query": original.rstrip(".")},
            92.0,
        )

    ov_match = OFFICIAL_VIDEO_RE.match(original)
    if ov_match and len(ov_match.group(1).strip()) >= 2:
        return _make_route(
            "search_youtube",
            {"query": original.rstrip(".")},
            90.0,
        )

    open_route = _try_open_app_route(text)
    if open_route:
        return open_route

    fuzzy_open = _try_fuzzy_open_app(text, high_confidence)
    if fuzzy_open:
        return fuzzy_open

    word_count = len(text.split())
    if word_count < _MIN_WORDS_FOR_MEDIA_FUZZY:
        keys = [k for k in BUILTIN_TOOLS if k in _FUZZY_MEDIA_KEYS]
        cleaned_map = {_strip_punctuation(k): k for k in keys}
        result = process.extractOne(
            _strip_punctuation(text), list(cleaned_map.keys()),
            scorer=fuzz.WRatio, score_cutoff=high_confidence,
        )
        if result:
            matched, score, _ = result
            tool_name, args = BUILTIN_TOOLS[cleaned_map[matched]]
            return _make_route(tool_name, args, score)

    for alias, wf_name in WORKFLOW_ALIASES.items():
        if fuzz.ratio(text, alias) >= high_confidence:
            return _make_route("run_workflow", {"name": wf_name}, fuzz.ratio(text, alias))

    return None


def _make_route(tool_name: str, arguments: dict, confidence: float) -> dict[str, Any]:
    needs_confirm = tool_name in RISKY_TOOLS
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "confidence": confidence,
        "needs_confirmation": needs_confirm,
        "confirmation_message": RISKY_MESSAGES.get(tool_name),
        "tier": 0,
    }
