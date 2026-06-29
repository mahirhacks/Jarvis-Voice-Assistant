"""Tool registry — whitelist execution."""

import logging
from typing import Any, Callable, Optional

from src.core.fast_router import RISKY_TOOLS
from src.core.models import ToolResult
from src.tools import apps, browser, media, system
from src.tools.workflows import run_workflow

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, settings: dict, music_tools=None):
        self._settings = settings
        self._music = music_tools
        self._safe_mode = settings.get("safe_mode", True)
        self._handlers: dict[str, Callable[..., str]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        from src.voice import tts as tts_mod

        def _noarg(fn):
            def wrapper(**_):
                return fn()
            return wrapper

        def _arg(fn, key):
            def wrapper(**kw):
                return fn(kw.get(key, ""))
            return wrapper

        def _speak(text="", **_):
            tts_mod.speak(text, blocking=True)
            return text or "Speaking."

        self._handlers["speak"] = _speak

        for name in (
            "media_play_pause", "media_next", "media_previous", "media_forward",
            "media_backward", "media_volume_up", "media_volume_down", "media_mute",
            "media_loop_toggle", "media_fullscreen", "media_exit_fullscreen",
        ):
            self._handlers[name] = _noarg(getattr(media, name))

        browser_noarg = {
            "browser_new_tab", "browser_close_tab", "browser_reopen_tab",
            "browser_next_tab", "browser_previous_tab", "browser_refresh",
            "browser_stop_loading", "browser_back", "browser_forward",
            "browser_scroll_down", "browser_scroll_up", "browser_zoom_in",
            "browser_zoom_out", "browser_reset_zoom", "browser_focus_address_bar",
            "browser_open_downloads", "browser_open_history", "browser_bookmark_page",
        }
        browser_arg = {
            "browser_search": "query",
            "browser_go_to_website": "url",
            "browser_find_on_page": "query",
            "open_url": "url",
        }
        for name in browser_noarg:
            self._handlers[name] = _noarg(getattr(browser, name))
        for name, key in browser_arg.items():
            self._handlers[name] = _arg(getattr(browser, name), key)

        for name in ("shutdown", "restart", "cancel_shutdown", "lock"):
            self._handlers[name] = _noarg(getattr(system, name))

        self._handlers["open_app"] = lambda app_key="", **_: apps.open_app(app_key, self._settings)
        self._handlers["open_folder"] = lambda folder_path="", **_: apps.open_folder(folder_path)
        self._handlers["run_workflow"] = lambda name="", **_: run_workflow(
            name, self._settings, self.execute,
        )
        self._handlers["restart_ollama"] = _noarg(self._restart_ollama)

        from src.tools import web_browse

        self._handlers["browse_search"] = lambda query="", **_: web_browse.browse_search(
            query, self._settings,
        )
        self._handlers["browse_open"] = lambda url="", **_: web_browse.browse_open(
            url, self._settings,
        )
        self._handlers["browse_read"] = lambda **_: web_browse.browse_read(self._settings)

    def set_music_tools(self, music_tools) -> None:
        self._music = music_tools
        if music_tools:
            self._handlers["search_youtube"] = lambda query, raw_transcript="", **_: music_tools.search_youtube(query, raw_transcript)
            self._handlers["play_video"] = lambda video_id, **_: music_tools.play_video(video_id)
            self._handlers["next_candidate"] = lambda **_: music_tools.next_candidate()
            self._handlers["previous_candidate"] = lambda **_: music_tools.previous_candidate()
            self._handlers["replay_last"] = lambda **_: music_tools.replay_last()
            # Legacy workflow alias
            self._handlers["play_youtube_search"] = lambda query, **_: music_tools.search_youtube(query, query)

    def is_known(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    def is_risky(self, tool_name: str) -> bool:
        return tool_name in RISKY_TOOLS

    def is_blocked(self, tool_name: str) -> bool:
        return self._safe_mode and tool_name in RISKY_TOOLS

    def execute(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> ToolResult:
        arguments = arguments or {}
        if tool_name not in self._handlers:
            logger.warning("Rejected unknown tool: %s", tool_name)
            return ToolResult(tool_name=tool_name, success=False, message=f"Unknown tool '{tool_name}'.")

        if self.is_blocked(tool_name):
            return ToolResult(tool_name=tool_name, success=False, message=f"Tool '{tool_name}' is disabled in safe mode.")

        try:
            logger.info("Executing tool: %s args=%s", tool_name, arguments)
            handler = self._handlers[tool_name]
            message = handler(**arguments)
            return ToolResult(tool_name=tool_name, success=True, message=message or "Done.")
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult(tool_name=tool_name, success=False, message=str(e))

    @staticmethod
    def _restart_ollama() -> str:
        import subprocess
        import time
        import requests
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            resp = requests.get("http://localhost:11434/", timeout=5)
            if resp.status_code == 200:
                return "Ollama restarted."
        except Exception:
            pass
        return "Attempted Ollama restart."
