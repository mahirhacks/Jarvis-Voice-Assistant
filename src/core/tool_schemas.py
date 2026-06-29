"""Ollama tool JSON schemas for native tool calling."""

from src.tools.apps import list_app_keys


def _tool(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def build_tool_schemas(settings: dict | None = None) -> list[dict]:
    app_keys = list_app_keys(settings)
    app_hint = ", ".join(app_keys) if app_keys else "chrome, vscode, notepad, task_manager, explorer"

    return [
        _tool("search_youtube", "Search YouTube and play music when user wants to hear a song."),
        _tool("play_video", "Play a YouTube video by ID.", {"video_id": {"type": "string"}}, ["video_id"]),
        _tool("next_candidate", "Show or play the next search result."),
        _tool("previous_candidate", "Show or play the previous search result."),
        _tool("replay_last", "Replay the last played song."),
        _tool("media_play_pause", "Toggle play/pause for media."),
        _tool("media_next", "Next track."),
        _tool("media_previous", "Previous track."),
        _tool("media_forward", "Skip forward."),
        _tool("media_backward", "Skip backward."),
        _tool("media_volume_up", "Increase system volume."),
        _tool("media_volume_down", "Decrease system volume."),
        _tool("media_mute", "Mute or unmute."),
        _tool("media_fullscreen", "Enter fullscreen."),
        _tool("media_exit_fullscreen", "Exit fullscreen."),
        _tool("browser_new_tab", "Open a new browser tab."),
        _tool("browser_close_tab", "Close the current browser tab."),
        _tool("browser_next_tab", "Switch to next browser tab."),
        _tool("browser_previous_tab", "Switch to previous browser tab."),
        _tool("browser_refresh", "Refresh the current page."),
        _tool("browser_back", "Go back in browser."),
        _tool("browser_forward", "Go forward in browser."),
        _tool("browser_search", "Search Google in browser.", {"query": {"type": "string"}}, ["query"]),
        _tool("browser_go_to_website", "Open a website URL.", {"url": {"type": "string"}}, ["url"]),
        _tool(
            "browse_search",
            "Headless web search; returns result snippets as text (DuckDuckGo).",
            {"query": {"type": "string"}},
            ["query"],
        ),
        _tool(
            "browse_open",
            "Open a URL in headless Firefox and return page text preview.",
            {"url": {"type": "string"}},
            ["url"],
        ),
        _tool("browse_read", "Read text from the current headless browser page."),
        _tool(
            "open_app",
            f"Open a Windows app or utility. Allowed keys: {app_hint}. "
            "Use for task manager, settings, calculator, chrome, vscode, explorer, etc.",
            {"app_key": {"type": "string", "description": f"One of: {app_hint}"}},
            ["app_key"],
        ),
        _tool("open_folder", "Open a folder in File Explorer.", {"folder_path": {"type": "string"}}, ["folder_path"]),
        _tool(
            "run_workflow",
            "Run multi-step workflow: study_mode, assignment_mode, bug_bounty_mode, play_music, shutdown, lock_laptop.",
            {"name": {"type": "string"}},
            ["name"],
        ),
        _tool("shutdown", "Shut down the computer."),
        _tool("restart", "Restart/reboot the computer."),
        _tool("lock", "Lock the workstation."),
        _tool("cancel_shutdown", "Cancel a pending shutdown or restart."),
        _tool("speak", "Speak a short answer to the user.", {"text": {"type": "string"}}, ["text"]),
    ]


TOOL_ALTERNATIVES: dict[str, list[str]] = {
    "search_youtube": ["play_video", "replay_last"],
    "open_app": ["open_folder", "browser_go_to_website"],
    "browser_search": ["browse_search", "browser_go_to_website"],
    "browser_go_to_website": ["browser_search", "open_app"],
    "browse_search": ["browser_search", "speak"],
    "media_volume_up": ["media_volume_down", "media_mute"],
    "media_volume_down": ["media_volume_up", "media_mute"],
    "media_next": ["media_previous", "media_forward"],
    "media_previous": ["media_next", "media_backward"],
    "run_workflow": ["open_app"],
    "speak": ["browse_search"],
}


def narrow_tool_schemas(primary_tool: str, settings: dict | None = None) -> list[dict]:
    """Primary tool plus a few related alternatives for Layer 2."""
    all_schemas = build_tool_schemas(settings)
    by_name = {s["function"]["name"]: s for s in all_schemas}
    names = [primary_tool]
    for alt in TOOL_ALTERNATIVES.get(primary_tool, []):
        if alt in by_name and alt not in names:
            names.append(alt)
    return [by_name[n] for n in names if n in by_name]


# Default schemas (used in tests)
TOOL_SCHEMAS = build_tool_schemas()
