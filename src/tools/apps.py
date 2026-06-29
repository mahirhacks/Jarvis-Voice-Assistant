"""Application launch tools."""

import json
import logging
import os
import subprocess
from pathlib import Path

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Spoken phrase -> allowed_apps.json key
APP_ALIASES: dict[str, str] = {
    "task manager": "task_manager",
    "taskmanager": "task_manager",
    "file explorer": "explorer",
    "windows explorer": "explorer",
    "explorer": "explorer",
    "settings": "settings",
    "windows settings": "settings",
    "calculator": "calculator",
    "calc": "calculator",
    "command prompt": "cmd",
    "cmd": "cmd",
    "powershell": "powershell",
    "control panel": "control_panel",
    "device manager": "device_manager",
    "paint": "paint",
    "snipping tool": "snipping_tool",
    "disk cleanup": "disk_cleanup",
    "resource monitor": "resource_monitor",
    "notepad": "notepad",
    "chrome": "chrome",
    "google chrome": "chrome",
    "vscode": "vscode",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "kiro": "kiro",
}


def _load_apps(settings: dict | None = None) -> dict:
    path = "config/allowed_apps.json"
    if settings:
        path = settings.get("allowed_apps_path", path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load allowed apps: %s", e)
        return {}


def list_app_keys(settings: dict | None = None) -> list[str]:
    return sorted(_load_apps(settings).keys())


def resolve_app_key(phrase: str, settings: dict | None = None) -> str | None:
    """Map a spoken app name to an allowed_apps.json key."""
    apps = _load_apps(settings)
    if not apps:
        return None

    normalized = phrase.strip().lower()
    if normalized in apps:
        return normalized
    if normalized in APP_ALIASES and APP_ALIASES[normalized] in apps:
        return APP_ALIASES[normalized]

    # Fuzzy match aliases then keys
    alias_result = process.extractOne(
        normalized, list(APP_ALIASES.keys()), scorer=fuzz.WRatio, score_cutoff=80,
    )
    if alias_result:
        key = APP_ALIASES[alias_result[0]]
        if key in apps:
            return key

    key_result = process.extractOne(
        normalized, list(apps.keys()), scorer=fuzz.WRatio, score_cutoff=80,
    )
    if key_result:
        return key_result[0]

    return None


def open_app(app_key: str, settings: dict) -> str:
    apps = _load_apps(settings)
    key = app_key.strip().lower()
    resolved = resolve_app_key(key, settings) or key
    if resolved not in apps:
        raise ValueError(
            f"App '{app_key}' is not allowed. "
            f"Available: {', '.join(sorted(apps.keys()))}"
        )
    entry = apps[resolved]
    command = entry.get("command", "")
    name = entry.get("name", resolved)
    subprocess.Popen(command, shell=True)
    return f"Opened {name}."


def open_folder(folder_path: str) -> str:
    resolved = Path(folder_path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"Folder does not exist: {resolved}")
    os.startfile(str(resolved))
    return f"Opened folder {resolved.name}."
