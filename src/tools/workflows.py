"""Workflow execution tool."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ACTION_ARG_MAP = {
    "speak": lambda v: {"text": v},
    "open_url": lambda v: {"url": v},
    "open_app": lambda v: {"app_key": v},
    "open_folder": lambda v: {"folder_path": v},
    "browser_search": lambda v: {"query": v},
    "browser_go_to_website": lambda v: {"url": v},
    "browser_find_on_page": lambda v: {"query": v},
    "play_youtube_search": lambda v: {"query": v},
    "search_youtube": lambda v: {"query": v},
    "run_workflow": lambda v: {"name": v},
    "wait": lambda v: {"seconds": int(v) if v else 1},
}


def run_workflow(name: str, settings: dict, execute_tool) -> str:
    workflows_dir = Path(settings.get("workflows_dir", "config/workflows"))
    path = workflows_dir / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Workflow '{name}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    steps = wf.get("steps", [])
    if not steps:
        raise ValueError(f"Workflow '{name}' has no steps.")

    import time as _time

    for i, step in enumerate(steps, 1):
        action = step.get("action")
        value = step.get("value")
        logger.info("Workflow %s step %d: %s", name, i, action)

        if action == "wait":
            _time.sleep(int(value) if value else 1)
            continue

        mapper = _ACTION_ARG_MAP.get(action)
        args = mapper(value) if mapper else {}
        result = execute_tool(action, args)
        if not result.success:
            raise RuntimeError(f"Workflow failed at step {i}: {result.message}")

    return f"Workflow {name.replace('_', ' ')} complete."
