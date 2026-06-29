"""System tools (risky)."""

import ctypes
import logging
import subprocess

logger = logging.getLogger(__name__)


def shutdown() -> str:
    subprocess.run(["shutdown", "/s", "/t", "5"], check=True)
    return "Shutting down in 5 seconds."


def restart() -> str:
    subprocess.run(["shutdown", "/r", "/t", "5"], check=True)
    return "Restarting in 5 seconds."


def cancel_shutdown() -> str:
    subprocess.run(["shutdown", "/a"], check=True)
    return "Shutdown cancelled."


def lock() -> str:
    ctypes.windll.user32.LockWorkStation()
    return "Locked."
