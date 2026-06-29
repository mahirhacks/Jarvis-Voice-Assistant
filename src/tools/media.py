"""Media control tools."""

import logging

import pyautogui

from src.voice import tts

logger = logging.getLogger(__name__)


def media_play_pause() -> str:
    pyautogui.press("playpause")
    return "Paused or resumed."


def media_next() -> str:
    pyautogui.press("nexttrack")
    return "Skipping forward."


def media_previous() -> str:
    pyautogui.press("prevtrack")
    return "Going back."


def media_forward() -> str:
    pyautogui.press("right")
    return "Skipping forward."


def media_backward() -> str:
    pyautogui.press("left")
    return "Going back."


def media_volume_up() -> str:
    pyautogui.press("volumeup")
    return "Volume up."


def media_volume_down() -> str:
    pyautogui.press("volumedown")
    return "Volume down."


def media_mute() -> str:
    pyautogui.press("volumemute")
    return "Muted or unmuted."


def media_loop_toggle() -> str:
    pyautogui.hotkey("shift", "l")
    return "Toggled loop."


def media_fullscreen() -> str:
    pyautogui.press("f")
    return "Full screen."


def media_exit_fullscreen() -> str:
    pyautogui.press("escape")
    return "Exited full screen."
