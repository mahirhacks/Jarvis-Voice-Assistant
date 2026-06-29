"""Tests for tool registry."""

import pytest

from src.core.tool_registry import ToolRegistry
from src.core.settings_loader import load_settings


def test_unknown_tool_rejected():
    reg = ToolRegistry(load_settings())
    result = reg.execute("nonexistent_tool", {})
    assert not result.success


def test_speak_tool():
    reg = ToolRegistry(load_settings())
    assert reg.is_known("speak")
    result = reg.execute("speak", {"text": "test"})
    assert result.success


def test_media_tool_registered():
    reg = ToolRegistry(load_settings())
    assert reg.is_known("media_play_pause")
    assert reg.is_risky("shutdown")
