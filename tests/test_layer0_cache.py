"""Tests for Layer 0 cache."""

from src.core.layer0_cache import Layer0Cache


def test_cache_set_and_get():
    cache = Layer0Cache({"layer0_cache_max_entries": 10, "layer0_cache_ttl_seconds": 3600})
    cache.set("play:en nuit", "results here")
    assert cache.get("play:en nuit") == "results here"


def test_cache_miss():
    cache = Layer0Cache({})
    assert cache.get("missing") is None
