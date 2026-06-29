"""Tests for search cache."""

import json
import tempfile
from pathlib import Path

from src.music.search_cache import SearchCache


def test_correction_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        corr = Path(tmp) / "corr.json"
        ent = Path(tmp) / "ent.json"
        settings = {
            "music_corrections_cache_path": str(corr),
            "music_entity_cache_path": str(ent),
            "music_entity_cache_enabled": True,
        }
        cache = SearchCache(settings)
        cache.save_correction("the marinas", "The Marías")
        assert cache.lookup_correction("The Marinas") == "The Marías"
        assert corr.exists()
