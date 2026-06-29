"""Search corrections and entity cache."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SearchCache:
    def __init__(self, settings: dict):
        self._corrections_path = settings.get(
            "music_corrections_cache_path", "config/music_corrections_cache.json"
        )
        self._entity_path = settings.get(
            "music_entity_cache_path", "config/music_entity_cache.json"
        )
        self._entity_enabled = settings.get("music_entity_cache_enabled", True)
        self._corrections = self._read_json(self._corrections_path)
        self._entities = (
            self._read_json(self._entity_path) if self._entity_enabled else {}
        )

    def lookup_correction(self, raw_transcript: str) -> Optional[str]:
        return self._corrections.get(self._normalize(raw_transcript))

    def save_correction(self, raw_transcript: str, corrected: str) -> None:
        self._corrections[self._normalize(raw_transcript)] = corrected
        self._write_json(self._corrections_path, self._corrections)

    def lookup_entity(self, query: str) -> Optional[dict]:
        if not self._entity_enabled:
            return None
        return self._entities.get(self._normalize(query))

    def save_entity(self, query: str, entities: dict) -> None:
        if not self._entity_enabled:
            return
        key = self._normalize(query)
        entities["cached_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._entities[key] = entities
        self._write_json(self._entity_path, self._entities)

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _read_json(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _write_json(path: str, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("Failed to write cache %s: %s", path, e)
