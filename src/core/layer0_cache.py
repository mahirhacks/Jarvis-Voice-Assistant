"""In-memory cache for Layer 0 search results."""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class Layer0Cache:
    def __init__(self, settings: dict):
        self._entries: dict[str, tuple[str, float]] = {}
        self._max = settings.get("layer0_cache_max_entries", 200)
        self._ttl = settings.get("layer0_cache_ttl_seconds", 86400)

    def get(self, key: str) -> Optional[str]:
        normalised = key.strip().lower()
        entry = self._entries.get(normalised)
        if not entry:
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._entries[normalised]
            return None
        logger.info("[Layer0Cache] hit for %r", normalised[:60])
        return value

    def set(self, key: str, value: str) -> None:
        if not value or not value.strip():
            return
        normalised = key.strip().lower()
        if len(self._entries) >= self._max:
            oldest = min(self._entries, key=lambda k: self._entries[k][1])
            del self._entries[oldest]
        self._entries[normalised] = (value, time.time())
