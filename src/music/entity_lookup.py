"""MusicBrainz and Wikidata enrichment with parallel lookups."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

from src.core.models import EntityEnrichmentResult, MusicBrainzEntity, WikidataEntity, VocabularyMatch
from src.music.search_cache import SearchCache

logger = logging.getLogger(__name__)


class EntityLookup:
    def __init__(self, settings: dict, cache: SearchCache):
        self._settings = settings
        self._cache = cache
        self._enabled = settings.get("music_entity_lookup_enabled", True)
        self._mb_enabled = settings.get("musicbrainz_enabled", True)
        self._wd_enabled = settings.get("wikidata_enabled", True)
        self._mb_rate = settings.get("musicbrainz_rate_limit_seconds", 1.1)
        self._wd_rate = settings.get("wikidata_rate_limit_seconds", 1.1)
        self._timeout = settings.get("entity_lookup_timeout_seconds", 5)
        self._ua = settings.get("entity_lookup_user_agent", "Jarvis3.0/1.0")
        self._last_call: dict[str, float] = {}

    def enrich(
        self,
        raw_transcript: str,
        vocabulary_matches: list[VocabularyMatch] | None = None,
    ) -> EntityEnrichmentResult:
        if not self._enabled:
            return EntityEnrichmentResult(raw_transcript=raw_transcript)

        cached = self._cache.lookup_entity(raw_transcript)
        if cached:
            return EntityEnrichmentResult(
                raw_transcript=raw_transcript,
                musicbrainz_entities=self._parse_mb(cached),
                wikidata_entities=self._parse_wd(cached),
            )

        mb_entities: list[MusicBrainzEntity] = []
        wd_entities: list[WikidataEntity] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            mb_fut = pool.submit(self._lookup_musicbrainz, raw_transcript) if self._mb_enabled else None
            wd_fut = pool.submit(self._lookup_wikidata, raw_transcript) if self._wd_enabled else None
            if mb_fut:
                mb_entities = mb_fut.result()
            if wd_fut:
                wd_entities = wd_fut.result()

        if mb_entities or wd_entities:
            self._cache.save_entity(raw_transcript, {
                "musicbrainz_entities": [e.model_dump() for e in mb_entities],
                "wikidata_entities": [e.model_dump() for e in wd_entities],
            })

        return EntityEnrichmentResult(
            raw_transcript=raw_transcript,
            musicbrainz_entities=mb_entities,
            wikidata_entities=wd_entities,
            vocabulary_matches=vocabulary_matches or [],
        )

    def _lookup_musicbrainz(self, query: str) -> list[MusicBrainzEntity]:
        entities: list[MusicBrainzEntity] = []
        headers = {"User-Agent": self._ua}

        def fetch_type(entity_type: str) -> list[MusicBrainzEntity]:
            out: list[MusicBrainzEntity] = []
            try:
                self._rate_limit("musicbrainz", self._mb_rate)
                url = f"https://musicbrainz.org/ws/2/{entity_type}/"
                resp = requests.get(
                    url, params={"query": query, "fmt": "json", "limit": "5"},
                    headers=headers, timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                items_key = "release-groups" if entity_type == "release-group" else entity_type + "s"
                model_type = "release" if entity_type == "release-group" else entity_type
                for item in data.get(items_key, []):
                    name = item.get("name") or item.get("title", "")
                    artist_name = None
                    if entity_type in ("recording", "release-group"):
                        ac = item.get("artist-credit", [])
                        if ac:
                            artist_name = ac[0].get("name") or ac[0].get("artist", {}).get("name")
                    out.append(MusicBrainzEntity(
                        type=model_type, name=name, artist=artist_name,
                        score=float(item.get("score", 0)),
                    ))
            except Exception as exc:
                logger.warning("[MusicBrainz] %s failed: %s", entity_type, exc)
            return out

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(fetch_type, t) for t in ("artist", "recording", "release-group")]
            for fut in futs:
                entities.extend(fut.result())
        return entities

    def _lookup_wikidata(self, query: str) -> list[WikidataEntity]:
        try:
            self._rate_limit("wikidata", self._wd_rate)
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={"action": "wbsearchentities", "search": query, "format": "json", "language": "en"},
                headers={"User-Agent": self._ua}, timeout=self._timeout,
            )
            resp.raise_for_status()
            entities = []
            for idx, item in enumerate(resp.json().get("search", [])):
                entities.append(WikidataEntity(
                    label=item.get("label", ""),
                    description=item.get("description"),
                    score=max(0.0, min(1.0, 1.0 - idx * 0.1)),
                ))
            return entities
        except Exception as exc:
            logger.warning("[Wikidata] failed: %s", exc)
            return []

    def _rate_limit(self, api: str, limit: float) -> None:
        last = self._last_call.get(api, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_call[api] = time.monotonic()

    @staticmethod
    def _parse_mb(cached: dict) -> list[MusicBrainzEntity]:
        out = []
        for item in cached.get("musicbrainz_entities", []):
            try:
                out.append(MusicBrainzEntity(**item))
            except Exception:
                pass
        return out

    @staticmethod
    def _parse_wd(cached: dict) -> list[WikidataEntity]:
        out = []
        for item in cached.get("wikidata_entities", []):
            try:
                out.append(WikidataEntity(**item))
            except Exception:
                pass
        return out
