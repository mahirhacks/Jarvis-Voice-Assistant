"""Tier 2 music enrichment pipeline."""

import logging
from typing import Optional

from src.core.models import YouTubeCandidate
from src.music.candidate_ranker import clean_query, entry_to_candidate_dict, rank_candidates
from src.music.entity_lookup import EntityLookup
from src.music.google_dym import google_dym
from src.music.search_cache import SearchCache
from src.music.vocabulary_matcher import match_vocabulary
from src.music.youtube_search import YouTubeSearch

logger = logging.getLogger(__name__)

_NOISE_ENTITY_NAMES = frozenset({
    "youtube", "played", "play", "music", "video", "song",
})


class EnrichPipeline:
    def __init__(
        self,
        settings: dict,
        cache: SearchCache,
        entity_lookup: EntityLookup,
        youtube: YouTubeSearch,
        vocabulary: dict,
        ollama_client=None,
    ):
        self._settings = settings
        self._cache = cache
        self._entity_lookup = entity_lookup
        self._youtube = youtube
        self._vocabulary = vocabulary
        self._ollama = ollama_client

    def build_queries(self, transcript: str) -> list[str]:
        queries: list[str] = []
        cached = self._cache.lookup_correction(transcript)
        if cached:
            queries.append(cached)

        cleaned = clean_query(transcript)
        vocab = match_vocabulary(transcript, self._vocabulary)
        for v in vocab:
            if v.canonical_form not in queries:
                queries.append(v.canonical_form)

        lookup_text = cleaned or transcript
        enrichment = self._entity_lookup.enrich(lookup_text, vocab)
        for e in enrichment.musicbrainz_entities[:5]:
            name = (e.name or "").strip()
            if not name or name.lower() in _NOISE_ENTITY_NAMES:
                continue
            q = f"{e.name} {e.artist}" if e.artist else e.name
            if q.strip() not in queries:
                queries.append(q.strip())
                if len([x for x in queries if x not in (cleaned, transcript)]) >= 3:
                    break
        for e in enrichment.wikidata_entities[:2]:
            if e.label not in queries:
                queries.append(e.label)

        if not self._cache.lookup_entity(transcript) and not vocab:
            dym = google_dym(transcript, self._settings)
            if dym and dym not in queries:
                queries.append(dym)

        if cleaned and cleaned not in queries:
            queries.insert(0, cleaned)
        if not queries:
            queries = [transcript]
        return queries[:5]

    def search_enriched(self, transcript: str) -> tuple[list[YouTubeCandidate], list[tuple[dict, float]]]:
        queries = self.build_queries(transcript)
        logger.info("[Enrich] queries: %s", queries)
        raw_candidates, _ = self._youtube.multi_search(queries)
        entries = [c.model_dump() for c in raw_candidates]
        ranked = rank_candidates(entries, clean_query(transcript))
        candidates = [
            YouTubeCandidate(**entry_to_candidate_dict(e)) for e, _ in ranked
        ]
        return candidates, ranked
