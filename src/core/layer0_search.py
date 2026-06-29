"""Layer 0 — gather context: music APIs, DDG lite, then browser search fallback."""

import logging
from typing import Any, Optional

from src.core.layer0_cache import Layer0Cache
from src.core.layer0_gates import build_search_query, quick_verb_hint, should_skip_web_search
from src.core.models import GatherResult
from src.core.search_lite import ddg_lite_search
from src.music.vocabulary_matcher import match_vocabulary

logger = logging.getLogger(__name__)


def build_music_context(
    transcript: str,
    vocabulary: dict[str, Any],
    entity_lookup,
) -> str:
    """MusicBrainz + vocabulary — faster than web for songs."""
    lines = ["Local music lookup:", ""]
    vocab = match_vocabulary(transcript, vocabulary)
    for v in vocab:
        lines.append(f"- Vocabulary: {v.matched_term} → {v.canonical_form}")

    enrichment = entity_lookup.enrich(transcript, vocab)
    for e in enrichment.musicbrainz_entities[:4]:
        label = f"{e.name} by {e.artist}" if e.artist else e.name
        lines.append(f"- MusicBrainz: {label} (score {e.score:.0f})")
    for e in enrichment.wikidata_entities[:2]:
        lines.append(f"- Wikidata: {e.label}")

    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


class Layer0Search:
    """Conditional search with cache, music APIs, lite HTTP, browser fallback."""

    def __init__(self, browser, settings: dict, entity_lookup=None, vocabulary: dict | None = None):
        self._browser = browser
        self._settings = settings
        self._entity_lookup = entity_lookup
        self._vocabulary = vocabulary or {}
        self._cache = Layer0Cache(settings)

    def gather(self, transcript: str) -> GatherResult:
        query = transcript.strip()
        if len(query) < 2:
            return GatherResult(transcript=query)

        verb_hint = quick_verb_hint(query, self._settings)
        parts: list[str] = []

        if self._entity_lookup and self._vocabulary:
            music_ctx = build_music_context(query, self._vocabulary, self._entity_lookup)
            if music_ctx:
                parts.append(music_ctx)

        if not self._settings.get("layer0_search_enabled", True):
            return GatherResult(
                transcript=query,
                context="\n\n".join(parts),
                verb_hint=verb_hint,
                skipped_web=True,
            )

        if should_skip_web_search(query, verb_hint, self._settings):
            logger.info("[Layer0] Skipping web search (verb_hint=%s)", verb_hint)
            return GatherResult(
                transcript=query,
                context="\n\n".join(parts),
                verb_hint=verb_hint,
                skipped_web=True,
            )

        cache_key = f"{verb_hint or 'any'}:{query.lower()}"
        cached = self._cache.get(cache_key)
        if cached:
            parts.append(cached)
            return GatherResult(
                transcript=query,
                context="\n\n".join(parts),
                verb_hint=verb_hint,
                skipped_web=False,
                from_cache=True,
            )

        search_query = build_search_query(query, verb_hint)
        max_chars = self._settings.get("browse_max_text_chars", 6000)

        web_text = ddg_lite_search(
            search_query,
            max_chars=max_chars,
            timeout=self._settings.get("layer0_lite_timeout_seconds", 8),
        )
        if self._is_usable(web_text):
            logger.info("[Layer0] DDG lite returned %d chars", len(web_text))
            parts.append(web_text)
            self._cache.set(cache_key, web_text)
            return GatherResult(
                transcript=query,
                context="\n\n".join(parts),
                verb_hint=verb_hint,
            )

        if self._browser:
            engines = self._settings.get("layer0_search_engines", ["google", "bing", "yahoo"])
            for engine in engines:
                try:
                    text = self._browser.search_engine(engine, search_query)
                    if self._is_usable(text):
                        logger.info("[Layer0] %s returned %d chars", engine, len(text))
                        parts.append(text)
                        self._cache.set(cache_key, text)
                        return GatherResult(
                            transcript=query,
                            context="\n\n".join(parts),
                            verb_hint=verb_hint,
                        )
                    logger.info("[Layer0] %s result too thin — trying next", engine)
                except Exception as exc:
                    logger.warning("[Layer0] %s failed: %s", engine, exc)

        logger.warning("[Layer0] Web search unavailable — using local context only")
        return GatherResult(
            transcript=query,
            context="\n\n".join(parts),
            verb_hint=verb_hint,
            skipped_web=not parts,
        )

    @staticmethod
    def _is_usable(text: str) -> bool:
        if not text or len(text.strip()) < 80:
            return False
        lower = text.lower()
        if "captcha" in lower or "unusual traffic" in lower:
            return False
        return True
