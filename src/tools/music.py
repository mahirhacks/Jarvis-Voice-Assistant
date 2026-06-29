"""Music search and playback tools."""

import logging
import webbrowser
from typing import Optional

from src.core.confirmation import ask_confirmation
from src.core.models import YouTubeCandidate
from src.core.session import Session
from src.core.utils import confidence_gate
from src.music.candidate_ranker import (
    clean_query, entry_to_candidate_dict, normalize_score_to_confidence, rank_candidates,
)
from src.music.enrich_pipeline import EnrichPipeline
from src.music.search_cache import SearchCache
from src.music.youtube_search import YouTubeSearch
from src.voice import tts

logger = logging.getLogger(__name__)


class MusicTools:
    def __init__(
        self,
        settings: dict,
        session: Session,
        cache: SearchCache,
        youtube: YouTubeSearch,
        enrich: EnrichPipeline,
        stt,
    ):
        self._settings = settings
        self._session = session
        self._cache = cache
        self._youtube = youtube
        self._enrich = enrich
        self._stt = stt
        self._auto_thresh = settings.get("youtube_auto_play_threshold", 82)
        self._confirm_thresh = settings.get("youtube_confirm_threshold", 65)

    def search_youtube(self, query: str, raw_transcript: str = "") -> str:
        raw = raw_transcript or query
        search_query = query

        cached = self._cache.lookup_correction(raw)
        if cached:
            search_query = cached
            logger.info("[Music] Cache hit: '%s' -> '%s'", raw, cached)

        cleaned = clean_query(search_query)
        if len(cleaned) < 2:
            return "I didn't catch the song name. Please repeat it."

        tts.speak(f"Searching for {cleaned}.", blocking=False)
        entries = [c.model_dump() for c in self._youtube.search(cleaned)]
        ranked = rank_candidates(entries, cleaned)

        if not ranked or ranked[0][1] < self._confirm_thresh:
            if self._settings.get("music_enrichment_enabled", True):
                logger.info("[Music] Tier 1 low confidence — trying enrichment")
                candidates, ranked = self._enrich.search_enriched(raw)
                if not ranked:
                    return "I could not find that song."
                entries = [c.model_dump() for c in candidates]
                ranked = [(e, s) for e, s in rank_candidates(entries, cleaned)]
            else:
                return "I could not confidently find that song."

        best_entry, best_score = ranked[0]
        candidates = [
            YouTubeCandidate(**entry_to_candidate_dict(e)) for e, _ in ranked[:15]
        ]
        self._session.update_search(raw, cleaned, candidates)

        confidence = normalize_score_to_confidence(best_score, self._auto_thresh)
        gate = confidence_gate(
            confidence,
            self._settings.get("music_auto_play_confidence", 0.85),
            self._settings.get("music_confirm_confidence", 0.65),
        )

        title = best_entry.get("title", "Unknown")
        channel = best_entry.get("uploader", best_entry.get("channel", "Unknown"))

        if gate == "auto_play":
            return self._play_entry(best_entry, raw, cleaned)

        if gate == "confirm":
            if ask_confirmation(
                self._stt,
                f"I found {title} by {channel}. Should I play it?",
                self._settings.get("confirmation_max_attempts", 2),
            ):
                return self._play_entry(best_entry, raw, cleaned)
            return "Cancelled."

        if len(ranked) >= 2 and ranked[1][1] >= self._settings.get("youtube_reject_threshold", 55):
            second = ranked[1][0]
            if ask_confirmation(
                self._stt,
                f"Did you mean {title}, or {second.get('title', 'the other result')}?",
                self._settings.get("confirmation_max_attempts", 2),
            ):
                return self._play_entry(best_entry, raw, cleaned)
        return "I'm not sure which song you meant. Could you be more specific?"

    def play_video(self, video_id: str) -> str:
        if not video_id:
            return "No video to play."
        url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(url)
        return f"Playing video."

    def next_candidate(self) -> str:
        state = self._session.get_state()
        candidates = state.last_candidates
        if not candidates:
            return "There are no search results to browse."
        idx = state.current_candidate_index
        if idx >= len(candidates) - 1:
            return "No more results."
        state.current_candidate_index = idx + 1
        c = candidates[state.current_candidate_index]
        if ask_confirmation(self._stt, f"Result {state.current_candidate_index + 1}: {c.title} by {c.uploader}. Play it?"):
            return self._play_candidate(c)
        return "Cancelled."

    def previous_candidate(self) -> str:
        state = self._session.get_state()
        candidates = state.last_candidates
        if not candidates:
            return "There are no search results to browse."
        idx = state.current_candidate_index
        if idx <= 0:
            return "No more results."
        state.current_candidate_index = idx - 1
        c = candidates[state.current_candidate_index]
        if ask_confirmation(self._stt, f"Result {state.current_candidate_index + 1}: {c.title} by {c.uploader}. Play it?"):
            return self._play_candidate(c)
        return "Cancelled."

    def replay_last(self) -> str:
        last = self._session.get_state().last_played_video
        if not last or not last.video_id:
            return "There's nothing to replay yet."
        webbrowser.open(last.webpage_url or f"https://www.youtube.com/watch?v={last.video_id}")
        return f"Replaying {last.title}."

    def _play_entry(self, entry: dict, raw: str, corrected: str) -> str:
        candidate = YouTubeCandidate(**entry_to_candidate_dict(entry))
        cache_value = clean_query(corrected) or corrected
        if len(cache_value) >= 3:
            self._cache.save_correction(raw, cache_value)
        return self._play_candidate(candidate)

    def _play_candidate(self, candidate: YouTubeCandidate) -> str:
        url = candidate.webpage_url or f"https://www.youtube.com/watch?v={candidate.video_id}"
        webbrowser.open(url)
        self._session.update_played(candidate)
        return f"Playing {candidate.title}."
