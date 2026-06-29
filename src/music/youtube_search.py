"""YouTube search via yt-dlp with parallel multi-search."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

from src.core.models import YouTubeCandidate

logger = logging.getLogger(__name__)

_YDL_SEARCH_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "ignoreerrors": True,
    # Flat playlist/search: metadata only — avoids per-video player requests
    # that trigger "not available on this app" and add ~15s+ per search.
    "extract_flat": "in_playlist",
}


class YouTubeSearch:
    def __init__(self, settings: dict):
        self._num_results = settings.get("youtube_search_results", 15)

    def search(self, query: str) -> list[YouTubeCandidate]:
        search_query = f"ytsearch{self._num_results}:{query}"
        t0 = time.time()
        try:
            with yt_dlp.YoutubeDL(_YDL_SEARCH_OPTS) as ydl:
                result = ydl.extract_info(search_query, download=False)
        except Exception:
            logger.exception("yt-dlp failed for '%s'", query)
            return []
        logger.info("[YouTube] '%s' took %.2fs", query, time.time() - t0)
        if not result:
            return []
        candidates = []
        for entry in result.get("entries") or []:
            if entry is None:
                continue
            try:
                candidates.append(YouTubeCandidate(
                    title=entry.get("title", ""),
                    video_id=entry.get("id", ""),
                    uploader=entry.get("uploader", ""),
                    duration=int(entry.get("duration") or 0),
                    view_count=int(entry.get("view_count") or 0),
                    webpage_url=entry.get("webpage_url", ""),
                ))
            except Exception:
                continue
        return candidates

    def multi_search(self, queries: list[str]) -> tuple[list[YouTubeCandidate], int]:
        all_candidates: list[YouTubeCandidate] = []
        unique_queries = list(dict.fromkeys(q for q in queries if q and q.strip()))
        if not unique_queries:
            return [], 0
        if len(unique_queries) == 1:
            results = self.search(unique_queries[0])
            return results, len(results)

        with ThreadPoolExecutor(max_workers=min(5, len(unique_queries))) as pool:
            futures = {pool.submit(self.search, q): q for q in unique_queries}
            for fut in as_completed(futures):
                try:
                    all_candidates.extend(fut.result())
                except Exception:
                    logger.exception("Parallel search failed for '%s'", futures[fut])

        total_before = len(all_candidates)
        seen: set[str] = set()
        deduped: list[YouTubeCandidate] = []
        for c in all_candidates:
            if c.video_id and c.video_id not in seen:
                seen.add(c.video_id)
                deduped.append(c)
        return deduped, total_before
