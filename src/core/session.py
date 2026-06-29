"""In-memory session state."""

from src.core.models import SessionState, YouTubeCandidate


class Session:
    def __init__(self) -> None:
        self._state = SessionState()

    def update_search(self, request: str, query: str, candidates: list[YouTubeCandidate]) -> None:
        self._state.last_user_request = request
        self._state.last_search_query = query
        self._state.last_candidates = candidates
        self._state.current_candidate_index = 0

    def update_played(self, video: YouTubeCandidate) -> None:
        self._state.last_played_video = video
        self._state.last_artist = video.uploader
        self._state.last_song = video.title

    def add_rejected(self, video: YouTubeCandidate) -> None:
        self._state.rejected_videos.append(video)

    def get_context(self) -> dict:
        return self._state.model_dump()

    def get_state(self) -> SessionState:
        return self._state
