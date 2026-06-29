"""Shared data models."""

from typing import Optional

from pydantic import BaseModel, Field


class YouTubeCandidate(BaseModel):
    title: str
    video_id: str
    uploader: str
    duration: int = 0
    view_count: int = 0
    webpage_url: str = ""


class VocabularyMatch(BaseModel):
    matched_term: str
    canonical_form: str
    match_type: str = "correction"
    score: float = Field(ge=0.0, le=1.0)


class MusicBrainzEntity(BaseModel):
    type: str
    name: str
    artist: Optional[str] = None
    score: float = Field(ge=0.0, le=100.0)
    source: str = "musicbrainz"


class WikidataEntity(BaseModel):
    type: str = "entity"
    label: str
    description: Optional[str] = None
    score: float = Field(ge=0.0, le=1.0)
    source: str = "wikidata"


class SessionState(BaseModel):
    last_user_request: Optional[str] = None
    last_search_query: Optional[str] = None
    last_candidates: list[YouTubeCandidate] = []
    last_played_video: Optional[YouTubeCandidate] = None
    rejected_videos: list[YouTubeCandidate] = []
    last_artist: Optional[str] = None
    last_song: Optional[str] = None
    current_candidate_index: int = 0


class EntityEnrichmentResult(BaseModel):
    raw_transcript: str
    musicbrainz_entities: list[MusicBrainzEntity] = []
    wikidata_entities: list[WikidataEntity] = []
    vocabulary_matches: list[VocabularyMatch] = []
    google_dym_correction: Optional[str] = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    message: str = ""
    data: dict = Field(default_factory=dict)


class LayerPlan(BaseModel):
    """Output of Layer 1 intent finding."""

    transcript: str
    verb: str
    phrase: str = ""
    tool: str
    search_context: str = ""
    confidence: str = "medium"
    source: str = "layer1"


class GatherResult(BaseModel):
    """Output of Layer 0 information gathering."""

    transcript: str
    context: str = ""
    verb_hint: Optional[str] = None
    skipped_web: bool = False
    from_cache: bool = False


class ResolvedIntent(BaseModel):
    action: str
    objective: str = ""
    confidence: str = "medium"
    raw_response: str = ""
