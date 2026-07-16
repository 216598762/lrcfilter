"""Data models for LRCFilter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AudioFile:
    """Represents an audio file on disk."""

    path: Path
    filename: str
    extension: str
    size_mb: float


@dataclass
class TrackMetadata:
    """Metadata extracted from an audio file."""

    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    duration_seconds: Optional[float]
    raw_tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class LyricsResult:
    """Lyrics fetched from an online source."""

    source: str  # 'lrclib' or 'genius'
    synced_lyrics: Optional[str]  # LRC format with timestamps
    plain_lyrics: Optional[str]  # Plain text lyrics
    matched_track_name: str  # Track name from lyrics API
    matched_artist_name: str  # Artist name from lyrics API
    matched_album_name: Optional[str]  # Album name from lyrics API
    match_score: float  # Confidence in the match (0.0 to 1.0)


@dataclass
class Word:
    """A single word with timing information."""

    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    """A transcription segment with timing."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription."""

    text: str
    segments: list[Segment]
    language: str
    duration: float
    has_speech: bool  # Based on VAD results


@dataclass
class CensorshipResult:
    """Result from censorship detection."""

    is_censored: bool
    mismatch_score: float  # 0.0 to 1.0
    profanity_count: int
    confidence: float
    details: str


@dataclass
class InstrumentalResult:
    """Result from instrumental detection."""

    is_instrumental: bool
    word_count: int
    speech_duration: float
    confidence: float


@dataclass
class MismatchResult:
    """Result from metadata mismatch detection."""

    is_mismatch: bool
    title_score: float  # 0.0 to 1.0 similarity
    artist_score: float  # 0.0 to 1.0 similarity
    duration_difference: float  # Seconds
    confidence: float  # 0.0 to 1.0
    details: str  # Human-readable explanation
