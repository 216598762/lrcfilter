"""Tests for lyrics module."""

from lrcfilter.lyrics import _calculate_match_score
from lrcfilter.models import TrackMetadata


def test_calculate_match_score_basic() -> None:
    """Test match score with basic metadata."""
    metadata = TrackMetadata(
        title="Hello World",
        artist="Test Artist",
        album=None,
        duration_seconds=None,
    )
    lrclib_result = {"trackName": "Hello World", "artistName": "Test Artist"}
    score = _calculate_match_score(metadata, lrclib_result)
    assert score == 1.0


def test_calculate_match_score() -> None:
    """Test match score calculation."""
    metadata = TrackMetadata(
        title="Test Song",
        artist="Test Artist",
        album=None,
        duration_seconds=None,
    )
    
    # Perfect match
    lrclib_result = {
        "trackName": "Test Song",
        "artistName": "Test Artist",
    }
    score = _calculate_match_score(metadata, lrclib_result)
    assert score == 1.0
    
    # Partial match
    lrclib_result = {
        "trackName": "Test Song (Remix)",
        "artistName": "Test Artist",
    }
    score = _calculate_match_score(metadata, lrclib_result)
    assert 0.0 < score < 1.0
    
    # No match
    lrclib_result = {
        "trackName": "Different Song",
        "artistName": "Different Artist",
    }
    score = _calculate_match_score(metadata, lrclib_result)
    assert score == 0.0


def test_calculate_match_score_missing_metadata() -> None:
    """Test match score with missing metadata."""
    # Missing title in file metadata
    metadata = TrackMetadata(
        title=None,
        artist="Test Artist",
        album=None,
        duration_seconds=None,
    )
    
    lrclib_result = {
        "trackName": "Test Song",
        "artistName": "Test Artist",
    }
    
    score = _calculate_match_score(metadata, lrclib_result)
    # Should only consider artist match
    assert 0.0 < score <= 1.0


def test_lyrics_result_dataclass() -> None:
    """Test LyricsResult dataclass creation."""
    from lrcfilter.models import LyricsResult
    
    result = LyricsResult(
        source="lrclib",
        synced_lyrics="[00:00.00] Line 1",
        plain_lyrics="Line 1",
        matched_track_name="Test Song",
        matched_artist_name="Test Artist",
        matched_album_name="Test Album",
        match_score=0.95,
    )
    
    assert result.source == "lrclib"
    assert result.synced_lyrics == "[00:00.00] Line 1"
    assert result.plain_lyrics == "Line 1"
    assert result.matched_track_name == "Test Song"
    assert result.matched_artist_name == "Test Artist"
    assert result.matched_album_name == "Test Album"
    assert result.match_score == 0.95
