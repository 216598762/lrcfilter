"""Tests for mismatch module."""

from lrcfilter.mismatch import (
    detect_metadata_mismatch,
    _calculate_title_similarity,
    _calculate_artist_similarity,
)
from lrcfilter.models import TrackMetadata, LyricsResult, MismatchResult
from lrcfilter.utils import normalize_for_mismatch


def test_normalize_text_mismatch() -> None:
    """Test text normalization for mismatch detection."""
    assert normalize_for_mismatch("Hello World") == "hello world"
    assert normalize_for_mismatch("  extra   spaces  ") == "extra spaces"
    assert normalize_for_mismatch("Song (Remix)") == "song"
    assert normalize_for_mismatch("Song (Live)") == "song"
    # Test new suffixes
    assert normalize_for_mismatch("Song (Deluxe Edition)") == "song"
    assert normalize_for_mismatch("Song (Remastered)") == "song"


def test_calculate_title_similarity_identical() -> None:
    """Test title similarity for identical titles."""
    score = _calculate_title_similarity("Test Song", "Test Song")
    assert score == 1.0


def test_calculate_title_similarity_different() -> None:
    """Test title similarity for different titles."""
    score = _calculate_title_similarity("Love Song", "Heavy Metal Anthem")
    assert score < 0.5


def test_calculate_title_similarity_partial() -> None:
    """Test title similarity for partially matching titles."""
    score = _calculate_title_similarity("Test Song", "Test Song Remix")
    assert 0.5 < score < 1.0


def test_calculate_title_similarity_empty() -> None:
    """Test title similarity with empty strings."""
    score = _calculate_title_similarity("", "Test Song")
    assert score == 0.0
    
    score = _calculate_title_similarity("Test Song", "")
    assert score == 0.0


def test_calculate_artist_similarity_identical() -> None:
    """Test artist similarity for identical artists."""
    score = _calculate_artist_similarity("Test Artist", "Test Artist")
    assert score == 1.0


def test_calculate_artist_similarity_different() -> None:
    """Test artist similarity for different artists."""
    score = _calculate_artist_similarity("The Beatles", "Black Sabbath")
    assert score < 0.5


def test_detect_metadata_mismatch_match() -> None:
    """Test mismatch detection when metadata matches."""
    file_metadata = TrackMetadata(
        title="Test Song",
        artist="Test Artist",
        album=None,
        duration_seconds=180.0,
    )
    
    lyrics_result = LyricsResult(
        source="lrclib",
        synced_lyrics=None,
        plain_lyrics="Lyrics here",
        matched_track_name="Test Song",
        matched_artist_name="Test Artist",
        matched_album_name=None,
        match_score=1.0,
    )
    
    result = detect_metadata_mismatch(file_metadata, lyrics_result)
    
    assert isinstance(result, MismatchResult)
    assert result.is_mismatch is False
    assert result.title_score == 1.0
    assert result.artist_score == 1.0


def test_detect_metadata_mismatch_mismatch() -> None:
    """Test mismatch detection when metadata doesn't match."""
    file_metadata = TrackMetadata(
        title="Love Song",
        artist="The Beatles",
        album=None,
        duration_seconds=180.0,
    )
    
    lyrics_result = LyricsResult(
        source="lrclib",
        synced_lyrics=None,
        plain_lyrics="Lyrics here",
        matched_track_name="Heavy Metal Anthem",
        matched_artist_name="Black Sabbath",
        matched_album_name=None,
        match_score=0.0,
    )
    
    result = detect_metadata_mismatch(file_metadata, lyrics_result)
    
    assert isinstance(result, MismatchResult)
    assert result.is_mismatch is True
    assert result.title_score < 0.5
    assert result.artist_score < 0.5


def test_detect_metadata_mismatch_missing_metadata() -> None:
    """Test mismatch detection with missing metadata."""
    file_metadata = TrackMetadata(
        title=None,
        artist=None,
        album=None,
        duration_seconds=None,
    )
    
    lyrics_result = LyricsResult(
        source="lrclib",
        synced_lyrics=None,
        plain_lyrics="Lyrics here",
        matched_track_name="Test Song",
        matched_artist_name="Test Artist",
        matched_album_name=None,
        match_score=0.5,
    )
    
    result = detect_metadata_mismatch(file_metadata, lyrics_result)
    
    # Should handle missing metadata gracefully
    assert isinstance(result, MismatchResult)
    assert result.is_mismatch is True


def test_mismatch_result_dataclass() -> None:
    """Test MismatchResult dataclass creation."""
    result = MismatchResult(
        is_mismatch=True,
        title_score=0.3,
        artist_score=0.4,
        duration_difference=45.0,
        confidence=0.8,
        details="Title and artist mismatch",
    )
    
    assert result.is_mismatch is True
    assert result.title_score == 0.3
    assert result.artist_score == 0.4
    assert result.duration_difference == 45.0
    assert result.confidence == 0.8
    assert result.details == "Title and artist mismatch"
