"""Tests for metadata module."""

from pathlib import Path

from lrcfilter.metadata import extract_metadata
from lrcfilter.models import AudioFile, TrackMetadata


def test_extract_metadata_nonexistent_file(tmp_path: Path) -> None:
    """Test extracting metadata from a nonexistent file."""
    audio_file = AudioFile(
        path=tmp_path / "nonexistent.mp3",
        filename="nonexistent.mp3",
        extension=".mp3",
        size_mb=0.0,
    )

    result = extract_metadata(audio_file)

    # Should return empty metadata, not raise exception
    assert isinstance(result, TrackMetadata)
    assert result.title == "nonexistent.mp3"
    assert result.artist is None
    assert result.album is None
    assert result.duration_seconds is None


def test_extract_metadata_empty_file(tmp_path: Path) -> None:
    """Test extracting metadata from an empty file."""
    test_file = tmp_path / "empty.mp3"
    test_file.touch()

    audio_file = AudioFile(
        path=test_file,
        filename="empty.mp3",
        extension=".mp3",
        size_mb=0.0,
    )

    result = extract_metadata(audio_file)

    # Should return empty metadata
    assert isinstance(result, TrackMetadata)
    assert result.title == "empty.mp3"


def test_track_metadata_dataclass() -> None:
    """Test TrackMetadata dataclass creation."""
    metadata = TrackMetadata(
        title="Test Song",
        artist="Test Artist",
        album="Test Album",
        duration_seconds=180.0,
        raw_tags={"key": "value"},
    )

    assert metadata.title == "Test Song"
    assert metadata.artist == "Test Artist"
    assert metadata.album == "Test Album"
    assert metadata.duration_seconds == 180.0
    assert metadata.raw_tags == {"key": "value"}


def test_track_metadata_defaults() -> None:
    """Test TrackMetadata default values."""
    metadata = TrackMetadata(
        title=None,
        artist=None,
        album=None,
        duration_seconds=None,
    )

    assert metadata.title is None
    assert metadata.artist is None
    assert metadata.album is None
    assert metadata.duration_seconds is None
    assert metadata.raw_tags == {}
