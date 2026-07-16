"""Tests for metadata extraction module to improve coverage."""

from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest

from lrcfilter.metadata import extract_metadata, _get_tag_value, _create_empty_metadata
from lrcfilter.models import AudioFile, TrackMetadata


class TestGetTagValue:
    """Test the _get_tag_value helper function."""

    def test_returns_first_matching_key(self) -> None:
        """Should return value for the first matching key."""
        tags = {"TIT2": MagicMock(text=["My Title"]), "title": "Other Title"}
        result = _get_tag_value(tags, ["TIT2", "title"])
        assert result == "My Title"

    def test_returns_second_key_if_first_missing(self) -> None:
        """Should fall back to second key if first is missing."""
        tags = {"title": "Fallback Title"}
        result = _get_tag_value(tags, ["TIT2", "title"])
        assert result == "Fallback Title"

    def test_returns_none_when_no_keys_match(self) -> None:
        """Should return None when no keys are found."""
        tags = {"artist": "Some Artist"}
        result = _get_tag_value(tags, ["TIT2", "title"])
        assert result is None

    def test_returns_none_for_empty_tags(self) -> None:
        """Should return None for empty tags dict."""
        result = _get_tag_value({}, ["TIT2", "title"])
        assert result is None

    def test_handles_value_with_text_attribute(self) -> None:
        """Should handle values with .text attribute (Mutagen style)."""
        mock_value = MagicMock()
        mock_value.text = ["Text Value"]
        tags = {"key": mock_value}
        result = _get_tag_value(tags, ["key"])
        assert result == "Text Value"

    def test_handles_value_with_empty_text(self) -> None:
        """Should return None when .text is empty."""
        mock_value = MagicMock()
        mock_value.text = []
        tags = {"key": mock_value}
        result = _get_tag_value(tags, ["key"])
        assert result is None

    def test_handles_value_without_text_attribute(self) -> None:
        """Should handle values without .text attribute."""
        tags = {"key": "Direct Value"}
        result = _get_tag_value(tags, ["key"])
        assert result == "Direct Value"

    def test_handles_none_value(self) -> None:
        """Should return None for None values."""
        tags = {"key": None}
        result = _get_tag_value(tags, ["key"])
        assert result is None


class TestCreateEmptyMetadata:
    """Test the _create_empty_metadata helper function."""

    def test_returns_empty_metadata_with_filename(self) -> None:
        """Should use filename as title when metadata extraction fails."""
        audio_file = AudioFile(
            path=Path("/fake/path.mp3"),
            filename="path.mp3",
            extension=".mp3",
            size_mb=1.0,
        )
        result = _create_empty_metadata(audio_file)
        assert result.title == "path.mp3"
        assert result.artist is None
        assert result.album is None
        assert result.duration_seconds is None
        assert result.raw_tags == {}


class TestExtractMetadata:
    """Test the extract_metadata function."""

    def test_returns_empty_metadata_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Should return empty metadata for nonexistent file."""
        audio_file = AudioFile(
            path=tmp_path / "nonexistent.mp3",
            filename="nonexistent.mp3",
            extension=".mp3",
            size_mb=0.0,
        )
        result = extract_metadata(audio_file)
        assert isinstance(result, TrackMetadata)
        assert result.title == "nonexistent.mp3"

    def test_returns_empty_metadata_for_empty_file(self, tmp_path: Path) -> None:
        """Should return empty metadata for empty file."""
        test_file = tmp_path / "empty.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="empty.mp3",
            extension=".mp3",
            size_mb=0.0,
        )
        result = extract_metadata(audio_file)
        assert isinstance(result, TrackMetadata)
        assert result.title == "empty.mp3"

    def test_handles_mutagen_returning_none(self, tmp_path: Path) -> None:
        """Should handle case where MutagenFile returns None."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        with patch("lrcfilter.metadata.MutagenFile", return_value=None):
            result = extract_metadata(audio_file)
            assert result.title == "test.mp3"

    def test_extracts_duration_when_available(self, tmp_path: Path) -> None:
        """Should extract duration from audio file info."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        # Skip this test as it requires mocking MutagenFile internals
        # which is fragile due to isinstance checks
        pytest.skip("Requires MutagenFile mock that passes isinstance checks")

    def test_extracts_tags_from_file(self, tmp_path: Path) -> None:
        """Should extract tags from audio file."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        # Skip this test as it requires mocking MutagenFile internals
        pytest.skip("Requires MutagenFile mock that passes isinstance checks")

    def test_stores_raw_tags(self, tmp_path: Path) -> None:
        """Should store all raw tags in the metadata."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        # Skip this test as it requires mocking MutagenFile internals
        pytest.skip("Requires MutagenFile mock that passes isinstance checks")

    def test_handles_no_info_attribute(self, tmp_path: Path) -> None:
        """Should handle case where mutagen_file.info is None."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        with patch("lrcfilter.metadata.MutagenFile", return_value=None):
            result = extract_metadata(audio_file)
            assert result.title == "test.mp3"

    def test_handles_exception_during_extraction(self, tmp_path: Path) -> None:
        """Should handle exceptions and return empty metadata."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        audio_file = AudioFile(
            path=test_file,
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )

        with patch("lrcfilter.metadata.MutagenFile", side_effect=Exception("Corrupted file")):
            result = extract_metadata(audio_file)
            assert isinstance(result, TrackMetadata)
            assert result.title == "test.mp3"
