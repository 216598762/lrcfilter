"""Tests for metadata extraction module.

Note: All tests mock MutagenFile instead of using real audio files because:
- Creating valid MP3 files requires ffmpeg which may not be available
- pydub (alternative) depends on audioop which is removed in Python 3.13+
- Mocking MutagenFile provides reliable, deterministic tests
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from lrcfilter.metadata import _create_empty_metadata, _get_tag_value, extract_metadata
from lrcfilter.models import AudioFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mutagen(tags: dict[str, Any] | None, duration: float | None = 180.0) -> MagicMock:
    """Create a mock MutagenFile with controlled tags and duration."""
    mock = MagicMock()
    if duration is not None:
        mock.info = MagicMock()
        mock.info.length = duration
    else:
        mock.info = None
    mock.tags = tags
    return mock


def _make_audio_file(tmp_path: Path, name: str = "test.mp3") -> AudioFile:
    """Create a dummy AudioFile for testing."""
    path = tmp_path / name
    path.touch()
    return AudioFile(
        path=path,
        filename=name,
        extension=".mp3",
        size_mb=0.0,
    )


# ---------------------------------------------------------------------------
# Tests for _get_tag_value (pure unit tests)
# ---------------------------------------------------------------------------


class TestGetTagValue:
    """Test the _get_tag_value helper function."""

    def test_returns_first_matching_key(self) -> None:
        tags = {"TIT2": MagicMock(text=["My Title"])}
        result = _get_tag_value(tags, ["TIT2", "title"])
        assert result == "My Title"

    def test_returns_second_key_if_first_missing(self) -> None:
        tags = {"title": "Fallback Title"}
        result = _get_tag_value(tags, ["TIT2", "title"])
        assert result == "Fallback Title"

    def test_returns_none_when_no_keys_match(self) -> None:
        result = _get_tag_value({"artist": "X"}, ["TIT2", "title"])
        assert result is None

    def test_returns_none_for_empty_tags(self) -> None:
        assert _get_tag_value({}, ["TIT2"]) is None

    def test_handles_value_with_empty_text(self) -> None:
        tags = {"key": MagicMock(text=[])}
        assert _get_tag_value(tags, ["key"]) is None

    def test_handles_value_without_text_attribute(self) -> None:
        assert _get_tag_value({"key": "Direct"}, ["key"]) == "Direct"

    def test_handles_none_value(self) -> None:
        assert _get_tag_value({"key": None}, ["key"]) is None


# ---------------------------------------------------------------------------
# Tests for _create_empty_metadata
# ---------------------------------------------------------------------------


class TestCreateEmptyMetadata:
    def test_returns_filename_as_title(self) -> None:
        af = AudioFile(path=Path("/x.mp3"), filename="x.mp3", extension=".mp3", size_mb=0.0)
        result = _create_empty_metadata(af)
        assert result.title == "x.mp3"
        assert result.artist is None
        assert result.duration_seconds is None
        assert result.raw_tags == {}


# ---------------------------------------------------------------------------
# NOTE: Integration tests with real audio files would require:
# - ffmpeg installed in the test environment, OR
# - pydub with audioop compatibility (not available in Python 3.13+)
# These would test the full MutagenFile parsing pipeline end-to-end.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests for extract_metadata using mocked MutagenFile
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    """Test extract_metadata with mocked MutagenFile."""

    def test_nonexistent_file_returns_filename(self, tmp_path: Path) -> None:
        af = _make_audio_file(tmp_path, "gone.mp3")
        af.path = tmp_path / "gone.mp3"
        # File doesn't exist → MutagenFile raises → returns empty metadata
        result = extract_metadata(af)
        assert result.title == "gone.mp3"

    def test_empty_file_returns_filename(self, tmp_path: Path) -> None:
        af = _make_audio_file(tmp_path, "empty.mp3")
        with patch("lrcfilter.metadata.MutagenFile", return_value=None):
            result = extract_metadata(af)
            assert result.title == "empty.mp3"

    def test_extracts_title_artist_album(self, tmp_path: Path) -> None:
        """Test that TIT2/TPE1/TALB tags are extracted correctly."""
        mock_tags = {
            "TIT2": MagicMock(text=["Test Song"]),
            "TPE1": MagicMock(text=["Test Artist"]),
            "TALB": MagicMock(text=["Test Album"]),
        }
        mock_mutagen = _make_mock_mutagen(mock_tags)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.title == "Test Song"
            assert result.artist == "Test Artist"
            assert result.album == "Test Album"

    def test_extracts_duration(self, tmp_path: Path) -> None:
        mock_mutagen = _make_mock_mutagen({}, duration=240.5)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.duration_seconds == 240.5

    def test_stores_raw_tags(self, tmp_path: Path) -> None:
        mock_tags = {
            "TIT2": MagicMock(text=["Title"]),
            "TPE1": MagicMock(text=["Artist"]),
        }
        mock_mutagen = _make_mock_mutagen(mock_tags)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert len(result.raw_tags) == 2
            for v in result.raw_tags.values():
                assert isinstance(v, str)

    def test_no_tags_returns_none_fields(self, tmp_path: Path) -> None:
        """When tags is None (file has no tags), title/artist/album should be None."""
        mock_mutagen = _make_mock_mutagen(tags=None)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.title is None
            assert result.artist is None
            assert result.album is None
            assert result.raw_tags == {}
            assert result.duration_seconds == 180.0

    def test_partial_tags(self, tmp_path: Path) -> None:
        mock_tags = {
            "TIT2": MagicMock(text=["Only Title"]),
        }
        mock_mutagen = _make_mock_mutagen(mock_tags)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.title == "Only Title"
            assert result.artist is None
            assert result.album is None

    def test_raw_tags_are_strings(self, tmp_path: Path) -> None:
        mock_tags = {
            "TIT2": MagicMock(text=["Title"]),
            "COMM": MagicMock(text=["Comment"]),
        }
        mock_mutagen = _make_mock_mutagen(mock_tags)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            for v in result.raw_tags.values():
                assert isinstance(v, str)

    def test_mutagen_none_returns_filename(self, tmp_path: Path) -> None:
        af = _make_audio_file(tmp_path, "x.mp3")
        with patch("lrcfilter.metadata.MutagenFile", return_value=None):
            result = extract_metadata(af)
            assert result.title == "x.mp3"
            assert result.artist is None
            assert result.duration_seconds is None

    def test_exception_returns_filename(self, tmp_path: Path) -> None:
        af = _make_audio_file(tmp_path, "x.mp3")
        with patch("lrcfilter.metadata.MutagenFile", side_effect=RuntimeError("boom")):
            result = extract_metadata(af)
            assert result.title == "x.mp3"

    def test_no_info_attribute(self, tmp_path: Path) -> None:
        """When mutagen_file.info is None, duration should be None."""
        mock_mutagen = MagicMock()
        mock_mutagen.info = None
        mock_mutagen.tags = MagicMock()
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.duration_seconds is None

    def test_empty_text_in_tag(self, tmp_path: Path) -> None:
        """Tag with empty text list should return None."""
        mock_tags = {
            "TIT2": MagicMock(text=[]),
            "TPE1": MagicMock(text=["Artist"]),
        }
        mock_mutagen = _make_mock_mutagen(mock_tags)
        af = _make_audio_file(tmp_path)

        with patch("lrcfilter.metadata.MutagenFile", return_value=mock_mutagen):
            result = extract_metadata(af)
            assert result.title is None
            assert result.artist == "Artist"
