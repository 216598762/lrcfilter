"""Tests for lyrics fetching module to improve coverage."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lrcfilter.lyrics import (
    _calculate_match_score,
    _fetch_from_genius,
    _fetch_from_lrclib,
    fetch_lyrics,
)
from lrcfilter.models import LyricsResult, TrackMetadata


class TestFetchLyrics:
    """Test the fetch_lyrics function."""

    def test_returns_none_when_no_title(self) -> None:
        """Should return None when metadata has no title."""
        metadata = TrackMetadata(title=None, artist="Artist", album="Album", duration_seconds=180.0)
        result = fetch_lyrics(metadata)
        assert result is None

    def test_raises_on_negative_api_delay(self) -> None:
        """Should raise ValueError for negative api_delay."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)
        with pytest.raises(ValueError, match="api_delay must be non-negative"):
            fetch_lyrics(metadata, api_delay=-1.0)

    def test_lrclib_only_skips_genius(self) -> None:
        """Should skip Genius fallback when lrclib_only is True."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)

        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics._fetch_from_genius") as mock_genius,
        ):
            result = fetch_lyrics(metadata, lrclib_only=True)
            assert result is None
            mock_genius.assert_not_called()

    def test_returns_lrclib_result_when_available(self) -> None:
        """Should return LRCLib result when lyrics are found."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)
        mock_result = LyricsResult(
            source="lrclib",
            synced_lyrics="[00:00.00]Test lyrics",
            plain_lyrics="Test lyrics",
            matched_track_name="Song",
            matched_artist_name="Artist",
            matched_album_name=None,
            match_score=0.9,
        )

        with patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=mock_result):
            result = fetch_lyrics(metadata)
            assert result is not None
            assert result.source == "lrclib"

    def test_falls_back_to_genius_when_lrclib_empty(self) -> None:
        """Should fall back to Genius when LRCLib returns no lyrics."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)
        mock_genius_result = LyricsResult(
            source="genius",
            synced_lyrics=None,
            plain_lyrics="Genius lyrics here",
            matched_track_name="Song",
            matched_artist_name="Artist",
            matched_album_name=None,
            match_score=0.8,
        )

        # Must patch both _fetch_from_genius AND GENIUS_AVAILABLE
        # because the function checks GENIUS_AVAILABLE before calling _fetch_from_genius
        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics._fetch_from_genius", return_value=mock_genius_result),
            patch.dict(os.environ, {"GENIUS_ACCESS_TOKEN": "test_token"}),
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
        ):
            result = fetch_lyrics(metadata)
            assert result is not None
            assert result.source == "genius"

    def test_returns_none_when_both_sources_fail(self) -> None:
        """Should return None when both LRCLib and Genius fail."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)

        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics._fetch_from_genius", return_value=None),
        ):
            result = fetch_lyrics(metadata)
            assert result is None

    def test_uses_genius_token_from_env(self) -> None:
        """Should use Genius token from environment variable."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)

        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics._fetch_from_genius", return_value=None) as mock_genius,
            patch.dict(os.environ, {"GENIUS_ACCESS_TOKEN": "test_token"}),
        ):
            fetch_lyrics(metadata)
            mock_genius.assert_called_once()

    def test_prefers_provided_token_over_env(self) -> None:
        """Should prefer provided token over environment variable."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)

        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics._fetch_from_genius", return_value=None) as mock_genius,
            patch.dict(os.environ, {"GENIUS_ACCESS_TOKEN": "env_token"}),
        ):
            fetch_lyrics(metadata, genius_token="provided_token")
            call_args = mock_genius.call_args
            assert call_args[0][1] == "provided_token"

    def test_returns_synced_lyrics_when_available(self) -> None:
        """Should return LRCLib result even if only synced lyrics available."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)
        mock_result = LyricsResult(
            source="lrclib",
            synced_lyrics="[00:00.00]Synced lyrics",
            plain_lyrics=None,
            matched_track_name="Song",
            matched_artist_name="Artist",
            matched_album_name=None,
            match_score=0.9,
        )

        with patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=mock_result):
            result = fetch_lyrics(metadata)
            assert result is not None
            assert result.synced_lyrics is not None


class TestFetchFromLrclib:
    """Test the _fetch_from_lrclib function."""

    def test_returns_lyrics_on_success(self) -> None:
        """Should return LyricsResult on successful API call."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "trackName": "Test Song",
                "artistName": "Test Artist",
                "albumName": "Test Album",
                "syncedLyrics": "[00:00.00]Test lyrics",
                "plainLyrics": "Test lyrics",
            }
        ]

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_lrclib(metadata, api_delay=0)
            assert result is not None
            assert result.source == "lrclib"

    def test_returns_none_on_non_200_status(self) -> None:
        """Should return None when API returns non-200 status."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_lrclib(metadata, api_delay=0)
            assert result is None

    def test_returns_none_on_empty_results(self) -> None:
        """Should return None when API returns empty results."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_lrclib(metadata, api_delay=0)
            assert result is None

    def test_includes_artist_in_request(self) -> None:
        """Should include artist in API request parameters."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response) as mock_get,
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            _fetch_from_lrclib(metadata, api_delay=0)
            call_kwargs = mock_get.call_args
            assert "artist_name" in call_kwargs[1]["params"]
            assert call_kwargs[1]["params"]["artist_name"] == "Test Artist"

    def test_includes_album_in_request(self) -> None:
        """Should include album in API request parameters."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album="Test Album", duration_seconds=None
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response) as mock_get,
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            _fetch_from_lrclib(metadata, api_delay=0)
            call_kwargs = mock_get.call_args
            assert "album_name" in call_kwargs[1]["params"]

    def test_includes_duration_in_request(self) -> None:
        """Should include duration in API request parameters."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=180.0
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with (
            patch("lrcfilter.lyrics.requests.get", return_value=mock_response) as mock_get,
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            _fetch_from_lrclib(metadata, api_delay=0)
            call_kwargs = mock_get.call_args
            assert "duration" in call_kwargs[1]["params"]
            assert call_kwargs[1]["params"]["duration"] == "180"

    def test_handles_request_exception(self) -> None:
        """Should return None when request raises exception."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        with (
            patch("lrcfilter.lyrics.requests.get", side_effect=Exception("Network error")),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_lrclib(metadata, api_delay=0)
            assert result is None

    def test_sleeps_for_api_delay(self) -> None:
        """Should sleep for the specified API delay."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        with (
            patch(
                "lrcfilter.lyrics.requests.get", return_value=MagicMock(status_code=200, json=list)
            ),
            patch("lrcfilter.lyrics.time.sleep") as mock_sleep,
        ):
            _fetch_from_lrclib(metadata, api_delay=1.5)
            mock_sleep.assert_called_with(1.5)


class TestFetchFromGenius:
    """Test the _fetch_from_genius function."""

    def test_returns_none_when_genius_not_available(self) -> None:
        """Should return None when lyricsgenius is not installed."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        with patch("lrcfilter.lyrics.GENIUS_AVAILABLE", False):
            result = _fetch_from_genius(metadata, "test_token")
            assert result is None

    def test_returns_lyrics_on_success(self) -> None:
        """Should return LyricsResult on successful Genius search."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_song = MagicMock()
        mock_song.title = "Test Song"
        mock_song.artist = "Test Artist"
        mock_song.album = "Test Album"
        mock_song.lyrics = "Genius lyrics here"

        mock_genius = MagicMock()
        mock_genius.search_song.return_value = mock_song

        with (
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
            patch("lrcfilter.lyrics.lyricsgenius.Genius", return_value=mock_genius),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_genius(metadata, "test_token", api_delay=0)
            assert result is not None
            assert result.source == "genius"
            assert result.plain_lyrics == "Genius lyrics here"

    def test_returns_none_when_no_lyrics_found(self) -> None:
        """Should return None when Genius search returns no lyrics."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_genius = MagicMock()
        mock_genius.search_song.return_value = None

        with (
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
            patch("lrcfilter.lyrics.lyricsgenius.Genius", return_value=mock_genius),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_genius(metadata, "test_token", api_delay=0)
            assert result is None

    def test_returns_none_when_lyrics_empty(self) -> None:
        """Should return None when Genius search returns empty lyrics."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        mock_song = MagicMock()
        mock_song.lyrics = ""

        mock_genius = MagicMock()
        mock_genius.search_song.return_value = mock_song

        with (
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
            patch("lrcfilter.lyrics.lyricsgenius.Genius", return_value=mock_genius),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_genius(metadata, "test_token", api_delay=0)
            assert result is None

    def test_handles_genius_exception(self) -> None:
        """Should return None when Genius raises exception."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )

        with (
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
            patch("lrcfilter.lyrics.lyricsgenius.Genius", side_effect=Exception("API error")),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            result = _fetch_from_genius(metadata, "test_token", api_delay=0)
            assert result is None

    def test_builds_search_term_correctly(self) -> None:
        """Should combine title and artist for search term."""
        metadata = TrackMetadata(
            title="My Song", artist="My Artist", album=None, duration_seconds=None
        )

        mock_genius = MagicMock()
        mock_genius.search_song.return_value = None

        with (
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", True),
            patch("lrcfilter.lyrics.lyricsgenius.Genius", return_value=mock_genius),
            patch("lrcfilter.lyrics.time.sleep"),
        ):
            _fetch_from_genius(metadata, "test_token", api_delay=0)
            mock_genius.search_song.assert_called_with("My Song My Artist")


class TestCalculateMatchScore:
    """Test the _calculate_match_score function."""

    def test_perfect_match(self) -> None:
        """Should return 1.0 for perfect title and artist match."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "Test Artist",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 1.0

    def test_no_match(self) -> None:
        """Should return low score for completely different names."""
        metadata = TrackMetadata(
            title="Song A", artist="Artist A", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Song B",
            "artistName": "Artist B",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.0

    def test_partial_title_match(self) -> None:
        """Should return 0.5 for partial title match (substring)."""
        metadata = TrackMetadata(
            title="Song", artist="Test Artist", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "Test Artist",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.75  # 0.5 title + 1.0 artist / 2.0

    def test_case_insensitive_match(self) -> None:
        """Should match case-insensitively."""
        metadata = TrackMetadata(
            title="test song", artist="test artist", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "Test Artist",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 1.0

    def test_missing_metadata_title(self) -> None:
        """Should handle missing title in metadata."""
        metadata = TrackMetadata(
            title=None, artist="Test Artist", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "Test Artist",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 1.0  # Only artist matched

    def test_missing_api_result(self) -> None:
        """Should handle missing fields in API result."""
        metadata = TrackMetadata(
            title="Test Song", artist="Test Artist", album=None, duration_seconds=None
        )
        lrclib_result = {}
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.0

    def test_empty_metadata(self) -> None:
        """Should return 0.0 for empty metadata."""
        metadata = TrackMetadata(title=None, artist=None, album=None, duration_seconds=None)
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "Test Artist",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.0

    def test_partial_artist_match(self) -> None:
        """Should return 0.75 for partial artist match (substring)."""
        metadata = TrackMetadata(
            title="Test Song", artist="Beatles", album=None, duration_seconds=None
        )
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "The Beatles",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.75  # 1.0 title + 0.5 artist / 2.0

    def test_both_partial_matches(self) -> None:
        """Should return 0.5 for both partial title and artist matches."""
        metadata = TrackMetadata(title="Song", artist="Beatles", album=None, duration_seconds=None)
        lrclib_result = {
            "trackName": "Test Song",
            "artistName": "The Beatles",
        }
        score = _calculate_match_score(metadata, lrclib_result)
        assert score == 0.5  # 0.5 title + 0.5 artist / 2.0


class TestFetchLyricsNoGenius:
    """Test fetch_lyrics when Genius is not available (covers branch 83->85)."""

    def test_no_genius_fallback_when_unavailable(self) -> None:
        """Should skip Genius fallback when GENIUS_AVAILABLE is False."""
        metadata = TrackMetadata(title="Song", artist="Artist", album=None, duration_seconds=None)

        with (
            patch("lrcfilter.lyrics._fetch_from_lrclib", return_value=None),
            patch("lrcfilter.lyrics.GENIUS_AVAILABLE", False),
        ):
            result = fetch_lyrics(metadata)
            assert result is None
