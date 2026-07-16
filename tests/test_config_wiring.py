"""Tests to verify CLI config options are correctly passed through to functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lrcfilter.models import (
    AudioFile,
    CensorshipResult,
    InstrumentalResult,
    LyricsResult,
    MismatchResult,
    TrackMetadata,
    TranscriptionResult,
)
from lrcfilter.pipeline import PipelineConfig, process_single_track, run_pipeline


@pytest.fixture
def mock_audio_file(tmp_path: Path) -> AudioFile:
    """Create a mock AudioFile for testing."""
    test_file = tmp_path / "test.mp3"
    test_file.touch()
    return AudioFile(
        path=test_file,
        filename="test.mp3",
        extension=".mp3",
        size_mb=0.0,
    )


@pytest.fixture
def mock_metadata() -> TrackMetadata:
    """Create mock metadata."""
    return TrackMetadata(
        title="Test Song",
        artist="Test Artist",
        album="Test Album",
        duration_seconds=180.0,
    )


@pytest.fixture
def mock_lyrics() -> LyricsResult:
    """Create mock lyrics result."""
    return LyricsResult(
        source="lrclib",
        synced_lyrics="[00:00.00] Line 1",
        plain_lyrics="Line 1",
        matched_track_name="Test Song",
        matched_artist_name="Test Artist",
        matched_album_name="Test Album",
        match_score=0.95,
    )


@pytest.fixture
def mock_transcription() -> TranscriptionResult:
    """Create mock transcription result."""
    from lrcfilter.models import Segment, Word

    segment = Segment(
        start=0.0,
        end=5.0,
        text="Test transcription",
        words=[Word(start=0.0, end=5.0, word="Test", probability=0.9)],
    )
    return TranscriptionResult(
        text="Test transcription",
        segments=[segment],
        language="en",
        duration=5.0,
        has_speech=True,
    )


class TestProcessSingleTrackConfigWiring:
    """Test that config options are passed correctly to functions in process_single_track."""

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    def test_beam_size_passed_to_analyze_audio(
        self,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that beam_size is passed to analyze_audio."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig(beam_size=8)
        process_single_track(mock_audio_file, config)

        # Verify beam_size was passed
        call_kwargs = mock_analyze_audio.call_args[1]
        assert call_kwargs["beam_size"] == 8

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    def test_vad_filter_passed_to_analyze_audio(
        self,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that vad_filter is passed to analyze_audio."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig(vad_filter=False)
        process_single_track(mock_audio_file, config)

        # Verify vad_filter was passed
        call_kwargs = mock_analyze_audio.call_args[1]
        assert call_kwargs["vad_filter"] is False

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    def test_genius_token_passed_to_fetch_lyrics(
        self,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that genius_token is passed to fetch_lyrics."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig(genius_token="test_token_123")
        process_single_track(mock_audio_file, config)

        # Verify genius_token was passed
        call_kwargs = mock_fetch_lyrics.call_args[1]
        assert call_kwargs["genius_token"] == "test_token_123"

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    def test_lrclib_only_passed_to_fetch_lyrics(
        self,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that lrclib_only is passed to fetch_lyrics."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig(lrclib_only=True)
        process_single_track(mock_audio_file, config)

        # Verify lrclib_only was passed
        call_kwargs = mock_fetch_lyrics.call_args[1]
        assert call_kwargs["lrclib_only"] is True

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    def test_api_delay_passed_to_fetch_lyrics(
        self,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that api_delay is passed to fetch_lyrics."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig(api_delay=2.5)
        process_single_track(mock_audio_file, config)

        # Verify api_delay was passed
        call_kwargs = mock_fetch_lyrics.call_args[1]
        assert call_kwargs["api_delay"] == 2.5

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_censorship")
    def test_censorship_threshold_passed_to_detect_censorship(
        self,
        mock_detect_censorship: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that censorship_threshold is passed to detect_censorship."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_censorship.return_value = CensorshipResult(
            is_censored=False,
            mismatch_score=0.0,
            profanity_count=0,
            confidence=0.0,
            details="No censorship detected",
        )

        config = PipelineConfig(censorship_threshold=0.5)
        process_single_track(mock_audio_file, config)

        # Verify threshold was passed
        call_kwargs = mock_detect_censorship.call_args[1]
        assert call_kwargs["threshold"] == 0.5

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_instrumental")
    def test_min_words_vocals_passed_to_detect_instrumental(
        self,
        mock_detect_instrumental: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that min_words_vocals is passed to detect_instrumental."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_instrumental.return_value = InstrumentalResult(
            is_instrumental=False, word_count=10, speech_duration=5.0, confidence=0.2
        )

        config = PipelineConfig(min_words_vocals=20)
        process_single_track(mock_audio_file, config)

        # Verify min_words_vocals was passed
        call_kwargs = mock_detect_instrumental.call_args[1]
        assert call_kwargs["min_words_vocals"] == 20

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_instrumental")
    def test_min_speech_duration_passed_to_detect_instrumental(
        self,
        mock_detect_instrumental: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that min_speech_duration is passed to detect_instrumental."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_instrumental.return_value = InstrumentalResult(
            is_instrumental=False, word_count=10, speech_duration=5.0, confidence=0.2
        )

        config = PipelineConfig(min_speech_duration=10.0)
        process_single_track(mock_audio_file, config)

        # Verify min_speech_duration was passed
        call_kwargs = mock_detect_instrumental.call_args[1]
        assert call_kwargs["min_speech_duration"] == 10.0

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_metadata_mismatch")
    def test_title_threshold_passed_to_detect_metadata_mismatch(
        self,
        mock_detect_mismatch: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that title_threshold is passed to detect_metadata_mismatch."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_mismatch.return_value = MismatchResult(
            is_mismatch=False,
            title_score=0.9,
            artist_score=0.9,
            duration_difference=0.0,
            confidence=0.1,
            details="Metadata matches",
        )

        config = PipelineConfig(title_threshold=0.8)
        process_single_track(mock_audio_file, config)

        # Verify title_threshold was passed
        call_kwargs = mock_detect_mismatch.call_args[1]
        assert call_kwargs["title_threshold"] == 0.8

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_metadata_mismatch")
    def test_artist_threshold_passed_to_detect_metadata_mismatch(
        self,
        mock_detect_mismatch: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that artist_threshold is passed to detect_metadata_mismatch."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_mismatch.return_value = MismatchResult(
            is_mismatch=False,
            title_score=0.9,
            artist_score=0.9,
            duration_difference=0.0,
            confidence=0.1,
            details="Metadata matches",
        )

        config = PipelineConfig(artist_threshold=0.85)
        process_single_track(mock_audio_file, config)

        # Verify artist_threshold was passed
        call_kwargs = mock_detect_mismatch.call_args[1]
        assert call_kwargs["artist_threshold"] == 0.85

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_metadata_mismatch")
    def test_duration_tolerance_passed_to_detect_metadata_mismatch(
        self,
        mock_detect_mismatch: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_lyrics: LyricsResult,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that duration_tolerance is passed to detect_metadata_mismatch."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = mock_lyrics
        mock_analyze_audio.return_value = mock_transcription
        mock_detect_mismatch.return_value = MismatchResult(
            is_mismatch=False,
            title_score=0.9,
            artist_score=0.9,
            duration_difference=0.0,
            confidence=0.1,
            details="Metadata matches",
        )

        config = PipelineConfig(duration_tolerance=60.0)
        process_single_track(mock_audio_file, config)

        # Verify duration_tolerance was passed
        call_kwargs = mock_detect_mismatch.call_args[1]
        assert call_kwargs["duration_tolerance"] == 60.0

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_metadata_mismatch")
    @patch("lrcfilter.pipeline.detect_censorship")
    def test_lyrics_none_skips_mismatch_and_censorship(
        self,
        mock_detect_censorship: MagicMock,
        mock_detect_mismatch: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that when lyrics is None, mismatch and censorship are not called."""
        mock_extract_metadata.return_value = mock_metadata
        mock_fetch_lyrics.return_value = None
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig()
        process_single_track(mock_audio_file, config)

        # Verify mismatch and censorship were not called
        mock_detect_mismatch.assert_not_called()
        mock_detect_censorship.assert_not_called()

    @patch("lrcfilter.pipeline.analyze_audio")
    @patch("lrcfilter.pipeline.fetch_lyrics")
    @patch("lrcfilter.pipeline.extract_metadata")
    @patch("lrcfilter.pipeline.detect_censorship")
    def test_lyrics_plain_lyrics_none_skips_censorship(
        self,
        mock_detect_censorship: MagicMock,
        mock_extract_metadata: MagicMock,
        mock_fetch_lyrics: MagicMock,
        mock_analyze_audio: MagicMock,
        mock_audio_file: AudioFile,
        mock_metadata: TrackMetadata,
        mock_transcription: TranscriptionResult,
    ) -> None:
        """Test that when lyrics.plain_lyrics is None, censorship is not called."""
        mock_extract_metadata.return_value = mock_metadata
        # Create lyrics with None plain_lyrics
        lyrics = LyricsResult(
            source="lrclib",
            synced_lyrics="[00:00.00] Line 1",
            plain_lyrics=None,
            matched_track_name="Test Song",
            matched_artist_name="Test Artist",
            matched_album_name=None,
            match_score=0.9,
        )
        mock_fetch_lyrics.return_value = lyrics
        mock_analyze_audio.return_value = mock_transcription

        config = PipelineConfig()
        process_single_track(mock_audio_file, config)

        # Verify censorship was not called
        mock_detect_censorship.assert_not_called()


class TestRunPipelineConfigWiring:
    """Test that config options are passed correctly to functions in run_pipeline."""

    @patch("lrcfilter.pipeline.write_results")
    @patch("lrcfilter.pipeline.process_single_track")
    @patch("lrcfilter.pipeline.scan_audio_files")
    def test_formats_passed_to_scan_audio_files(
        self,
        mock_scan: MagicMock,
        mock_process: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that formats is passed to scan_audio_files."""
        mock_scan.return_value = []

        custom_formats = {".mp3", ".flac"}
        config = PipelineConfig(formats=custom_formats)
        run_pipeline(tmp_path, config)

        # Verify formats was passed
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["formats"] == custom_formats

    @patch("lrcfilter.pipeline.write_results")
    @patch("lrcfilter.pipeline.process_single_track")
    @patch("lrcfilter.pipeline.scan_audio_files")
    def test_no_censored_flag_skips_censored_output(
        self,
        mock_scan: MagicMock,
        mock_process: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
        mock_audio_file: AudioFile,
    ) -> None:
        """Test that no_censored flag prevents writing censored tracks."""
        from lrcfilter.models import CensorshipResult

        mock_scan.return_value = [mock_audio_file]
        mock_process.return_value = MagicMock(
            censorship=CensorshipResult(
                is_censored=True,
                mismatch_score=0.5,
                profanity_count=1,
                confidence=0.7,
                details="Censored",
            ),
            instrumental=InstrumentalResult(
                is_instrumental=False, word_count=10, speech_duration=5.0, confidence=0.2
            ),
            mismatch=None,
        )

        config = PipelineConfig(no_censored=True)
        run_pipeline(tmp_path, config)

        # Verify write_results was called with empty censored_tracks
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["censored_tracks"] == []

    @patch("lrcfilter.pipeline.write_results")
    @patch("lrcfilter.pipeline.process_single_track")
    @patch("lrcfilter.pipeline.scan_audio_files")
    def test_no_instrumental_flag_skips_instrumental_output(
        self,
        mock_scan: MagicMock,
        mock_process: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
        mock_audio_file: AudioFile,
    ) -> None:
        """Test that no_instrumental flag prevents writing instrumental tracks."""
        mock_scan.return_value = [mock_audio_file]
        mock_process.return_value = MagicMock(
            censorship=None,
            instrumental=InstrumentalResult(
                is_instrumental=True, word_count=2, speech_duration=1.0, confidence=0.9
            ),
            mismatch=None,
        )

        config = PipelineConfig(no_instrumental=True)
        run_pipeline(tmp_path, config)

        # Verify write_results was called with empty instrumental_tracks
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["instrumental_tracks"] == []

    @patch("lrcfilter.pipeline.write_results")
    @patch("lrcfilter.pipeline.process_single_track")
    @patch("lrcfilter.pipeline.scan_audio_files")
    def test_no_mismatches_flag_skips_mismatch_output(
        self,
        mock_scan: MagicMock,
        mock_process: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
        mock_audio_file: AudioFile,
    ) -> None:
        """Test that no_mismatches flag prevents writing metadata mismatches."""
        mock_scan.return_value = [mock_audio_file]
        mock_process.return_value = MagicMock(
            censorship=None,
            instrumental=InstrumentalResult(
                is_instrumental=False, word_count=10, speech_duration=5.0, confidence=0.2
            ),
            mismatch=MismatchResult(
                is_mismatch=True,
                title_score=0.3,
                artist_score=0.4,
                duration_difference=0.0,
                confidence=0.8,
                details="Mismatch",
            ),
        )

        config = PipelineConfig(no_mismatches=True)
        run_pipeline(tmp_path, config)

        # Verify write_results was called with empty metadata_mismatches
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["metadata_mismatches"] == []

    @patch("lrcfilter.pipeline.write_results")
    @patch("lrcfilter.pipeline.process_single_track")
    @patch("lrcfilter.pipeline.scan_audio_files")
    def test_output_dir_passed_to_write_results(
        self,
        mock_scan: MagicMock,
        mock_process: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
        mock_audio_file: AudioFile,
    ) -> None:
        """Test that output_dir is passed to write_results."""
        from lrcfilter.models import CensorshipResult

        mock_scan.return_value = [mock_audio_file]
        mock_process.return_value = MagicMock(
            censorship=CensorshipResult(
                is_censored=True,
                mismatch_score=0.5,
                profanity_count=1,
                confidence=0.7,
                details="Censored",
            ),
            instrumental=None,
            mismatch=None,
        )

        output_dir = tmp_path / "output"
        config = PipelineConfig(output_dir=output_dir)
        run_pipeline(tmp_path, config)

        # Verify output_dir was passed to write_results
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["output_dir"] == output_dir


class TestPipelineConfigDefaults:
    """Test PipelineConfig default values match expected defaults."""

    def test_default_beam_size(self) -> None:
        """Test default beam_size is 5."""
        config = PipelineConfig()
        assert config.beam_size == 5

    def test_default_vad_filter(self) -> None:
        """Test default vad_filter is True."""
        config = PipelineConfig()
        assert config.vad_filter is True

    def test_default_censorship_threshold(self) -> None:
        """Test default censorship_threshold is 0.3."""
        config = PipelineConfig()
        assert config.censorship_threshold == 0.3

    def test_default_min_words_vocals(self) -> None:
        """Test default min_words_vocals is 10."""
        config = PipelineConfig()
        assert config.min_words_vocals == 10

    def test_default_min_speech_duration(self) -> None:
        """Test default min_speech_duration is 5.0."""
        config = PipelineConfig()
        assert config.min_speech_duration == 5.0

    def test_default_title_threshold(self) -> None:
        """Test default title_threshold is 0.6."""
        config = PipelineConfig()
        assert config.title_threshold == 0.6

    def test_default_artist_threshold(self) -> None:
        """Test default artist_threshold is 0.7."""
        config = PipelineConfig()
        assert config.artist_threshold == 0.7

    def test_default_duration_tolerance(self) -> None:
        """Test default duration_tolerance is 30.0."""
        config = PipelineConfig()
        assert config.duration_tolerance == 30.0
