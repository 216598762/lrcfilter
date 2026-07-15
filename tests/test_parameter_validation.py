"""Tests for parameter validation across lrcfilter modules."""

import pytest
from pathlib import Path

from lrcfilter.pipeline import PipelineConfig
from lrcfilter.analyzer import analyze_audio
from lrcfilter.censorship import detect_censorship
from lrcfilter.instrumental import detect_instrumental
from lrcfilter.mismatch import detect_metadata_mismatch
from lrcfilter.lyrics import fetch_lyrics
from lrcfilter.scanner import scan_audio_files
from lrcfilter.models import (
    AudioFile,
    TrackMetadata,
    LyricsResult,
    TranscriptionResult,
    Segment,
    Word,
)


class TestPipelineConfigValidation:
    """Test PipelineConfig __post_init__ validation."""
    
    def test_valid_config(self) -> None:
        """Test that valid config creates without error."""
        config = PipelineConfig()
        assert config.beam_size == 5
        assert config.censorship_threshold == 0.3
    
    def test_beam_size_zero_raises_error(self) -> None:
        """Test that beam_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="beam_size must be positive"):
            PipelineConfig(beam_size=0)
    
    def test_beam_size_negative_raises_error(self) -> None:
        """Test that beam_size=-1 raises ValueError."""
        with pytest.raises(ValueError, match="beam_size must be positive"):
            PipelineConfig(beam_size=-1)
    
    def test_api_delay_negative_raises_error(self) -> None:
        """Test that api_delay=-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="api_delay must be non-negative"):
            PipelineConfig(api_delay=-1.0)
    
    def test_censorship_threshold_too_low_raises_error(self) -> None:
        """Test that censorship_threshold=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="censorship_threshold must be between 0.0 and 1.0"):
            PipelineConfig(censorship_threshold=-0.1)
    
    def test_censorship_threshold_too_high_raises_error(self) -> None:
        """Test that censorship_threshold=1.1 raises ValueError."""
        with pytest.raises(ValueError, match="censorship_threshold must be between 0.0 and 1.0"):
            PipelineConfig(censorship_threshold=1.1)
    
    def test_min_words_vocals_negative_raises_error(self) -> None:
        """Test that min_words_vocals=-1 raises ValueError."""
        with pytest.raises(ValueError, match="min_words_vocals must be non-negative"):
            PipelineConfig(min_words_vocals=-1)
    
    def test_min_speech_duration_negative_raises_error(self) -> None:
        """Test that min_speech_duration=-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_speech_duration must be non-negative"):
            PipelineConfig(min_speech_duration=-1.0)
    
    def test_title_threshold_too_low_raises_error(self) -> None:
        """Test that title_threshold=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="title_threshold must be between 0.0 and 1.0"):
            PipelineConfig(title_threshold=-0.1)
    
    def test_title_threshold_too_high_raises_error(self) -> None:
        """Test that title_threshold=1.1 raises ValueError."""
        with pytest.raises(ValueError, match="title_threshold must be between 0.0 and 1.0"):
            PipelineConfig(title_threshold=1.1)
    
    def test_artist_threshold_too_low_raises_error(self) -> None:
        """Test that artist_threshold=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="artist_threshold must be between 0.0 and 1.0"):
            PipelineConfig(artist_threshold=-0.1)
    
    def test_artist_threshold_too_high_raises_error(self) -> None:
        """Test that artist_threshold=1.1 raises ValueError."""
        with pytest.raises(ValueError, match="artist_threshold must be between 0.0 and 1.0"):
            PipelineConfig(artist_threshold=1.1)
    
    def test_duration_tolerance_negative_raises_error(self) -> None:
        """Test that duration_tolerance=-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="duration_tolerance must be non-negative"):
            PipelineConfig(duration_tolerance=-1.0)
    
    def test_boundary_values_valid(self) -> None:
        """Test that boundary values (0.0 and 1.0) are valid."""
        config = PipelineConfig(
            censorship_threshold=0.0,
            title_threshold=1.0,
            artist_threshold=0.0,
        )
        assert config.censorship_threshold == 0.0
        assert config.title_threshold == 1.0
        assert config.artist_threshold == 0.0


class TestAnalyzeAudioValidation:
    """Test analyze_audio beam_size validation."""
    
    def test_beam_size_zero_raises_error(self) -> None:
        """Test that beam_size=0 raises ValueError."""
        # Create minimal AudioFile for testing
        audio_file = AudioFile(
            path=Path("/tmp/test.mp3"),
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )
        
        with pytest.raises(ValueError, match="beam_size must be positive"):
            analyze_audio(audio_file, beam_size=0)
    
    def test_beam_size_negative_raises_error(self) -> None:
        """Test that beam_size=-1 raises ValueError."""
        audio_file = AudioFile(
            path=Path("/tmp/test.mp3"),
            filename="test.mp3",
            extension=".mp3",
            size_mb=0.0,
        )
        
        with pytest.raises(ValueError, match="beam_size must be positive"):
            analyze_audio(audio_file, beam_size=-1)


class TestDetectCensorshipValidation:
    """Test detect_censorship threshold validation."""
    
    def test_threshold_too_low_raises_error(self) -> None:
        """Test that threshold=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            detect_censorship("lyrics", "transcription", threshold=-0.1)
    
    def test_threshold_too_high_raises_error(self) -> None:
        """Test that threshold=1.1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            detect_censorship("lyrics", "transcription", threshold=1.1)
    
    def test_boundary_values_valid(self) -> None:
        """Test that boundary values (0.0 and 1.0) are valid."""
        # These should not raise errors
        result = detect_censorship("lyrics", "transcription", threshold=0.0)
        assert result is not None
        
        result = detect_censorship("lyrics", "transcription", threshold=1.0)
        assert result is not None


class TestDetectInstrumentalValidation:
    """Test detect_instrumental validation."""
    
    def _mock_transcription(self) -> TranscriptionResult:
        """Create a mock transcription for testing."""
        segment = Segment(
            start=0.0,
            end=5.0,
            text="Test",
            words=[Word(start=0.0, end=5.0, word="Test", probability=0.9)],
        )
        return TranscriptionResult(
            text="Test transcription",
            segments=[segment],
            language="en",
            duration=5.0,
            has_speech=True,
        )
    
    def test_min_words_vocals_negative_raises_error(self) -> None:
        """Test that min_words_vocals=-1 raises ValueError."""
        transcription = self._mock_transcription()
        
        with pytest.raises(ValueError, match="min_words_vocals must be non-negative"):
            detect_instrumental(transcription, min_words_vocals=-1)
    
    def test_min_speech_duration_negative_raises_error(self) -> None:
        """Test that min_speech_duration=-1.0 raises ValueError."""
        transcription = self._mock_transcription()
        
        with pytest.raises(ValueError, match="min_speech_duration must be non-negative"):
            detect_instrumental(transcription, min_speech_duration=-1.0)


class TestDetectMetadataMismatchValidation:
    """Test detect_metadata_mismatch validation."""
    
    def _mock_metadata(self) -> TrackMetadata:
        """Create mock metadata for testing."""
        return TrackMetadata(
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            duration_seconds=180.0,
        )
    
    def _mock_lyrics(self) -> LyricsResult:
        """Create mock lyrics for testing."""
        return LyricsResult(
            source="lrclib",
            synced_lyrics="[00:00.00] Line 1",
            plain_lyrics="Line 1",
            matched_track_name="Test Song",
            matched_artist_name="Test Artist",
            matched_album_name="Test Album",
            match_score=0.95,
        )
    
    def test_title_threshold_too_low_raises_error(self) -> None:
        """Test that title_threshold=-0.1 raises ValueError."""
        metadata = self._mock_metadata()
        lyrics = self._mock_lyrics()
        
        with pytest.raises(ValueError, match="title_threshold must be between 0.0 and 1.0"):
            detect_metadata_mismatch(metadata, lyrics, title_threshold=-0.1)
    
    def test_title_threshold_too_high_raises_error(self) -> None:
        """Test that title_threshold=1.1 raises ValueError."""
        metadata = self._mock_metadata()
        lyrics = self._mock_lyrics()
        
        with pytest.raises(ValueError, match="title_threshold must be between 0.0 and 1.0"):
            detect_metadata_mismatch(metadata, lyrics, title_threshold=1.1)
    
    def test_artist_threshold_too_low_raises_error(self) -> None:
        """Test that artist_threshold=-0.1 raises ValueError."""
        metadata = self._mock_metadata()
        lyrics = self._mock_lyrics()
        
        with pytest.raises(ValueError, match="artist_threshold must be between 0.0 and 1.0"):
            detect_metadata_mismatch(metadata, lyrics, artist_threshold=-0.1)
    
    def test_artist_threshold_too_high_raises_error(self) -> None:
        """Test that artist_threshold=1.1 raises ValueError."""
        metadata = self._mock_metadata()
        lyrics = self._mock_lyrics()
        
        with pytest.raises(ValueError, match="artist_threshold must be between 0.0 and 1.0"):
            detect_metadata_mismatch(metadata, lyrics, artist_threshold=1.1)
    
    def test_duration_tolerance_negative_raises_error(self) -> None:
        """Test that duration_tolerance=-1.0 raises ValueError."""
        metadata = self._mock_metadata()
        lyrics = self._mock_lyrics()
        
        with pytest.raises(ValueError, match="duration_tolerance must be non-negative"):
            detect_metadata_mismatch(metadata, lyrics, duration_tolerance=-1.0)


class TestFetchLyricsValidation:
    """Test fetch_lyrics api_delay validation."""
    
    def test_api_delay_negative_raises_error(self) -> None:
        """Test that api_delay=-1.0 raises ValueError."""
        metadata = TrackMetadata(
            title="Test Song",
            artist="Test Artist",
            album=None,
            duration_seconds=None,
        )
        
        with pytest.raises(ValueError, match="api_delay must be non-negative"):
            fetch_lyrics(metadata, api_delay=-1.0)


class TestScanAudioFilesValidation:
    """Test scan_audio_files formats validation."""
    
    def test_format_without_dot_raises_error(self, tmp_path: Path) -> None:
        """Test that format without leading dot raises ValueError."""
        with pytest.raises(ValueError, match="Format must start with a dot"):
            scan_audio_files(tmp_path, formats={"mp3"})
    
    def test_format_with_dot_is_valid(self, tmp_path: Path) -> None:
        """Test that format with leading dot is valid."""
        # This should not raise an error
        result = scan_audio_files(tmp_path, formats={".mp3"})
        assert result == []  # Empty directory
    
    def test_empty_formats_set_is_valid(self, tmp_path: Path) -> None:
        """Test that empty formats set is valid."""
        result = scan_audio_files(tmp_path, formats=set())
        assert result == []
    
    def test_none_formats_is_valid(self, tmp_path: Path) -> None:
        """Test that None formats is valid."""
        result = scan_audio_files(tmp_path, formats=None)
        assert result == []
    
    def test_multiple_formats_mixed_valid_invalid(self, tmp_path: Path) -> None:
        """Test that mix of valid and invalid formats raises error."""
        with pytest.raises(ValueError, match="Format must start with a dot"):
            scan_audio_files(tmp_path, formats={".mp3", "flac"})  # flac missing dot
    
    def test_format_non_string_raises_error(self, tmp_path: Path) -> None:
        """Test that non-string format raises ValueError."""
        with pytest.raises(ValueError, match="Format must be a string"):
            scan_audio_files(tmp_path, formats={123})  # type: ignore


class TestParameterValidationIntegration:
    """Integration tests for parameter validation."""
    
    def test_invalid_config_raises_error_on_creation(self) -> None:
        """Test that invalid config raises error on creation."""
        with pytest.raises(ValueError):
            PipelineConfig(beam_size=-1)
    
    def test_valid_config_can_be_used(self) -> None:
        """Test that valid config can be created and used."""
        config = PipelineConfig(
            beam_size=10,
            censorship_threshold=0.5,
            min_words_vocals=20,
        )
        assert config.beam_size == 10
        assert config.censorship_threshold == 0.5
        assert config.min_words_vocals == 20
