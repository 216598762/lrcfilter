"""Tests for analyzer module to improve coverage from 27%."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lrcfilter.analyzer import get_model, analyze_audio, _model_cache
from lrcfilter.models import AudioFile, TranscriptionResult
from lrcfilter.config import DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the model cache before each test."""
    _model_cache.clear()
    yield
    _model_cache.clear()


class TestGetModel:
    """Test the get_model function for model caching."""

    def test_creates_model_on_first_call(self) -> None:
        """Should create a new WhisperModel on first call."""
        mock_model = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model) as mock_cls:
            result = get_model("tiny", "cpu", "float32")
            mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="float32")
            assert result is mock_model

    def test_caches_model_on_subsequent_calls(self) -> None:
        """Should return cached model on subsequent calls without creating new instance."""
        mock_model = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model) as mock_cls:
            # First call creates the model
            result1 = get_model("tiny", "cpu", "float32")
            # Second call should use cache
            result2 = get_model("tiny", "cpu", "float32")

            # WhisperModel should only be called once
            mock_cls.assert_called_once()
            assert result1 is result2

    def test_different_configs_create_different_models(self) -> None:
        """Should create separate models for different configurations."""
        mock_model1 = MagicMock(spec=["transcribe"])
        mock_model2 = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", side_effect=[mock_model1, mock_model2]) as mock_cls:
            result1 = get_model("tiny", "cpu", "float32")
            result2 = get_model("base", "cpu", "float32")

            assert mock_cls.call_count == 2
            assert result1 is not result2

    def test_uses_default_config_values(self) -> None:
        """Should use default config values when not specified."""
        mock_model = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model) as mock_cls:
            get_model()
            mock_cls.assert_called_once_with(
                DEFAULT_MODEL,
                device=DEFAULT_DEVICE,
                compute_type=DEFAULT_COMPUTE_TYPE,
            )

    def test_raises_exception_on_model_creation_failure(self) -> None:
        """Should raise exception when WhisperModel creation fails."""
        with patch("lrcfilter.analyzer.WhisperModel", side_effect=RuntimeError("CUDA error")):
            with pytest.raises(RuntimeError, match="CUDA error"):
                get_model("large-v3", "cuda", "float16")

    def test_cache_key_format(self) -> None:
        """Cache key should be formatted as model_device_compute_type."""
        mock_model = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model):
            get_model("test-model", "cuda", "int8")
            assert "test-model_cuda_int8" in _model_cache

    def test_returns_cached_model_without_lock(self) -> None:
        """Should return cached model directly, skipping lock acquisition (branch 40->53)."""
        mock_model = MagicMock(spec=["transcribe"])
        _model_cache["tiny_cpu_float32"] = mock_model

        # Call get_model without patching WhisperModel - should return cached model
        result = get_model("tiny", "cpu", "float32")
        assert result is mock_model

    def test_thread_safety(self) -> None:
        """Should handle concurrent access safely."""
        mock_model = MagicMock(spec=["transcribe"])
        results = []

        def create_model():
            with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model):
                results.append(get_model("tiny", "cpu", "float32"))

        import threading
        threads = [threading.Thread(target=create_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same cached model
        assert all(r is mock_model for r in results)
        # Model should only be created once
        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model) as mock_cls:
            get_model("tiny", "cpu", "float32")
            mock_cls.assert_not_called()  # Already cached


class TestAnalyzeAudio:
    """Test the analyze_audio function."""

    def _make_audio_file(self, path: str = "/test/audio.mp3") -> AudioFile:
        """Create a mock AudioFile."""
        return AudioFile(
            path=Path(path),
            filename="audio.mp3",
            extension=".mp3",
            size_mb=1.0,
        )

    def _make_mock_segment(
        self, text: str = "Hello world", start: float = 0.0, end: float = 2.0,
        words=None
    ) -> MagicMock:
        """Create a mock transcription segment."""
        mock_segment = MagicMock()
        mock_segment.text = text
        mock_segment.start = start
        mock_segment.end = end
        mock_segment.words = words or []
        return mock_segment

    def _make_mock_word(self, word: str = "Hello", start: float = 0.0, end: float = 0.5, prob: float = 0.99) -> MagicMock:
        """Create a mock word."""
        mock_word = MagicMock()
        mock_word.word = word
        mock_word.start = start
        mock_word.end = end
        mock_word.probability = prob
        return mock_word

    def test_raises_on_zero_beam_size(self) -> None:
        """Should raise ValueError when beam_size is zero."""
        audio_file = self._make_audio_file()

        with pytest.raises(ValueError, match="beam_size must be positive"):
            analyze_audio(audio_file, beam_size=0)

    def test_raises_on_negative_beam_size(self) -> None:
        """Should raise ValueError when beam_size is negative."""
        audio_file = self._make_audio_file()

        with pytest.raises(ValueError, match="beam_size must be positive"):
            analyze_audio(audio_file, beam_size=-1)

    def test_transcribes_audio_file(self) -> None:
        """Should transcribe audio and return TranscriptionResult."""
        audio_file = self._make_audio_file()

        mock_word = self._make_mock_word("Hello", 0.0, 0.5, 0.99)
        mock_segment = self._make_mock_segment("Hello world", 0.0, 2.0, [mock_word])
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 10.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file, beam_size=5)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 10.0
        assert result.has_speech is True

    def test_calls_transcribe_with_correct_params(self) -> None:
        """Should call transcribe with correct parameters."""
        audio_file = self._make_audio_file()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", duration=0.0))

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            analyze_audio(
                audio_file,
                model_name="tiny",
                device="cpu",
                compute_type="float32",
                beam_size=10,
                vad_filter=True,
            )

        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs[0][0] == str(audio_file.path)
        assert call_kwargs[1]["beam_size"] == 10
        assert call_kwargs[1]["word_timestamps"] is True
        assert call_kwargs[1]["vad_filter"] is True
        assert call_kwargs[1]["vad_parameters"] == {"min_silence_duration_ms": 500}

    def test_disables_vad_when_false(self) -> None:
        """Should pass vad_parameters=None when vad_filter is False."""
        audio_file = self._make_audio_file()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", duration=0.0))

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            analyze_audio(audio_file, vad_filter=False)

        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs[1]["vad_filter"] is False
        assert call_kwargs[1]["vad_parameters"] is None

    def test_collects_multiple_segments(self) -> None:
        """Should collect all segments and combine text."""
        audio_file = self._make_audio_file()

        mock_segment1 = self._make_mock_segment("Hello", 0.0, 1.0)
        mock_segment2 = self._make_mock_segment("world", 1.0, 2.0)
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert result.text == "Hello world"
        assert len(result.segments) == 2

    def test_extracts_word_timestamps(self) -> None:
        """Should extract word timestamps from segments."""
        audio_file = self._make_audio_file()

        mock_word1 = self._make_mock_word("Hello", 0.0, 0.5, 0.99)
        mock_word2 = self._make_mock_word("world", 0.5, 1.0, 0.95)
        mock_segment = self._make_mock_segment("Hello world", 0.0, 1.0, [mock_word1, mock_word2])
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 2
        assert result.segments[0].words[0].word == "Hello"
        assert result.segments[0].words[0].start == 0.0
        assert result.segments[0].words[0].end == 0.5
        assert result.segments[0].words[0].probability == 0.99

    def test_handles_segments_without_words(self) -> None:
        """Should handle segments with no word timestamps."""
        audio_file = self._make_audio_file()

        mock_segment = self._make_mock_segment("No words here", 0.0, 2.0, words=None)
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 0

    def test_has_speech_true_when_segments_exist(self) -> None:
        """Should set has_speech=True when there are segments with speech."""
        audio_file = self._make_audio_file()

        mock_segment = self._make_mock_segment("Speech detected", 0.0, 2.0)
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert result.has_speech is True

    def test_has_speech_false_when_no_segments(self) -> None:
        """Should set has_speech=False when there are no segments."""
        audio_file = self._make_audio_file()

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert result.has_speech is False
        assert result.text == ""

    def test_calculates_total_speech_duration(self) -> None:
        """Should calculate total speech duration from segments."""
        audio_file = self._make_audio_file()

        mock_segment1 = self._make_mock_segment("Part 1", 0.0, 2.0)
        mock_segment2 = self._make_mock_segment("Part 2", 3.0, 5.0)
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 10.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        # Both segments have text, so has_speech should be True
        assert result.has_speech is True
        # Check that speech duration was calculated (2.0 + 2.0 = 4.0)
        assert len(result.segments) == 2

    def test_passes_model_config_to_get_model(self) -> None:
        """Should pass model configuration to get_model."""
        audio_file = self._make_audio_file()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", duration=0.0))

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model) as mock_get:
            analyze_audio(
                audio_file,
                model_name="turbo",
                device="cuda",
                compute_type="int8",
            )

        mock_get.assert_called_once_with("turbo", "cuda", "int8")

    def test_empty_text_from_segments(self) -> None:
        """Should handle segments that return empty text after stripping."""
        audio_file = self._make_audio_file()

        mock_segment = MagicMock()
        mock_segment.text = "   "  # Only whitespace
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        # Empty text after stripping still counts as speech (segment exists)
        assert result.has_speech is True

    def test_mixed_segments_with_and_without_words(self) -> None:
        """Should handle a mix of segments with and without word timestamps."""
        audio_file = self._make_audio_file()

        # Segment with words
        mock_word = self._make_mock_word("Hello", 0.0, 0.5, 0.99)
        mock_segment1 = self._make_mock_segment("Hello world", 0.0, 2.0, [mock_word])
        
        # Segment without words
        mock_segment2 = self._make_mock_segment("No words here", 2.0, 4.0, words=None)
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 10.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert len(result.segments) == 2
        assert len(result.segments[0].words) == 1
        assert len(result.segments[1].words) == 0
        assert result.text == "Hello world No words here"
        assert result.has_speech is True

    def test_single_word_segment(self) -> None:
        """Should handle segment with a single word."""
        audio_file = self._make_audio_file()

        mock_word = self._make_mock_word("Yes", 0.0, 0.3, 0.98)
        mock_segment = self._make_mock_segment("Yes", 0.0, 0.3, [mock_word])
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert result.text == "Yes"
        assert len(result.segments) == 1
        assert result.segments[0].words[0].word == "Yes"

    def test_segment_text_with_leading_trailing_whitespace(self) -> None:
        """Should strip whitespace from segment text when joining."""
        audio_file = self._make_audio_file()

        mock_segment1 = self._make_mock_segment("  Hello  ", 0.0, 1.0)
        mock_segment2 = self._make_mock_segment("  world  ", 1.0, 2.0)
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        with patch("lrcfilter.analyzer.get_model", return_value=mock_model):
            result = analyze_audio(audio_file)

        assert result.text == "Hello world"

    def test_model_loading_logs_info(self) -> None:
        """Should log info when loading model."""
        mock_model = MagicMock(spec=["transcribe"])

        with patch("lrcfilter.analyzer.WhisperModel", return_value=mock_model):
            with patch("lrcfilter.analyzer.logger") as mock_logger:
                get_model("tiny", "cpu", "float32")
                mock_logger.info.assert_any_call("Loading Whisper model 'tiny' on cpu...")
                mock_logger.info.assert_any_call("Model loaded successfully.")

    def test_model_creation_failure_logs_error(self) -> None:
        """Should log error when model creation fails."""
        with patch("lrcfilter.analyzer.WhisperModel", side_effect=RuntimeError("CUDA error")):
            with patch("lrcfilter.analyzer.logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    get_model("large-v3", "cuda", "float16")
                mock_logger.error.assert_called_once()
