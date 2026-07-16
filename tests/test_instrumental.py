"""Tests for instrumental module."""

from lrcfilter.instrumental import detect_instrumental
from lrcfilter.models import Segment, TranscriptionResult, Word


def test_detect_instrumental_no_speech() -> None:
    """Test instrumental detection when no speech is detected."""
    transcription = TranscriptionResult(
        text="",
        segments=[],
        language="en",
        duration=180.0,
        has_speech=False,
    )

    result = detect_instrumental(transcription)

    assert result.is_instrumental is True
    assert result.word_count == 0
    assert result.speech_duration == 0.0
    assert result.confidence > 0.9


def test_detect_instrumental_minimal_speech() -> None:
    """Test instrumental detection with minimal speech."""
    transcription = TranscriptionResult(
        text="yeah",
        segments=[
            Segment(
                start=0.0,
                end=0.5,
                text="yeah",
                words=[Word(start=0.0, end=0.5, word="yeah", probability=0.9)],
            )
        ],
        language="en",
        duration=180.0,
        has_speech=True,
    )

    result = detect_instrumental(transcription)

    assert result.is_instrumental is True
    assert result.word_count == 1
    assert result.speech_duration == 0.5
    assert result.confidence > 0.8


def test_detect_instrumental_with_vocals() -> None:
    """Test instrumental detection when vocals are present."""
    # Create transcription with many words
    words = [Word(start=i, end=i + 0.5, word=f"word{i}", probability=0.9) for i in range(20)]
    text = " ".join([f"word{i}" for i in range(20)])

    transcription = TranscriptionResult(
        text=text,
        segments=[Segment(start=0.0, end=10.0, text=text, words=words)],
        language="en",
        duration=180.0,
        has_speech=True,
    )

    result = detect_instrumental(transcription)

    assert result.is_instrumental is False
    assert result.word_count == 20
    assert result.speech_duration == 10.0
    assert result.confidence < 0.5


def test_detect_instrumental_short_duration() -> None:
    """Test instrumental detection with short speech duration."""
    words = [Word(start=i, end=i + 0.5, word=f"word{i}", probability=0.9) for i in range(15)]
    text = " ".join([f"word{i}" for i in range(15)])

    transcription = TranscriptionResult(
        text=text,
        segments=[
            Segment(start=0.0, end=3.0, text=text, words=words)  # Only 3 seconds
        ],
        language="en",
        duration=180.0,
        has_speech=True,
    )

    result = detect_instrumental(transcription)

    assert result.is_instrumental is True
    assert result.word_count == 15
    assert result.speech_duration == 3.0


def test_instrumental_result_dataclass() -> None:
    """Test InstrumentalResult dataclass creation."""
    from lrcfilter.models import InstrumentalResult

    result = InstrumentalResult(
        is_instrumental=True,
        word_count=5,
        speech_duration=2.5,
        confidence=0.85,
    )

    assert result.is_instrumental is True
    assert result.word_count == 5
    assert result.speech_duration == 2.5
    assert result.confidence == 0.85
