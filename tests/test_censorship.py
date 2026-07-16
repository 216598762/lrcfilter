"""Tests for censorship module."""

from lrcfilter.censorship import (
    detect_censorship,
    _normalize_text,
    _calculate_mismatch_score,
    _detect_profanity,
)
from lrcfilter.models import CensorshipResult


def test_normalize_text_censorship() -> None:
    """Test text normalization for censorship detection."""
    assert _normalize_text("Hello World") == "hello world"
    assert _normalize_text("  extra   spaces  ") == "extra spaces"
    # Filler words are removed, apostrophes are normalized
    assert _normalize_text("it's a test") == "it's test"


def test_calculate_mismatch_score_identical() -> None:
    """Test mismatch score for identical texts."""
    lyrics = "This is a test song"
    transcription = "This is a test song"
    
    score = _calculate_mismatch_score(lyrics, transcription)
    assert score == 0.0  # No mismatch


def test_calculate_mismatch_score_different() -> None:
    """Test mismatch score for completely different texts."""
    lyrics = "This is a love song about roses"
    transcription = "Completely unrelated spoken words here"
    
    score = _calculate_mismatch_score(lyrics, transcription)
    assert score > 0.5  # Significant mismatch


def test_calculate_mismatch_score_empty() -> None:
    """Test mismatch score with empty texts."""
    score = _calculate_mismatch_score("", "some text")
    assert score == 0.0
    
    score = _calculate_mismatch_score("some text", "")
    assert score == 0.0


def test_detect_profanity_clean() -> None:
    """Test profanity detection on clean text."""
    count = _detect_profanity("This is a clean song with no bad words")
    assert count == 0


def test_detect_profanity_dirty() -> None:
    """Test profanity detection on text with profanity."""
    count = _detect_profanity("This is a damn song with shit in it")
    assert count == 2


def test_detect_profanity_empty() -> None:
    """Test profanity detection on empty text."""
    count = _detect_profanity("")
    assert count == 0
    count = _detect_profanity(None)
    assert count == 0


def test_detect_censorship_clean() -> None:
    """Test censorship detection on clean content."""
    lyrics = "This is a beautiful song about love and happiness"
    transcription = "This is a beautiful song about love and happiness"
    
    result = detect_censorship(lyrics, transcription)
    
    assert isinstance(result, CensorshipResult)
    assert result.is_censored is False
    assert result.profanity_count == 0
    assert result.mismatch_score < 0.3


def test_detect_censorship_censored() -> None:
    """Test censorship detection on censored content."""
    lyrics = "I want to fuck you so bad tonight"
    transcription = "Completely different words spoken here"
    
    result = detect_censorship(lyrics, transcription)
    
    assert isinstance(result, CensorshipResult)
    assert result.is_censored is True
    assert result.mismatch_score > 0.3


def test_detect_censorship_explicit() -> None:
    """Test censorship detection on explicit content."""
    lyrics = "Normal lyrics here"
    transcription = "Normal lyrics with damn and shit"
    
    result = detect_censorship(lyrics, transcription)
    
    assert isinstance(result, CensorshipResult)
    assert result.is_censored is True
    assert result.profanity_count == 2


def test_censorship_result_dataclass() -> None:
    """Test CensorshipResult dataclass creation."""
    result = CensorshipResult(
        is_censored=True,
        mismatch_score=0.5,
        profanity_count=3,
        confidence=0.8,
        details="Lyrics mismatch and profanity detected",
    )
    
    assert result.is_censored is True
    assert result.mismatch_score == 0.5
    assert result.profanity_count == 3
    assert result.confidence == 0.8
    assert result.details == "Lyrics mismatch and profanity detected"
