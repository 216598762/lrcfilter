"""Tests for censorship module."""

from unittest.mock import patch

import pytest

from lrcfilter.censorship import (
    _calculate_mismatch_score,
    _detect_profanity,
    _generate_details,
    detect_censorship,
)
from lrcfilter.models import CensorshipResult
from lrcfilter.utils import normalize_for_censorship


def test_normalize_text_censorship() -> None:
    """Test text normalization for censorship detection."""
    assert normalize_for_censorship("Hello World") == "hello world"
    assert normalize_for_censorship("  extra   spaces  ") == "extra spaces"
    # Filler words are removed, apostrophes are normalized
    assert normalize_for_censorship("it's a test") == "it's test"


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


def test_detect_profanity_fallback_no_better_profanity() -> None:
    """Test profanity detection fallback when better_profanity is not available."""
    # When better_profanity is unavailable, COMMON_PROFANITY must also be defined
    fallback_words = {"damn", "shit", "hell"}
    with (
        patch("lrcfilter.censorship.BETTER_PROFANITY_AVAILABLE", False),
        patch("lrcfilter.censorship.COMMON_PROFANITY", fallback_words),
    ):
        # Clean text
        count = _detect_profanity("This is a clean song")
        assert count == 0

        # Text with profanity from fallback list
        count = _detect_profanity("This is a damn song with shit in it")
        assert count == 2

        # Empty text
        count = _detect_profanity("")
        assert count == 0

        # None text
        count = _detect_profanity(None)
        assert count == 0


def test_detect_censorship_fallback_profanity() -> None:
    """Test censorship detection using fallback profanity detection."""
    fallback_words = {"damn", "shit"}
    with (
        patch("lrcfilter.censorship.BETTER_PROFANITY_AVAILABLE", False),
        patch("lrcfilter.censorship.COMMON_PROFANITY", fallback_words),
    ):
        lyrics = "Normal lyrics here"
        transcription = "Normal lyrics with damn and shit"

        result = detect_censorship(lyrics, transcription)

        assert result.is_censored is True
        assert result.profanity_count == 2


def test_detect_censorship_clean() -> None:
    """Test censorship detection on clean content."""
    lyrics = "This is a beautiful song about love and happiness"
    transcription = "This is a beautiful song about love and happiness"

    result = detect_censorship(lyrics, transcription)

    assert isinstance(result, CensorshipResult)
    assert result.is_censored is False
    assert result.profanity_count == 0
    assert result.mismatch_score < 0.3
    assert result.details == "No censorship detected"


def test_detect_censorship_censored() -> None:
    """Test censorship detection on censored content."""
    lyrics = "I want to fuck you so bad tonight"
    transcription = "Completely different words spoken here"

    result = detect_censorship(lyrics, transcription)

    assert isinstance(result, CensorshipResult)
    assert result.is_censored is True
    assert result.mismatch_score > 0.3
    assert "mismatch" in result.details.lower() or "lyrics" in result.details.lower()


def test_detect_censorship_explicit() -> None:
    """Test censorship detection on explicit content."""
    lyrics = "Normal lyrics here"
    transcription = "Normal lyrics with damn and shit"

    result = detect_censorship(lyrics, transcription)

    assert isinstance(result, CensorshipResult)
    assert result.is_censored is True
    assert result.profanity_count == 2
    assert "profanity" in result.details


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


def test_detect_censorship_threshold_boundary() -> None:
    """Test censorship detection at threshold boundaries."""
    # Threshold = 0.0 should detect any mismatch
    lyrics = "Completely different lyrics"
    transcription = "Totally unrelated words"
    result = detect_censorship(lyrics, transcription, threshold=0.0)
    assert result.is_censored is True

    # Threshold = 1.0 should not detect mismatch-based censorship
    result = detect_censorship(lyrics, transcription, threshold=1.0)
    assert result.mismatch_score < 1.0  # Mismatch exists but below threshold


def test_detect_censorship_invalid_threshold() -> None:
    """Test that invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match=r"threshold must be between 0\.0 and 1\.0"):
        detect_censorship("test", "test", threshold=-0.1)
    with pytest.raises(ValueError, match=r"threshold must be between 0\.0 and 1\.0"):
        detect_censorship("test", "test", threshold=1.1)


def test_generate_details_mismatch_only() -> None:
    """Test _generate_details with mismatch only (no profanity)."""
    details = _generate_details(mismatch_score=0.8, profanity_count=0, is_censored=True, threshold=0.5)
    assert "mismatch" in details.lower()
    assert "profanity" not in details


def test_generate_details_profanity_only() -> None:
    """Test _generate_details with profanity only (no mismatch)."""
    details = _generate_details(mismatch_score=0.1, profanity_count=3, is_censored=True, threshold=0.5)
    assert "profanity" in details
    assert "3 profanity" in details


def test_generate_details_both() -> None:
    """Test _generate_details with both mismatch and profanity."""
    details = _generate_details(mismatch_score=0.8, profanity_count=2, is_censored=True, threshold=0.5)
    assert "mismatch" in details.lower()
    assert "profanity" in details
    assert "; " in details  # Both parts joined with semicolon
