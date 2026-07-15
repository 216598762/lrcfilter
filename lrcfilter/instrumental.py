"""Instrumental detection module for identifying tracks with no vocals."""

from lrcfilter.config import MIN_WORDS_FOR_VOCALS, MIN_SPEECH_DURATION
from lrcfilter.models import TranscriptionResult, InstrumentalResult


def detect_instrumental(
    transcription: TranscriptionResult,
    min_words_vocals: int = MIN_WORDS_FOR_VOCALS,
    min_speech_duration: float = MIN_SPEECH_DURATION,
) -> InstrumentalResult:
    """
    Detect if a track is instrumental (no vocals).
    
    Args:
        transcription: Transcription result from Whisper
        min_words_vocals: Minimum word count to consider track as having vocals.
                         Default from config.MIN_WORDS_FOR_VOCALS.
        min_speech_duration: Minimum total speech duration in seconds to consider
                             track as having vocals. Default from config.MIN_SPEECH_DURATION.
        
    Returns:
        InstrumentalResult with detection results
        
    Raises:
        ValueError: If min_words_vocals is negative or min_speech_duration is negative
    """
    if min_words_vocals < 0:
        raise ValueError(f"min_words_vocals must be non-negative, got {min_words_vocals}")
    if min_speech_duration < 0:
        raise ValueError(f"min_speech_duration must be non-negative, got {min_speech_duration}")
    
    # Count words in transcription
    word_count = len(transcription.text.split()) if transcription.text else 0
    
    # Calculate total speech duration from segments
    speech_duration = 0.0
    for segment in transcription.segments:
        speech_duration += segment.end - segment.start
    
    # Determine if instrumental
    is_instrumental = (
        not transcription.has_speech or
        word_count < min_words_vocals or
        speech_duration < min_speech_duration
    )
    
    # Calculate confidence
    confidence = _calculate_confidence(
        word_count=word_count,
        speech_duration=speech_duration,
        has_speech=transcription.has_speech,
        min_words_vocals=min_words_vocals,
        min_speech_duration=min_speech_duration,
    )
    
    return InstrumentalResult(
        is_instrumental=is_instrumental,
        word_count=word_count,
        speech_duration=speech_duration,
        confidence=confidence,
    )


def _calculate_confidence(
    word_count: int,
    speech_duration: float,
    has_speech: bool,
    min_words_vocals: int = MIN_WORDS_FOR_VOCALS,
    min_speech_duration: float = MIN_SPEECH_DURATION,
) -> float:
    """
    Calculate confidence score for instrumental detection.
    
    Args:
        word_count: Number of words detected
        speech_duration: Duration of detected speech in seconds
        has_speech: Whether any speech was detected
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not has_speech:
        # No speech detected = high confidence
        return 0.95
    
    if word_count < 5:
        # Very few words = high confidence
        return 0.9
    
    if word_count < min_words_vocals:
        # Below threshold = medium-high confidence
        return 0.8
    
    if speech_duration < min_speech_duration:
        # Short speech duration = medium confidence
        return 0.7
    
    # Above thresholds = low confidence (likely has vocals)
    return 0.2
