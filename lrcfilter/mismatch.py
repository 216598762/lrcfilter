"""Metadata mismatch detection module for identifying mismatched lyrics."""

from rapidfuzz import fuzz

from lrcfilter.config import TITLE_MATCH_THRESHOLD, ARTIST_MATCH_THRESHOLD, DURATION_TOLERANCE
from lrcfilter.models import TrackMetadata, LyricsResult, MismatchResult


def detect_metadata_mismatch(
    file_metadata: TrackMetadata,
    lyrics_result: LyricsResult,
) -> MismatchResult:
    """
    Detect if lyrics metadata doesn't match file metadata.
    
    Args:
        file_metadata: Metadata from the audio file
        lyrics_result: Lyrics fetched from API
        
    Returns:
        MismatchResult with detection results
    """
    # Title similarity
    title_score = _calculate_title_similarity(
        file_metadata.title,
        lyrics_result.matched_track_name,
    )
    
    # Artist similarity
    artist_score = _calculate_artist_similarity(
        file_metadata.artist,
        lyrics_result.matched_artist_name,
    )
    
    # Duration difference (None if not available)
    duration_difference = _calculate_duration_difference(
        file_metadata.duration_seconds,
    )
    
    # Determine if mismatch
    has_duration_mismatch = (
        duration_difference is not None and duration_difference > DURATION_TOLERANCE
    )
    is_mismatch = (
        title_score < TITLE_MATCH_THRESHOLD or
        artist_score < ARTIST_MATCH_THRESHOLD or
        has_duration_mismatch
    )
    
    # Calculate confidence
    confidence = _calculate_confidence(title_score, artist_score, duration_difference)
    
    # Generate details
    details = _generate_details(
        file_metadata,
        lyrics_result,
        title_score,
        artist_score,
        duration_difference,
        is_mismatch,
    )
    
    return MismatchResult(
        is_mismatch=is_mismatch,
        title_score=title_score,
        artist_score=artist_score,
        duration_difference=duration_difference or 0.0,
        confidence=confidence,
        details=details,
    )


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove common variations
    text = text.replace("'", "").replace("'", "")
    text = text.replace('"', '').replace('"', '')
    
    # Remove common suffixes/prefixes
    for suffix in ["(remix)", "(live)", "(acoustic)", "(instrumental)", "(radio edit)"]:
        text = text.replace(suffix, "")
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text.strip()


def _calculate_title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not title1 or not title2:
        return 0.0
    
    norm_title1 = _normalize_text(title1)
    norm_title2 = _normalize_text(title2)
    
    if not norm_title1 or not norm_title2:
        return 0.0
    
    # Use token sort ratio for flexible matching
    similarity = fuzz.token_sort_ratio(norm_title1, norm_title2)
    
    return similarity / 100.0


def _calculate_artist_similarity(artist1: str, artist2: str) -> float:
    """
    Calculate similarity between two artist names.
    
    Args:
        artist1: First artist name
        artist2: Second artist name
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not artist1 or not artist2:
        return 0.0
    
    norm_artist1 = _normalize_text(artist1)
    norm_artist2 = _normalize_text(artist2)
    
    if not norm_artist1 or not norm_artist2:
        return 0.0
    
    # Use token sort ratio for flexible matching
    similarity = fuzz.token_sort_ratio(norm_artist1, norm_artist2)
    
    return similarity / 100.0


def _calculate_duration_difference(
    file_duration: float,
) -> float:
    """
    Calculate duration difference between file and lyrics.
    
    Note: LyricsResult doesn't currently include duration, so this
    always returns None until duration is added to LyricsResult.
    
    Args:
        file_duration: Duration from audio file
        
    Returns:
        Duration difference in seconds, or None if not available
    """
    # LyricsResult doesn't have duration yet - return None for now
    return None


def _calculate_confidence(
    title_score: float,
    artist_score: float,
    duration_difference: float,
) -> float:
    """
    Calculate confidence score for mismatch detection.
    
    Args:
        title_score: Title similarity score
        artist_score: Artist similarity score
        duration_difference: Duration difference in seconds
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Lower scores = higher confidence in mismatch
    title_confidence = 1.0 - title_score
    artist_confidence = 1.0 - artist_score
    
    # Duration difference contribution
    duration_confidence = 0.0
    if duration_difference is not None:
        duration_confidence = min(duration_difference / 60.0, 1.0)
    
    # Weighted average
    return (title_confidence * 0.4 + artist_confidence * 0.4 + duration_confidence * 0.2)


def _generate_details(
    file_metadata: TrackMetadata,
    lyrics_result: LyricsResult,
    title_score: float,
    artist_score: float,
    duration_difference: float,
    is_mismatch: bool,
) -> str:
    """
    Generate human-readable details about metadata mismatch.
    
    Args:
        file_metadata: File metadata
        lyrics_result: Lyrics result
        title_score: Title similarity
        artist_score: Artist similarity
        duration_difference: Duration difference
        is_mismatch: Whether mismatch was detected
        
    Returns:
        Details string
    """
    if not is_mismatch:
        return "Metadata matches lyrics"
    
    parts = []
    
    file_title = file_metadata.title or "Unknown"
    lyrics_title = lyrics_result.matched_track_name or "Unknown"
    parts.append(f"File: '{file_title}' vs Lyrics: '{lyrics_title}' ({title_score:.0%} match)")
    
    file_artist = file_metadata.artist or "Unknown"
    lyrics_artist = lyrics_result.matched_artist_name or "Unknown"
    parts.append(f"Artist: '{file_artist}' vs '{lyrics_artist}' ({artist_score:.0%} match)")
    
    if duration_difference is not None and duration_difference > DURATION_TOLERANCE:
        parts.append(f"Duration differs by {duration_difference:.1f}s")
    
    return "; ".join(parts)
