"""Metadata mismatch detection module for identifying mismatched lyrics."""

from rapidfuzz import fuzz

from lrcfilter.config import TITLE_MATCH_THRESHOLD, ARTIST_MATCH_THRESHOLD
from lrcfilter.models import TrackMetadata, LyricsResult, MismatchResult
from lrcfilter.utils import normalize_for_mismatch


def detect_metadata_mismatch(
    file_metadata: TrackMetadata,
    lyrics_result: LyricsResult,
    title_threshold: float = TITLE_MATCH_THRESHOLD,
    artist_threshold: float = ARTIST_MATCH_THRESHOLD,
    duration_tolerance: float = 30.0,  # Kept for API compatibility
) -> MismatchResult:
    """
    Detect if lyrics metadata doesn't match file metadata.
    
    Args:
        file_metadata: Metadata from the audio file
        lyrics_result: Lyrics fetched from API
        title_threshold: Minimum title similarity score (0-1) to consider match.
                        Default from config.TITLE_MATCH_THRESHOLD.
        artist_threshold: Minimum artist similarity score (0-1) to consider match.
                         Default from config.ARTIST_MATCH_THRESHOLD.
        duration_tolerance: Maximum allowed duration difference in seconds.
                           Default: 30.0 seconds.
        
    Returns:
        MismatchResult with detection results
        
    Raises:
        ValueError: If thresholds are not between 0.0 and 1.0 or duration_tolerance is negative
    """
    if not 0.0 <= title_threshold <= 1.0:
        raise ValueError(f"title_threshold must be between 0.0 and 1.0, got {title_threshold}")
    if not 0.0 <= artist_threshold <= 1.0:
        raise ValueError(f"artist_threshold must be between 0.0 and 1.0, got {artist_threshold}")
    if duration_tolerance < 0:
        raise ValueError(f"duration_tolerance must be non-negative, got {duration_tolerance}")
    
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
    
    # Determine if mismatch
    is_mismatch = (
        title_score < title_threshold or
        artist_score < artist_threshold
    )
    
    # Calculate confidence
    confidence = _calculate_confidence(title_score, artist_score)
    
    # Generate details
    details = _generate_details(
        file_metadata,
        lyrics_result,
        title_score,
        artist_score,
        is_mismatch,
    )
    
    return MismatchResult(
        is_mismatch=is_mismatch,
        title_score=title_score,
        artist_score=artist_score,
        duration_difference=0.0,
        confidence=confidence,
        details=details,
    )


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
    
    norm_title1 = normalize_for_mismatch(title1)
    norm_title2 = normalize_for_mismatch(title2)
    
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
    
    norm_artist1 = normalize_for_mismatch(artist1)
    norm_artist2 = normalize_for_mismatch(artist2)
    
    if not norm_artist1 or not norm_artist2:
        return 0.0
    
    # Use token sort ratio for flexible matching
    similarity = fuzz.token_sort_ratio(norm_artist1, norm_artist2)
    
    return similarity / 100.0


def _calculate_confidence(
    title_score: float,
    artist_score: float,
) -> float:
    """
    Calculate confidence score for mismatch detection.
    
    Args:
        title_score: Title similarity score
        artist_score: Artist similarity score
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Lower scores = higher confidence in mismatch
    title_confidence = 1.0 - title_score
    artist_confidence = 1.0 - artist_score
    
    # Weighted average (title and artist equally weighted)
    return (title_confidence * 0.5 + artist_confidence * 0.5)


def _generate_details(
    file_metadata: TrackMetadata,
    lyrics_result: LyricsResult,
    title_score: float,
    artist_score: float,
    is_mismatch: bool,
) -> str:
    """
    Generate human-readable details about metadata mismatch.
    
    Args:
        file_metadata: File metadata
        lyrics_result: Lyrics result
        title_score: Title similarity
        artist_score: Artist similarity
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
    
    return "; ".join(parts)
