"""Censorship detection module for identifying censored or explicit content."""

import re
from typing import List, Tuple

from rapidfuzz import fuzz

from lrcfilter.config import CENSORSHIP_MISMATCH_THRESHOLD
from lrcfilter.models import CensorshipResult

# Optional better-profanity import
try:
    from better_profanity import profanity
    profanity.load_censor_words()
    BETTER_PROFANITY_AVAILABLE = True
except ImportError:
    BETTER_PROFANITY_AVAILABLE = False
    # Fallback to basic profanity list
    COMMON_PROFANITY = {
        "fuck", "shit", "damn", "hell", "ass", "bitch",
        "bastard", "crap", "dick", "cock", "pussy",
        "whore", "slut", "nigger", "nigga", "faggot",
        "retard", "retarded",
    }


def detect_censorship(
    lyrics: str,
    transcription: str,
    threshold: float = CENSORSHIP_MISMATCH_THRESHOLD,
) -> CensorshipResult:
    """
    Detect censorship using lyrics vs transcription mismatch and profanity detection.
    
    Args:
        lyrics: Original lyrics text
        transcription: Whisper transcription text
        threshold: Mismatch score threshold above which lyrics are considered censored.
                   Default from config.CENSORSHIP_MISMATCH_THRESHOLD.
        
    Returns:
        CensorshipResult with detection results
        
    Raises:
        ValueError: If threshold is not between 0.0 and 1.0
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Method 1: Lyrics vs transcription mismatch
    mismatch_score = _calculate_mismatch_score(lyrics, transcription)
    
    # Method 2: Profanity detection
    profanity_count = _detect_profanity(transcription)
    
    # Combined decision
    is_censored = mismatch_score > threshold or profanity_count > 0
    
    # Calculate confidence
    confidence = _calculate_confidence(mismatch_score, profanity_count)
    
    # Generate details
    details = _generate_details(mismatch_score, profanity_count, is_censored, threshold)
    
    return CensorshipResult(
        is_censored=is_censored,
        mismatch_score=mismatch_score,
        profanity_count=profanity_count,
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
    # Lowercase
    text = text.lower()
    
    # Remove common variations
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r'["""]', '"', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common filler words
    filler_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    words = text.split()
    words = [w for w in words if w not in filler_words]
    
    return " ".join(words)


def _calculate_mismatch_score(lyrics: str, transcription: str) -> float:
    """
    Calculate mismatch score between lyrics and transcription.
    
    Args:
        lyrics: Original lyrics
        transcription: Whisper transcription
        
    Returns:
        Mismatch score between 0.0 and 1.0
    """
    if not lyrics or not transcription:
        return 0.0
    
    normalized_lyrics = _normalize_text(lyrics)
    normalized_transcription = _normalize_text(transcription)
    
    # Use fuzzy matching for comparison
    # Token sort ratio handles different word orderings
    similarity = fuzz.token_sort_ratio(normalized_lyrics, normalized_transcription)
    
    # Convert similarity to mismatch score (100 - similarity)
    mismatch_score = (100 - similarity) / 100.0
    
    return mismatch_score


def _detect_profanity(text: str) -> int:
    """
    Detect profanity in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of profanity instances found
    """
    if not text:
        return 0
    
    if BETTER_PROFANITY_AVAILABLE:
        # Use better-profanity library for comprehensive detection
        # Find all words in text and check each one
        words = re.findall(r'\b\w+\b', text)
        count = 0
        for word in words:
            if profanity.contains_profanity(word):
                count += 1
        return count
    else:
        # Fallback to basic profanity list
        words = re.findall(r'\b\w+\b', text.lower())
        
        count = 0
        for word in words:
            if word in COMMON_PROFANITY:
                count += 1
        
        return count


def _calculate_confidence(mismatch_score: float, profanity_count: int) -> float:
    """
    Calculate confidence score for censorship detection.
    
    Args:
        mismatch_score: Mismatch score (0.0 to 1.0)
        profanity_count: Number of profanity instances
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Higher mismatch = higher confidence
    mismatch_confidence = min(mismatch_score * 2, 1.0)
    
    # Any profanity = high confidence
    profanity_confidence = min(profanity_count * 0.5, 1.0)
    
    # Weighted average
    return (mismatch_confidence * 0.6 + profanity_confidence * 0.4)


def _generate_details(
    mismatch_score: float,
    profanity_count: int,
    is_censored: bool,
    threshold: float = CENSORSHIP_MISMATCH_THRESHOLD,
) -> str:
    """
    Generate human-readable details about censorship detection.
    
    Args:
        mismatch_score: Mismatch score
        profanity_count: Profanity count
        is_censored: Whether censorship was detected
        
    Returns:
        Details string
    """
    if not is_censored:
        return "No censorship detected"
    
    parts = []
    
    if mismatch_score > threshold:
        parts.append(f"Lyrics mismatch ({mismatch_score:.1%})")
    
    if profanity_count > 0:
        parts.append(f"{profanity_count} profanity instance(s)")
    
    return "; ".join(parts)
