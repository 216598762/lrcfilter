"""Shared utility functions for text normalization and comparison."""

import re
from typing import Optional


def normalize_text(
    text: str,
    *,
    remove_quotes: bool = False,
    remove_filler_words: bool = False,
    remove_suffixes: bool = False,
    filler_words: Optional[set[str]] = None,
    suffixes: Optional[set[str]] = None,
) -> str:
    """
    Normalize text for comparison.

    This is a shared utility used by both censorship and mismatch detection.
    Behavior is controlled via keyword arguments.

    Args:
        text: Input text to normalize
        remove_quotes: If True, remove all quotes entirely.
                      If False, normalize different quote types to standard ones.
        remove_filler_words: If True, remove common filler words (the, a, an, etc.)
        remove_suffixes: If True, remove common suffixes like (remix), (live), etc.
        filler_words: Custom set of filler words to remove. Ignored if remove_filler_words is False.
                     Defaults to common English filler words.
        suffixes: Custom set of suffixes to remove. Ignored if remove_suffixes is False.
                Defaults to common audio metadata suffixes.

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Handle quotes
    if remove_quotes:
        text = text.replace("'", "").replace("'", "")
        text = text.replace('"', "").replace('"', "")
    else:
        # Normalize different quote types to standard ones
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r'["""]', '"', text)

    # Remove suffixes before removing extra whitespace
    if remove_suffixes:
        if suffixes is None:
            suffixes = {
                "(remix)",
                "(live)",
                "(acoustic)",
                "(instrumental)",
                "(radio edit)",
                "(album version)",
                "(single version)",
                "(bonus track)",
                "(deluxe edition)",
                "(remastered)",
                "(explicit)",
                "(clean)",
                "(clean version)",
                "(explicit version)",
            }
        for suffix in suffixes:
            text = text.replace(suffix, "")

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove filler words
    if remove_filler_words:
        if filler_words is None:
            filler_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
            }
        words = text.split()
        words = [w for w in words if w not in filler_words]
        text = " ".join(words)

    return text.strip()


def normalize_for_censorship(text: str) -> str:
    """
    Normalize text for censorship detection (lyrics vs transcription comparison).

    - Normalizes quotes to standard types
    - Removes common filler words
    - Preserves parenthetical content

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return normalize_text(
        text,
        remove_quotes=False,
        remove_filler_words=True,
        remove_suffixes=False,
    )


def normalize_for_mismatch(text: str) -> str:
    """
    Normalize text for metadata mismatch detection (title/artist comparison).

    - Removes quotes entirely
    - Removes common parenthetical suffixes
    - Preserves filler words for better fuzzy matching

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return normalize_text(
        text,
        remove_quotes=True,
        remove_filler_words=False,
        remove_suffixes=True,
    )
