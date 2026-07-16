"""Lyrics fetching module with LRCLib and Genius fallback."""

import os
import time
from typing import Optional

import requests

from lrcfilter.config import API_RATE_LIMIT_DELAY, GENIUS_TOKEN_ENV, LRCLIB_BASE_URL
from lrcfilter.logging_config import get_logger
from lrcfilter.models import LyricsResult, TrackMetadata

logger = get_logger(__name__)

# Optional Genius import
GENIUS_AVAILABLE = False
try:
    import lyricsgenius

    GENIUS_AVAILABLE = True
except ImportError:
    pass


def fetch_lyrics(
    metadata: TrackMetadata,
    genius_token: Optional[str] = None,
    lrclib_only: bool = False,
    api_delay: float = API_RATE_LIMIT_DELAY,
) -> Optional[LyricsResult]:
    """
    Fetch lyrics from LRCLib with Genius fallback.

    Args:
        metadata: Track metadata to search for
        genius_token: Optional Genius API token (or set GENIUS_ACCESS_TOKEN env var)
        lrclib_only: If True, only use LRCLib API and skip Genius fallback.
        api_delay: Delay in seconds between API requests to avoid rate limiting.
                  Default from config.API_RATE_LIMIT_DELAY.

    Returns:
        LyricsResult if lyrics found, None otherwise

    Raises:
        ValueError: If api_delay is negative
    """
    if api_delay < 0:
        raise ValueError(f"api_delay must be non-negative, got {api_delay}")

    if not metadata.title:
        return None

    # Try LRCLib first
    lrclib_result = _fetch_from_lrclib(metadata, api_delay)
    if lrclib_result and (lrclib_result.synced_lyrics or lrclib_result.plain_lyrics):
        return lrclib_result

    # Fall back to Genius if token provided and not lrclib_only
    if lrclib_only:
        return None
    genius_token = genius_token or os.getenv(GENIUS_TOKEN_ENV)
    if genius_token and GENIUS_AVAILABLE:
        genius_result = _fetch_from_genius(metadata, genius_token, api_delay)
        if genius_result and genius_result.plain_lyrics:
            return genius_result

    return None


def _fetch_from_lrclib(
    metadata: TrackMetadata, api_delay: float = API_RATE_LIMIT_DELAY
) -> Optional[LyricsResult]:
    """
    Fetch lyrics from LRCLib API.

    Args:
        metadata: Track metadata to search for

    Returns:
        LyricsResult if found, None otherwise
    """
    try:
        # Search for lyrics
        params = {
            "track_name": metadata.title,
        }
        if metadata.artist:
            params["artist_name"] = metadata.artist
        if metadata.album:
            params["album_name"] = metadata.album
        if metadata.duration_seconds:
            params["duration"] = str(int(metadata.duration_seconds))

        response = requests.get(
            f"{LRCLIB_BASE_URL}/search",
            params=params,
            timeout=10,
        )

        if response.status_code != 200:
            return None

        results = response.json()

        if not results:
            return None

        # Get the first result
        result = results[0]

        # Calculate match score based on title/artist similarity
        match_score = _calculate_match_score(metadata, result)

        return LyricsResult(
            source="lrclib",
            synced_lyrics=result.get("syncedLyrics"),
            plain_lyrics=result.get("plainLyrics"),
            matched_track_name=result.get("trackName", ""),
            matched_artist_name=result.get("artistName", ""),
            matched_album_name=result.get("albumName"),
            match_score=match_score,
        )

    except Exception as e:
        logger.warning(f"LRCLib fetch failed: {e}")
        return None
    finally:
        time.sleep(api_delay)


def _fetch_from_genius(
    metadata: TrackMetadata,
    genius_token: str,
    api_delay: float = API_RATE_LIMIT_DELAY,
) -> Optional[LyricsResult]:
    """
    Fetch lyrics from Genius API.

    Args:
        metadata: Track metadata to search for
        genius_token: Genius API access token

    Returns:
        LyricsResult if found, None otherwise
    """
    if not GENIUS_AVAILABLE:
        return None

    try:
        genius = lyricsgenius.Genius(genius_token, timeout=10)
        genius.verbose = False

        # Search for song
        search_term = f"{metadata.title} {metadata.artist or ''}".strip()
        song = genius.search_song(search_term)

        if not song or not song.lyrics:
            return None

        return LyricsResult(
            source="genius",
            synced_lyrics=None,
            plain_lyrics=song.lyrics,
            matched_track_name=song.title,
            matched_artist_name=song.artist,
            matched_album_name=song.album,
            match_score=0.8,  # Default score for Genius matches
        )

    except Exception as e:
        logger.warning(f"Genius fetch failed: {e}")
        return None
    finally:
        time.sleep(api_delay)


def _calculate_match_score(metadata: TrackMetadata, lrclib_result: dict) -> float:
    """
    Calculate a match score between file metadata and LRCLib result.

    Args:
        metadata: File metadata
        lrclib_result: LRCLib API result

    Returns:
        Match score between 0.0 and 1.0
    """
    score = 0.0
    max_score = 0.0

    # Title match
    file_title = metadata.title
    api_title = lrclib_result.get("trackName")
    if file_title and api_title:
        max_score += 1.0
        if file_title.lower() == api_title.lower():
            score += 1.0
        elif file_title.lower() in api_title.lower():
            score += 0.5

    # Artist match
    file_artist = metadata.artist
    api_artist = lrclib_result.get("artistName")
    if file_artist and api_artist:
        max_score += 1.0
        if file_artist.lower() == api_artist.lower():
            score += 1.0
        elif file_artist.lower() in api_artist.lower():
            score += 0.5

    return score / max_score if max_score > 0 else 0.0
