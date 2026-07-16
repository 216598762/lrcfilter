"""Metadata extraction module for audio files."""

from typing import Any, Optional

from mutagen import File as MutagenFile

from lrcfilter.logging_config import get_logger
from lrcfilter.models import AudioFile, TrackMetadata

logger = get_logger(__name__)


def extract_metadata(audio_file: AudioFile) -> TrackMetadata:
    """
    Extract metadata from an audio file.

    Args:
        audio_file: AudioFile object to extract metadata from

    Returns:
        TrackMetadata with extracted information
    """
    try:
        mutagen_file = MutagenFile(audio_file.path)

        if mutagen_file is None:
            return _create_empty_metadata(audio_file)

        title = None
        artist = None
        album = None
        duration = None
        raw_tags: dict[str, Any] = {}

        # Get duration
        if mutagen_file.info and hasattr(mutagen_file.info, "length"):
            duration = mutagen_file.info.length

        # Extract tags from the mutagen file
        tags = mutagen_file.tags

        if tags:
            # Store all raw tags
            raw_tags = {k: str(v) for k, v in tags.items()}

            # Extract common fields
            title = _get_tag_value(tags, ["TIT2", "title"])
            artist = _get_tag_value(tags, ["TPE1", "artist"])
            album = _get_tag_value(tags, ["TALB", "album"])

        return TrackMetadata(
            title=title,
            artist=artist,
            album=album,
            duration_seconds=duration,
            raw_tags=raw_tags,
        )

    except Exception as e:
        logger.warning(f"Error extracting metadata from {audio_file.path}: {e}")
        return _create_empty_metadata(audio_file)


def _get_tag_value(tags: Any, possible_keys: list) -> Optional[str]:
    """
    Try to get a tag value using multiple possible key names.

    Args:
        tags: Mutagen tags object
        possible_keys: List of possible tag key names

    Returns:
        Tag value if found, None otherwise
    """
    for key in possible_keys:
        if key in tags:
            value = tags[key]
            if hasattr(value, "text"):
                return str(value.text[0]) if value.text else None
            return str(value) if value else None
    return None


def _create_empty_metadata(audio_file: AudioFile) -> TrackMetadata:
    """Create empty metadata when extraction fails."""
    return TrackMetadata(
        title=audio_file.filename,
        artist=None,
        album=None,
        duration_seconds=None,
        raw_tags={},
    )
