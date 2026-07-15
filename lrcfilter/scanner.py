"""File scanner module for discovering audio files."""

import os
from pathlib import Path
from typing import List, Optional, Set

from lrcfilter.config import SUPPORTED_FORMATS
from lrcfilter.models import AudioFile
from lrcfilter.logging_config import get_logger

logger = get_logger(__name__)


def scan_audio_files(directory: Path, formats: Optional[Set[str]] = None) -> List[AudioFile]:
    """
    Recursively scan a directory for supported audio files.
    
    Args:
        directory: Root directory to scan
        formats: Optional set of file extensions to scan for (e.g., {'.mp3', '.flac'}).
                 If None, uses SUPPORTED_FORMATS from config.
                 If empty set is provided, no files will match.
        
    Returns:
        List of AudioFile objects for discovered files
        
    Raises:
        ValueError: If formats contains invalid extensions (not starting with '.')
    """
    # Validate formats parameter
    if formats is not None:
        for fmt in formats:
            if not isinstance(fmt, str):
                raise ValueError(f"Format must be a string, got {type(fmt).__name__}: {fmt}")
            if not fmt.startswith('.'):
                raise ValueError(f"Format must start with a dot (e.g., '.mp3'), got: {fmt}")
    
    audio_files = []
    visited_inodes: Set[int] = set()  # Track inodes to prevent symlink loops
    
    def _scan_recursive(current_dir: Path) -> None:
        nonlocal audio_files, visited_inodes
        
        try:
            # Get inode to detect symlink loops
            dir_stat = current_dir.stat(follow_symlinks=False)
            dir_inode = dir_stat.st_ino
            
            if dir_inode in visited_inodes:
                return  # Already visited, skip to prevent loop
            visited_inodes.add(dir_inode)
        except (OSError, PermissionError):
            return
        
        try:
            entries = list(current_dir.iterdir())
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not access {current_dir}: {e}")
            return
        
        for entry in entries:
            # Skip hidden files/directories
            if entry.name.startswith('.'):
                continue
            
            if entry.is_dir(follow_symlinks=False):
                _scan_recursive(entry)
            elif entry.is_file(follow_symlinks=False):
                extension = entry.suffix.lower()
                
                if extension in (formats if formats is not None else SUPPORTED_FORMATS):
                    try:
                        size_bytes = entry.stat().st_size
                        size_mb = size_bytes / (1024 * 1024)
                        
                        audio_file = AudioFile(
                            path=entry,
                            filename=entry.name,
                            extension=extension,
                            size_mb=size_mb,
                        )
                        audio_files.append(audio_file)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not access {entry}: {e}")
                        continue
    
    _scan_recursive(directory)
    return sorted(audio_files, key=lambda x: x.path)
