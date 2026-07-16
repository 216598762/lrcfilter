"""Tests for scanner module."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import lrcfilter.scanner as scanner_module
from lrcfilter.config import SUPPORTED_FORMATS
from lrcfilter.scanner import scan_audio_files


def test_scan_empty_directory(tmp_path: Path) -> None:
    """Test scanning an empty directory."""
    result = scan_audio_files(tmp_path)
    assert result == []


def test_scan_supported_formats(tmp_path: Path) -> None:
    """Test scanning for supported audio formats."""
    for ext in SUPPORTED_FORMATS:
        test_file = tmp_path / f"test{ext}"
        test_file.touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == len(SUPPORTED_FORMATS)

    found_extensions = {f.extension for f in result}
    assert found_extensions == SUPPORTED_FORMATS


def test_scan_ignores_unsupported_formats(tmp_path: Path) -> None:
    """Test that unsupported formats are ignored."""
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.wav").touch()
    (tmp_path / "test.jpg").touch()
    (tmp_path / "test.mp3").touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == 1
    assert result[0].extension == ".mp3"


def test_scan_recursive(tmp_path: Path) -> None:
    """Test recursive directory scanning."""
    sub_dir1 = tmp_path / "subdir1"
    sub_dir2 = tmp_path / "subdir1" / "subdir2"
    sub_dir1.mkdir()
    sub_dir2.mkdir()

    (tmp_path / "root.mp3").touch()
    (sub_dir1 / "sub1.flac").touch()
    (sub_dir2 / "sub2.ogg").touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == 3

    filenames = {f.filename for f in result}
    assert filenames == {"root.mp3", "sub1.flac", "sub2.ogg"}


def test_scan_ignores_hidden_files(tmp_path: Path) -> None:
    """Test that hidden files are ignored."""
    (tmp_path / ".hidden.mp3").touch()
    (tmp_path / "visible.mp3").touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == 1
    assert result[0].filename == "visible.mp3"


def test_scan_ignores_hidden_directories(tmp_path: Path) -> None:
    """Test that hidden directories are ignored."""
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.mp3").touch()
    (tmp_path / "visible.mp3").touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == 1
    assert result[0].filename == "visible.mp3"


def test_audio_file_dataclass(tmp_path: Path) -> None:
    """Test AudioFile dataclass creation."""
    from lrcfilter.models import AudioFile

    test_file = tmp_path / "test.mp3"
    test_file.write_bytes(b"test content")

    audio_file = AudioFile(
        path=test_file,
        filename="test.mp3",
        extension=".mp3",
        size_mb=0.0,
    )

    assert audio_file.path == test_file
    assert audio_file.filename == "test.mp3"
    assert audio_file.extension == ".mp3"


def test_scan_symlink_loop_prevention(tmp_path: Path) -> None:
    """Test that symlink loops are detected and skipped (line 50).

    The scanner tracks directory inodes to prevent infinite recursion
    on symlink loops. We replace _stat_fn so the symlink's inode matches
    the target directory's inode, simulating a hardlink-based loop.
    """
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "loop.mp3").touch()

    loop_link = subdir / "loop_dir"
    loop_link.symlink_to(subdir)

    # Get the real inode of subdir
    real_stat = scanner_module._default_stat(str(subdir), follow_symlinks=False)
    subdir_inode = real_stat.st_ino

    # Save original stat function
    original_stat_fn = scanner_module._stat_fn

    def mock_stat(path: str, *, follow_symlinks: bool = True):  # type: ignore[no-untyped-def]
        result = original_stat_fn(path, follow_symlinks=follow_symlinks)
        # Make the symlink return the same inode as the target directory
        if os.path.basename(path) == "loop_dir":
            mock_result = MagicMock()
            mock_result.st_ino = subdir_inode  # Same inode as subdir
            mock_result.st_size = result.st_size
            return mock_result
        return result

    scanner_module._stat_fn = mock_stat
    try:
        result = scan_audio_files(tmp_path)
    finally:
        scanner_module._stat_fn = original_stat_fn

    # Should find the file without infinite recursion
    assert len(result) == 1
    assert result[0].filename == "loop.mp3"


def test_scan_broken_symlink_skipped(tmp_path: Path) -> None:
    """Test that broken symlinks are skipped (branch 67->60).

    A broken symlink is neither is_dir() nor is_file() when checked
    with follow_symlinks=False, so it falls through both branches.
    """
    broken_link = tmp_path / "broken.mp3"
    broken_link.symlink_to(tmp_path / "nonexistent.mp3")

    (tmp_path / "real.mp3").touch()

    result = scan_audio_files(tmp_path)
    assert len(result) == 1
    assert result[0].filename == "real.mp3"


def test_scan_custom_formats(tmp_path: Path) -> None:
    """Test scanning with custom format set."""
    (tmp_path / "test.mp3").touch()
    (tmp_path / "test.flac").touch()
    (tmp_path / "test.ogg").touch()

    result = scan_audio_files(tmp_path, formats={".mp3"})
    assert len(result) == 1
    assert result[0].extension == ".mp3"


def test_scan_empty_formats_set(tmp_path: Path) -> None:
    """Test scanning with empty format set finds nothing."""
    (tmp_path / "test.mp3").touch()

    result = scan_audio_files(tmp_path, formats=set())
    assert result == []


def test_scan_invalid_format_not_string(tmp_path: Path) -> None:
    """Test that non-string format raises ValueError."""
    with pytest.raises(ValueError, match="Format must be a string"):
        scan_audio_files(tmp_path, formats={123})  # type: ignore


def test_scan_invalid_format_no_dot(tmp_path: Path) -> None:
    """Test that format without leading dot raises ValueError."""
    with pytest.raises(ValueError, match="Format must start with a dot"):
        scan_audio_files(tmp_path, formats={"mp3"})


def test_scan_unreadable_file_stat(tmp_path: Path) -> None:
    """Test that files with unreadable stat are skipped (lines 83-85).

    When os.stat() raises OSError or PermissionError for a file, the scanner
    logs a warning and continues to the next file.
    """
    test_file = tmp_path / "test.mp3"
    test_file.touch()
    (tmp_path / "other.mp3").touch()

    original_stat_fn = scanner_module._stat_fn

    def mock_stat(path: str, *, follow_symlinks: bool = True):  # type: ignore[no-untyped-def]
        if os.path.basename(path) == "test.mp3":
            raise PermissionError("Access denied")
        return original_stat_fn(path, follow_symlinks=follow_symlinks)

    scanner_module._stat_fn = mock_stat
    try:
        result = scan_audio_files(tmp_path)
    finally:
        scanner_module._stat_fn = original_stat_fn

    # Only the accessible file should be found
    assert len(result) == 1
    assert result[0].filename == "other.mp3"


def test_scan_dir_stat_oserror_skipped(tmp_path: Path) -> None:
    """Test that directories with unreadable stat are skipped (lines 52-53).

    When os.stat() raises OSError for a directory, the scanner
    returns early and skips that directory entirely.
    """
    inaccessible_dir = tmp_path / "inaccessible"
    inaccessible_dir.mkdir()
    (inaccessible_dir / "track.mp3").touch()

    (tmp_path / "normal.mp3").touch()

    original_stat_fn = scanner_module._stat_fn

    def mock_stat(path: str, *, follow_symlinks: bool = True):  # type: ignore[no-untyped-def]
        if os.path.basename(path) == "inaccessible":
            raise OSError("Permission denied")
        return original_stat_fn(path, follow_symlinks=follow_symlinks)

    scanner_module._stat_fn = mock_stat
    try:
        result = scan_audio_files(tmp_path)
    finally:
        scanner_module._stat_fn = original_stat_fn

    # Only the normal file should be found
    assert len(result) == 1
    assert result[0].filename == "normal.mp3"


def test_scan_iterdir_oserror_skipped(tmp_path: Path) -> None:
    """Test that directories with unreadable listing are skipped (lines 57-59).

    When iterdir() raises OSError or PermissionError, the scanner
    logs a warning and returns early from that directory.
    """
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    (protected_dir / "hidden.mp3").touch()

    (tmp_path / "visible.mp3").touch()

    original_iterdir = Path.iterdir

    def mock_iterdir(self: Path):  # type: ignore[no-untyped-def]
        if self.name == "protected":
            raise PermissionError("Access denied")
        return original_iterdir(self)

    Path.iterdir = mock_iterdir  # type: ignore[assignment]
    try:
        result = scan_audio_files(tmp_path)
    finally:
        Path.iterdir = original_iterdir

    # Only the visible file should be found
    assert len(result) == 1
    assert result[0].filename == "visible.mp3"


def test_scan_sorted_output(tmp_path: Path) -> None:
    """Test that results are sorted by path."""
    (tmp_path / "c.mp3").touch()
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.mp3").touch()

    result = scan_audio_files(tmp_path)
    paths = [f.filename for f in result]
    assert paths == ["a.mp3", "b.mp3", "c.mp3"]
