"""Tests for scanner module."""

from pathlib import Path
from lrcfilter.scanner import scan_audio_files
from lrcfilter.config import SUPPORTED_FORMATS


def test_scan_empty_directory(tmp_path: Path) -> None:
    """Test scanning an empty directory."""
    result = scan_audio_files(tmp_path)
    assert result == []


def test_scan_supported_formats(tmp_path: Path) -> None:
    """Test scanning for supported audio formats."""
    # Create test files with supported extensions
    for ext in SUPPORTED_FORMATS:
        test_file = tmp_path / f"test{ext}"
        test_file.touch()
    
    result = scan_audio_files(tmp_path)
    assert len(result) == len(SUPPORTED_FORMATS)
    
    # Verify all extensions are found
    found_extensions = {f.extension for f in result}
    assert found_extensions == SUPPORTED_FORMATS


def test_scan_ignores_unsupported_formats(tmp_path: Path) -> None:
    """Test that unsupported formats are ignored."""
    # Create unsupported files
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.wav").touch()
    (tmp_path / "test.jpg").touch()
    
    # Create one supported file
    (tmp_path / "test.mp3").touch()
    
    result = scan_audio_files(tmp_path)
    assert len(result) == 1
    assert result[0].extension == ".mp3"


def test_scan_recursive(tmp_path: Path) -> None:
    """Test recursive directory scanning."""
    # Create nested directories
    sub_dir1 = tmp_path / "subdir1"
    sub_dir2 = tmp_path / "subdir1" / "subdir2"
    sub_dir1.mkdir()
    sub_dir2.mkdir()
    
    # Create files in different locations
    (tmp_path / "root.mp3").touch()
    (sub_dir1 / "sub1.flac").touch()
    (sub_dir2 / "sub2.ogg").touch()
    
    result = scan_audio_files(tmp_path)
    assert len(result) == 3
    
    # Verify all files are found
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
