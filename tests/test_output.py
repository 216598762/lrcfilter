"""Tests for output module."""

from pathlib import Path
from typing import Optional
from unittest.mock import patch

from lrcfilter.models import (
    AudioFile,
    CensorshipResult,
    InstrumentalResult,
    MismatchResult,
)
from lrcfilter.output import write_results


def _make_audio_file(tmp_path: Path, name: str = "test.mp3") -> AudioFile:
    """Create a test AudioFile."""
    path = tmp_path / name
    path.touch()
    return AudioFile(
        path=path,
        filename=name,
        extension=".mp3",
        size_mb=1.0,
    )


def _make_censorship_result(details: Optional[str] = None) -> CensorshipResult:
    """Create a test CensorshipResult."""
    return CensorshipResult(
        is_censored=True,
        mismatch_score=0.4,
        profanity_count=3,
        confidence=0.85,
        details=details if details is not None else "Profanity detected: damn, hell, crap",
    )


def _make_instrumental_result() -> InstrumentalResult:
    """Create a test InstrumentalResult."""
    return InstrumentalResult(
        is_instrumental=True,
        word_count=2,
        speech_duration=1.5,
        confidence=0.92,
    )


def _make_mismatch_result() -> MismatchResult:
    """Create a test MismatchResult."""
    return MismatchResult(
        is_mismatch=True,
        title_score=0.45,
        artist_score=0.3,
        duration_difference=15.0,
        confidence=0.78,
        details="Title mismatch: 'Wrong Title' vs 'Right Title'",
    )


class TestWriteResults:
    """Tests for the write_results function."""

    def test_write_results_all_empty(self, tmp_path: Path) -> None:
        """Test write_results with all empty lists creates no files."""
        write_results(
            censored_tracks=[],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )
        assert list(tmp_path.glob("*.txt")) == []

    def test_write_results_censored_only(self, tmp_path: Path) -> None:
        """Test write_results with only censored tracks."""
        audio_file = _make_audio_file(tmp_path, "censored_song.mp3")
        result = _make_censorship_result()

        write_results(
            censored_tracks=[(audio_file, result)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        output_file = tmp_path / "censored.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Censored/Non-Explicit Tracks" in content
        assert "censored_song.mp3" in content
        assert "damn, hell, crap" in content

    def test_write_results_instrumental_only(self, tmp_path: Path) -> None:
        """Test write_results with only instrumental tracks."""
        audio_file = _make_audio_file(tmp_path, "instrumental_song.flac")
        result = _make_instrumental_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[(audio_file, result)],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        output_file = tmp_path / "instrumental.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Instrumental Tracks" in content
        assert "instrumental_song.flac" in content
        assert "Words detected: 2" in content

    def test_write_results_mismatch_only(self, tmp_path: Path) -> None:
        """Test write_results with only metadata mismatches."""
        audio_file = _make_audio_file(tmp_path, "mismatched.ogg")
        result = _make_mismatch_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[],
            metadata_mismatches=[(audio_file, result)],
            output_dir=tmp_path,
        )

        output_file = tmp_path / "metadata_mismatches.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Metadata Mismatches" in content
        assert "mismatched.ogg" in content
        assert "Title mismatch" in content

    def test_write_results_all_with_data(self, tmp_path: Path) -> None:
        """Test write_results with all three track types populated."""
        audio1 = _make_audio_file(tmp_path, "censored.mp3")
        audio2 = _make_audio_file(tmp_path, "instrumental.mp3")
        audio3 = _make_audio_file(tmp_path, "mismatch.mp3")

        write_results(
            censored_tracks=[(audio1, _make_censorship_result())],
            instrumental_tracks=[(audio2, _make_instrumental_result())],
            metadata_mismatches=[(audio3, _make_mismatch_result())],
            output_dir=tmp_path,
        )

        assert (tmp_path / "censored.txt").exists()
        assert (tmp_path / "instrumental.txt").exists()
        assert (tmp_path / "metadata_mismatches.txt").exists()


class TestFilenamePrefix:
    """Tests for filename_prefix parameter (covers filename_prefix=True/False branches)."""

    def test_filename_prefix_none_uses_default_names(self, tmp_path: Path) -> None:
        """Test that filename_prefix=None uses default filenames (covers False branch)."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_censorship_result()

        write_results(
            censored_tracks=[(audio_file, result)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
            filename_prefix=None,
        )

        assert (tmp_path / "censored.txt").exists()

    def test_filename_prefix_set_uses_prefixed_names(self, tmp_path: Path) -> None:
        """Test that filename_prefix uses prefixed filenames (covers True branch)."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_censorship_result()

        write_results(
            censored_tracks=[(audio_file, result)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
            filename_prefix="my_run",
        )

        assert (tmp_path / "my_run_censored.txt").exists()
        assert not (tmp_path / "censored.txt").exists()

    def test_prefix_instrumental_file(self, tmp_path: Path) -> None:
        """Test prefix is applied to instrumental file."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_instrumental_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[(audio_file, result)],
            metadata_mismatches=[],
            output_dir=tmp_path,
            filename_prefix="batch_01",
        )

        assert (tmp_path / "batch_01_instrumental.txt").exists()

    def test_prefix_mismatch_file(self, tmp_path: Path) -> None:
        """Test prefix is applied to mismatch file."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_mismatch_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[],
            metadata_mismatches=[(audio_file, result)],
            output_dir=tmp_path,
            filename_prefix="scan_v2",
        )

        assert (tmp_path / "scan_v2_metadata_mismatches.txt").exists()

    def test_prefix_all_files(self, tmp_path: Path) -> None:
        """Test prefix is applied to all files simultaneously."""
        audio1 = _make_audio_file(tmp_path, "a.mp3")
        audio2 = _make_audio_file(tmp_path, "b.mp3")
        audio3 = _make_audio_file(tmp_path, "c.mp3")

        write_results(
            censored_tracks=[(audio1, _make_censorship_result())],
            instrumental_tracks=[(audio2, _make_instrumental_result())],
            metadata_mismatches=[(audio3, _make_mismatch_result())],
            output_dir=tmp_path,
            filename_prefix="full_scan",
        )

        assert (tmp_path / "full_scan_censored.txt").exists()
        assert (tmp_path / "full_scan_instrumental.txt").exists()
        assert (tmp_path / "full_scan_metadata_mismatches.txt").exists()


class TestIncludeTimestamp:
    """Tests for INCLUDE_TIMESTAMP config (covers True/False branches)."""

    def test_timestamp_included_when_true(self, tmp_path: Path) -> None:
        """Test that timestamp is written when INCLUDE_TIMESTAMP=True."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_censorship_result()

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", True):
            write_results(
                censored_tracks=[(audio_file, result)],
                instrumental_tracks=[],
                metadata_mismatches=[],
                output_dir=tmp_path,
            )

        content = (tmp_path / "censored.txt").read_text(encoding="utf-8")
        assert "Generated:" in content

    def test_timestamp_excluded_when_false(self, tmp_path: Path) -> None:
        """Test that timestamp is NOT written when INCLUDE_TIMESTAMP=False."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_censorship_result()

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", False):
            write_results(
                censored_tracks=[(audio_file, result)],
                instrumental_tracks=[],
                metadata_mismatches=[],
                output_dir=tmp_path,
            )

        content = (tmp_path / "censored.txt").read_text(encoding="utf-8")
        assert "Generated:" not in content
        assert "Censored/Non-Explicit Tracks" in content

    def test_timestamp_excluded_instrumental(self, tmp_path: Path) -> None:
        """Test timestamp excluded in instrumental file (covers branch 77->79)."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_instrumental_result()

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", False):
            write_results(
                censored_tracks=[],
                instrumental_tracks=[(audio_file, result)],
                metadata_mismatches=[],
                output_dir=tmp_path,
            )

        content = (tmp_path / "instrumental.txt").read_text(encoding="utf-8")
        assert "Generated:" not in content
        assert "Instrumental Tracks" in content

    def test_timestamp_included_instrumental(self, tmp_path: Path) -> None:
        """Test timestamp included in instrumental file (covers branch 77->79)."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_instrumental_result()

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", True):
            write_results(
                censored_tracks=[],
                instrumental_tracks=[(audio_file, result)],
                metadata_mismatches=[],
                output_dir=tmp_path,
            )

        content = (tmp_path / "instrumental.txt").read_text(encoding="utf-8")
        assert "Generated:" in content

    def test_timestamp_excluded_mismatch(self, tmp_path: Path) -> None:
        """Test timestamp excluded in mismatch file."""
        audio_file = _make_audio_file(tmp_path)
        result = _make_mismatch_result()

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", False):
            write_results(
                censored_tracks=[],
                instrumental_tracks=[],
                metadata_mismatches=[(audio_file, result)],
                output_dir=tmp_path,
            )

        content = (tmp_path / "metadata_mismatches.txt").read_text(encoding="utf-8")
        assert "Generated:" not in content

    def test_timestamp_included_all_files(self, tmp_path: Path) -> None:
        """Test timestamp is included in all three files when True."""
        audio1 = _make_audio_file(tmp_path, "a.mp3")
        audio2 = _make_audio_file(tmp_path, "b.mp3")
        audio3 = _make_audio_file(tmp_path, "c.mp3")

        with patch("lrcfilter.output.INCLUDE_TIMESTAMP", True):
            write_results(
                censored_tracks=[(audio1, _make_censorship_result())],
                instrumental_tracks=[(audio2, _make_instrumental_result())],
                metadata_mismatches=[(audio3, _make_mismatch_result())],
                output_dir=tmp_path,
            )

        for filename in ["censored.txt", "instrumental.txt", "metadata_mismatches.txt"]:
            content = (tmp_path / filename).read_text(encoding="utf-8")
            assert "Generated:" in content


class TestOutputContent:
    """Tests for specific output content formatting."""

    def test_censored_output_format(self, tmp_path: Path) -> None:
        """Test the exact format of censored output file."""
        audio_file = _make_audio_file(tmp_path, "song.mp3")
        result = _make_censorship_result()

        write_results(
            censored_tracks=[(audio_file, result)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        content = (tmp_path / "censored.txt").read_text(encoding="utf-8")
        assert "# Censored/Non-Explicit Tracks" in content
        assert "# Total: 1 tracks" in content
        assert "song.mp3" in content
        assert "# Confidence: 85.0%" in content

    def test_instrumental_output_format(self, tmp_path: Path) -> None:
        """Test the exact format of instrumental output file."""
        audio_file = _make_audio_file(tmp_path, "beat.flac")
        result = _make_instrumental_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[(audio_file, result)],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        content = (tmp_path / "instrumental.txt").read_text(encoding="utf-8")
        assert "# Instrumental Tracks (No Vocals)" in content
        assert "# Total: 1 tracks" in content
        assert "beat.flac" in content
        assert "Words detected: 2" in content
        assert "Speech duration: 1.5s" in content
        assert "# Confidence: 92.0%" in content

    def test_mismatch_output_format(self, tmp_path: Path) -> None:
        """Test the exact format of mismatch output file."""
        audio_file = _make_audio_file(tmp_path, "wrong.flac")
        result = _make_mismatch_result()

        write_results(
            censored_tracks=[],
            instrumental_tracks=[],
            metadata_mismatches=[(audio_file, result)],
            output_dir=tmp_path,
        )

        content = (tmp_path / "metadata_mismatches.txt").read_text(encoding="utf-8")
        assert "# Metadata Mismatches" in content
        assert "# Total: 1 tracks with mismatched metadata" in content
        assert "# File:" in content
        assert "wrong.flac" in content
        assert "Title mismatch" in content
        assert "# Confidence: 78.0%" in content

    def test_multiple_tracks_format(self, tmp_path: Path) -> None:
        """Test formatting with multiple tracks."""
        audio1 = _make_audio_file(tmp_path, "song1.mp3")
        audio2 = _make_audio_file(tmp_path, "song2.mp3")

        result1 = _make_censorship_result()
        result2 = _make_censorship_result(details="Profanity detected: damn")

        write_results(
            censored_tracks=[(audio1, result1), (audio2, result2)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        content = (tmp_path / "censored.txt").read_text(encoding="utf-8")
        assert "# Total: 2 tracks" in content
        assert "song1.mp3" in content
        assert "song2.mp3" in content

    def test_censorship_without_details(self, tmp_path: Path) -> None:
        """Test censored track with no details (covers details branch)."""
        audio_file = _make_audio_file(tmp_path, "clean.mp3")
        result = _make_censorship_result(details="")

        write_results(
            censored_tracks=[(audio_file, result)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=tmp_path,
        )

        content = (tmp_path / "censored.txt").read_text(encoding="utf-8")
        assert "clean.mp3" in content
        assert "# Confidence: 85.0%" in content
