"""Integration tests for the full pipeline with mocked dependencies."""

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from lrcfilter.models import (
    AudioFile,
    LyricsResult,
    Segment,
    TrackMetadata,
    TranscriptionResult,
    Word,
)
from lrcfilter.pipeline import (
    PipelineConfig,
    PipelineResult,
    TrackResult,
    print_summary,
    process_single_track,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock audio files."""
    (tmp_path / "clean_song.mp3").touch()
    (tmp_path / "censored_song.mp3").touch()
    (tmp_path / "instrumental_track.flac").touch()
    (tmp_path / "mismatched_track.m4a").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested_track.ogg").touch()
    return tmp_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


def _make_audio_file(path: Path) -> AudioFile:
    """Helper to build an AudioFile from a path."""
    return AudioFile(
        path=path,
        filename=path.name,
        extension=path.suffix.lower(),
        size_mb=path.stat().st_size / (1024 * 1024) if path.exists() else 0.0,
    )


def _make_transcription(
    text: str = "",
    segments: Optional[list[Segment]] = None,
    has_speech: bool = True,
    duration: float = 180.0,
) -> TranscriptionResult:
    """Helper to build a TranscriptionResult."""
    if segments is None:
        words = text.split() if text else []
        segments = [
            Segment(
                start=0.0,
                end=float(len(words)) * 0.5,
                text=text,
                words=[
                    Word(start=i * 0.5, end=(i + 1) * 0.5, word=w, probability=0.99)
                    for i, w in enumerate(words)
                ],
            )
        ]
    return TranscriptionResult(
        text=text,
        segments=segments,
        language="en",
        duration=duration,
        has_speech=has_speech,
    )


def _make_lyrics(
    title: str = "Test Song",
    artist: str = "Test Artist",
    plain_lyrics: str = "Test lyrics here",
    synced_lyrics: Optional[str] = None,
    source: str = "lrclib",
) -> LyricsResult:
    """Helper to build a LyricsResult."""
    return LyricsResult(
        source=source,
        synced_lyrics=synced_lyrics,
        plain_lyrics=plain_lyrics,
        matched_track_name=title,
        matched_artist_name=artist,
        matched_album_name="Test Album",
        match_score=0.95,
    )


def _make_metadata(
    title: str = "Test Song",
    artist: str = "Test Artist",
    album: str = "Test Album",
    duration: float = 180.0,
) -> TrackMetadata:
    """Helper to build a TrackMetadata."""
    return TrackMetadata(
        title=title,
        artist=artist,
        album=album,
        duration_seconds=duration,
        raw_tags={},
    )


# ---------------------------------------------------------------------------
# Mocked pipeline helpers
# ---------------------------------------------------------------------------

# Default transcript long enough to exceed min_words_vocals (10) threshold
_LONG_TRANSCRIPTION_TEXT = (
    "This is a long transcription with many words that exceeds "
    "the minimum word count threshold for instrumental detection"
)
_LONG_TRANSCRIPTION = _make_transcription(_LONG_TRANSCRIPTION_TEXT)


def _run_mocked_pipeline(
    audio_files: list[AudioFile],
    config: PipelineConfig,
    metadata: Optional[TrackMetadata] = None,
    lyrics: Optional[LyricsResult] = "__SENTINEL__",
    transcription: Optional[TranscriptionResult] = None,
    progress_callback=None,
) -> PipelineResult:
    """
    Run the pipeline with all external calls mocked.

    Uses ``"__SENTINEL__"`` as the default for *lyrics* so that a caller
    can explicitly pass ``None`` to simulate "no lyrics found".  When the
    caller does not supply *lyrics* at all the helper defaults to a
    normal LyricsResult.
    """
    if metadata is None:
        metadata = _make_metadata()
    if lyrics == "__SENTINEL__":
        lyrics = _make_lyrics()
    if transcription is None:
        transcription = _LONG_TRANSCRIPTION

    with (
        patch("lrcfilter.pipeline.scan_audio_files", return_value=audio_files),
        patch("lrcfilter.pipeline.extract_metadata", return_value=metadata),
        patch("lrcfilter.pipeline.fetch_lyrics", return_value=lyrics),
        patch("lrcfilter.pipeline.analyze_audio", return_value=transcription),
    ):
        return run_pipeline(
            directory=Path("."),
            config=config,
            progress_callback=progress_callback,
        )


# ===========================================================================
# Integration Tests – Clean / Normal Content
# ===========================================================================


class TestPipelineCleanContent:
    """Pipeline should pass clean tracks through without flagging them."""

    def test_clean_song_not_flagged(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        lyrics = _make_lyrics(
            plain_lyrics="This is a clean test song about happiness",
        )
        transcription = _make_transcription(
            "This is a clean test song about happiness that has many words here"
        )

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        assert result.total_files == 1
        assert result.processed_files == 1
        assert len(result.censored_tracks) == 0
        assert len(result.instrumental_tracks) == 0
        assert len(result.metadata_mismatches) == 0

    def test_clean_song_writes_no_files(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        # Provide matching lyrics and transcription so no censorship is triggered
        matching_text = "This is a beautiful song with many words that are happy"
        lyrics = _make_lyrics(plain_lyrics=matching_text)
        transcription = _make_transcription(matching_text)

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        # No result files should be written for clean tracks
        assert not (output_dir / "censored.txt").exists()
        assert not (output_dir / "instrumental.txt").exists()
        assert not (output_dir / "metadata_mismatches.txt").exists()

    def test_track_result_fields_populated(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        metadata = _make_metadata(title="Clean Song", artist="Good Artist")
        lyrics = _make_lyrics(plain_lyrics="Beautiful lyrics")
        transcription = _make_transcription(
            "Beautiful lyrics that are long enough to pass the word count check"
        )

        result = _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        track = result.track_results[0]
        assert track.metadata.title == "Clean Song"
        assert track.lyrics is not None
        assert track.transcription is not None
        assert track.censorship is not None
        assert track.instrumental is not None


# ===========================================================================
# Integration Tests – Censored / Explicit Content
# ===========================================================================


class TestPipelineCensoredContent:
    """Pipeline should detect tracks with profanity or lyrics mismatch."""

    def test_profanity_detected(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        lyrics = _make_lyrics(plain_lyrics="Normal lyrics about love")
        transcription = _make_transcription(
            "Normal lyrics with damn and shit words here that are many"
        )

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        assert result.processed_files == 1
        assert len(result.censored_tracks) == 1
        audio_file, censorship = result.censored_tracks[0]
        assert audio_file.filename == "censored_song.mp3"
        assert censorship.is_censored is True
        assert censorship.profanity_count >= 2

    def test_profanity_writes_output_file(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        lyrics = _make_lyrics(plain_lyrics="Clean lyrics")
        transcription = _make_transcription(
            "This song has damn and shit in it with plenty of words here"
        )

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        output_file = output_dir / "censored.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "censored_song.mp3" in content

    def test_lyrics_mismatch_triggers_censorship(
        self, tmp_audio_dir: Path, output_dir: Path
    ) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        lyrics = _make_lyrics(
            title="Censored Song",
            artist="Censored Artist",
            plain_lyrics="This is the original unedited version of the lyrics that are here",
        )
        transcription = _make_transcription(
            "Completely different spoken words that do not match the lyrics at all"
        )

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        assert len(result.censored_tracks) == 1
        _, censorship = result.censored_tracks[0]
        assert censorship.mismatch_score > 0.3

    def test_censored_with_prefix(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]

        from lrcfilter.censorship import detect_censorship
        from lrcfilter.output import write_results

        lyrics = _make_lyrics(plain_lyrics="Clean")
        transcription = _make_transcription("This has damn and shit words here and many more")

        censorship = detect_censorship(lyrics.plain_lyrics, transcription.text)
        write_results(
            censored_tracks=[(audio_files[0], censorship)],
            instrumental_tracks=[],
            metadata_mismatches=[],
            output_dir=output_dir,
            filename_prefix="test_run",
        )

        output_file = output_dir / "test_run_censored.txt"
        assert output_file.exists()


# ===========================================================================
# Integration Tests – Instrumental Tracks
# ===========================================================================


class TestPipelineInstrumentalContent:
    """Pipeline should detect instrumental tracks with no vocals."""

    def test_instrumental_detected(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "instrumental_track.flac")]
        config = PipelineConfig(output_dir=output_dir)

        # Transcription with no words → instrumental
        transcription = _make_transcription("", segments=[], has_speech=False, duration=200.0)
        lyrics = _make_lyrics(plain_lyrics=None)

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        assert result.processed_files == 1
        assert len(result.instrumental_tracks) == 1
        audio_file, instrumental = result.instrumental_tracks[0]
        assert audio_file.filename == "instrumental_track.flac"
        assert instrumental.is_instrumental is True

    def test_instrumental_writes_output_file(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "instrumental_track.flac")]
        config = PipelineConfig(output_dir=output_dir)

        transcription = _make_transcription("", segments=[], has_speech=False)
        lyrics = _make_lyrics(plain_lyrics=None)

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        output_file = output_dir / "instrumental.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "instrumental_track.flac" in content

    def test_low_word_count_instrumental(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        """Fewer words than threshold should be flagged as instrumental."""
        audio_files = [_make_audio_file(tmp_audio_dir / "instrumental_track.flac")]
        config = PipelineConfig(output_dir=output_dir, min_words_vocals=10)

        # Only 3 words → below threshold
        transcription = _make_transcription("yeah uh huh", duration=200.0)
        lyrics = _make_lyrics(plain_lyrics=None)

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        assert len(result.instrumental_tracks) == 1
        _audio_file, instrumental = result.instrumental_tracks[0]
        assert instrumental.is_instrumental is True
        assert instrumental.word_count == 3


# ===========================================================================
# Integration Tests – Metadata Mismatch
# ===========================================================================


class TestPipelineMetadataMismatch:
    """Pipeline should detect metadata mismatches between file and lyrics API."""

    def test_title_mismatch_detected(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "mismatched_track.m4a")]
        config = PipelineConfig(output_dir=output_dir)

        metadata = _make_metadata(title="Wrong Title Here", artist="Test Artist")
        lyrics = _make_lyrics(title="Completely Different Song Name", artist="Test Artist")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass word count"
        )

        result = _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        assert len(result.metadata_mismatches) == 1
        audio_file, mismatch = result.metadata_mismatches[0]
        assert audio_file.filename == "mismatched_track.m4a"
        assert mismatch.is_mismatch is True
        assert mismatch.title_score < 0.6

    def test_artist_mismatch_detected(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "mismatched_track.m4a")]
        config = PipelineConfig(output_dir=output_dir)

        metadata = _make_metadata(title="Test Song", artist="The Beatles")
        lyrics = _make_lyrics(title="Test Song", artist="Black Sabbath")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass word count"
        )

        result = _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        assert len(result.metadata_mismatches) == 1
        _, mismatch = result.metadata_mismatches[0]
        assert mismatch.is_mismatch is True
        assert mismatch.artist_score < 0.7

    def test_mismatch_writes_output_file(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "mismatched_track.m4a")]
        config = PipelineConfig(output_dir=output_dir)

        metadata = _make_metadata(title="File Title", artist="File Artist")
        lyrics = _make_lyrics(title="API Title", artist="API Artist")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass word count"
        )

        _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        output_file = output_dir / "metadata_mismatches.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "mismatched_track.m4a" in content


# ===========================================================================
# Integration Tests – No Lyrics Found
# ===========================================================================


class TestPipelineNoLyrics:
    """Pipeline should gracefully handle missing lyrics."""

    def test_no_lyrics_no_censorship_check(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        transcription = _make_transcription(
            "Some spoken words here that are long enough to pass the threshold"
        )

        result = _run_mocked_pipeline(audio_files, config, lyrics=None, transcription=transcription)

        assert result.processed_files == 1
        track = result.track_results[0]
        assert track.lyrics is None
        # Censorship check requires lyrics, so it should be None
        assert track.censorship is None
        # Instrumental check still runs
        assert track.instrumental is not None

    def test_no_lyrics_skips_mismatch_check(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        transcription = _make_transcription(
            "Some words here that are long enough to pass the threshold check"
        )

        result = _run_mocked_pipeline(audio_files, config, lyrics=None, transcription=transcription)

        assert len(result.metadata_mismatches) == 0


# ===========================================================================
# Integration Tests – Multiple Tracks
# ===========================================================================


class TestPipelineMultipleTracks:
    """Pipeline should handle multiple tracks in a single run."""

    def test_mixed_results(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        """Run pipeline with a mix of clean, censored, and instrumental tracks."""
        audio_files = [
            _make_audio_file(tmp_audio_dir / "clean_song.mp3"),
            _make_audio_file(tmp_audio_dir / "censored_song.mp3"),
            _make_audio_file(tmp_audio_dir / "instrumental_track.flac"),
        ]
        config = PipelineConfig(output_dir=output_dir)

        # Track 1: clean lyrics that match transcription (needs 10+ words)
        clean_lyrics = _make_lyrics(
            plain_lyrics="Happy clean lyrics here that are many words and more"
        )
        clean_transcription = _make_transcription(
            "Happy clean lyrics here that are many words and more"
        )

        # Track 2: censored (profanity in transcription)
        censored_lyrics = _make_lyrics(plain_lyrics="Normal lyrics here that are many words")
        censored_transcription = _make_transcription(
            "Lyrics with damn and shit words here that are many"
        )

        # Track 3: instrumental (no speech)
        instrumental_lyrics = _make_lyrics(plain_lyrics=None)
        instrumental_transcription = _make_transcription("", segments=[], has_speech=False)

        metadata = _make_metadata()
        track_index = 0

        def mock_fetch_lyrics(m: TrackMetadata, **kwargs) -> Optional[LyricsResult]:
            nonlocal track_index
            lyrics_list = [clean_lyrics, censored_lyrics, instrumental_lyrics]
            return lyrics_list[track_index % len(lyrics_list)]

        def mock_analyze_audio(audio_file: AudioFile, **kwargs) -> TranscriptionResult:
            nonlocal track_index
            transcriptions = [
                clean_transcription,
                censored_transcription,
                instrumental_transcription,
            ]
            result = transcriptions[track_index % len(transcriptions)]
            track_index += 1
            return result

        with (
            patch("lrcfilter.pipeline.scan_audio_files", return_value=audio_files),
            patch("lrcfilter.pipeline.extract_metadata", return_value=metadata),
            patch("lrcfilter.pipeline.fetch_lyrics", side_effect=mock_fetch_lyrics),
            patch("lrcfilter.pipeline.analyze_audio", side_effect=mock_analyze_audio),
        ):
            result = run_pipeline(directory=Path("."), config=config)

        assert result.total_files == 3
        assert result.processed_files == 3
        assert len(result.censored_tracks) == 1
        assert len(result.instrumental_tracks) == 1

    def test_progress_callback_invoked(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        progress_calls = []

        def track_progress(current: int, total: int, filename: str) -> None:
            progress_calls.append((current, total, filename))

        _run_mocked_pipeline(audio_files, config, progress_callback=track_progress)

        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, "clean_song.mp3")


# ===========================================================================
# Integration Tests – Output Writing
# ===========================================================================


class TestPipelineOutput:
    """Verify that output files are written correctly."""

    def test_censored_output_format(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        lyrics = _make_lyrics(plain_lyrics="Clean lyrics")
        transcription = _make_transcription(
            "This has damn and shit in it with plenty of words here"
        )

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        output_file = output_dir / "censored.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Censored/Non-Explicit Tracks" in content
        assert "censored_song.mp3" in content
        assert "# Confidence:" in content

    def test_instrumental_output_format(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "instrumental_track.flac")]
        config = PipelineConfig(output_dir=output_dir)

        transcription = _make_transcription("", segments=[], has_speech=False)
        lyrics = _make_lyrics(plain_lyrics=None)

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        output_file = output_dir / "instrumental.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Instrumental Tracks (No Vocals)" in content
        assert "instrumental_track.flac" in content

    def test_mismatch_output_format(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "mismatched_track.m4a")]
        config = PipelineConfig(output_dir=output_dir)

        metadata = _make_metadata(title="Wrong Title", artist="Wrong Artist")
        lyrics = _make_lyrics(title="Correct Title", artist="Correct Artist")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass the threshold"
        )

        _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        output_file = output_dir / "metadata_mismatches.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Metadata Mismatches" in content
        assert "mismatched_track.m4a" in content


# ===========================================================================
# Integration Tests – Config Options
# ===========================================================================


class TestPipelineConfigOptions:
    """Verify that pipeline configuration options are respected."""

    def test_no_censored_flag(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "censored_song.mp3")]
        config = PipelineConfig(output_dir=output_dir, no_censored=True)

        lyrics = _make_lyrics(plain_lyrics="Clean")
        transcription = _make_transcription("Words with damn in them that are many and long")

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        # Output file should not be written when flag is set
        assert not (output_dir / "censored.txt").exists()

    def test_no_instrumental_flag(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "instrumental_track.flac")]
        config = PipelineConfig(output_dir=output_dir, no_instrumental=True)

        transcription = _make_transcription("", segments=[], has_speech=False)
        lyrics = _make_lyrics(plain_lyrics=None)

        _run_mocked_pipeline(audio_files, config, lyrics=lyrics, transcription=transcription)

        assert not (output_dir / "instrumental.txt").exists()

    def test_no_mismatches_flag(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "mismatched_track.m4a")]
        config = PipelineConfig(output_dir=output_dir, no_mismatches=True)

        metadata = _make_metadata(title="Wrong", artist="Wrong")
        lyrics = _make_lyrics(title="Different", artist="Different")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass the threshold"
        )

        _run_mocked_pipeline(
            audio_files, config, metadata=metadata, lyrics=lyrics, transcription=transcription
        )

        assert not (output_dir / "metadata_mismatches.txt").exists()

    def test_custom_thresholds(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        """Custom thresholds should affect detection sensitivity."""
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(
            output_dir=output_dir,
            censorship_threshold=0.9,  # Very high threshold → fewer flags
        )

        lyrics = _make_lyrics(plain_lyrics="Some lyrics here that are many words")
        transcription = _make_transcription("Slightly different lyrics here that are many words")

        result = _run_mocked_pipeline(
            audio_files, config, lyrics=lyrics, transcription=transcription
        )

        # With high threshold, slight mismatch should NOT trigger censorship
        assert len(result.censored_tracks) == 0


# ===========================================================================
# Integration Tests – Empty / Edge Cases
# ===========================================================================


class TestPipelineEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_directory(self, output_dir: Path) -> None:
        """Pipeline with no audio files should return empty result."""
        config = PipelineConfig(output_dir=output_dir)

        with patch("lrcfilter.pipeline.scan_audio_files", return_value=[]):
            result = run_pipeline(directory=Path("."), config=config)

        assert result.total_files == 0
        assert result.processed_files == 0
        assert len(result.track_results) == 0

    def test_default_config_when_none(self, output_dir: Path) -> None:
        """Pipeline should create default PipelineConfig when config is None (line 213)."""
        with patch("lrcfilter.pipeline.scan_audio_files", return_value=[]):
            result = run_pipeline(directory=Path("."), config=None)

        assert result.total_files == 0
        assert result.processed_files == 0

    def test_exception_in_track_continues_pipeline(
        self, tmp_audio_dir: Path, output_dir: Path
    ) -> None:
        """If one track raises an exception, the pipeline should continue."""
        audio_files = [
            _make_audio_file(tmp_audio_dir / "clean_song.mp3"),
            _make_audio_file(tmp_audio_dir / "censored_song.mp3"),
        ]
        config = PipelineConfig(output_dir=output_dir)

        call_count = 0

        def mock_analyze(audio_file: AudioFile, **kwargs) -> TranscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated transcription failure")
            return _make_transcription("Normal lyrics here that are long enough to pass the check")

        with (
            patch("lrcfilter.pipeline.scan_audio_files", return_value=audio_files),
            patch("lrcfilter.pipeline.extract_metadata", return_value=_make_metadata()),
            patch("lrcfilter.pipeline.fetch_lyrics", return_value=_make_lyrics()),
            patch("lrcfilter.pipeline.analyze_audio", side_effect=mock_analyze),
        ):
            result = run_pipeline(directory=Path("."), config=config)

        # First track failed, second succeeded
        assert result.total_files == 2
        assert result.processed_files == 1

    def test_print_summary_does_not_raise(self, tmp_audio_dir: Path, output_dir: Path) -> None:
        audio_files = [_make_audio_file(tmp_audio_dir / "clean_song.mp3")]
        config = PipelineConfig(output_dir=output_dir)

        result = _run_mocked_pipeline(audio_files, config)

        # Should not raise
        print_summary(result)


# ===========================================================================
# Integration Tests – Process Single Track
# ===========================================================================


class TestProcessSingleTrack:
    """Test the process_single_track function in isolation."""

    def test_single_track_returns_track_result(self, tmp_audio_dir: Path) -> None:
        audio_file = _make_audio_file(tmp_audio_dir / "clean_song.mp3")
        config = PipelineConfig()

        metadata = _make_metadata(title="Test Song", artist="Test Artist")
        lyrics = _make_lyrics(plain_lyrics="Beautiful lyrics")
        transcription = _make_transcription(
            "Beautiful lyrics here that are long enough to pass the word count"
        )

        with (
            patch("lrcfilter.pipeline.extract_metadata", return_value=metadata),
            patch("lrcfilter.pipeline.fetch_lyrics", return_value=lyrics),
            patch("lrcfilter.pipeline.analyze_audio", return_value=transcription),
        ):
            result = process_single_track(audio_file, config)

        assert isinstance(result, TrackResult)
        assert result.audio_file.filename == "clean_song.mp3"
        assert result.metadata.title == "Test Song"
        assert result.lyrics is not None
        assert result.transcription is not None
        assert result.censorship is not None
        assert result.instrumental is not None

    def test_single_track_mismatch_detected(self, tmp_audio_dir: Path) -> None:
        audio_file = _make_audio_file(tmp_audio_dir / "mismatched_track.m4a")
        config = PipelineConfig()

        metadata = _make_metadata(title="File Title", artist="File Artist")
        lyrics = _make_lyrics(title="API Title", artist="API Artist")
        transcription = _make_transcription(
            "Some lyrics here that are long enough to pass the threshold"
        )

        with (
            patch("lrcfilter.pipeline.extract_metadata", return_value=metadata),
            patch("lrcfilter.pipeline.fetch_lyrics", return_value=lyrics),
            patch("lrcfilter.pipeline.analyze_audio", return_value=transcription),
        ):
            result = process_single_track(audio_file, config)

        assert result.mismatch is not None
        assert result.mismatch.is_mismatch is True
