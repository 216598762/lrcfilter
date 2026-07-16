"""Main pipeline orchestration module for LRCFilter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from lrcfilter.analyzer import analyze_audio
from lrcfilter.censorship import detect_censorship
from lrcfilter.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL
from lrcfilter.instrumental import detect_instrumental
from lrcfilter.logging_config import get_logger
from lrcfilter.lyrics import fetch_lyrics
from lrcfilter.metadata import extract_metadata
from lrcfilter.mismatch import detect_metadata_mismatch
from lrcfilter.models import (
    AudioFile,
    CensorshipResult,
    InstrumentalResult,
    LyricsResult,
    MismatchResult,
    TrackMetadata,
    TranscriptionResult,
)
from lrcfilter.output import write_results
from lrcfilter.scanner import scan_audio_files

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    # Whisper model settings
    model_name: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    beam_size: int = 5
    vad_filter: bool = True

    # API settings
    genius_token: Optional[str] = None
    lrclib_only: bool = False
    api_delay: float = 1.0

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("."))
    verbose: bool = False
    formats: Optional[set] = None
    no_censored: bool = False
    no_instrumental: bool = False
    no_mismatches: bool = False

    # Detection thresholds
    censorship_threshold: float = 0.3
    min_words_vocals: int = 10
    min_speech_duration: float = 5.0
    title_threshold: float = 0.6
    artist_threshold: float = 0.7
    duration_tolerance: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate beam_size
        if self.beam_size <= 0:
            raise ValueError(f"beam_size must be positive, got {self.beam_size}")

        # Validate api_delay
        if self.api_delay < 0:
            raise ValueError(f"api_delay must be non-negative, got {self.api_delay}")

        # Validate censorship_threshold
        if not 0.0 <= self.censorship_threshold <= 1.0:
            raise ValueError(
                f"censorship_threshold must be between 0.0 and 1.0, got {self.censorship_threshold}"
            )

        # Validate min_words_vocals
        if self.min_words_vocals < 0:
            raise ValueError(f"min_words_vocals must be non-negative, got {self.min_words_vocals}")

        # Validate min_speech_duration
        if self.min_speech_duration < 0:
            raise ValueError(
                f"min_speech_duration must be non-negative, got {self.min_speech_duration}"
            )

        # Validate title_threshold
        if not 0.0 <= self.title_threshold <= 1.0:
            raise ValueError(
                f"title_threshold must be between 0.0 and 1.0, got {self.title_threshold}"
            )

        # Validate artist_threshold
        if not 0.0 <= self.artist_threshold <= 1.0:
            raise ValueError(
                f"artist_threshold must be between 0.0 and 1.0, got {self.artist_threshold}"
            )

        # Validate duration_tolerance
        if self.duration_tolerance < 0:
            raise ValueError(
                f"duration_tolerance must be non-negative, got {self.duration_tolerance}"
            )


@dataclass
class TrackResult:
    """Complete analysis result for a single track."""

    audio_file: AudioFile
    metadata: TrackMetadata
    lyrics: Optional[LyricsResult]
    transcription: Optional[TranscriptionResult]
    censorship: Optional[CensorshipResult]
    instrumental: Optional[InstrumentalResult]
    mismatch: Optional[MismatchResult]


@dataclass
class PipelineResult:
    """Complete results from running the analysis pipeline."""

    total_files: int
    processed_files: int
    censored_tracks: list[tuple[AudioFile, CensorshipResult]]
    instrumental_tracks: list[tuple[AudioFile, InstrumentalResult]]
    metadata_mismatches: list[tuple[AudioFile, MismatchResult]]
    track_results: list[TrackResult]


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]


def process_single_track(
    audio_file: AudioFile,
    config: PipelineConfig,
) -> TrackResult:
    """
    Process a single audio file through the complete analysis pipeline.

    Args:
        audio_file: The audio file to process
        config: Pipeline configuration

    Returns:
        TrackResult with all analysis results
    """
    # Step 1: Extract metadata
    metadata = extract_metadata(audio_file)

    # Step 2: Fetch lyrics
    lyrics = fetch_lyrics(
        metadata,
        genius_token=config.genius_token,
        lrclib_only=config.lrclib_only,
        api_delay=config.api_delay,
    )

    # Step 3: Check for metadata mismatch
    mismatch = None
    if lyrics:
        mismatch = detect_metadata_mismatch(
            metadata,
            lyrics,
            title_threshold=config.title_threshold,
            artist_threshold=config.artist_threshold,
            duration_tolerance=config.duration_tolerance,
        )

    # Step 4: Transcribe audio with Whisper
    transcription = analyze_audio(
        audio_file,
        model_name=config.model_name,
        device=config.device,
        compute_type=config.compute_type,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
    )

    # Step 5: Detect censorship
    censorship = None
    if lyrics and lyrics.plain_lyrics:
        censorship = detect_censorship(
            lyrics.plain_lyrics,
            transcription.text,
            threshold=config.censorship_threshold,
        )

    # Step 6: Detect instrumental
    instrumental = detect_instrumental(
        transcription,
        min_words_vocals=config.min_words_vocals,
        min_speech_duration=config.min_speech_duration,
    )

    return TrackResult(
        audio_file=audio_file,
        metadata=metadata,
        lyrics=lyrics,
        transcription=transcription,
        censorship=censorship,
        instrumental=instrumental,
        mismatch=mismatch,
    )


def run_pipeline(
    directory: Path,
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """
    Run the complete analysis pipeline on a directory.

    Args:
        directory: Directory to scan for audio files
        config: Pipeline configuration (uses defaults if None)
        progress_callback: Optional callback for progress updates

    Returns:
        PipelineResult with all analysis results
    """
    if config is None:
        config = PipelineConfig()

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for audio files
    audio_files = scan_audio_files(directory, formats=config.formats)
    total_files = len(audio_files)

    if total_files == 0:
        return PipelineResult(
            total_files=0,
            processed_files=0,
            censored_tracks=[],
            instrumental_tracks=[],
            metadata_mismatches=[],
            track_results=[],
        )

    # Process each file
    censored_tracks = []
    instrumental_tracks = []
    metadata_mismatches = []
    track_results = []

    for i, audio_file in enumerate(audio_files, 1):
        # Report progress
        if progress_callback:
            progress_callback(i, total_files, audio_file.filename)

        try:
            # Process the track
            result = process_single_track(audio_file, config)
            track_results.append(result)

            # Collect results
            if result.censorship and result.censorship.is_censored:
                censored_tracks.append((audio_file, result.censorship))

            if result.instrumental and result.instrumental.is_instrumental:
                instrumental_tracks.append((audio_file, result.instrumental))

            if result.mismatch and result.mismatch.is_mismatch:
                metadata_mismatches.append((audio_file, result.mismatch))

        except Exception as e:
            logger.error(f"Error processing {audio_file.filename}: {e}")
            continue

    # Create pipeline result
    pipeline_result = PipelineResult(
        total_files=total_files,
        processed_files=len(track_results),
        censored_tracks=censored_tracks,
        instrumental_tracks=instrumental_tracks,
        metadata_mismatches=metadata_mismatches,
        track_results=track_results,
    )

    # Write results to files
    write_results(
        censored_tracks=censored_tracks if not config.no_censored else [],
        instrumental_tracks=instrumental_tracks if not config.no_instrumental else [],
        metadata_mismatches=metadata_mismatches if not config.no_mismatches else [],
        output_dir=config.output_dir,
    )

    return pipeline_result


def print_summary(result: PipelineResult) -> None:
    """
    Print a summary of the pipeline results.

    Args:
        result: PipelineResult to summarize
    """
    logger.info("=" * 50)
    logger.info("Analysis Summary")
    logger.info("=" * 50)
    logger.info(f"  Total files scanned: {result.total_files}")
    logger.info(f"  Files processed: {result.processed_files}")
    logger.info(f"  Censored/non-explicit: {len(result.censored_tracks)}")
    logger.info(f"  Instrumental: {len(result.instrumental_tracks)}")
    logger.info(f"  Metadata mismatches: {len(result.metadata_mismatches)}")
    logger.info("=" * 50)
