"""Main pipeline orchestration module for LRCFilter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from lrcfilter.scanner import scan_audio_files
from lrcfilter.metadata import extract_metadata
from lrcfilter.lyrics import fetch_lyrics
from lrcfilter.analyzer import analyze_audio
from lrcfilter.censorship import detect_censorship
from lrcfilter.instrumental import detect_instrumental
from lrcfilter.mismatch import detect_metadata_mismatch
from lrcfilter.output import write_results
from lrcfilter.config import DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE
from lrcfilter.models import (
    AudioFile,
    TrackMetadata,
    LyricsResult,
    TranscriptionResult,
    CensorshipResult,
    InstrumentalResult,
    MismatchResult,
)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    model_name: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    genius_token: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("."))
    verbose: bool = False


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
    censored_tracks: List[Tuple[AudioFile, CensorshipResult]]
    instrumental_tracks: List[Tuple[AudioFile, InstrumentalResult]]
    metadata_mismatches: List[Tuple[AudioFile, MismatchResult]]
    track_results: List[TrackResult]


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
    )
    
    # Step 3: Check for metadata mismatch
    mismatch = None
    if lyrics:
        mismatch = detect_metadata_mismatch(metadata, lyrics)
    
    # Step 4: Transcribe audio with Whisper
    transcription = analyze_audio(
        audio_file,
        model_name=config.model_name,
        device=config.device,
        compute_type=config.compute_type,
    )
    
    # Step 5: Detect censorship
    censorship = None
    if lyrics and lyrics.plain_lyrics:
        censorship = detect_censorship(
            lyrics.plain_lyrics,
            transcription.text,
        )
    
    # Step 6: Detect instrumental
    instrumental = detect_instrumental(transcription)
    
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
    audio_files = scan_audio_files(directory)
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
            if config.verbose:
                print(f"Error processing {audio_file.filename}: {e}")
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
        censored_tracks=censored_tracks,
        instrumental_tracks=instrumental_tracks,
        metadata_mismatches=metadata_mismatches,
        output_dir=config.output_dir,
    )
    
    return pipeline_result


def print_summary(result: PipelineResult) -> None:
    """
    Print a summary of the pipeline results.
    
    Args:
        result: PipelineResult to summarize
    """
    print("\n" + "=" * 50)
    print("Analysis Summary")
    print("=" * 50)
    print(f"  Total files scanned: {result.total_files}")
    print(f"  Files processed: {result.processed_files}")
    print(f"  Censored/non-explicit: {len(result.censored_tracks)}")
    print(f"  Instrumental: {len(result.instrumental_tracks)}")
    print(f"  Metadata mismatches: {len(result.metadata_mismatches)}")
    print("=" * 50)
