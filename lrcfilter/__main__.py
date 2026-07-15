"""CLI entry point for LRCFilter."""

import argparse
import sys
from pathlib import Path
from typing import List

from lrcfilter.pipeline import run_pipeline, print_summary, PipelineConfig
from lrcfilter.config import (
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
    SUPPORTED_FORMATS,
    CENSORSHIP_MISMATCH_THRESHOLD,
    MIN_WORDS_FOR_VOCALS,
    MIN_SPEECH_DURATION,
    TITLE_MATCH_THRESHOLD,
    ARTIST_MATCH_THRESHOLD,
    DURATION_TOLERANCE,
)
from lrcfilter.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="lrcfilter",
        description="Audio analysis tool for detecting censored/explicit content and instrumental tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lrcfilter /path/to/music
  lrcfilter /path/to/music -o ./results -m turbo --device cpu
  lrcfilter /path/to/music --log-file analysis.log --formats .mp3 .flac
        """,
    )

    # Required arguments
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for audio files",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory for result files (default: current directory)",
    )
    output_group.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=None,
        help=f"Audio formats to process (default: {', '.join(SUPPORTED_FORMATS)})",
    )
    output_group.add_argument(
        "--no-censored",
        action="store_true",
        help="Skip generating censored tracks list",
    )
    output_group.add_argument(
        "--no-instrumental",
        action="store_true",
        help="Skip generating instrumental tracks list",
    )
    output_group.add_argument(
        "--no-mismatches",
        action="store_true",
        help="Skip generating metadata mismatches list",
    )

    # Whisper model options
    whisper_group = parser.add_argument_group("whisper model options")
    whisper_group.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"],
        help=f"Whisper model to use (default: {DEFAULT_MODEL})",
    )
    whisper_group.add_argument(
        "--device",
        "-d",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to run Whisper on (default: {DEFAULT_DEVICE})",
    )
    whisper_group.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        choices=["float16", "int8", "int8_float16", "float32"],
        help=f"Compute type for Whisper (default: {DEFAULT_COMPUTE_TYPE})",
    )
    whisper_group.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for Whisper transcription (default: 5)",
    )
    whisper_group.add_argument(
        "--vad-filter",
        action="store_true",
        default=True,
        help="Enable Voice Activity Detection filter (default: enabled)",
    )
    whisper_group.add_argument(
        "--no-vad-filter",
        action="store_false",
        dest="vad_filter",
        help="Disable Voice Activity Detection filter",
    )

    # API options
    api_group = parser.add_argument_group("API options")
    api_group.add_argument(
        "--genius-token",
        type=str,
        default=None,
        help="Genius API access token (or set GENIUS_ACCESS_TOKEN env var)",
    )
    api_group.add_argument(
        "--lrclib-only",
        action="store_true",
        help="Only use LRCLib API, skip Genius fallback",
    )
    api_group.add_argument(
        "--api-delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )

    # Detection threshold options
    threshold_group = parser.add_argument_group("detection thresholds")
    threshold_group.add_argument(
        "--censorship-threshold",
        type=float,
        default=CENSORSHIP_MISMATCH_THRESHOLD,
        help=f"Censorship mismatch threshold (default: {CENSORSHIP_MISMATCH_THRESHOLD})",
    )
    threshold_group.add_argument(
        "--min-words-vocals",
        type=int,
        default=MIN_WORDS_FOR_VOCALS,
        help=f"Minimum words to consider vocal track (default: {MIN_WORDS_FOR_VOCALS})",
    )
    threshold_group.add_argument(
        "--min-speech-duration",
        type=float,
        default=MIN_SPEECH_DURATION,
        help=f"Minimum speech duration in seconds (default: {MIN_SPEECH_DURATION})",
    )
    threshold_group.add_argument(
        "--title-threshold",
        type=float,
        default=TITLE_MATCH_THRESHOLD,
        help=f"Title match threshold for mismatch detection (default: {TITLE_MATCH_THRESHOLD})",
    )
    threshold_group.add_argument(
        "--artist-threshold",
        type=float,
        default=ARTIST_MATCH_THRESHOLD,
        help=f"Artist match threshold for mismatch detection (default: {ARTIST_MATCH_THRESHOLD})",
    )
    threshold_group.add_argument(
        "--duration-tolerance",
        type=float,
        default=DURATION_TOLERANCE,
        help=f"Duration tolerance in seconds for mismatch (default: {DURATION_TOLERANCE})",
    )

    # Logging options
    logging_group = parser.add_argument_group("logging options")
    logging_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )
    logging_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    logging_group.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write logs to file",
    )

    return parser.parse_args()


def progress_callback(current: int, total: int, filename: str) -> None:
    """Progress callback for the pipeline."""
    logger.info(f"[{current}/{total}] Processing: {filename}")


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Configure logging
    setup_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        log_file=args.log_file,
    )

    # Validate directory
    if not args.directory.exists():
        logger.error(f"Directory '{args.directory}' does not exist.")
        sys.exit(1)

    if not args.directory.is_dir():
        logger.error(f"'{args.directory}' is not a directory.")
        sys.exit(1)

    # Create pipeline config from arguments
    config = PipelineConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        genius_token=args.genius_token,
        output_dir=args.output_dir,
        verbose=args.verbose,
        # Additional config options
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        lrclib_only=args.lrclib_only,
        api_delay=args.api_delay,
        censorship_threshold=args.censorship_threshold,
        min_words_vocals=args.min_words_vocals,
        min_speech_duration=args.min_speech_duration,
        title_threshold=args.title_threshold,
        artist_threshold=args.artist_threshold,
        duration_tolerance=args.duration_tolerance,
        formats=set(args.formats) if args.formats else None,
        no_censored=args.no_censored,
        no_instrumental=args.no_instrumental,
        no_mismatches=args.no_mismatches,
    )

    logger.info(f"Scanning {args.directory} for audio files...")
    logger.debug(f"Config: {config}")

    # Run the pipeline
    result = run_pipeline(
        directory=args.directory,
        config=config,
        progress_callback=progress_callback,
    )

    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()
