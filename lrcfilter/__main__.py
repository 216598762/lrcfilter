"""CLI entry point for LRCFilter."""

import argparse
import sys
from pathlib import Path

from lrcfilter.pipeline import run_pipeline, print_summary, PipelineConfig
from lrcfilter.config import DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="lrcfilter",
        description="Audio analysis tool for detecting censored/explicit content and instrumental tracks",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for audio files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory for result files (default: current directory)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to run Whisper on (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help=f"Compute type for Whisper (default: {DEFAULT_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--genius-token",
        type=str,
        default=None,
        help="Genius API access token (or set GENIUS_ACCESS_TOKEN env var)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def progress_callback(current: int, total: int, filename: str) -> None:
    """Progress callback for the pipeline."""
    print(f"[{current}/{total}] Processing: {filename}")


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Create pipeline config
    config = PipelineConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        genius_token=args.genius_token,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print(f"Scanning {args.directory} for audio files...")

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
