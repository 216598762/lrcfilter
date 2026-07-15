"""LRCFilter - Audio analysis tool for detecting censored/explicit content and instrumental tracks."""

__version__ = "0.1.0"
__author__ = "LRCFilter"

from lrcfilter.scanner import scan_audio_files
from lrcfilter.metadata import extract_metadata
from lrcfilter.lyrics import fetch_lyrics
from lrcfilter.analyzer import analyze_audio
from lrcfilter.censorship import detect_censorship
from lrcfilter.instrumental import detect_instrumental
from lrcfilter.mismatch import detect_metadata_mismatch
from lrcfilter.output import write_results
from lrcfilter.pipeline import run_pipeline, PipelineConfig, PipelineResult

__all__ = [
    "scan_audio_files",
    "extract_metadata",
    "fetch_lyrics",
    "analyze_audio",
    "detect_censorship",
    "detect_instrumental",
    "detect_metadata_mismatch",
    "write_results",
    "run_pipeline",
    "PipelineConfig",
    "PipelineResult",
]
