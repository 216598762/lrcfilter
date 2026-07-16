"""LRCFilter - Audio analysis tool for detecting censored/explicit content and instrumental tracks."""

__version__ = "0.1.0"
__author__ = "LRCFilter"

from lrcfilter.analyzer import analyze_audio
from lrcfilter.censorship import detect_censorship
from lrcfilter.instrumental import detect_instrumental
from lrcfilter.lyrics import fetch_lyrics
from lrcfilter.metadata import extract_metadata
from lrcfilter.mismatch import detect_metadata_mismatch
from lrcfilter.output import write_results
from lrcfilter.pipeline import PipelineConfig, PipelineResult, run_pipeline
from lrcfilter.scanner import scan_audio_files

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "analyze_audio",
    "detect_censorship",
    "detect_instrumental",
    "detect_metadata_mismatch",
    "extract_metadata",
    "fetch_lyrics",
    "run_pipeline",
    "scan_audio_files",
    "write_results",
]
