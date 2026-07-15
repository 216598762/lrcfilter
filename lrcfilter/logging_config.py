"""Logging configuration for LRCFilter."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: Enable debug-level logging
        quiet: Suppress non-error output
        log_file: Optional file to write logs to
    """
    # Set log level based on verbosity/quiet flags
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Configure root logger
    root_logger = logging.getLogger("lrcfilter")
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"lrcfilter.{name}")
