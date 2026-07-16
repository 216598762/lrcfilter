"""Tests for logging configuration module."""

import logging
import sys
from pathlib import Path
from typing import Optional

import pytest

from lrcfilter.logging_config import setup_logging, get_logger


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_default_logging_level(self) -> None:
        """Default logging should be INFO level."""
        setup_logging()
        logger = get_logger("test_module")
        assert logger.isEnabledFor(logging.INFO)
        assert not logger.isEnabledFor(logging.DEBUG)

    def test_verbose_logging_level(self) -> None:
        """Verbose mode should enable DEBUG level."""
        setup_logging(verbose=True)
        logger = get_logger("test_module")
        assert logger.isEnabledFor(logging.DEBUG)

    def test_quiet_logging_level(self) -> None:
        """Quiet mode should only show WARNING and above."""
        setup_logging(quiet=True)
        logger = get_logger("test_module")
        assert logger.isEnabledFor(logging.WARNING)
        assert not logger.isEnabledFor(logging.INFO)

    def test_verbose_overrides_quiet(self) -> None:
        """If both verbose and quiet are set, verbose should win."""
        setup_logging(verbose=True, quiet=True)
        logger = get_logger("test_module")
        assert logger.isEnabledFor(logging.DEBUG)

    def test_creates_console_handler(self) -> None:
        """Should add a console handler to the root logger."""
        setup_logging()
        root_logger = logging.getLogger("lrcfilter")
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_console_handler_writes_to_stdout(self) -> None:
        """Console handler should be configured to write to stdout."""
        setup_logging()
        root_logger = logging.getLogger("lrcfilter")
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                assert handler.stream is sys.stdout

    def test_creates_file_handler_when_log_file_provided(
        self, tmp_path: Path
    ) -> None:
        """Should add a file handler when log_file is provided."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        root_logger = logging.getLogger("lrcfilter")
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_file_handler_always_logs_debug(
        self, tmp_path: Path
    ) -> None:
        """File handler should always log at DEBUG level regardless of console level."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        root_logger = logging.getLogger("lrcfilter")
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                assert handler.level == logging.DEBUG

    def test_log_file_parent_directory_created(
        self, tmp_path: Path
    ) -> None:
        """Should create parent directories for log file if they don't exist."""
        log_file = tmp_path / "subdir" / "deep" / "test.log"
        setup_logging(log_file=log_file)
        assert log_file.parent.exists()

    def test_clears_existing_handlers(self) -> None:
        """Should clear existing handlers before adding new ones."""
        root_logger = logging.getLogger("lrcfilter")
        # Add a dummy handler
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)

        setup_logging()

        # Dummy handler should be removed
        assert dummy_handler not in root_logger.handlers

    def test_formatter_format_string(self, tmp_path: Path) -> None:
        """Formatter should use the expected format string."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        root_logger = logging.getLogger("lrcfilter")

        for handler in root_logger.handlers:
            formatter = handler.formatter
            assert formatter is not None
            # Check that the format contains expected components
            assert "%(asctime)s" in formatter._fmt
            assert "%(levelname)s" in formatter._fmt
            assert "%(name)s" in formatter._fmt
            assert "%(message)s" in formatter._fmt

    def test_datefmt_format(self, tmp_path: Path) -> None:
        """Date format should be YYYY-MM-DD HH:MM:SS."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        root_logger = logging.getLogger("lrcfilter")

        for handler in root_logger.handlers:
            formatter = handler.formatter
            assert formatter is not None
            assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"


class TestGetLogger:
    """Test the get_logger function."""

    def test_returns_logger_instance(self) -> None:
        """Should return a logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefixed_with_lrcfilter(self) -> None:
        """Logger name should be prefixed with 'lrcfilter.'."""
        logger = get_logger("my_module")
        assert logger.name == "lrcfilter.my_module"

    def test_different_loggers_for_different_names(self) -> None:
        """Different module names should return different logger instances."""
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_same_name_returns_same_logger(self) -> None:
        """Same module name should return the same logger instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2

    def test_logger_inherits_root_config(self) -> None:
        """Module loggers should inherit configuration from root logger."""
        setup_logging(verbose=True)
        logger = get_logger("child_module")
        assert logger.isEnabledFor(logging.DEBUG)

    def test_logger_propagation(self) -> None:
        """Logger should propagate to parent loggers."""
        logger = get_logger("propagation_test")
        assert logger.propagate is True
