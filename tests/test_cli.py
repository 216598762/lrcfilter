"""Tests for CLI entry point module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from lrcfilter.__main__ import main, parse_args, progress_callback


class TestParseArgs:
    """Test the parse_args function."""

    def test_directory_argument(self) -> None:
        """Should parse directory argument correctly."""
        with patch("sys.argv", ["lrcfilter", "/path/to/music"]):
            args = parse_args()
            assert args.directory == Path("/path/to/music")

    def test_output_dir_option(self) -> None:
        """Should parse --output-dir option."""
        with patch("sys.argv", ["lrcfilter", "/music", "-o", "./results"]):
            args = parse_args()
            assert args.output_dir == Path("./results")

    def test_model_option(self) -> None:
        """Should parse --model option."""
        with patch("sys.argv", ["lrcfilter", "/music", "-m", "turbo"]):
            args = parse_args()
            assert args.model == "turbo"

    def test_device_option(self) -> None:
        """Should parse --device option."""
        with patch("sys.argv", ["lrcfilter", "/music", "-d", "cpu"]):
            args = parse_args()
            assert args.device == "cpu"

    def test_compute_type_option(self) -> None:
        """Should parse --compute-type option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--compute-type", "int8"]):
            args = parse_args()
            assert args.compute_type == "int8"

    def test_beam_size_option(self) -> None:
        """Should parse --beam-size option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--beam-size", "10"]):
            args = parse_args()
            assert args.beam_size == 10

    def test_vad_filter_enabled(self) -> None:
        """Should enable VAD filter by default."""
        with patch("sys.argv", ["lrcfilter", "/music"]):
            args = parse_args()
            assert args.vad_filter is True

    def test_no_vad_filter_option(self) -> None:
        """Should parse --no-vad-filter option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--no-vad-filter"]):
            args = parse_args()
            assert args.vad_filter is False

    def test_genius_token_option(self) -> None:
        """Should parse --genius-token option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--genius-token", "abc123"]):
            args = parse_args()
            assert args.genius_token == "abc123"

    def test_lrclib_only_option(self) -> None:
        """Should parse --lrclib-only option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--lrclib-only"]):
            args = parse_args()
            assert args.lrclib_only is True

    def test_api_delay_option(self) -> None:
        """Should parse --api-delay option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--api-delay", "2.5"]):
            args = parse_args()
            assert args.api_delay == 2.5

    def test_censorship_threshold_option(self) -> None:
        """Should parse --censorship-threshold option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--censorship-threshold", "0.5"]):
            args = parse_args()
            assert args.censorship_threshold == 0.5

    def test_min_words_vocals_option(self) -> None:
        """Should parse --min-words-vocals option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--min-words-vocals", "15"]):
            args = parse_args()
            assert args.min_words_vocals == 15

    def test_min_speech_duration_option(self) -> None:
        """Should parse --min-speech-duration option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--min-speech-duration", "10.0"]):
            args = parse_args()
            assert args.min_speech_duration == 10.0

    def test_title_threshold_option(self) -> None:
        """Should parse --title-threshold option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--title-threshold", "0.8"]):
            args = parse_args()
            assert args.title_threshold == 0.8

    def test_artist_threshold_option(self) -> None:
        """Should parse --artist-threshold option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--artist-threshold", "0.9"]):
            args = parse_args()
            assert args.artist_threshold == 0.9

    def test_duration_tolerance_option(self) -> None:
        """Should parse --duration-tolerance option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--duration-tolerance", "60"]):
            args = parse_args()
            assert args.duration_tolerance == 60.0

    def test_verbose_option(self) -> None:
        """Should parse --verbose option."""
        with patch("sys.argv", ["lrcfilter", "/music", "-v"]):
            args = parse_args()
            assert args.verbose is True

    def test_quiet_option(self) -> None:
        """Should parse --quiet option."""
        with patch("sys.argv", ["lrcfilter", "/music", "-q"]):
            args = parse_args()
            assert args.quiet is True

    def test_log_file_option(self) -> None:
        """Should parse --log-file option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--log-file", "analysis.log"]):
            args = parse_args()
            assert args.log_file == Path("analysis.log")

    def test_formats_option(self) -> None:
        """Should parse --formats option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--formats", ".mp3", ".flac"]):
            args = parse_args()
            assert args.formats == [".mp3", ".flac"]

    def test_no_censored_option(self) -> None:
        """Should parse --no-censored option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--no-censored"]):
            args = parse_args()
            assert args.no_censored is True

    def test_no_instrumental_option(self) -> None:
        """Should parse --no-instrumental option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--no-instrumental"]):
            args = parse_args()
            assert args.no_instrumental is True

    def test_no_mismatches_option(self) -> None:
        """Should parse --no-mismatches option."""
        with patch("sys.argv", ["lrcfilter", "/music", "--no-mismatches"]):
            args = parse_args()
            assert args.no_mismatches is True

    def test_defaults(self) -> None:
        """Should use correct default values."""
        with patch("sys.argv", ["lrcfilter", "/music"]):
            args = parse_args()
            assert args.output_dir == Path(".")
            assert args.model == "large-v3"
            assert args.device == "cuda"
            assert args.compute_type == "float16"
            assert args.beam_size == 5
            assert args.vad_filter is True
            assert args.lrclib_only is False
            assert args.api_delay == 1.0
            assert args.verbose is False
            assert args.quiet is False
            assert args.no_censored is False
            assert args.no_instrumental is False
            assert args.no_mismatches is False


class TestProgressCallback:
    """Test the progress_callback function."""

    def test_logs_progress(self) -> None:
        """Should log progress message."""
        with patch("lrcfilter.__main__.logger") as mock_logger:
            progress_callback(1, 10, "test.mp3")
            mock_logger.info.assert_called_once_with("[1/10] Processing: test.mp3")


class TestMain:
    """Test the main function."""

    def test_exits_when_directory_not_exists(self, tmp_path: Path) -> None:
        """Should exit with code 1 when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        with patch("sys.argv", ["lrcfilter", str(nonexistent)]), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_with(1)

    def test_exits_when_not_a_directory(self, tmp_path: Path) -> None:
        """Should exit with code 1 when path is not a directory."""
        test_file = tmp_path / "file.txt"
        test_file.touch()
        with patch("sys.argv", ["lrcfilter", str(test_file)]), patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_with(1)

    def test_runs_pipeline_successfully(self, tmp_path: Path) -> None:
        """Should run pipeline when directory exists."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        mock_result = MagicMock()
        mock_result.total_files = 0
        mock_result.processed_files = 0
        mock_result.censored_tracks = []
        mock_result.instrumental_tracks = []
        mock_result.metadata_mismatches = []

        with (
            patch("sys.argv", ["lrcfilter", str(music_dir)]),
            patch("lrcfilter.__main__.run_pipeline", return_value=mock_result) as mock_pipeline,
            patch("lrcfilter.__main__.print_summary"),
        ):
            main()
            mock_pipeline.assert_called_once()

    def test_creates_pipeline_config(self, tmp_path: Path) -> None:
        """Should create PipelineConfig with parsed arguments."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        mock_result = MagicMock()
        mock_result.total_files = 0
        mock_result.processed_files = 0
        mock_result.censored_tracks = []
        mock_result.instrumental_tracks = []
        mock_result.metadata_mismatches = []

        with (
            patch(
                "sys.argv",
                [
                    "lrcfilter",
                    str(music_dir),
                    "-m",
                    "turbo",
                    "-d",
                    "cpu",
                    "--no-censored",
                ],
            ),
            patch("lrcfilter.__main__.run_pipeline", return_value=mock_result) as mock_pipeline,
            patch("lrcfilter.__main__.print_summary"),
        ):
            main()
            config = mock_pipeline.call_args[1]["config"]
            assert config.model_name == "turbo"
            assert config.device == "cpu"
            assert config.no_censored is True

    def test_sets_up_logging(self, tmp_path: Path) -> None:
        """Should configure logging based on arguments."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        mock_result = MagicMock()
        mock_result.total_files = 0
        mock_result.processed_files = 0
        mock_result.censored_tracks = []
        mock_result.instrumental_tracks = []
        mock_result.metadata_mismatches = []

        with (
            patch("sys.argv", ["lrcfilter", str(music_dir), "-v"]),
            patch("lrcfilter.__main__.setup_logging") as mock_setup,
            patch("lrcfilter.__main__.run_pipeline", return_value=mock_result),
            patch("lrcfilter.__main__.print_summary"),
        ):
            main()
            mock_setup.assert_called_once()
            assert mock_setup.call_args[1]["verbose"] is True

    def test_passes_progress_callback(self, tmp_path: Path) -> None:
        """Should pass progress callback to pipeline."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        mock_result = MagicMock()
        mock_result.total_files = 0
        mock_result.processed_files = 0
        mock_result.censored_tracks = []
        mock_result.instrumental_tracks = []
        mock_result.metadata_mismatches = []

        with (
            patch("sys.argv", ["lrcfilter", str(music_dir)]),
            patch("lrcfilter.__main__.run_pipeline", return_value=mock_result) as mock_pipeline,
            patch("lrcfilter.__main__.print_summary"),
        ):
            main()
            assert "progress_callback" in mock_pipeline.call_args[1]

    def test_prints_summary(self, tmp_path: Path) -> None:
        """Should print summary after pipeline completes."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        mock_result = MagicMock()

        with (
            patch("sys.argv", ["lrcfilter", str(music_dir)]),
            patch("lrcfilter.__main__.run_pipeline", return_value=mock_result),
            patch("lrcfilter.__main__.print_summary") as mock_summary,
        ):
            main()
            mock_summary.assert_called_once_with(mock_result)
