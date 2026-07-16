# LRCFilter

Audio analysis tool for detecting censored/explicit content, instrumental tracks, and metadata mismatches using AI-powered speech recognition.

## Features

- **Censorship Detection**: Identifies censored or explicit content by comparing lyrics with AI transcription
- **Instrumental Detection**: Detects tracks with no vocals
- **Metadata Mismatch Detection**: Finds tracks where lyrics metadata doesn't match file tags
- **Recursive Scanning**: Processes entire directory trees automatically
- **Multiple Formats**: Supports FLAC, MP3, M4A, OGG, and OPUS files
- **Flexible Lyrics Sources**: LRCLib (primary) with Genius API fallback

## Installation

```bash
# Clone the repository
git clone https://github.com/216598762/lrcfilter.git
cd lrcfilter

# Install the package
pip install .

# Or install in development mode (editable install)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg installed on your system

### Dependencies

All dependencies are managed in `pyproject.toml` and installed automatically:

- **mutagen** - Audio metadata extraction
- **faster-whisper** - Speech recognition
- **requests** - HTTP requests
- **lyricsgenius** - Genius API integration
- **rapidfuzz** - Fuzzy string matching
- **better-profanity** - Profanity detection
- **tqdm** - Progress bars

Development dependencies (install with `pip install -e ".[dev]"`):

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **ruff** - Linter and formatter
- **pre-commit** - Git hooks

## Quick Start

```bash
# Basic usage - scan a directory
lrcfilter /path/to/music

# With custom output directory
lrcfilter /path/to/music -o ./results

# Use faster model on CPU
lrcfilter /path/to/music -m turbo --device cpu

# Only process MP3 and FLAC files
lrcfilter /path/to/music --formats .mp3 .flac
```

## Usage

### Basic Command

```bash
lrcfilter [OPTIONS] DIRECTORY
```

**Arguments:**
- `DIRECTORY` - Path to the directory containing audio files to analyze

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output-dir DIR` | Output directory for result files (default: current directory) |
| `--formats FMT [FMT ...]` | Audio formats to process (default: all supported) |
| `--no-censored` | Skip generating censored tracks list |
| `--no-instrumental` | Skip generating instrumental tracks list |
| `--no-mismatches` | Skip generating metadata mismatches list |

### Whisper Model Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Whisper model: tiny, base, small, medium, large-v2, large-v3, turbo (default: large-v3) |
| `-d, --device DEVICE` | Compute device: cpu, cuda (default: cuda) |
| `--compute-type TYPE` | Compute type: float16, int8, int8_float16, float32 (default: float16) |
| `--beam-size SIZE` | Beam size for transcription (default: 5) |
| `--vad-filter` | Enable Voice Activity Detection (default: enabled) |
| `--no-vad-filter` | Disable Voice Activity Detection |

### API Options

| Option | Description |
|--------|-------------|
| `--genius-token TOKEN` | Genius API token (or set GENIUS_ACCESS_TOKEN env var) |
| `--lrclib-only` | Only use LRCLib API, skip Genius fallback |
| `--api-delay SECONDS` | Delay between API requests (default: 1.0) |

### Detection Thresholds

| Option | Description |
|--------|-------------|
| `--censorship-threshold FLOAT` | Mismatch threshold for censorship (default: 0.3) |
| `--min-words-vocals INT` | Minimum words to consider vocal track (default: 10) |
| `--min-speech-duration FLOAT` | Minimum speech duration in seconds (default: 5.0) |
| `--title-threshold FLOAT` | Title match threshold (default: 0.6) |
| `--artist-threshold FLOAT` | Artist match threshold (default: 0.7) |
| `--duration-tolerance FLOAT` | Duration tolerance in seconds (default: 30.0) |

### Logging Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose/debug output |
| `-q, --quiet` | Suppress non-error output |
| `--log-file FILE` | Write logs to file |

## Examples

### Basic Analysis

```bash
# Analyze music library
lrcfilter ~/Music

# Analyze with verbose output
lrcfilter ~/Music --verbose

# Analyze quietly (errors only)
lrcfilter ~/Music --quiet
```

### Model Selection

```bash
# Use large-v3 for best accuracy (default)
lrcfilter ~/Music -m large-v3

# Use turbo for faster processing
lrcfilter ~/Music -m turbo

# Use CPU if no GPU available
lrcfilter ~/Music --device cpu --compute-type int8
```

### Filtering Formats

```bash
# Only process FLAC files
lrcfilter ~/Music --formats .flac

# Process MP3 and M4A files
lrcfilter ~/Music --formats .mp3 .m4a
```

### Custom Thresholds

```bash
# More sensitive censorship detection
lrcfilter ~/Music --censorship-threshold 0.2

# Strict metadata matching
lrcfilter ~/Music --title-threshold 0.8 --artist-threshold 0.9

# Shorter instrumental detection
lrcfilter ~/Music --min-speech-duration 2.0 --min-words-vocals 5
```

### API Configuration

```bash
# Use only LRCLib (no Genius)
lrcfilter ~/Music --lrclib-only

# Use Genius with custom token
lrcfilter ~/Music --genius-token YOUR_TOKEN_HERE

# Slower API requests to avoid rate limiting
lrcfilter ~/Music --api-delay 2.0
```

### Logging

```bash
# Save logs to file
lrcfilter ~/Music --log-file analysis.log

# Verbose logging for debugging
lrcfilter ~/Music -v --log-file debug.log
```

## Output Files

After processing, LRCFilter generates three text files in the output directory:

### censored.txt

List of tracks identified as censored or containing explicit content.

```
# Censored/Non-Explicit Tracks
# Generated: 2026-07-15 14:30:00
# Total: 42 tracks

/path/to/song1.mp3
  # Lyrics mismatch (45.2%)
  # Confidence: 72.3%

/path/to/song2.flac
  # 3 profanity instance(s)
  # Confidence: 85.0%
```

### instrumental.txt

List of tracks identified as instrumental (no vocals).

```
# Instrumental Tracks (No Vocals)
# Generated: 2026-07-15 14:30:00
# Total: 15 tracks

/path/to/intro.mp3
  # Words detected: 2, Speech duration: 0.5s
  # Confidence: 90.0%

/path/to/outro.flac
  # Words detected: 0, Speech duration: 0.0s
  # Confidence: 95.0%
```

### metadata_mismatches.txt

List of tracks where lyrics metadata doesn't match file tags.

```
# Metadata Mismatches
# Generated: 2026-07-15 14:30:00
# Total: 8 tracks with mismatched metadata

# File: /path/to/song.mp3
  File: 'Original Title' vs Lyrics: 'Different Title' (45% match)
  Artist: 'Artist A' vs 'Artist B' (62% match)
  # Confidence: 78.5%
---
```

## Programmatic Usage

```python
from pathlib import Path
from lrcfilter.pipeline import run_pipeline, PipelineConfig, print_summary

# Configure the pipeline
config = PipelineConfig(
    model_name="large-v3",
    device="cuda",
    beam_size=5,
    censorship_threshold=0.3,
    output_dir=Path("./results"),
)

# Run the pipeline
result = run_pipeline(
    directory=Path("/path/to/music"),
    config=config,
)

# Print summary
print_summary(result)

# Access detailed results
for track in result.track_results:
    if track.censorship and track.censorship.is_censored:
        print(f"Censored: {track.audio_file.filename}")
    if track.instrumental and track.instrumental.is_instrumental:
        print(f"Instrumental: {track.audio_file.filename}")
    if track.mismatch and track.mismatch.is_mismatch:
        print(f"Mismatch: {track.audio_file.filename}")
```

## Docker

### Using Docker

```bash
# Build the image
docker build -t lrcfilter .

# Run with your music directory
# Linux/macOS
docker run --rm -v ~/Music:/music:ro -v ./output:/output lrcfilter /music -o /output

# Windows
docker run --rm -v C:/Users/You/Music:/music:ro -v ./output:/output lrcfilter /music -o /output
```

### Using Docker Compose

```bash
# Edit docker-compose.yml to set your music directory
# Then run:
docker-compose run lrcfilter /music -o /output

# For GPU support (requires nvidia-docker):
# Uncomment the deploy section in docker-compose.yml
docker-compose --profile gpu run lrcfilter /music -o /output
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `MUSIC_DIR` | Path to music directory | `./music` |
| `OUTPUT_DIR` | Path to output directory | `./output` |

## Troubleshooting

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
- Use a smaller model: `--model turbo` or `--model medium`
- Use CPU mode: `--device cpu --compute-type int8`
- Process fewer files at once

### API Rate Limiting
If you hit API rate limits:
- Increase delay between requests: `--api-delay 2.0`
- Use `--lrclib-only` to skip Genius API
- Wait and retry later

### Files Not Processing
- Check file permissions
- Verify file format is supported: FLAC, MP3, M4A, OGG, OPUS
- Check logs for specific errors: `--log-file debug.log`

### No Lyrics Found
- Some tracks may not have lyrics in LRCLib or Genius
- Try adding a Genius API token for better coverage
- Check if track metadata (title/artist) is correct

### Slow Processing
- Use a smaller model: `--model turbo`
- Disable VAD filter: `--no-vad-filter`
- Use GPU acceleration: `--device cuda`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

```
File Scanner → Metadata Extractor → Lyrics Fetcher → Whisper Analyzer
                                                      ↓
                              ┌────────────────────────┼────────────────────────┐
                              ↓                        ↓                        ↓
                      Censorship Detector    Instrumental Detector    Metadata Mismatch
                              ↓                        ↓                        ↓
                              └────────────────────────┼────────────────────────┘
                                                      ↓
                                                Output Writer
                                                      ↓
                              censored.txt    instrumental.txt    metadata_mismatches.txt
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config_wiring.py -v

# Run with coverage
pytest tests/ --cov=lrcfilter

# Generate coverage report
pytest tests/ --cov=lrcfilter --cov-report=html
```

## Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/216598762/lrcfilter.git
cd lrcfilter
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Run linter
ruff check lrcfilter/ tests/

# Run formatter
ruff format lrcfilter/ tests/

# Run linter with auto-fix
ruff check --fix lrcfilter/ tests/
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality:
- **ruff**: Linting and formatting
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **check-json**: Validate JSON files

To run hooks manually:
```bash
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`pip install -e ".[dev]"`)
4. Make your changes
5. Run tests (`pytest tests/ -v`)
6. Run linter (`ruff check lrcfilter/ tests/`)
7. Commit your changes (`git commit -S -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- [LRCLib](https://lrclib.net/) for free lyrics API
- [Genius](https://genius.com/) for lyrics fallback
- [mutagen](https://github.com/quodlibet/mutagen) for audio metadata
