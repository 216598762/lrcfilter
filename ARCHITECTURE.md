# LRCFilter - Audio Analysis & Censorship Detection Architecture

## Overview

A Python tool that recursively scans audio files, fetches lyrics, and uses AI-powered speech recognition to detect censored/explicit content and instrumental tracks.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LRCFilter Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ File Scanner  │───▶│   Metadata   │───▶│   Lyrics     │               │
│  │   (Recursive) │    │  Extractor   │    │   Fetcher    │               │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                    │                       │
│         │                   │    ┌───────────────┘                       │
│         │                   │    │                                       │
│         │                   ▼    ▼                                       │
         │            ┌──────────────────┐                                │
         │            │    Metadata      │                                │
         │            │   Mismatch      │                                │
         │            │    Detector     │                                │
         │            └────────┬─────────┘                                │
         │                     │                                          │
         │                     ▼                                          │
         │            ┌──────────────────┐                                │
         │            │     Whisper      │                                │
         │            │    Analyzer      │                                │
         │            └────────┬─────────┘                                │
         │                     │                                          │
         │           ┌─────────┴─────────┐                               │
         │           ▼                   ▼                                │
         │  ┌──────────────┐    ┌──────────────┐                         │
         │  │   Censorship │    │  Instrumental │                         │
         │  │   Detector   │    │   Detector    │                         │
         │  └──────┬───────┘    └──────┬───────┘                         │
         │         │                   │                                  │
         │         └─────────┬─────────┘                                 │
         │                   ▼                                            │
         │            ┌──────────────┐                                    │
         └───────────▶│   Output     │                                    │
                      │   Writer     │                                    │
                      └──────┬───────┘                                    │
                             │                                            │
              ┌──────────────┼──────────────┐                             │
              ▼              ▼              ▼                              │
        censored.txt  instrumental.txt  metadata_mismatches.txt           │
                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. File Scanner Module (`scanner.py`)

**Purpose**: Recursively discover all supported audio files in a directory.

```python
SUPPORTED_FORMATS = {'.flac', '.mp3', '.m4a', '.ogg', '.opus'}
```

**Responsibilities**:
- Walk directory tree recursively
- Filter files by supported extensions
- Return list of `AudioFile` dataclass instances
- Handle symlinks and permission errors gracefully
- **Symlink loop prevention**: Tracks visited inodes to prevent infinite recursion
- **Format validation**: Validates that custom format sets use dot-prefixed extensions (e.g., `'.mp3'`)
- Skip hidden files/directories (names starting with `.`)

**Output**:
```python
@dataclass
class AudioFile:
    path: Path
    filename: str
    extension: str
    size_mb: float
```

---

### 2. Metadata Extractor Module (`metadata.py`)

**Purpose**: Extract track metadata (artist, title, duration) from audio file tags.

**Library**: `mutagen` (supports all target formats)

**Responsibilities**:
- Parse ID3 tags (MP3), Vorbis comments (FLAC/OGG), MP4 atoms (M4A)
- Extract: title, artist, album, duration
- Handle missing/empty tags gracefully
- Return `TrackMetadata` dataclass

**Output**:
```python
@dataclass
class TrackMetadata:
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    duration_seconds: Optional[float]
    raw_tags: Dict[str, Any]
```

---

### 3. Lyrics Fetcher Module (`lyrics.py`)

**Purpose**: Retrieve synchronized or plain lyrics from online sources.

**Primary Source**: LRCLib API (free, no auth required)
**Fallback Source**: Genius API (requires API token)

**LRCLib Endpoints**:
- `GET https://lrclib.net/api/search` - Search by track/artist
- `GET https://lrclib.net/api/get` - Get specific lyrics

**Genius Integration**:
- Library: `lyricsgenius`
- Requires `GENIUS_ACCESS_TOKEN` environment variable
- Scrapes lyrics from Genius pages

**Responsibilities**:
- Try LRCLib first
- Fall back to Genius if LRCLib fails/returns no results
- Cache results to avoid duplicate API calls
- Return `LyricsResult` dataclass with matched metadata

**Output**:
```python
@dataclass
class LyricsResult:
    source: str  # 'lrclib' or 'genius'
    synced_lyrics: Optional[str]  # LRC format with timestamps
    plain_lyrics: Optional[str]   # Plain text lyrics
    # Metadata from the lyrics source (may differ from file metadata)
    matched_track_name: str       # Track name from lyrics API
    matched_artist_name: str      # Artist name from lyrics API
    matched_album_name: Optional[str]  # Album name from lyrics API
    match_score: float            # Confidence in the match (0.0 to 1.0)
```

---

### 4. Whisper Analyzer Module (`analyzer.py`)

**Purpose**: Transcribe audio files using faster-whisper for speech analysis.

**Library**: `faster-whisper`

**Model Selection**:
- `large-v3` - Best accuracy (recommended)
- `turbo` - Faster with good accuracy
- `medium` - Balance of speed/accuracy

**Thread-Safe Model Cache**:
Models are cached using double-checked locking (`threading.Lock`) to ensure thread safety and avoid redundant model loading:

```python
_model_cache = {}  # cache_key -> WhisperModel
_model_cache_lock = threading.Lock()

def get_model(model_name, device, compute_type) -> WhisperModel:
    cache_key = f"{model_name}_{device}_{compute_type}"
    if cache_key not in _model_cache:
        with _model_cache_lock:
            if cache_key not in _model_cache:
                _model_cache[cache_key] = WhisperModel(...)
    return _model_cache[cache_key]
```

**Transcription Parameters**:
```python
segments, info = model.transcribe(
    audio_path,
    beam_size=5,
    word_timestamps=True,  # Enable word-level timestamps
    vad_filter=True,       # Enable Voice Activity Detection
    vad_parameters=dict(
        min_silence_duration_ms=500
    )
)
```

**Responsibilities**:
- Transcribe audio to text with word timestamps
- Detect speech presence/absence (for instrumental detection)
- Return transcription segments with timing data
- Validate beam_size > 0

**Output**:
```python
@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    language: str
    duration: float
    has_speech: bool  # Based on VAD results
```

---

### 5. Censorship Detector Module (`censorship.py`)

**Purpose**: Identify censored or explicit content using dual detection methods.

#### Method 1: Lyrics vs Transcription Mismatch
- Normalize both lyrics and transcription text
- Align sequences using fuzzy matching
- Detect: skipped words, replaced words, missing sections
- Calculate mismatch percentage

#### Method 2: Explicit Language Detection
- Use profanity word list (e.g., `better-profanity` library)
- Scan transcribed text for explicit terms
- Count occurrences and severity

**Scoring Logic**:
```python
def detect_censorship(lyrics: str, transcription: str) -> CensorshipResult:
    # Method 1: Mismatch detection
    mismatch_score = calculate_mismatch(lyrics, transcription)
    
    # Method 2: Profanity detection
    profanity_score = detect_profanity(transcription)
    
    # Combined decision
    is_censored = mismatch_score > THRESHOLD or profanity_score > 0
    
    return CensorshipResult(
        is_censored=is_censored,
        mismatch_score=mismatch_score,
        profanity_score=profanity_score,
        confidence=calculate_confidence(mismatch_score, profanity_score)
    )
```

**Output**:
```python
@dataclass
class CensorshipResult:
    is_censored: bool
    mismatch_score: float  # 0.0 to 1.0
    profanity_count: int
    confidence: float
    details: str
```

---

### 6. Instrumental Detector Module (`instrumental.py`)

**Purpose**: Detect tracks with no vocals (instrumental songs).

**Detection Logic**:
1. Check if faster-whisper detected any speech segments
2. If `has_speech == False` or transcription is empty/minimal
3. Consider track as instrumental

**Thresholds**:
```python
MIN_WORDS_FOR_VOCALS = 10      # Minimum words to consider "vocal"
MIN_SPEECH_DURATION = 5.0      # Minimum seconds of detected speech
```

**Output**:
```python
@dataclass
class InstrumentalResult:
    is_instrumental: bool
    word_count: int
    speech_duration: float
    confidence: float
```

---

### 7. Metadata Mismatch Detector Module (`mismatch.py`)

**Purpose**: Detect when fetched lyrics don't match the audio file's embedded metadata.

**Use Cases**:
- Mislabeled files (wrong artist/title in tags)
- API returned wrong lyrics (fuzzy match error)
- Remix/live versions with different metadata
- Compilation tracks with incorrect attribution

**Detection Methods**:

#### Method 1: Title/Artist Fuzzy Matching
- Compare file metadata title vs lyrics API title
- Compare file metadata artist vs lyrics API artist
- Use fuzzy string matching with configurable threshold

#### Method 2: Duration Comparison
- Compare file duration vs lyrics API duration (if available)
- Significant differences indicate mismatched tracks

#### Method 3: Album Cross-Reference
- Verify album name matches between file and API
- Flag when album is completely different

**Scoring Logic**:
```python
def detect_metadata_mismatch(
    file_metadata: TrackMetadata,
    lyrics_result: LyricsResult
) -> MismatchResult:
    
    # Title similarity (0.0 to 1.0)
    title_score = fuzz.ratio(
        normalize(file_metadata.title),
        normalize(lyrics_result.matched_track_name)
    ) / 100.0
    
    # Artist similarity (0.0 to 1.0)
    artist_score = fuzz.ratio(
        normalize(file_metadata.artist),
        normalize(lyrics_result.matched_artist_name)
    ) / 100.0
    
    # Duration difference (if available)
    duration_diff = abs(file_metadata.duration - lyrics_duration)
    
    # Combined mismatch detection
    is_mismatch = (
        title_score < TITLE_MATCH_THRESHOLD or
        artist_score < ARTIST_MATCH_THRESHOLD or
        duration_diff > DURATION_TOLERANCE
    )
    
    return MismatchResult(
        is_mismatch=is_mismatch,
        title_score=title_score,
        artist_score=artist_score,
        duration_difference=duration_diff,
        confidence=calculate_mismatch_confidence(title_score, artist_score)
    )
```

**Thresholds**:
```python
TITLE_MATCH_THRESHOLD = 0.6    # Below 60% = potential mismatch
ARTIST_MATCH_THRESHOLD = 0.7   # Below 70% = potential mismatch
DURATION_TOLERANCE = 30.0      # 30 seconds difference tolerance
```

**Output**:
```python
@dataclass
class MismatchResult:
    is_mismatch: bool
    title_score: float          # 0.0 to 1.0 similarity
    artist_score: float         # 0.0 to 1.0 similarity
    duration_difference: float  # Seconds
    confidence: float           # 0.0 to 1.0
    details: str                # Human-readable explanation
```

---

### 8. Output Writer Module (`output.py`)

**Purpose**: Write results to categorized text files.

**Output Files**:
- `censored.txt` - List of censored/non-explicit tracks
- `instrumental.txt` - List of instrumental tracks (no vocals)
- `metadata_mismatches.txt` - List of tracks with metadata/lyrics mismatches

**Filename Prefix Support**: Output filenames can be customized with a prefix:
```python
write_results(..., filename_prefix="my_analysis")
# Produces: my_analysis_censored.txt, my_analysis_instrumental.txt, etc.
```

**Format**:
```
# Censored/Non-Explicit Tracks
# Generated: 2026-07-15 14:30:00
# Total: 42 tracks

/path/to/song1.mp3
/path/to/song2.flac
Artist - Title (album)
...
```

**Metadata Mismatches Format**:
```
# Metadata Mismatches
# Generated: 2026-07-15 14:30:00
# Total: 15 tracks with mismatched metadata

# File: /path/to/song.mp3
# File Metadata: "Song Title" by "Artist A"
# Lyrics Metadata: "Different Song" by "Artist B"
# Mismatch Score: 0.45 (title: 0.32, artist: 0.58)
# Confidence: 0.85
---
...
```

**Responsibilities**:
- Create output files with headers
- Append results as they're processed
- Handle duplicate entries
- Provide summary statistics
- Include detailed mismatch information for debugging

---

### 9. Pipeline Orchestrator (`pipeline.py`)

**Purpose**: Orchestrate the entire analysis pipeline and manage configuration.

**Configuration Dataclass**:
```python
@dataclass
class PipelineConfig:
    # Whisper model settings
    model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    
    # API settings
    genius_token: Optional[str] = None
    lrclib_only: bool = False
    api_delay: float = 1.0
    
    # Output settings
    output_dir: Path = Path(".")
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
```

**Parameter Validation** (`__post_init__`):
- `beam_size` must be positive
- `api_delay` must be non-negative
- Thresholds must be between 0.0 and 1.0
- Duration and word counts must be non-negative

**Pipeline Flow**:
```python
def process_single_track(audio_file, config) -> TrackResult:
    metadata = extract_metadata(audio_file)
    lyrics = fetch_lyrics(metadata, ...)
    mismatch = detect_metadata_mismatch(metadata, lyrics, ...)
    transcription = analyze_audio(audio_file, ...)
    censorship = detect_censorship(lyrics.plain_lyrics, transcription.text, ...)
    instrumental = detect_instrumental(transcription, ...)
    return TrackResult(...)
```

**Data Classes**:
```python
@dataclass
class TrackResult:
    audio_file: AudioFile
    metadata: TrackMetadata
    lyrics: Optional[LyricsResult]
    transcription: Optional[TranscriptionResult]
    censorship: Optional[CensorshipResult]
    instrumental: Optional[InstrumentalResult]
    mismatch: Optional[MismatchResult]

@dataclass
class PipelineResult:
    total_files: int
    processed_files: int
    censored_tracks: List[Tuple[AudioFile, CensorshipResult]]
    instrumental_tracks: List[Tuple[AudioFile, InstrumentalResult]]
    metadata_mismatches: List[Tuple[AudioFile, MismatchResult]]
    track_results: List[TrackResult]
```

**Responsibilities**:
- Validate all configuration parameters before pipeline start
- Orchestrate the complete analysis flow per-track
- Collect and aggregate results across all files
- Write results to output files (single `write_results` call)
- Provide `print_summary()` for CLI output
- Support optional `progress_callback` for progress reporting

---

### 10. Logging Configuration (`logging_config.py`)

**Purpose**: Provide consistent logging across all modules.

**Setup**:
```python
setup_logging(
    verbose=False,   # DEBUG level
    quiet=False,     # WARNING level
    log_file=None,   # Optional file handler
)
```

**Logger Access**:
```python
from lrcfilter.logging_config import get_logger
logger = get_logger(__name__)
```

**Features**:
- Formatted timestamps: `2026-07-15 14:30:00 [INFO] lrcfilter.scanner: ...`
- Console handler (stdout) with configurable level
- Optional file handler (always logs DEBUG to file)
- Hierarchical logger namespace: `lrcfilter.<module>`

---

## Data Flow

The pipeline is orchestrated by `pipeline.py`, which processes each track individually:

```
1. scan_audio_files(directory)
   └─▶ List[AudioFile]

2. For each AudioFile in process_single_track():
   a. extract_metadata(audio_file)
      └─▶ TrackMetadata
   b. fetch_lyrics(metadata)
      └─▶ Optional[LyricsResult]
   c. detect_metadata_mismatch(metadata, lyrics)
      └─▶ Optional[MismatchResult]
   d. analyze_audio(audio_file)
      └─▶ TranscriptionResult
   e. detect_censorship(lyrics.plain_lyrics, transcription.text)
      └─▶ Optional[CensorshipResult]
   f. detect_instrumental(transcription)
      └─▶ InstrumentalResult
   g. Return TrackResult(...)

3. run_pipeline() collects all TrackResults into PipelineResult

4. write_results(censored, instrumental, mismatches)
   └─▶ censored.txt, instrumental.txt, metadata_mismatches.txt
```

---

## Project Structure

```
lrcfilter/
├── lrcfilter/
│   ├── __init__.py          # Package init, public API exports
│   ├── __main__.py          # CLI entry point (argparse)
│   ├── config.py            # Configuration constants & thresholds
│   ├── logging_config.py    # Logging setup (console + file handlers)
│   ├── models.py            # Dataclasses: AudioFile, TrackMetadata, etc.
│   ├── scanner.py           # Recursive file discovery with symlink protection
│   ├── metadata.py          # Audio tag extraction via mutagen
│   ├── lyrics.py            # LRCLib + Genius lyrics fetching
│   ├── analyzer.py          # faster-whisper wrapper with model caching
│   ├── censorship.py        # Lyrics mismatch + profanity detection
│   ├── instrumental.py      # No-vocal / instrumental detection
│   ├── mismatch.py          # Metadata vs lyrics mismatch detection
│   ├── output.py            # Result file writer (censored/instrumental/mismatch)
│   └── pipeline.py          # Pipeline orchestration & config validation
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py
│   ├── test_metadata.py
│   ├── test_lyrics.py
│   ├── test_censorship.py
│   ├── test_instrumental.py
│   ├── test_mismatch.py
│   ├── test_config_wiring.py
│   └── test_parameter_validation.py
├── .github/
│   ├── workflows/ci.yml     # GitHub Actions CI/CD
│   └── dependabot.yml       # Dependabot config
├── pyproject.toml           # Package metadata, deps, ruff config
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Docker Compose for GPU/CPU
├── .dockerignore
├── .pre-commit-config.yaml  # Pre-commit hooks (ruff, trailing-ws, etc.)
├── .gitignore
├── LICENSE                  # MIT License
├── README.md                # Usage instructions
├── ARCHITECTURE.md          # This file
├── CONTRIBUTING.md           # Contributor guidelines
├── CODE_OF_CONDUCT.md        # Contributor Covenant v2.1
└── SECURITY.md               # Vulnerability reporting
```

---

## Dependencies

All dependencies are managed in `pyproject.toml`. See that file for exact version constraints.

**Runtime**: mutagen, faster-whisper, requests, lyricsgenius, rapidfuzz, better-profanity, tqdm

**Development** (`pip install -e ".[dev]"`): pytest, pytest-cov, ruff, pre-commit

---

## CLI Interface

Entry point: `lrcfilter.__main__:main` (installed as `lrcfilter` console script)

```bash
# Basic usage
lrcfilter /path/to/music

# With options
lrcfilter /path/to/music \
    -o ./results \
    -m large-v3 \
    -d cuda \
    --genius-token YOUR_TOKEN \
    -v

# Restrict formats and skip outputs
lrcfilter /path/to/music \
    --formats .mp3 .flac \
    --no-instrumental \
    --no-mismatches

# CPU with int8 quantization
lrcfilter /path/to/music --device cpu --compute-type int8

# Log to file
lrcfilter /path/to/music --log-file debug.log -v
```

**Argument Groups**:
- **Output options**: `--output-dir`, `--formats`, `--no-censored`, `--no-instrumental`, `--no-mismatches`
- **Whisper model options**: `--model`, `--device`, `--compute-type`, `--beam-size`, `--vad-filter/--no-vad-filter`
- **API options**: `--genius-token`, `--lrclib-only`, `--api-delay`
- **Detection thresholds**: `--censorship-threshold`, `--min-words-vocals`, `--min-speech-duration`, `--title-threshold`, `--artist-threshold`, `--duration-tolerance`
- **Logging options**: `--verbose`, `--quiet`, `--log-file`

---

## Configuration

### Default Constants (`config.py`)

Default values used when `PipelineConfig` is not explicitly set:

```python
# Supported audio formats
SUPPORTED_FORMATS = {'.flac', '.mp3', '.m4a', '.ogg', '.opus'}

# Whisper model settings
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"  # or "cpu"
DEFAULT_COMPUTE_TYPE = "float16"

# Detection thresholds
CENSORSHIP_MISMATCH_THRESHOLD = 0.3
MIN_WORDS_FOR_VOCALS = 10
MIN_SPEECH_DURATION = 5.0
TITLE_MATCH_THRESHOLD = 0.6
ARTIST_MATCH_THRESHOLD = 0.7
DURATION_TOLERANCE = 30.0

# API settings
LRCLIB_BASE_URL = "https://lrclib.net/api"
GENIUS_TOKEN_ENV = "GENIUS_ACCESS_TOKEN"
API_RATE_LIMIT_DELAY = 1.0

# Output settings
OUTPUT_ENCODING = "utf-8"
INCLUDE_TIMESTAMP = True
```

### Pipeline Configuration (`PipelineConfig`)

All parameters are passed through `PipelineConfig` which validates them in `__post_init__`. See [Pipeline Orchestrator](#9-pipeline-orchestrator-pipelinepy) for the full dataclass.

---

## Error Handling Strategy

1. **File Not Found / Permission Error**: Log warning, skip file/directory, continue processing
2. **Unsupported Format**: Filtered out by `scanner.py` before processing
3. **Metadata Parse Error**: Use filename as fallback title, log warning, return empty metadata
4. **Lyrics Not Found**: Return `None` from `fetch_lyrics()`, censorship/mismatch checks skipped
5. **API Rate Limit**: Fixed delay between requests via `api_delay` parameter (default 1.0s)
6. **Whisper Error**: Log error, skip file, continue processing
7. **Symlink Loop**: Detected via inode tracking in `scanner.py`, directory skipped
8. **Invalid Config**: `PipelineConfig.__post_init__` raises `ValueError` with descriptive message

---

## Public API (`__init__.py`)

The following symbols are exported from the package for programmatic use:

```python
from lrcfilter import (
    scan_audio_files,
    extract_metadata,
    fetch_lyrics,
    analyze_audio,
    detect_censorship,
    detect_instrumental,
    detect_metadata_mismatch,
    write_results,
    run_pipeline,
    PipelineConfig,
    PipelineResult,
)
```

---

## Performance Considerations

1. **Sequential Processing**: Files are processed one at a time due to GPU memory constraints with Whisper
2. **GPU Utilization**: Single GPU with model caching (thread-safe) to avoid redundant loads
3. **Caching**: 
   - Whisper models cached by `(model_name, device, compute_type)` key
   - Lyrics fetched once per unique metadata
4. **Memory Management**: Process files sequentially; no batch loading of all audio into memory

---

## Testing Strategy

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lrcfilter
```

**Test Files**:
- `test_scanner.py` - File discovery, symlink loop prevention, format validation
- `test_metadata.py` - Tag extraction from various formats
- `test_lyrics.py` - LRCLib/Genius fetching, error handling
- `test_censorship.py` - Mismatch detection, profanity detection
- `test_instrumental.py` - No-vocal detection thresholds
- `test_mismatch.py` - Title/artist similarity scoring
- `test_config_wiring.py` - CLI args correctly passed through PipelineConfig to functions
- `test_parameter_validation.py` - Validation of beam_size, thresholds, formats, etc.

**Approach**: Mock external services (LRCLib, Genius APIs) to avoid network calls in tests.

---

## Future Enhancements

1. **Web UI**: Flask/FastAPI dashboard
2. **Database**: SQLite for persistent results
3. **Batch Processing**: Queue system for large libraries
4. **Custom Word Lists**: User-defined profanity lists
5. **Export Formats**: JSON, CSV output options
6. **Progress Tracking**: Real-time CLI progress bar
7. **Auto-Fix Metadata**: Option to update file tags based on matched lyrics
8. **Batch Metadata Correction**: Apply fixes across entire library
