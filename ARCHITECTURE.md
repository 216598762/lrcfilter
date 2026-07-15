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

**Configuration**:
```python
model = WhisperModel(
    "large-v3",
    device="cuda",  # or "cpu"
    compute_type="float16"  # or "int8" for CPU
)
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

**Output**:
```python
@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    language: str
    duration: float
    has_speech: bool  # Based on VAD results

@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word]

@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float
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

## Data Flow

```
1. File Scanner
   └─▶ List[AudioFile]

2. Metadata Extractor
   └─▶ List[(AudioFile, TrackMetadata)]

3. Lyrics Fetcher
   └─▶ List[(AudioFile, TrackMetadata, LyricsResult)]

4. Metadata Mismatch Detector
   └─▶ List[(AudioFile, TrackMetadata, LyricsResult, MismatchResult)]
   └─▶ Flag mismatches for output

5. Whisper Analyzer
   └─▶ List[(AudioFile, TrackMetadata, LyricsResult, TranscriptionResult)]

6. Censorship Detector
   └─▶ List[(AudioFile, CensorshipResult)]

7. Instrumental Detector
   └─▶ List[(AudioFile, InstrumentalResult)]

8. Output Writer
   └─▶ censored.txt, instrumental.txt, metadata_mismatches.txt
```

---

## Project Structure

```
lrcfilter/
├── lrcfilter/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── scanner.py           # File discovery
│   ├── metadata.py          # Audio tag extraction
│   ├── lyrics.py            # LRCLib + Genius integration
│   ├── analyzer.py          # faster-whisper wrapper
│   ├── censorship.py        # Censorship detection logic
│   ├── instrumental.py      # Instrumental detection logic
│   ├── mismatch.py          # Metadata mismatch detection
│   ├── output.py            # File output handling
│   ├── models.py            # Dataclasses/models
│   └── config.py            # Configuration constants
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py
│   ├── test_metadata.py
│   ├── test_lyrics.py
│   ├── test_analyzer.py
│   ├── test_censorship.py
│   ├── test_instrumental.py
│   └── test_mismatch.py
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md          # This file
```

---

## Dependencies

Dependencies are defined in `pyproject.toml`:

```toml
[project]
dependencies = [
    "mutagen>=1.47.0",
    "faster-whisper>=1.0.0",
    "requests>=2.31.0",
    "lyricsgenius>=1.5.0",
    "rapidfuzz>=3.6.0",
    "better-profanity>=0.7.0",
    "tqdm>=4.66.0",
]
```

---

## CLI Interface

```bash
# Basic usage
python -m lrcfilter /path/to/music

# With options
python -m lrcfilter /path/to/music \
    --output-dir ./results \
    --model large-v3 \
    --device cuda \
    --genius-token YOUR_TOKEN \
    --verbose
```

---

## Configuration Options

```python
# config.py

# Supported audio formats
SUPPORTED_FORMATS = {'.flac', '.mp3', '.m4a', '.ogg', '.opus'}

# Whisper model settings
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"  # or "cpu"
DEFAULT_COMPUTE_TYPE = "float16"

# Censorship detection thresholds
CENSORSHIP_MISMATCH_THRESHOLD = 0.3  # 30% mismatch = censored
MIN_WORDS_FOR_VOCALS = 10
MIN_SPEECH_DURATION = 5.0

# Metadata mismatch detection thresholds
TITLE_MATCH_THRESHOLD = 0.6    # Below 60% = potential mismatch
ARTIST_MATCH_THRESHOLD = 0.7   # Below 70% = potential mismatch
DURATION_TOLERANCE = 30.0      # 30 seconds difference tolerance

# API settings
LRCLIB_BASE_URL = "https://lrclib.net/api"
GENIUS_TOKEN_ENV = "GENIUS_ACCESS_TOKEN"
API_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Output settings
OUTPUT_ENCODING = "utf-8"
INCLUDE_TIMESTAMP = True
```

---

## Error Handling Strategy

1. **File Not Found**: Log warning, skip file, continue
2. **Unsupported Format**: Log info, skip file
3. **Metadata Parse Error**: Use filename as fallback, log warning
4. **Lyrics Not Found**: Mark as "unknown", continue analysis
5. **API Rate Limit**: Implement exponential backoff
6. **Whisper Error**: Log error, skip file
7. **Permission Error**: Log warning, skip directory

---

## Performance Considerations

1. **Parallel Processing**: Use `concurrent.futures` for:
   - Metadata extraction (I/O bound)
   - Lyrics fetching (network bound)
   
2. **GPU Utilization**: Single GPU with batch processing for Whisper

3. **Caching**: 
   - Cache lyrics results to avoid duplicate API calls
   - Optional: Cache Whisper transcriptions for re-runs

4. **Memory Management**: Process files in chunks for large libraries

---

## Testing Strategy

1. **Unit Tests**: Each module independently
2. **Integration Tests**: End-to-end pipeline with sample files
3. **Mock External Services**: LRCLib, Genius APIs
4. **Test Audio Files**: Include small sample files for testing

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
