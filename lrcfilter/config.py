"""Configuration constants for LRCFilter."""

# Supported audio formats
SUPPORTED_FORMATS = {".flac", ".mp3", ".m4a", ".ogg", ".opus"}

# Whisper model settings
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"  # or "cpu"
DEFAULT_COMPUTE_TYPE = "float16"

# Censorship detection thresholds
CENSORSHIP_MISMATCH_THRESHOLD = 0.3  # 30% mismatch = censored
MIN_WORDS_FOR_VOCALS = 10
MIN_SPEECH_DURATION = 5.0

# Metadata mismatch detection thresholds
TITLE_MATCH_THRESHOLD = 0.6  # Below 60% = potential mismatch
ARTIST_MATCH_THRESHOLD = 0.7  # Below 70% = potential mismatch
DURATION_TOLERANCE = 30.0  # 30 seconds difference tolerance

# API settings
LRCLIB_BASE_URL = "https://lrclib.net/api"
GENIUS_TOKEN_ENV = "GENIUS_ACCESS_TOKEN"
API_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Output settings
OUTPUT_ENCODING = "utf-8"
INCLUDE_TIMESTAMP = True
