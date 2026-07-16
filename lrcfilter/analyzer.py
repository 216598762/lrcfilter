"""Whisper analyzer module for audio transcription."""

import threading

from faster_whisper import WhisperModel

from lrcfilter.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL
from lrcfilter.logging_config import get_logger
from lrcfilter.models import AudioFile, Segment, TranscriptionResult, Word

logger = get_logger(__name__)


# Thread-safe model cache
_model_cache = {}
_model_cache_lock = threading.Lock()


def get_model(
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> WhisperModel:
    """
    Get or create a Whisper model instance.

    Args:
        model_name: Name of the Whisper model
        device: Device to run on ('cpu' or 'cuda')
        compute_type: Compute type for the model

    Returns:
        WhisperModel instance
    """
    cache_key = f"{model_name}_{device}_{compute_type}"

    # Double-checked locking for thread safety
    if cache_key not in _model_cache:
        with _model_cache_lock:
            if cache_key not in _model_cache:
                logger.info(f"Loading Whisper model '{model_name}' on {device}...")
                try:
                    _model_cache[cache_key] = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                    logger.info("Model loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise

    return _model_cache[cache_key]


def analyze_audio(
    audio_file: AudioFile,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        audio_file: AudioFile to transcribe
        model_name: Whisper model name (tiny, base, small, medium, large-v2, large-v3, turbo)
        device: Device to run on ('cpu' or 'cuda')
        compute_type: Compute type ('float16', 'int8', 'int8_float16', 'float32')
        beam_size: Beam size for transcription (default: 5)
        vad_filter: Enable Voice Activity Detection filter (default: True)

    Returns:
        TranscriptionResult with transcription data

    Raises:
        ValueError: If beam_size is not positive
    """
    if beam_size <= 0:
        raise ValueError(f"beam_size must be positive, got {beam_size}")

    model = get_model(model_name, device, compute_type)

    # Transcribe with VAD and word timestamps
    segments_iter, info = model.transcribe(
        str(audio_file.path),
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 500,
        }
        if vad_filter
        else None,
    )

    # Collect segments
    segments = []
    full_text_parts = []
    total_speech_duration = 0.0

    for segment in segments_iter:
        words = []
        if segment.words:
            for word in segment.words:
                words.append(
                    Word(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                )

        segments.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words,
            )
        )

        full_text_parts.append(segment.text.strip())
        total_speech_duration += segment.end - segment.start

    # Determine if speech was detected
    has_speech = total_speech_duration > 0 and len(full_text_parts) > 0

    return TranscriptionResult(
        text=" ".join(full_text_parts),
        segments=segments,
        language=info.language,
        duration=info.duration,
        has_speech=has_speech,
    )
