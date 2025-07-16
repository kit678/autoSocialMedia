"""Slideshow module for creating smart video content."""

from .create_smart_video import run, create_with_scroll_video, create_with_webpage_video
from .exceptions import (
    SlideshowError,
    ValidationError,
    MissingFileError,
    InvalidDurationError,
    AudioMismatchError,
    FFmpegNotFoundError,
    TimelineValidationError
)
from .validation import (
    validate_slideshow_inputs,
    validate_mixed_media_inputs,
    validate_ffmpeg_availability,
    validate_audio_compatibility
)

__all__ = [
    'run',
    'create_with_scroll_video',
    'create_with_webpage_video',
    'SlideshowError',
    'ValidationError',
    'MissingFileError',
    'InvalidDurationError',
    'AudioMismatchError',
    'FFmpegNotFoundError',
    'TimelineValidationError',
    'validate_slideshow_inputs',
    'validate_mixed_media_inputs',
    'validate_ffmpeg_availability',
    'validate_audio_compatibility'
]
