"""
Custom exception classes for the slideshow module.

Provides descriptive error types for different slideshow creation failures.
"""


class SlideshowError(Exception):
    """Base exception class for slideshow-related errors."""
    pass


class ValidationError(SlideshowError):
    """Raised when validation of timeline segments or assets fails."""
    pass


class MissingFileError(ValidationError):
    """Raised when a required file is missing."""
    
    def __init__(self, file_path: str, message: str = None):
        self.file_path = file_path
        if message is None:
            message = f"Required file not found: {file_path}"
        else:
            message = f"{message}: {file_path}"
        super().__init__(message)


class InvalidDurationError(ValidationError):
    """Raised when a segment has invalid duration."""
    
    def __init__(self, duration: float, segment_id: str = None, message: str = None):
        self.duration = duration
        self.segment_id = segment_id
        if message is None:
            segment_info = f" (segment: {segment_id})" if segment_id else ""
            message = f"Invalid duration: {duration}{segment_info}. Duration must be > 0."
        super().__init__(message)


class AudioMismatchError(ValidationError):
    """Raised when audio duration doesn't match expected timeline."""
    
    def __init__(self, expected_duration: float, actual_duration: float, message: str = None):
        self.expected_duration = expected_duration
        self.actual_duration = actual_duration
        if message is None:
            message = f"Audio duration mismatch: expected {expected_duration}s, got {actual_duration}s"
        super().__init__(message)


class FFmpegNotFoundError(SlideshowError):
    """Raised when FFmpeg or ffprobe is not available."""
    
    def __init__(self, command: str = "ffmpeg", message: str = None):
        self.command = command
        if message is None:
            message = f"{command} not found. Please install FFmpeg and ensure it's in your PATH."
        super().__init__(message)


class TimelineValidationError(ValidationError):
    """Raised when timeline validation fails."""
    
    def __init__(self, issues: list, message: str = None):
        self.issues = issues
        if message is None:
            issue_list = "\n".join(f"- {issue}" for issue in issues)
            message = f"Timeline validation failed:\n{issue_list}"
        super().__init__(message)
