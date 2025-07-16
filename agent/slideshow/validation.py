"""
Validation functions for slideshow timeline and assets.

Provides comprehensive validation of timeline segments, file existence,
and audio/video compatibility before FFmpeg processing.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from agent.utils import run_command, get_audio_duration
from .exceptions import (
    FFmpegNotFoundError,
    MissingFileError,
    InvalidDurationError,
    AudioMismatchError,
    TimelineValidationError
)


def validate_ffmpeg_availability() -> None:
    """
    Validate that FFmpeg and ffprobe are available in the system.
    
    Raises:
        FFmpegNotFoundError: If FFmpeg or ffprobe is not found
    """
    # Check FFmpeg
    success, _, _ = run_command(['ffmpeg', '-version'], timeout=10)
    if not success:
        raise FFmpegNotFoundError("ffmpeg")
    
    # Check ffprobe
    success, _, _ = run_command(['ffprobe', '-version'], timeout=10)
    if not success:
        raise FFmpegNotFoundError("ffprobe")


def validate_file_exists(file_path: str, description: str = None) -> None:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to the file to validate
        description: Optional description for better error messages
        
    Raises:
        MissingFileError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        desc = f" ({description})" if description else ""
        raise MissingFileError(file_path, f"Required file not found{desc}: {file_path}")


def validate_timeline_segment(segment: Dict[str, Any], segment_index: int) -> List[str]:
    """
    Validate a single timeline segment.
    
    Args:
        segment: Timeline segment dictionary
        segment_index: Index of the segment for error reporting
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check required fields
    if 'cue_id' not in segment:
        issues.append(f"Segment {segment_index}: missing 'cue_id' field")
    
    if 'duration_frames' not in segment:
        issues.append(f"Segment {segment_index}: missing 'duration_frames' field")
    else:
        duration_frames = segment['duration_frames']
        if not isinstance(duration_frames, (int, float)) or duration_frames <= 0:
            issues.append(f"Segment {segment_index}: invalid duration_frames {duration_frames} (must be > 0)")
    
    return issues


def validate_timeline_segments(timeline: List[Dict[str, Any]], 
                             all_image_paths: Dict[str, str]) -> None:
    """
    Validate all timeline segments for consistency and file availability.
    
    Args:
        timeline: List of timeline segment dictionaries
        all_image_paths: Dictionary mapping cue_id to file paths
        
    Raises:
        TimelineValidationError: If validation fails
    """
    issues = []
    
    if not timeline:
        issues.append("Timeline is empty")
    
    for i, segment in enumerate(timeline):
        # Validate segment structure
        segment_issues = validate_timeline_segment(segment, i)
        issues.extend(segment_issues)
        
        # Validate file existence
        cue_id = segment.get('cue_id')
        if cue_id:
            if cue_id not in all_image_paths:
                issues.append(f"Segment {i}: cue_id '{cue_id}' not found in image paths")
            else:
                file_path = all_image_paths[cue_id]
                if not os.path.exists(file_path):
                    issues.append(f"Segment {i}: file not found for cue_id '{cue_id}': {file_path}")
    
    if issues:
        raise TimelineValidationError(issues)


def validate_audio_compatibility(audio_path: str, expected_duration: float, 
                                tolerance: float = 0.1) -> float:
    """
    Validate audio file compatibility and duration match.
    
    Args:
        audio_path: Path to the audio file
        expected_duration: Expected duration in seconds
        tolerance: Acceptable difference in seconds (default: 0.1s)
        
    Returns:
        Actual audio duration in seconds
        
    Raises:
        MissingFileError: If audio file doesn't exist
        AudioMismatchError: If duration doesn't match within tolerance
    """
    validate_file_exists(audio_path, "audio file")
    
    actual_duration = get_audio_duration(audio_path)
    if actual_duration <= 0:
        raise AudioMismatchError(expected_duration, actual_duration, 
                               f"Could not determine audio duration for {audio_path}")
    
    duration_diff = abs(actual_duration - expected_duration)
    if duration_diff > tolerance:
        raise AudioMismatchError(expected_duration, actual_duration,
                               f"Audio duration mismatch: expected {expected_duration:.2f}s, "
                               f"got {actual_duration:.2f}s (diff: {duration_diff:.2f}s)")
    
    return actual_duration


def validate_slideshow_inputs(visual_analysis: Dict[str, Any], 
                             all_image_paths: Dict[str, str],
                             audio_path: str, 
                             audio_duration: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    Comprehensive validation of all slideshow inputs.
    
    Args:
        visual_analysis: Visual analysis dictionary containing segments
        all_image_paths: Dictionary mapping cue_id to file paths
        audio_path: Path to the audio file
        audio_duration: Expected audio duration in seconds
        
    Returns:
        Tuple of (validated_timeline, actual_audio_duration)
        
    Raises:
        ValidationError: If any validation fails
        FFmpegNotFoundError: If FFmpeg is not available
    """
    # Validate FFmpeg availability once
    validate_ffmpeg_availability()
    
    # Validate visual analysis structure
    if not visual_analysis or not isinstance(visual_analysis, dict):
        raise TimelineValidationError(["Visual analysis is empty or invalid"])
    
    segments = visual_analysis.get('segments', [])
    if not segments:
        raise TimelineValidationError(["No segments found in visual analysis"])
    
    # Calculate timeline from segments
    from agent.slideshow.create_smart_video import _calculate_segment_durations
    timeline = _calculate_segment_durations(segments, audio_duration, fps=30)
    
    if not timeline:
        raise TimelineValidationError(["Timeline calculation resulted in no segments"])
    
    # Validate timeline segments
    validate_timeline_segments(timeline, all_image_paths)
    
    # Validate audio compatibility
    actual_audio_duration = validate_audio_compatibility(audio_path, audio_duration)
    
    logging.info(f"Validation passed: {len(timeline)} segments, "
                f"audio duration {actual_audio_duration:.2f}s")
    
    return timeline, actual_audio_duration


def validate_mixed_media_inputs(timeline: List[Dict[str, Any]], 
                               all_paths: Dict[str, str]) -> None:
    """
    Validate inputs for mixed media slideshow (video + images).
    
    Args:
        timeline: Timeline with video and image segments
        all_paths: Dictionary mapping source IDs to file paths
        
    Raises:
        TimelineValidationError: If validation fails
    """
    issues = []
    
    if not timeline:
        issues.append("Timeline is empty")
    
    for i, segment in enumerate(timeline):
        # Check required fields
        if 'image_source' not in segment:
            issues.append(f"Segment {i}: missing 'image_source' field")
            continue
            
        if 'start_time' not in segment or 'end_time' not in segment:
            issues.append(f"Segment {i}: missing time fields")
            continue
            
        # Validate duration
        duration = segment['end_time'] - segment['start_time']
        if duration <= 0:
            issues.append(f"Segment {i}: invalid duration {duration} (end_time - start_time must be > 0)")
        
        # Validate file existence
        image_source = segment['image_source']
        if image_source not in all_paths:
            issues.append(f"Segment {i}: image_source '{image_source}' not found in paths")
        else:
            file_path = all_paths[image_source]
            if not os.path.exists(file_path):
                issues.append(f"Segment {i}: file not found: {file_path}")
    
    if issues:
        raise TimelineValidationError(issues)
