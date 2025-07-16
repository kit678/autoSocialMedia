"""
Video Configuration Module for AutoSocialMedia Pipeline

This module provides configuration management for video generation parameters including
orientation handling, resolution settings, and visual effects configuration.

Orientation Support:
    - Portrait (1080x1920): Vertical videos for TikTok, Instagram Reels, YouTube Shorts
    - Landscape (1920x1080): Horizontal videos for YouTube, Facebook, LinkedIn
    - Square (1080x1080): Square videos for Instagram posts, Twitter

Configuration Parameters:
    - Resolution: Width and height in pixels (must be even numbers for FFmpeg)
    - FPS: Frames per second for video generation
    - Transitions: Duration and effects for visual segment transitions
    - Ken Burns Effects: Zoom and pan settings for static images

Usage:
    >>> from agent.video_config import get_default_config
    >>> config = get_default_config("portrait")
    >>> print(f"Resolution: {config.width}x{config.height}")
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VideoConfig:
    """
    Configuration object for video generation parameters.
    
    This class encapsulates all video generation settings including resolution,
    frame rate, orientation, and visual effects parameters. It provides validation
    to ensure FFmpeg compatibility and proper video generation.
    
    Attributes:
        width (int): Video width in pixels. Must be even number for FFmpeg compatibility.
        height (int): Video height in pixels. Must be even number for FFmpeg compatibility.
        fps (int): Frames per second for video generation. Must be positive.
        orientation (str): Video orientation - "portrait", "landscape", or "square".
        transition_duration (float): Duration in seconds for transitions between segments.
        ken_burns_defaults (Dict[str, Any]): Default parameters for Ken Burns effects on static images.
    
    Example:
        >>> config = VideoConfig(
        ...     width=1920, height=1080, fps=30, orientation="landscape",
        ...     transition_duration=1.0, ken_burns_defaults={"zoom_factor": 1.2}
        ... )
        >>> config.validate()  # Raises ValueError if invalid
    """
    width: int
    height: int
    fps: int
    orientation: str
    transition_duration: float
    ken_burns_defaults: Dict[str, Any]
    
    def validate(self) -> None:
        """
        Validate video configuration parameters for FFmpeg compatibility.
        
        Ensures that all configuration parameters are valid and compatible with
        FFmpeg video generation requirements.
        
        Raises:
            ValueError: If any parameter is invalid:
                - Width or height is not even (FFmpeg requirement)
                - FPS is not positive
                - Transition duration is negative
                - Orientation is not supported
                
        Example:
            >>> config = VideoConfig(width=1081, height=1920, fps=30, 
            ...                      orientation="portrait", transition_duration=1.0,
            ...                      ken_burns_defaults={})
            >>> config.validate()  # Raises ValueError for odd width
        """
        if self.width % 2 != 0:
            raise ValueError(f"Width must be even, got {self.width}")
        if self.height % 2 != 0:
            raise ValueError(f"Height must be even, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")
        if self.transition_duration < 0:
            raise ValueError(f"Transition duration must be non-negative, got {self.transition_duration}")
        if self.orientation not in ["portrait", "landscape", "square"]:
            raise ValueError(f"Orientation must be 'portrait', 'landscape', or 'square', got {self.orientation}")


def get_default_config(orientation: str = "portrait") -> VideoConfig:
    """
    Get default video configuration based on orientation.
    
    Creates a VideoConfig object with platform-optimized settings for the specified
    orientation. Includes Ken Burns effects defaults and proper FFmpeg-compatible
    resolution settings.
    
    Args:
        orientation (str, optional): Video orientation mode. Defaults to "portrait".
            - "portrait": 1080x1920 for TikTok, Instagram Reels, YouTube Shorts
            - "landscape": 1920x1080 for YouTube, Facebook, LinkedIn
            - "square": 1080x1080 for Instagram posts, Twitter
    
    Returns:
        VideoConfig: Validated configuration object with the following defaults:
            - FPS: 30 for all orientations
            - Transition duration: 1.0 second
            - Ken Burns effects: Zoom factor 1.2, duration 3.0s, pan speed 0.5
            - Even-numbered resolutions for FFmpeg compatibility
    
    Raises:
        ValueError: If orientation is not one of the supported values
        
    Example:
        >>> # Get portrait config for TikTok-style videos
        >>> config = get_default_config("portrait")
        >>> print(f"Resolution: {config.width}x{config.height}")
        Resolution: 1080x1920
        
        >>> # Get landscape config for YouTube
        >>> config = get_default_config("landscape")
        >>> print(f"Orientation: {config.orientation}")
        Orientation: landscape
    
    Note:
        The returned configuration is pre-validated and ready for use with
        FFmpeg-based video generation. Ken Burns effects are enabled by default
        to add visual interest to static images.
    """
    default_settings = {
        "portrait": {"width": 1080, "height": 1920, "fps": 30},
        "landscape": {"width": 1920, "height": 1080, "fps": 30},
        "square": {"width": 1080, "height": 1080, "fps": 30},
    }
    
    if orientation not in default_settings:
        raise ValueError(f"Unsupported orientation: {orientation}. Must be one of {list(default_settings.keys())}")
    
    settings = default_settings[orientation]
    
    config = VideoConfig(
        width=settings["width"],
        height=settings["height"],
        fps=settings["fps"],
        orientation=orientation,
        transition_duration=1.0,
        ken_burns_defaults={
            "zoom_factor": 1.2,
            "zoom_duration": 3.0,
            "pan_speed": 0.5,
            "ease_in_out": True
        }
    )
    
    # Validate the default configuration
    config.validate()
    
    return config
