"""media_utils.py
Utility functions for inspecting basic media properties (width, height, aspect ratio).

We want a very light-weight solution without bringing in heavy dependencies that
need compiled binaries.  Pillow is already a transitive dependency elsewhere in
this project, so we use it for images.  For video we rely on ffprobe which is
expected to be present because the pipeline already depends on FFmpeg.

The helpers never raise on failure – they return (None, None) so callers can
handle missing metadata gracefully.
"""

from __future__ import annotations

import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

try:
    from PIL import Image  # Pillow is available in the project
except ImportError:  # pragma: no cover – Pillow should be installed
    Image = None  # type: ignore

import subprocess
from agent.video_config import VideoConfig, get_default_config

__all__ = [
    "get_media_dimensions",
    "get_orientation",
    "get_aspect_ratio_str",
]

def _dimensions_from_image(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception as exc:  # pragma: no cover – corrupted images, etc.
        logging.debug(f"Failed to read image dimensions for {path}: {exc}")
        return None, None

def _dimensions_from_video(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Use ffprobe to fetch the first video stream's width/height."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None, None
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            w = streams[0].get("width")
            h = streams[0].get("height")
            return int(w) if w else None, int(h) if h else None
    except Exception as exc:  # pragma: no cover
        logging.debug(f"ffprobe failed for {path}: {exc}")
    return None, None

def get_media_dimensions(file_path: str | Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (width, height) of an image or video, or (None, None) if unknown."""
    path = Path(file_path)
    if not path.exists():
        return None, None

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        return _dimensions_from_image(path)
    elif suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return _dimensions_from_video(path)
    else:
        # Try image first, fall back to video – vague extension (e.g., none)
        w, h = _dimensions_from_image(path)
        if w is not None and h is not None:
            return w, h
        return _dimensions_from_video(path)

def get_orientation(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if width is None or height is None:
        return None
    if height > width:
        return "portrait"
    elif width > height:
        return "landscape"
    else:
        return "square"

def get_aspect_ratio_str(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if width is None or height is None or width == 0 or height == 0:
        return None
    from math import gcd

    g = gcd(width, height)
    return f"{width // g}:{height // g}"


def standardize_image(image_path: str, target_width: int = 1080, target_height: int = 1920) -> bool:
    try:
        with Image.open(image_path) as img:
            target_ratio = target_width / target_height
            current_ratio = img.width / img.height
            if current_ratio > target_ratio:
                new_width = int(target_ratio * img.height)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            elif current_ratio < target_ratio:
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))
            img = img.resize((target_width, target_height), Image.LANCZOS)
            img.save(image_path)
            return True
    except Exception as e:
        logging.error(f"Failed to standardize {image_path}: {e}")
        return False


def standardize_video(video_path: str, target_width: int = 1080, target_height: int = 1920, config: VideoConfig = None) -> bool:
    if config is None:
        config = get_default_config()
    try:
        temp_path = video_path + '.temp.mp4'
        scale_crop = scale_crop_str(config)
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', scale_crop, '-c:a', 'copy', temp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.replace(temp_path, video_path)
            return True
        else:
            logging.error(f"FFmpeg error: {result.stderr}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    except Exception as e:
        logging.error(f"Failed to standardize video {video_path}: {e}")
        return False

