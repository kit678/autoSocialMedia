import re
import subprocess
from typing import List, Optional
from agent.video_config import VideoConfig

__all__ = ["ffmpeg_filter_exists", "scale_crop_str", "scale_pad_str", "ken_burns_filter", "prepend_ffmpeg_path"]

# Module level cache for detected filters
_available_filters: Optional[List[str]] = None

# ---------------------------------------------------------------------------
# Public helper to dynamically adjust which ffmpeg binary is invoked
# ---------------------------------------------------------------------------

def prepend_ffmpeg_path(directory: str) -> None:
    """Pre-pend *directory* to the *PATH* so the contained *ffmpeg.exe* wins.

    The list of available filters is cleared so that the next call to
    ``ffmpeg_filter_exists`` reflects the new binary.
    """
    import os
    import logging
    global _available_filters

    if not directory:
        return

    directory = os.path.abspath(directory)
    # Only prepend if directory actually exists and contains an ffmpeg binary.
    ffmpeg_exe = os.path.join(directory, "ffmpeg.exe")
    if not os.path.isfile(ffmpeg_exe):
        logging.warning(f"FFmpeg not found at {ffmpeg_exe}")
        return

    # Pre-pend to PATH only once to avoid infinitely growing PATH.
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if directory not in path_parts:
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        _available_filters = None  # Invalidate cache so next probe uses new ffmpeg
        logging.info(f"Prepended FFmpeg path: {directory}")


def _load_available_filters() -> List[str]:
    """Load available ffmpeg filters by invoking the ffmpeg CLI once.

    Returns
    -------
    list[str]
        List of filter names detected from ffmpeg output.
    """
    global _available_filters

    if _available_filters is not None:
        return _available_filters

    try:
        # ``-v 0`` disables logging, ``-hide_banner`` removes banner information.
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "0",
                "-hide_banner",
                "-filters",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffmpeg is not present or errors, assume no filters available.
        _available_filters = []
        return _available_filters

    filters: List[str] = []
    # Parse each line for a filter name. The typical format is::
    #   T.. vflip             V->V       Flip the input video vertically.
    # where the filter name is the first word after the flag column.
    pattern = re.compile(r"^\s*[A-Z\.]*\s+([a-zA-Z0-9_]+)\s")

    for line in completed.stdout.splitlines():
        m = pattern.match(line)
        if m:
            filters.append(m.group(1))

    _available_filters = filters
    return _available_filters


def ffmpeg_filter_exists(filter_name: str) -> bool:
    """Return ``True`` if *filter_name* is reported by ``ffmpeg -filters``.

    The list of filters is cached after the first invocation to avoid the
    overhead of repeatedly spawning *ffmpeg*.
    """
    return filter_name in _load_available_filters()


def scale_crop_str(config: VideoConfig) -> str:
    """Generate standardized scale and crop filter string for FFmpeg.

    This helper provides a *single* filter string that first scales the input so
    that the **smallest** side fits the target dimension (while preserving the
    aspect ratio) and then crops the *overshoot* on the other axis so the frame
    exactly matches the requested resolution.

    * Lanczos is chosen because it offers the best quality-to-performance ratio
      for photographic images.
    * ``force_original_aspect_ratio=increase`` guarantees that we never distort
      the source.  The increased dimension will be trimmed away by the crop.

    Parameters
    ----------
    config : VideoConfig
        Target video configuration – only the ``width`` and ``height`` fields are
        used here.

    Returns
    -------
    str
        An FFmpeg filter string that can be dropped into a ``-vf`` argument.
    """
    # One *continuous* string – **not** a tuple!  Keep the comma to chain the
    # two individual filters inside FFmpeg.
    # The comma at the end of the scale filter seamlessly chains into the crop filter.
    return (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=increase:flags=lanczos,"  # scale while preserving AR
        f"crop={config.width}:{config.height}"  # then crop to exact dims
    )


def ken_burns_filter(config: VideoConfig, duration_frames: int, 
                    start_zoom: float = 1.0, end_zoom: float = 1.2,
                    start_x: str = '0', start_y: str = '0',
                    end_x: str = '0', end_y: str = '0') -> str:
    """Generate Ken Burns effect filter string with standardized scaling.
    
    Combines the scale_crop_str with zoompan to create smooth Ken Burns effects
    with consistent aspect-ratio handling.
    
    Args:
        config: VideoConfig containing target dimensions and fps
        duration_frames: Duration of the effect in frames
        start_zoom: Initial zoom level (default: 1.0)
        end_zoom: Final zoom level (default: 1.2)
        start_x: Initial X position offset (default: '0')
        start_y: Initial Y position offset (default: '0')
        end_x: Final X position offset (default: '0')
        end_y: Final Y position offset (default: '0')
        
    Returns:
        str: Complete FFmpeg filter string for Ken Burns effect
        
    Example:
        >>> config = VideoConfig(width=1920, height=1080, fps=30, ...)
        >>> ken_burns_filter(config, 90, start_zoom=1.0, end_zoom=1.3)
        'scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,zoompan=z=...'
    """
    # Start with the standardised scale/crop so *all* sources adhere to the
    # expected output aspect ratio *before* the Ken-Burns movement is applied.
    base_filter = scale_crop_str(config)

    # Build frame-accurate progression expressions.  FFmpeg evaluates these once
    # per output frame with the variable ``on`` holding the 0-based frame index.
    zoom_expr = (
        f"if(lte(on\,1)\,{start_zoom}\,"  # hold start value for the first frame
        f"{start_zoom}+({end_zoom}-{start_zoom})*on/{duration_frames})"
    )
    x_expr = (
        f"if(lte(on\,1)\,{start_x}\,"  # keep start offset for first frame
        f"{start_x}+({end_x}-{start_x})*on/{duration_frames})"
    )
    y_expr = (
        f"if(lte(on\,1)\,{start_y}\,"  # keep start offset for first frame
        f"{start_y}+({end_y}-{start_y})*on/{duration_frames})"
    )

    zoompan_filter = (
        "zoompan="
        f"z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={duration_frames}:s={config.width}x{config.height}:fps={config.fps}"
    )

    # Concatenate the two parts – they are already comma-separated internally.
    return f"{base_filter},{zoompan_filter}"

# -----------------------------------------------------------------------------
# NEW helper: scale_pad_str
# -----------------------------------------------------------------------------

def scale_pad_str(config: VideoConfig) -> str:
    """Return a filter that letter-boxes any input into the landscape frame.

    Usage is identical to *scale_crop_str* but without cropping. Portrait clips
    are pillar-boxed, ultra-wide are letter-boxed. Keeps full content visible –
    ideal for webpage captures.
    """
    w, h = config.width, config.height
    ar_expr = f"{w}/{h}"
    scale = (
        f"scale='if(gt(iw/ih,{ar_expr}),{w},-1)':'if(gt(iw/ih,{ar_expr}),-1,{h})'"
    )
    pad   = f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    return f"{scale},{pad}"

