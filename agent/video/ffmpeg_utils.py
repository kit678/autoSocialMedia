import re
import subprocess
from typing import List, Optional

__all__ = ["ffmpeg_filter_exists"]

# Module level cache for detected filters
_available_filters: Optional[List[str]] = None


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

