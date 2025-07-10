"""
Visual Gathering Module - Smart visual collection from multiple sources
"""

from .smart_visual_collector import (
    gather_visuals_for_timeline,
    process_opening_screenshot,
    determine_source_priority
)

__all__ = [
    'gather_visuals_for_timeline',
    'process_opening_screenshot', 
    'determine_source_priority'
] 