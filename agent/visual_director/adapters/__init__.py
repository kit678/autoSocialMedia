"""Visual asset adapters for various sources.

This module contains adapters for fetching visual content from different sources
like SearXNG, Pexels, Openverse, Wikimedia, Tenor, etc.
"""

from .searxng_adapter import SearXNGAdapter
from .pexels_adapter import PexelsAdapter
from .tenor_adapter import TenorAdapter
from .openverse_adapter import OpenverseAdapter
from .wikimedia_adapter import WikimediaAdapter
from .nasa_adapter import NASAAdapter
from .archive_tv_adapter import ArchiveTVAdapter
from .gdelt_adapter import GDELTAdapter
from .coverr_adapter import CoverrAdapter

__all__ = [
    'SearXNGAdapter',
    'PexelsAdapter',
    'TenorAdapter',
    'OpenverseAdapter',
    'WikimediaAdapter',
    'NASAAdapter',
    'ArchiveTVAdapter',
    'GDELTAdapter',
    'CoverrAdapter',
]

# Register all adapters when module is imported
def register_all_adapters():
    """Register all available adapters with the registry."""
    from ..asset_registry import register_adapter
    
    # Register each adapter
    register_adapter('searxng', SearXNGAdapter)
    register_adapter('pexels', PexelsAdapter)
    register_adapter('tenor', TenorAdapter)
    
    register_adapter('openverse', OpenverseAdapter)
    register_adapter('wikimedia', WikimediaAdapter)
    register_adapter('nasa', NASAAdapter)
    register_adapter('archive_tv', ArchiveTVAdapter)
    register_adapter('gdelt', GDELTAdapter)
    register_adapter('coverr', CoverrAdapter)
