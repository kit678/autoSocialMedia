"""Visual Director module for automated visual asset acquisition.

This module provides a sophisticated system for finding, scoring, and selecting
visual assets (images, videos, reactions) for social media video content.
"""

from .asset_types import Asset, AssetSearchResult, AssetScoringConfig
from .asset_registry import AssetAdapter, AssetAdapterRegistry, get_registry, initialize_adapters
from .asset_scorer import score_asset, rank_assets, select_best_asset
from .cache_manager import AssetCacheManager, get_cache_manager
from .llm_tagger import tag_segments_with_intent

__all__ = [
    # Types
    'Asset',
    'AssetSearchResult', 
    'AssetScoringConfig',
    
    # Registry
    'AssetAdapter',
    'AssetAdapterRegistry',
    'get_registry',
    'initialize_adapters',
    
    # Scoring
    'score_asset',
    'rank_assets',
    'select_best_asset',
    
    # Cache
    'AssetCacheManager',
    'get_cache_manager',
    
    # Tagging
    'tag_segments_with_intent',
]
