"""Asset type definitions for the visual director.

This module defines the core data structures for managing visual assets
(images, videos, reactions) throughout the visual acquisition pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, Any
from enum import Enum


@dataclass
class Asset:
    """Represents a visual asset (image, video, or reaction GIF/MP4).
    
    This is the primary data structure for tracking visual content throughout
    the acquisition, scoring, and selection process.
    
    Attributes:
        id: Unique identifier for this asset (typically source_assetid format)
        url: Original source URL where the asset was found
        type: Type of asset - image, video, or reaction
        source: Name of the adapter that found this asset (e.g., 'searxng', 'pexels')
        licence: License type (e.g., 'CC0', 'CC-BY', 'Pexels', 'editorial')
        attribution: Required attribution text if any
        dimensions: Width, height tuple in pixels
        duration: Duration in seconds for video assets
        relevance_score: How well this matches the search intent (0-1)
        quality_score: Technical quality assessment (0-1)
        diversity_score: How different from other selected assets (0-1)
        local_path: Path to downloaded file if cached locally
        metadata: Additional source-specific metadata
    """
    id: str
    url: str
    type: Literal["image", "video", "reaction"]
    source: str
    licence: str
    attribution: Optional[str] = None
    dimensions: Tuple[int, int] = (0, 0)
    duration: Optional[float] = None
    relevance_score: float = 0.0
    quality_score: float = 0.0
    diversity_score: float = 0.0
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _composite_score: float = 0.0  # Internal field for storing calculated composite score
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        if self.dimensions[0] > 0 and self.dimensions[1] > 0:
            return self.dimensions[0] / self.dimensions[1]
        return 0.0
    
    @property
    def is_portrait(self) -> bool:
        """Check if asset is portrait orientation."""
        return self.aspect_ratio < 1.0 if self.aspect_ratio > 0 else False
    
    @property
    def is_landscape(self) -> bool:
        """Check if asset is landscape orientation."""
        return self.aspect_ratio > 1.0 if self.aspect_ratio > 0 else False
    
    @property
    def composite_score(self) -> float:
        """Get the calculated composite score.
        
        If not set externally, calculates a default weighted score.
        """
        if self._composite_score > 0:
            return self._composite_score
        
        # Default weights - can be overridden by config
        weights = {
            'relevance': 0.5,
            'quality': 0.3,
            'diversity': 0.2
        }
        return (
            self.relevance_score * weights['relevance'] +
            self.quality_score * weights['quality'] +
            self.diversity_score * weights['diversity']
        )
    
    @composite_score.setter
    def composite_score(self, value: float) -> None:
        """Set the composite score."""
        self._composite_score = value
    
    def requires_attribution(self) -> bool:
        """Check if this asset requires attribution."""
        return bool(self.attribution) or self.licence in ['CC-BY', 'CC-BY-SA', 'editorial']


@dataclass
class AssetSearchResult:
    """Results from an asset search operation.
    
    Attributes:
        query: The search query used
        segment: The segment data that triggered this search
        assets: List of found assets
        adapter_name: Name of the adapter that performed the search
        search_time: Time taken for the search in seconds
        error: Error message if search failed
    """
    query: str
    segment: Dict[str, Any]
    assets: list[Asset] = field(default_factory=list)
    adapter_name: str = ""
    search_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return self.error is None and len(self.assets) > 0


@dataclass
class AssetScoringConfig:
    """Configuration for asset scoring weights and parameters.
    
    Attributes:
        relevance_weight: Weight for relevance score (0-1)
        quality_weight: Weight for quality score (0-1)
        diversity_weight: Weight for diversity score (0-1)
        prefer_portrait: Bonus score for portrait orientation
        prefer_video: Bonus score for video over images
        license_penalties: Score penalties for different license types
        source_bonuses: Score bonuses for preferred sources
    """
    relevance_weight: float = 0.5
    quality_weight: float = 0.3
    diversity_weight: float = 0.2
    prefer_portrait: float = 0.1
    prefer_video: float = 0.05
    license_penalties: Dict[str, float] = field(default_factory=lambda: {
        'editorial': 0.2,
        'CC-BY-SA': 0.1,
        'unknown': 0.3
    })
    source_bonuses: Dict[str, float] = field(default_factory=lambda: {
        'pexels': 0.1,
        'openverse': 0.05,
        'wikimedia': 0.05
    })


# License type constants
LICENSE_TYPES = {
    'CC0': 'Creative Commons Zero - No rights reserved',
    'CC-BY': 'Creative Commons Attribution',
    'CC-BY-SA': 'Creative Commons Attribution-ShareAlike',
    'Pexels': 'Pexels License - Free to use',
    'Unsplash': 'Unsplash License - Free to use',
    'Pixabay': 'Pixabay License - Free to use',
    'editorial': 'Editorial use only - may have restrictions',
    'commercial': 'Commercial license - check terms',
    'unknown': 'Unknown license - use with caution'
}

# Media type constants  
MEDIA_TYPES = {
    'image': ['jpg', 'jpeg', 'png', 'webp', 'gif'],
    'video': ['mp4', 'mov', 'webm', 'avi'],
    'reaction': ['gif', 'mp4']  # Reactions can be either format
}

# Asset source constants
class AssetSource(Enum):
    """Available asset sources/adapters."""
    SEARXNG = "searxng"
    PEXELS = "pexels"
    TENOR = "tenor"
    OPENVERSE = "openverse"
    WIKIMEDIA = "wikimedia"
    NASA = "nasa"
    ARCHIVE_TV = "archive_tv"
    COVERR = "coverr"
    GDELT = "gdelt"
    
    @classmethod
    def get_display_name(cls, source: str) -> str:
        """Get display name for a source."""
        display_names = {
            cls.SEARXNG.value: "SearXNG",
            cls.PEXELS.value: "Pexels",
            cls.TENOR.value: "Tenor",
            cls.OPENVERSE.value: "Openverse",
            cls.WIKIMEDIA.value: "Wikimedia Commons",
            cls.NASA.value: "NASA Image Gallery",
            cls.ARCHIVE_TV.value: "Archive.tv",
            cls.COVERR.value: "Coverr",
            cls.GDELT.value: "GDELT"
        }
        return display_names.get(source, source.title())
