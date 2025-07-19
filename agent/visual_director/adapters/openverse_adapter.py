"""
Openverse Adapter for Visual Director

This adapter integrates with Openverse API to fetch Creative Commons
licensed images with proper attribution metadata.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class OpenverseAdapter(AssetAdapter):
    """
    Adapter for Openverse API (formerly CC Search).
    
    Provides access to millions of CC-licensed images from various sources
    including museums, libraries, and other cultural institutions.
    """
    
    name = "openverse"
    supported_types = ["image"]
    requires_api_key = False  # Public API, but rate limited
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Openverse adapter."""
        super().__init__(config)
        self.base_url = "https://api.openverse.engineering/v1"
        self.headers = {
            "User-Agent": "AutoSocialMedia/1.0"
        }
        
        # Optional API key for higher rate limits
        self.api_key = config.get('api_key') if config else None
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search Openverse for CC-licensed images.
        
        Args:
            query: Search query string
            segment: Segment context (for filtering by license needs)
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        # Build search parameters
        params = {
            "q": query,
            "page_size": min(limit, 20),  # Openverse max is 20 per page
            "mature": "false"  # Safe for social media
        }
        
        # Filter by license if needed
        license_requirement = segment.get('licence_requirement', 'any')
        if license_requirement == 'commercial':
            params["license_type"] = "commercial"
        elif license_requirement == 'modification':
            params["license_type"] = "modification"
        
        # Add category filter based on segment type
        if segment.get('visual_type') == 'photograph':
            params["category"] = "photograph"
        elif segment.get('visual_type') == 'illustration':
            params["category"] = "illustration"
        
        url = f"{self.base_url}/images/?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Openverse API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data)
                    
        except Exception as e:
            self.logger.error(f"Openverse search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Asset]:
        """Parse Openverse API response into Asset objects."""
        assets = []
        
        for item in data.get('results', []):
            # Extract dimensions
            width = item.get('width', 0)
            height = item.get('height', 0)
            
            # Map Openverse license to our format
            license_info = self._map_license(item.get('license', 'unknown'))
            
            # Build attribution
            attribution = self._build_attribution(item)
            
            asset = Asset(
                id=f"openverse_{item.get('id', '')}",
                url=item.get('url', ''),
                type="image",
                source="openverse",
                licence=license_info['code'],
                attribution=attribution,
                dimensions=(width, height),
                relevance_score=self._calculate_relevance(item),
                quality_score=self._calculate_quality(item),
                metadata={
                    'title': item.get('title', ''),
                    'creator': item.get('creator', ''),
                    'creator_url': item.get('creator_url', ''),
                    'foreign_landing_url': item.get('foreign_landing_url', ''),
                    'detail_url': item.get('detail_url', ''),
                    'related_url': item.get('related_url', ''),
                    'tags': [tag['name'] for tag in item.get('tags', [])],
                    'source': item.get('source', ''),
                    'license_version': item.get('license_version', ''),
                    'license_url': item.get('license_url', ''),
                    'preview_url': item.get('thumbnail', ''),
                }
            )
            
            assets.append(asset)
        
        return assets
    
    def _map_license(self, license_code: str) -> Dict[str, str]:
        """Map Openverse license codes to our format."""
        license_map = {
            'cc0': {'code': 'CC0', 'name': 'Creative Commons Zero'},
            'pdm': {'code': 'CC0', 'name': 'Public Domain Mark'},
            'by': {'code': 'CC-BY', 'name': 'Creative Commons Attribution'},
            'by-sa': {'code': 'CC-BY-SA', 'name': 'Creative Commons Attribution-ShareAlike'},
            'by-nc': {'code': 'CC-BY-NC', 'name': 'Creative Commons Attribution-NonCommercial'},
            'by-nd': {'code': 'CC-BY-ND', 'name': 'Creative Commons Attribution-NoDerivatives'},
            'by-nc-sa': {'code': 'CC-BY-NC-SA', 'name': 'CC Attribution-NonCommercial-ShareAlike'},
            'by-nc-nd': {'code': 'CC-BY-NC-ND', 'name': 'CC Attribution-NonCommercial-NoDerivatives'},
        }
        
        return license_map.get(license_code.lower(), {'code': 'unknown', 'name': 'Unknown License'})
    
    def _build_attribution(self, item: Dict[str, Any]) -> str:
        """Build proper attribution text for an Openverse image."""
        creator = item.get('creator', 'Unknown')
        title = item.get('title', 'Untitled')
        source = item.get('source', '')
        license_code = item.get('license', '').upper()
        license_version = item.get('license_version', '')
        
        # Format: "Title" by Creator via Source is licensed under CC BY 4.0
        attribution_parts = []
        
        if title and title != 'Untitled':
            attribution_parts.append(f'"{title}"')
        
        attribution_parts.append(f"by {creator}")
        
        if source:
            attribution_parts.append(f"via {source}")
        
        if license_code and license_code != 'UNKNOWN':
            license_str = f"CC {license_code.replace('CC-', '')}"
            if license_version:
                license_str += f" {license_version}"
            attribution_parts.append(f"is licensed under {license_str}")
        
        return " ".join(attribution_parts)
    
    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        """Calculate relevance score based on Openverse metadata."""
        score = 0.5  # Base score
        
        # Boost for having tags
        tags = item.get('tags', [])
        if tags:
            score += min(0.2, len(tags) * 0.02)
        
        # Boost for having a title
        if item.get('title') and item['title'] != 'Untitled':
            score += 0.1
        
        # Boost for having creator info
        if item.get('creator'):
            score += 0.1
        
        # Slight penalty for very generic sources
        source = item.get('source', '').lower()
        if source in ['flickr', 'wikimedia']:
            score += 0.05  # Common but good sources
        elif source in ['rawpixel', 'stockvault']:
            score -= 0.05  # More generic stock photo sites
        
        return min(score, 1.0)
    
    def _calculate_quality(self, item: Dict[str, Any]) -> float:
        """Calculate quality score based on image properties."""
        score = 0.5  # Base score
        
        # Resolution quality
        width = item.get('width', 0)
        height = item.get('height', 0)
        
        if width >= 1920 and height >= 1080:
            score += 0.3  # HD or better
        elif width >= 1280 and height >= 720:
            score += 0.2  # Good resolution
        elif width >= 640 and height >= 480:
            score += 0.1  # Acceptable
        else:
            score -= 0.1  # Low resolution
        
        # Aspect ratio quality (prefer standard ratios)
        if width > 0 and height > 0:
            ratio = width / height
            if 0.5 <= ratio <= 0.7:  # Portrait (9:16 range)
                score += 0.15
            elif 1.7 <= ratio <= 1.8:  # Landscape (16:9 range)
                score += 0.1
            elif 0.9 <= ratio <= 1.1:  # Square
                score += 0.05
        
        # Source quality
        trusted_sources = ['museums', 'met', 'smithsonian', 'europeana', 'brooklynmuseum']
        if any(source in item.get('source', '').lower() for source in trusted_sources):
            score += 0.15
        
        return min(score, 1.0)
    
    async def get_image_details(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific image.
        
        Args:
            image_id: Openverse image ID
            
        Returns:
            Detailed image metadata or None
        """
        url = f"{self.base_url}/images/{image_id}/"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get image details: {e}")
            return None
