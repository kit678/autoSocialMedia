"""
Coverr Adapter for Visual Director

Provides access to free stock video loops from Coverr.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class CoverrAdapter(AssetAdapter):
    """
    Adapter for Coverr stock videos.
    
    Provides high-quality stock video loops that are free to use (CC0).
    Note: Coverr API requires an API key.
    """
    
    name = "coverr"
    supported_types = ["video"]
    requires_api_key = True
    api_key_env_var = "COVERR_API_KEY"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Coverr adapter."""
        super().__init__(config)
        self.base_url = "https://api.coverr.co/v1"
        self.api_key = config.get('api_key') if config else None
        
        if not self.api_key:
            import os
            self.api_key = os.getenv(self.api_key_env_var)
    
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search Coverr for stock video loops.
        
        Args:
            query: Search query string
            segment: Segment context
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        if not self.api_key:
            self.logger.warning("Coverr API key not configured")
            return []
        
        # Build search parameters
        params = {
            "query": query,
            "page": 1,
            "per_page": limit,
            "api_key": self.api_key
        }
        
        # Add filters based on segment needs
        visual_type = segment.get('visual_type', '')
        if visual_type == 'background':
            params['category'] = 'backgrounds'
        elif visual_type == 'nature':
            params['category'] = 'nature'
        elif visual_type == 'technology':
            params['category'] = 'tech'
        
        url = f"{self.base_url}/videos/search?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Coverr API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data)
                    
        except Exception as e:
            self.logger.error(f"Coverr search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Asset]:
        """Parse Coverr API response into Asset objects."""
        assets = []
        
        for video in data.get('videos', []):
            # Get the best quality URL
            urls = video.get('urls', {})
            video_url = urls.get('mp4_1080p') or urls.get('mp4_720p') or urls.get('mp4_480p')
            
            if not video_url:
                continue
            
            # Extract dimensions from URL quality
            width, height = self._extract_dimensions(urls)
            
            asset = Asset(
                id=f"coverr_{video.get('id', '')}",
                url=video_url,
                type="video",
                source="coverr",
                licence="CC0",  # Coverr videos are CC0
                attribution=f"Video by {video.get('author', 'Coverr')}",
                dimensions=(width, height),
                duration=video.get('duration', 10.0),
                relevance_score=self._calculate_relevance(video, query),
                quality_score=0.9,  # Coverr videos are high quality
                metadata={
                    'title': video.get('title', ''),
                    'description': video.get('description', ''),
                    'author': video.get('author', ''),
                    'tags': video.get('tags', []),
                    'categories': video.get('categories', []),
                    'thumbnail': video.get('thumbnail', ''),
                    'preview_url': video.get('preview_url', ''),
                    'loop': True,  # Coverr videos are designed to loop
                    'urls': urls
                }
            )
            
            assets.append(asset)
        
        return assets
    
    def _extract_dimensions(self, urls: Dict[str, str]) -> tuple:
        """Extract video dimensions from available URLs."""
        if 'mp4_1080p' in urls:
            return (1920, 1080)
        elif 'mp4_720p' in urls:
            return (1280, 720)
        elif 'mp4_480p' in urls:
            return (854, 480)
        else:
            return (0, 0)
    
    def _calculate_relevance(self, video: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for Coverr video."""
        score = 0.5  # Base score
        
        query_lower = query.lower()
        
        # Check title match
        title = video.get('title', '').lower()
        if query_lower in title:
            score += 0.3
        
        # Check description match
        description = video.get('description', '').lower()
        if query_lower in description:
            score += 0.2
        
        # Check tag matches
        tags = [tag.lower() for tag in video.get('tags', [])]
        matching_tags = sum(1 for tag in tags if query_lower in tag or tag in query_lower)
        if matching_tags:
            score += min(0.2, matching_tags * 0.05)
        
        # Check category matches
        categories = [cat.lower() for cat in video.get('categories', [])]
        if any(query_lower in cat or cat in query_lower for cat in categories):
            score += 0.1
        
        return min(score, 1.0)
    
    async def get_collections(self) -> List[Dict[str, Any]]:
        """
        Get available collections from Coverr.
        
        Returns:
            List of collection metadata
        """
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/collections?api_key={self.api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('collections', [])
                    return []
        except Exception as e:
            self.logger.error(f"Failed to get collections: {e}")
            return []
