"""
GDELT TV 2.0 Adapter for Visual Director

Provides access to frame-level search of global TV news via GDELT's TV API.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp
from datetime import datetime, timedelta

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class GDELTAdapter(AssetAdapter):
    """
    Adapter for GDELT TV 2.0 API.
    
    Provides frame-level search across global TV news broadcasts with
    AI-tagged content and entity recognition.
    """
    
    name = "gdelt"
    supported_types = ["video"]
    requires_api_key = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GDELT adapter."""
        super().__init__(config)
        self.base_url = "https://api.gdeltproject.org/api/v2/tv/tv"
        
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search GDELT TV for video clips.
        
        Args:
            query: Search query string
            segment: Segment context
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        # Build search parameters
        params = {
            "query": query,
            "mode": "clipgallery",  # Get video clips
            "maxrecords": limit,
            "sort": "date",
            "format": "json"
        }
        
        # Add timespan filter (last 30 days by default)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
        params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")
        
        url = f"{self.base_url}?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"GDELT API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data)
                    
        except Exception as e:
            self.logger.error(f"GDELT search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Asset]:
        """Parse GDELT API response into Asset objects."""
        assets = []
        
        clips = data.get('clips', [])
        for clip in clips:
            # GDELT provides frame-level data with timestamps
            preview_url = clip.get('preview_url', '')
            video_url = clip.get('video_url', '')
            
            if not video_url:
                continue
            
            # Extract metadata
            station = clip.get('station', 'Unknown')
            show = clip.get('show', '')
            date = clip.get('date', '')
            
            # Create asset
            asset = Asset(
                id=f"gdelt_{clip.get('id', '')}",
                url=video_url,
                type="video",
                source="gdelt",
                licence="fair_use",  # TV news for research
                attribution=f"{station} - {show}",
                duration=clip.get('duration', 15.0),  # GDELT clips are usually short
                metadata={
                    'title': clip.get('snippet', ''),
                    'station': station,
                    'show': show,
                    'date': date,
                    'preview_url': preview_url,
                    'start_offset': clip.get('start_offset', 0),
                    'end_offset': clip.get('end_offset', 0),
                    'matched_terms': clip.get('matched_terms', []),
                    'entities': clip.get('entities', []),
                    'themes': clip.get('themes', []),
                    'usage_note': 'Fair use for research/commentary only'
                }
            )
            
            # Calculate scores
            asset.relevance_score = self._calculate_relevance(clip)
            asset.quality_score = 0.75  # GDELT content is generally broadcast quality
            
            assets.append(asset)
        
        return assets
    
    def _calculate_relevance(self, clip: Dict[str, Any]) -> float:
        """Calculate relevance score for GDELT clip."""
        score = 0.5  # Base score
        
        # Boost for matched terms
        matched_terms = clip.get('matched_terms', [])
        if matched_terms:
            score += min(0.2, len(matched_terms) * 0.05)
        
        # Boost for entities
        entities = clip.get('entities', [])
        if entities:
            score += min(0.15, len(entities) * 0.03)
        
        # Boost for themes
        themes = clip.get('themes', [])
        if themes:
            score += min(0.1, len(themes) * 0.02)
        
        # Boost for recent content
        try:
            date_str = clip.get('date', '')
            if date_str:
                clip_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                days_old = (datetime.now() - clip_date).days
                if days_old < 7:
                    score += 0.15
                elif days_old < 30:
                    score += 0.1
        except:
            pass
        
        return min(score, 1.0)
    
    async def get_show_context(self, show_id: str) -> Optional[Dict[str, Any]]:
        """
        Get additional context about a specific TV show.
        
        Args:
            show_id: GDELT show identifier
            
        Returns:
            Show metadata or None
        """
        # This would fetch additional show context from GDELT
        # Placeholder for now
        return None
