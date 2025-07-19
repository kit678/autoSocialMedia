"""
Internet Archive TV News Adapter for Visual Director

Provides access to TV news clips from the Internet Archive's TV News Archive.
Note: These clips are for research/fair use - proper usage guidelines apply.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, quote
import aiohttp
from datetime import datetime, timedelta

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class ArchiveTVAdapter(AssetAdapter):
    """
    Adapter for Internet Archive TV News Archive.
    
    Provides 60-second clips from TV news broadcasts for fair use/research.
    """
    
    name = "archive_tv"
    supported_types = ["video"]
    requires_api_key = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Archive TV adapter."""
        super().__init__(config)
        self.base_url = "https://archive.org/details/tv"
        self.api_url = "https://archive.org/advancedsearch.php"
        
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search the Internet Archive TV News Archive.
        
        Args:
            query: Search query string
            segment: Segment context
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        # Build search parameters for TV news
        # Search in TV news collection with closed captions
        search_query = f'collection:tvnews AND "{query}"'
        
        params = {
            "q": search_query,
            "fl": "identifier,title,description,date,source,creator,subject",
            "rows": limit,
            "page": 1,
            "output": "json",
            "sort": "-date"  # Most recent first
        }
        
        url = f"{self.api_url}?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Archive.org API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return await self._parse_results(data, query)
                    
        except Exception as e:
            self.logger.error(f"Archive TV search failed: {e}")
            return []
    
    async def _parse_results(self, data: Dict[str, Any], query: str) -> List[Asset]:
        """Parse Internet Archive API response into Asset objects."""
        assets = []
        
        for doc in data.get('response', {}).get('docs', []):
            identifier = doc.get('identifier')
            if not identifier:
                continue
            
            # For TV news, we need to search within the clip for the query
            # This would require additional API calls to get clip segments
            # For now, we'll create a reference to the full program
            
            # Construct clip URL (this is a simplified version)
            # In production, you'd want to search within the transcript for exact timestamps
            clip_url = f"https://archive.org/download/{identifier}/{identifier}.mp4"
            
            asset = Asset(
                id=f"archive_tv_{identifier}",
                url=clip_url,
                type="video",
                source="archive_tv",
                licence="fair_use",  # TV news clips for research/commentary
                attribution=f"{doc.get('source', 'TV News')} - {doc.get('title', '')}",
                duration=60.0,  # Default 60 second clips
                metadata={
                    'title': doc.get('title', ''),
                    'description': doc.get('description', ''),
                    'date': doc.get('date', ''),
                    'source': doc.get('source', ''),
                    'creator': doc.get('creator', ''),
                    'subjects': doc.get('subject', []),
                    'identifier': identifier,
                    'query_context': query,
                    'usage_note': 'Fair use for research/commentary only'
                }
            )
            
            # Calculate relevance based on title/description match
            relevance = self._calculate_relevance(doc, query)
            asset.relevance_score = relevance
            
            # TV news clips are generally good quality
            asset.quality_score = 0.7
            
            assets.append(asset)
        
        return assets
    
    def _calculate_relevance(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for TV news clip."""
        score = 0.5  # Base score
        
        query_lower = query.lower()
        
        # Check title match
        title = doc.get('title', '').lower()
        if query_lower in title:
            score += 0.3
        
        # Check description match
        description = doc.get('description', '').lower() if isinstance(doc.get('description'), str) else ''
        if query_lower in description:
            score += 0.2
        
        # Recent content gets a boost
        try:
            date_str = doc.get('date', '')
            if date_str:
                clip_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                days_old = (datetime.now() - clip_date).days
                if days_old < 30:
                    score += 0.1
                elif days_old < 90:
                    score += 0.05
        except:
            pass
        
        return min(score, 1.0)
    
    async def get_clip_segments(self, identifier: str, query: str) -> List[Dict[str, Any]]:
        """
        Get specific clip segments from a TV program based on caption search.
        
        This would search within closed captions to find exact timestamps
        where the query appears.
        
        Args:
            identifier: Archive.org identifier
            query: Search query
            
        Returns:
            List of clip segments with timestamps
        """
        # This is a placeholder - full implementation would search captions
        # and return specific timestamp ranges
        return [{
            'start_time': 0,
            'end_time': 60,
            'caption_text': f'Segment containing "{query}"'
        }]
