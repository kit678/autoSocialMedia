"""SearXNG adapter for searching Google Images.

This adapter uses SearXNG instances to search for images without API limits.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from urllib.parse import quote

from ..asset_registry import AssetAdapter
from ..asset_types import Asset


class SearXNGAdapter(AssetAdapter):
    """Adapter for searching images via SearXNG (Google Images proxy)."""
    
    name = "searxng"
    supported_types = ["image"]
    requires_api_key = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SearXNG adapter.
        
        Args:
            config: Optional configuration with searxng_url
        """
        super().__init__(config)
        self.searxng_url = config.get('searxng_url') if config else None
        if not self.searxng_url:
            self.searxng_url = os.getenv('SEARXNG_URL')
    
    def is_available(self) -> bool:
        """Check if SearXNG is configured."""
        return bool(self.searxng_url)
    
    async def search(
        self, 
        query: str, 
        segment: Dict[str, Any], 
        limit: int = 10
    ) -> List[Asset]:
        """Search for images using SearXNG.
        
        Args:
            query: Search query
            segment: Segment context with intent, emotion, etc.
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        if not self.is_available():
            self.logger.warning("SearXNG URL not configured")
            return []
        
        try:
            # Enrich query based on segment context
            enriched_query = self._enrich_query(query, segment)
            
            # Build search parameters
            params = {
                'q': enriched_query,
                'categories': 'images',
                'format': 'json',
                'safesearch': 1,
                'engines': 'google images',
                'tbs': 'iar:t'  # Filter for tall/portrait images
            }
            
            self.logger.info(f"Searching SearXNG for: '{enriched_query}'")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.searxng_url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            results = data.get('results', [])[:limit]
            assets = []
            
            for i, result in enumerate(results):
                img_url = result.get('img_src')
                if not img_url:
                    continue
                
                # Extract metadata
                title = result.get('title', '')
                source_url = result.get('url', '')
                thumbnail = result.get('thumbnail_src', '')
                
                # Create asset
                asset = Asset(
                    id=f"searxng_{hash(img_url)}",
                    url=img_url,
                    type="image",
                    source="searxng",
                    licence="unknown",  # SearXNG doesn't provide license info
                    dimensions=(0, 0),  # Will be determined on download
                    relevance_score=self._calculate_relevance(i, len(results)),
                    quality_score=0.7,  # Default quality score
                    metadata={
                        'title': title,
                        'source_url': source_url,
                        'thumbnail': thumbnail,
                        'search_query': enriched_query,
                        'original_query': query
                    }
                )
                
                assets.append(asset)
            
            self.logger.info(f"Found {len(assets)} images from SearXNG")
            return assets
            
        except asyncio.TimeoutError:
            self.logger.error("SearXNG request timed out")
            return []
        except Exception as e:
            self.logger.error(f"SearXNG search failed: {e}")
            return []
    
    def _enrich_query(self, query: str, segment: Dict[str, Any]) -> str:
        """Enrich search query based on segment context.
        
        Args:
            query: Base search query
            segment: Segment with context
            
        Returns:
            Enriched query string
        """
        # Start with quoted exact query and add portrait/vertical preference
        enriched = f'"{query}" portrait vertical mobile'
        
        # Add intent-based keywords
        intent = segment.get('intent', 'inform')
        if intent == 'excite':
            enriched += ' exciting amazing breakthrough'
        elif intent == 'warn':
            enriched += ' warning danger risk alert'
        elif intent == 'explain':
            enriched += ' diagram illustration explanation'
        elif intent == 'celebrate':
            enriched += ' celebration success achievement'
        
        # Add emotion-based modifiers
        emotion = segment.get('emotion', 'neutral')
        if emotion == 'happy':
            enriched += ' positive bright colorful'
        elif emotion == 'concerned':
            enriched += ' serious concern worried'
        elif emotion == 'surprised':
            enriched += ' surprising unexpected shocking'
        
        # Add entity names if available
        entities = segment.get('entities', [])
        if entities:
            # Add first 2 entities to avoid query being too long
            enriched += ' ' + ' '.join(entities[:2])
        
        # Add context keywords from narrative
        narrative = segment.get('narrative_context', '')
        if narrative:
            # Extract key nouns/verbs from narrative
            words = narrative.lower().split()
            keywords = [w for w in words if len(w) > 4 and w.isalpha()][:3]
            if keywords:
                enriched += ' ' + ' '.join(keywords)
        
        return enriched
    
    def _calculate_relevance(self, position: int, total: int) -> float:
        """Calculate relevance score based on search result position.
        
        Args:
            position: 0-based position in results
            total: Total number of results
            
        Returns:
            Relevance score between 0 and 1
        """
        if total == 0:
            return 0.0
        
        # Linear decay from 1.0 to 0.5 based on position
        return 1.0 - (position / total) * 0.5
