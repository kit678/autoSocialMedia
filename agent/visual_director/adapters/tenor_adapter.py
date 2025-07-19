"""
Tenor GIF Adapter for Visual Director

This adapter integrates with Tenor's GIF API to fetch reaction GIFs
for emotional punctuation in videos.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

import aiohttp

from ..asset_types import Asset
from ..llm_intent_tagger import REACTION_TAG_MAP
from ..asset_registry import AssetAdapter


class TenorAdapter(AssetAdapter):
    """
    Adapter for Tenor GIF API.
    
    Provides access to reaction GIFs and short MP4 clips for
    emotional emphasis in video segments.
    """
    
    # Adapter metadata
    name = "tenor"
    supported_types = ["gif", "video", "reaction"]
    requires_api_key = True
    api_key_env_var = "TENOR_API_KEY"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Tenor adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        self.api_key = None
        if config and 'api_key' in config:
            self.api_key = config['api_key']
        else:
            self.api_key = os.getenv('TENOR_API_KEY')
        
        if not self.api_key:
            raise ValueError("Tenor API key required. Set TENOR_API_KEY environment variable.")
        
        self.base_url = "https://tenor.googleapis.com/v2"
        
        # Cache for popular reactions to reduce API calls
        self._reaction_cache = {}
    
    async def search(
        self, 
        query: str, 
        segment: Dict[str, Any], 
        limit: int = 10
    ) -> List[Asset]:
        """
        Search Tenor for reaction GIFs.
        
        Args:
            query: Search query string
            segment: Segment data containing context, intent, emotion, etc.
            limit: Maximum results to return
            
        Returns:
            List of Asset objects with GIF/MP4 data
        """
        emotion = segment.get('emotion')
        
        # If emotion is specified, use its search terms
        if emotion and emotion in REACTION_TAG_MAP:
            search_query = random.choice(REACTION_TAG_MAP[emotion])
        else:
            # Use provided query
            search_query = query
        
        # Check cache first
        cache_key = f"{search_query}_{limit}"
        if cache_key in self._reaction_cache:
            logging.info(f"Tenor: Using cached results for '{search_query}'")
            return self._reaction_cache[cache_key]
        
        params = {
            "q": search_query,
            "key": self.api_key,
            "limit": limit,
            "media_filter": "mp4,gif",  # Get both formats
            "ar_range": "standard",      # Prefer standard aspect ratios
            "contentfilter": "medium",   # Safe for social media
            "locale": "en_US"
        }
        
        url = f"{self.base_url}/search?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logging.error(f"Tenor API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    results = self._parse_results(data, emotion or query)
                    
                    # Cache successful results
                    if results:
                        self._reaction_cache[cache_key] = results
                    
                    return results
                    
        except Exception as e:
            logging.error(f"Tenor search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any], query: str) -> List[Asset]:
        """Parse Tenor API response into Asset objects."""
        assets = []
        
        for item in data.get('results', []):
            # Get the best quality MP4 (preferred) or GIF
            media_obj = None
            media_url = None
            media_type = "reaction"
            
            # Prefer MP4 for better quality and smaller size
            if 'mp4' in item.get('media_formats', {}):
                mp4_data = item['media_formats']['mp4']
                media_obj = mp4_data
                media_url = mp4_data.get('url')
            elif 'gif' in item.get('media_formats', {}):
                gif_data = item['media_formats']['gif']
                media_obj = gif_data
                media_url = gif_data.get('url')
            
            if not media_url or not media_obj:
                continue
            
            # Extract dimensions
            width = media_obj.get('dims', [0, 0])[0]
            height = media_obj.get('dims', [0, 0])[1]
            
            # Calculate duration (most reaction GIFs are 2-4 seconds)
            duration = media_obj.get('duration', 3.0)
            
            # Create asset
            asset = Asset(
                id=f"tenor_{item.get('id', '')}",
                url=media_url,
                type="reaction",
                source="tenor",
                licence="Tenor",  # Free with attribution
                attribution=f"Via Tenor",
                dimensions=(width, height),
                duration=duration,
                relevance_score=self._calculate_relevance(item, query),
                quality_score=0.8,  # Tenor content is generally good quality
                metadata={
                    'title': item.get('title', ''),
                    'tags': item.get('tags', []),
                    'shares': item.get('shares', 0),
                    'preview_url': item.get('media_formats', {}).get('tinygif', {}).get('url'),
                    'content_description': item.get('content_description', ''),
                    'emotion': query if query in REACTION_TAG_MAP else None
                }
            )
            
            assets.append(asset)
        
        return assets
    
    def _calculate_relevance(self, item: Dict[str, Any], query: str) -> float:
        """Calculate relevance score based on Tenor's data."""
        score = 0.5  # Base score
        
        # Boost for exact title match
        if query.lower() in item.get('title', '').lower():
            score += 0.2
        
        # Boost for tag match
        tags = [tag.lower() for tag in item.get('tags', [])]
        if query.lower() in tags:
            score += 0.15
        
        # Boost for popularity (shares)
        shares = item.get('shares', 0)
        if shares > 1000:
            score += 0.1
        elif shares > 100:
            score += 0.05
        
        # Ensure portrait orientation gets a boost for mobile videos
        dims = item.get('media_formats', {}).get('mp4', {}).get('dims', [0, 0])
        if len(dims) == 2 and dims[0] > 0 and dims[1] > 0:
            aspect_ratio = dims[0] / dims[1]
            if aspect_ratio < 1.0:  # Portrait
                score += 0.1
        
        return min(score, 1.0)
    
    async def get_trending_reactions(self, limit: int = 20) -> List[Asset]:
        """
        Get trending reaction GIFs.
        
        Args:
            limit: Number of trending GIFs to return
            
        Returns:
            List of trending reaction Assets
        """
        params = {
            "key": self.api_key,
            "limit": limit,
            "media_filter": "mp4,gif",
            "ar_range": "standard",
            "contentfilter": "medium",
            "locale": "en_US"
        }
        
        url = f"{self.base_url}/featured?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logging.error(f"Tenor trending API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data, "trending")
                    
        except Exception as e:
            logging.error(f"Tenor trending fetch failed: {e}")
            return []
    
    async def search_by_emotion(self, emotion: str, limit: int = 10) -> List[Asset]:
        """
        Search for reactions by specific emotion.
        
        Args:
            emotion: Emotion key from REACTION_TAG_MAP
            limit: Maximum results
            
        Returns:
            List of reaction Assets matching the emotion
        """
        if emotion not in REACTION_TAG_MAP:
            logging.warning(f"Unknown emotion: {emotion}")
            return []
        
        # Use random search term from the emotion's synonyms
        search_terms = REACTION_TAG_MAP[emotion]
        results = []
        
        # Try multiple search terms to get variety
        for term in search_terms[:3]:  # Try up to 3 terms
            assets = await self.search(term, {'emotion': emotion}, limit=limit // 3)
            results.extend(assets)
        
        # Deduplicate by ID
        unique_assets = {}
        for asset in results:
            if asset.id not in unique_assets:
                unique_assets[asset.id] = asset
        
        return list(unique_assets.values())[:limit]
    
    def format_attribution(self, asset: Asset) -> str:
        """
        Format attribution text for a Tenor asset.
        
        Args:
            asset: The Tenor asset
            
        Returns:
            Attribution text
        """
        return "GIF via Tenor"
    
    async def download_asset(self, asset: Asset, output_path: str) -> bool:
        """
        Download a Tenor asset to local storage.
        
        Args:
            asset: The asset to download
            output_path: Where to save the file
            
        Returns:
            True if successful
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(asset.url) as response:
                    if response.status != 200:
                        return False
                    
                    content = await response.read()
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Write file
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    
                    # Update asset's local path
                    asset.local_path = output_path
                    
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to download Tenor asset: {e}")
            return False


# Reaction overlay presets for different emotions
REACTION_OVERLAY_STYLES = {
    "mind_blown": {
        "position": "center",
        "scale": 1.2,
        "duration": 2.0,
        "fade_in": 0.2,
        "fade_out": 0.3
    },
    "facepalm": {
        "position": "bottom-right",
        "scale": 0.8,
        "duration": 1.5,
        "fade_in": 0.1,
        "fade_out": 0.2
    },
    "applause": {
        "position": "center",
        "scale": 1.0,
        "duration": 2.5,
        "fade_in": 0.3,
        "fade_out": 0.3
    },
    "shock": {
        "position": "center",
        "scale": 1.1,
        "duration": 1.8,
        "fade_in": 0.1,
        "fade_out": 0.2
    },
    "thinking": {
        "position": "top-right",
        "scale": 0.7,
        "duration": 2.0,
        "fade_in": 0.2,
        "fade_out": 0.3
    },
    "laugh": {
        "position": "center",
        "scale": 0.9,
        "duration": 2.0,
        "fade_in": 0.2,
        "fade_out": 0.2
    }
}


def get_reaction_overlay_params(emotion: str) -> Dict[str, Any]:
    """
    Get overlay parameters for a specific emotion.
    
    Args:
        emotion: The emotion type
        
    Returns:
        Dictionary with overlay styling parameters
    """
    return REACTION_OVERLAY_STYLES.get(emotion, {
        "position": "center",
        "scale": 1.0,
        "duration": 2.0,
        "fade_in": 0.2,
        "fade_out": 0.2
    })
