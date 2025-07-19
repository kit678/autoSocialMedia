"""
NASA Image and Video Library Adapter for Visual Director

Provides access to public domain images and videos from NASA's vast collection.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class NASAAdapter(AssetAdapter):
    """
    Adapter for NASA Image and Video Library.
    """
    
    name = "nasa"
    supported_types = ["image", "video"]
    requires_api_key = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NASA adapter."""
        super().__init__(config)
        self.base_url = "https://images-api.nasa.gov"
        
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search the NASA Image and Video Library.
        
        Args:
            query: Search query string
            segment: Segment context
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        # Build search parameters
        params = {
            "q": query,
            "media_type": ",".join(self.supported_types),
            "page": 1
        }
        
        if limit:
            params["page_size"] = limit
        
        url = f"{self.base_url}/search?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"NASA API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data)
                    
        except Exception as e:
            self.logger.error(f"NASA search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Asset]:
        """Parse NASA API response into Asset objects."""
        assets = []
        
        for item in data.get('collection', {}).get('items', []):
            # Only include images or videos
            if 'links' not in item:
                continue
            
            # Determine asset type
            href = item['links'][0]['href']
            asset_type = "video" if 'video' in item['data'][0].get('media_type', '') else "image"
            
            asset = Asset(
                id=item['data'][0]['nasa_id'],
                url=href,
                type=asset_type,
                source="nasa",
                licence='Public Domain',
                attribution=item['data'][0].get('title', 'NASA'),
                metadata={
                    'title': item['data'][0].get('title', ''),
                    'description': item['data'][0].get('description', ''),
                    'center': item['data'][0].get('center', ''),
                    'date_created': item['data'][0].get('date_created', ''),
                    'keywords': item['data'][0].get('keywords', [])
                }
            )
            
            assets.append(asset)
        
        return assets
    
    async def get_asset_info(self, nasa_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific NASA asset.
        
        Args:
            nasa_id: NASA asset ID
            
        Returns:
            Detailed asset metadata or None
        """
        url = f"{self.base_url}/asset/{nasa_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get NASA asset details: {e}")
            return None
