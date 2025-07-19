"""
Wikimedia Commons Adapter for Visual Director

Provides access to freely licensed images and some videos from Wikimedia Commons via the MediaWiki API.
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp

from ..asset_types import Asset
from ..asset_registry import AssetAdapter


class WikimediaAdapter(AssetAdapter):
    """
    Adapter for Wikimedia Commons using the MediaWiki API.
    
    Offers access to a vast repository of free images and multimedia content.
    """
    
    name = "wikimedia"
    supported_types = ["image", "video"]
    requires_api_key = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Wikimedia adapter."""
        super().__init__(config)
        self.base_url = "https://commons.wikimedia.org/w/api.php"
        
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """
        Search Wikimedia Commons via the MediaWiki API.
        
        Args:
            query: Search query string
            segment: Segment context
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        # Build search parameters
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo|description",
            "iiprop": "url|size|mime|metadata",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": min(limit, 50)  # Wikimedia API max 50 per request
        }
        
        url = f"{self.base_url}?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Wikimedia API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data)
                    
        except Exception as e:
            self.logger.error(f"Wikimedia search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Asset]:
        """Parse Wikimedia Commons API response into Asset objects."""
        assets = []
        
        for pageid, page in data.get('query', {}).get('pages', {}).items():
            imageinfo = page.get('imageinfo', [{}])[0]
            if not imageinfo:
                continue
            
            # Only include images or videos
            mime = imageinfo.get('mime', '')
            if not mime.startswith(('image/', 'video/')):
                continue
            
            # Determine asset type
            asset_type = "video" if mime.startswith('video/') else "image"
            
            # Create asset
            asset = Asset(
                id=f"wikimedia_{pageid}",
                url=imageinfo.get('url', ''),
                type=asset_type,
                source="wikimedia",
                licence='CC-BY-SA',  # Most commons content is under CC-BY-SA
                attribution=page.get('title', ''),
                dimensions=(imageinfo.get('thumbwidth', 0), imageinfo.get('thumbheight', 0)),
                metadata={
                    'title': page.get('title', ''),
                    'mime': mime,
                    'canonicalurl': page.get('canonicalurl', ''),
                    'descriptionurl': page.get('descriptionurl', ''),
                    'descriptiontext': page.get('extract', ''),
                    'categories': page.get('categories', []),
                    'metadata': imageinfo.get('metadata', {})
                }
            )
            
            assets.append(asset)
        
        return assets
    
    async def get_media_info(self, media_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific media file on Wikimedia Commons.
        
        Args:
            media_id: Wikimedia media ID
            
        Returns:
            Detailed media metadata or None
        """
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "pageids": media_id,
            "iiprop": "url|size|mime|metadata"
        }
        
        url = f"{self.base_url}?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get media details: {e}")
            return None
    
    def _build_attribution(self, page: Dict[str, Any], imageinfo: Dict[str, Any]) -> str:
        """Build attribution for a Wikimedia asset."""
        title = page.get('title', 'Unknown')
        
        # Example: "File:X.jpg - Wikimedia Commons"
        return f"{title} - Wikimedia Commons"  
        
