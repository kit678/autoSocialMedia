"""Pexels adapter for searching stock photos and videos.

This adapter uses the Pexels API to search for high-quality stock media.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp

from ..asset_registry import AssetAdapter
from ..asset_types import Asset


class PexelsAdapter(AssetAdapter):
    """Adapter for searching photos and videos via Pexels API."""
    
    name = "pexels"
    supported_types = ["image", "video"]
    requires_api_key = True
    api_key_env_var = "PEXELS_API_KEY"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Pexels adapter.
        
        Args:
            config: Optional configuration
        """
        super().__init__(config)
        self.api_key = os.getenv(self.api_key_env_var)
        self.photo_api_url = "https://api.pexels.com/v1/search"
        self.video_api_url = "https://api.pexels.com/videos/search"
    
    async def search(
        self, 
        query: str, 
        segment: Dict[str, Any], 
        limit: int = 10
    ) -> List[Asset]:
        """Search for assets using Pexels API.
        
        Args:
            query: Search query
            segment: Segment context with intent, emotion, etc.
            limit: Maximum number of results
            
        Returns:
            List of Asset objects
        """
        if not self.is_available():
            self.logger.warning("Pexels API key not configured")
            return []
        
        # Determine media type preference
        preferred_media = segment.get('preferred_media', 'image')
        
        assets = []
        
        # Search photos if needed
        if preferred_media in ['image', 'any']:
            photo_assets = await self._search_photos(query, segment, limit)
            assets.extend(photo_assets)
        
        # Search videos if needed
        if preferred_media in ['video', 'reaction', 'any'] and len(assets) < limit:
            video_assets = await self._search_videos(
                query, segment, limit - len(assets)
            )
            assets.extend(video_assets)
        
        return assets[:limit]
    
    async def _search_photos(
        self,
        query: str,
        segment: Dict[str, Any],
        limit: int
    ) -> List[Asset]:
        """Search for photos on Pexels."""
        try:
            headers = {'Authorization': self.api_key}
            
            # Map visual type to Pexels category if applicable
            category = self._map_visual_type_to_category(
                segment.get('visual_type', '')
            )
            
            # Enhance query with portrait keywords
            portrait_query = f"{query} portrait vertical mobile phone"
            
            params = {
                'query': portrait_query,
                'per_page': min(limit, 80),  # Pexels max is 80
                'orientation': 'portrait'
            }
            
            # Add category if mapped
            if category:
                params['category'] = category
            
            self.logger.info(f"Searching Pexels photos for: '{query}'" + 
                           (f" in category '{category}'" if category else ""))
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.photo_api_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            photos = data.get('photos', [])
            assets = []
            
            for i, photo in enumerate(photos):
                asset = Asset(
                    id=f"pexels_photo_{photo['id']}",
                    url=photo['src']['large'],  # Use large size
                    type="image",
                    source="pexels",
                    licence="Pexels",
                    attribution=f"Photo by {photo['photographer']} on Pexels",
                    dimensions=(photo['width'], photo['height']),
                    relevance_score=self._calculate_relevance(i, len(photos)),
                    quality_score=0.9,  # Pexels has high quality
                    metadata={
                        'photographer': photo['photographer'],
                        'photographer_url': photo['photographer_url'],
                        'pexels_url': photo['url'],
                        'alt': photo.get('alt', ''),
                        'avg_color': photo.get('avg_color', ''),
                        'dominant_color': photo.get('avg_color', ''),
                        'src_variants': photo['src']
                    }
                )
                assets.append(asset)
            
            self.logger.info(f"Found {len(assets)} photos from Pexels")
            return assets
            
        except Exception as e:
            self.logger.error(f"Pexels photo search failed: {e}")
            return []
    
    async def _search_videos(
        self,
        query: str,
        segment: Dict[str, Any],
        limit: int
    ) -> List[Asset]:
        """Search for videos on Pexels."""
        try:
            headers = {'Authorization': self.api_key}
            
            # Determine duration preference based on segment
            duration_pref = self._get_duration_preference(segment)
            
            # Enhance query with portrait keywords for videos
            portrait_query = f"{query} portrait vertical mobile"
            
            params = {
                'query': portrait_query,
                'per_page': min(limit, 80),
                'orientation': 'portrait',
                'size': 'medium'
            }
            
            # Add duration constraints
            if duration_pref == 'short':
                params['min_duration'] = 5
                params['max_duration'] = 15
            elif duration_pref == 'medium':
                params['min_duration'] = 10
                params['max_duration'] = 30
            
            self.logger.info(f"Searching Pexels videos for: '{query}'")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.video_api_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            videos = data.get('videos', [])
            assets = []
            
            for i, video in enumerate(videos):
                # Find best quality video file
                video_file = self._select_best_video_file(
                    video.get('video_files', [])
                )
                
                if not video_file:
                    continue
                
                # Determine if this could be a reaction video
                is_reaction = (
                    segment.get('preferred_media') == 'reaction' and
                    video['duration'] <= 5
                )
                
                asset = Asset(
                    id=f"pexels_video_{video['id']}",
                    url=video_file['link'],
                    type="reaction" if is_reaction else "video",
                    source="pexels",
                    licence="Pexels",
                    attribution=f"Video by {video['user']['name']} on Pexels",
                    dimensions=(video['width'], video['height']),
                    duration=float(video['duration']),
                    relevance_score=self._calculate_relevance(i, len(videos)),
                    quality_score=0.9,
                    metadata={
                        'videographer': video['user']['name'],
                        'videographer_url': video['user']['url'],
                        'pexels_url': video['url'],
                        'video_files': video['video_files'],
                        'selected_quality': video_file['quality'],
                        'file_type': video_file['file_type']
                    }
                )
                assets.append(asset)
            
            self.logger.info(f"Found {len(assets)} videos from Pexels")
            return assets
            
        except Exception as e:
            self.logger.error(f"Pexels video search failed: {e}")
            return []
    
    def _map_visual_type_to_category(self, visual_type: str) -> Optional[str]:
        """Map internal visual type to Pexels category."""
        mapping = {
            "person": "people",
            "company": "business",
            "product": "industry",
            "location": "places",
            "action": "people",
            "concept": "science",
            "Proper Noun": "business",
            "Concrete Object/Action": "industry",
            "Abstract Concept": "science",
        }
        
        # Valid Pexels categories
        valid_categories = [
            "backgrounds", "fashion", "nature", "science", "education",
            "feelings", "health", "people", "religion", "places", "animals",
            "industry", "computer", "food", "sports", "transportation",
            "travel", "buildings", "business", "music"
        ]
        
        category = mapping.get(visual_type)
        return category if category in valid_categories else None
    
    def _get_duration_preference(self, segment: Dict[str, Any]) -> str:
        """Determine video duration preference based on segment."""
        duration_target = segment.get('duration_target', '')
        
        if 'seconds' in duration_target:
            try:
                # Extract number from strings like "3-5 seconds"
                numbers = [int(s) for s in duration_target.split() if s.isdigit()]
                if numbers:
                    avg_duration = sum(numbers) / len(numbers)
                    if avg_duration <= 5:
                        return 'short'
                    elif avg_duration <= 15:
                        return 'medium'
            except:
                pass
        
        # Default based on motion hint
        motion = segment.get('motion_hint', 'static')
        if motion == 'dynamic':
            return 'medium'
        else:
            return 'short'
    
    def _select_best_video_file(self, video_files: List[Dict]) -> Optional[Dict]:
        """Select the best video file from options."""
        if not video_files:
            return None
        
        # Sort by quality preference
        quality_order = ['hd', 'sd', 'mobile']
        
        # First, filter for portrait orientation
        portrait_files = [
            f for f in video_files
            if f.get('height', 0) > f.get('width', 0)
        ]
        
        # Use all files if no portrait available
        files_to_check = portrait_files if portrait_files else video_files
        
        # Find best quality
        for quality in quality_order:
            for file in files_to_check:
                if file.get('quality') == quality and file.get('file_type') == 'video/mp4':
                    return file
        
        # Fallback to first available
        return files_to_check[0] if files_to_check else None
    
    def _calculate_relevance(self, position: int, total: int) -> float:
        """Calculate relevance score based on position."""
        if total == 0:
            return 0.0
        
        # Pexels API returns results in relevance order
        # Use exponential decay for more differentiation
        return 0.95 ** position
