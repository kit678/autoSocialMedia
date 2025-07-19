"""Cache manager for visual assets.

This module handles downloading, caching, and managing visual assets
to avoid redundant downloads and improve performance.
"""

import os
import logging
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from urllib.parse import urlparse
import asyncio
import aiohttp
import aiofiles

from ..media_utils import standardize_image, standardize_video
from .asset_types import Asset


class AssetCacheManager:
    """Manages downloading and caching of visual assets."""
    
    def __init__(self, cache_dir: str = "cache/visual_assets"):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cached assets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different asset types
        self.image_dir = self.cache_dir / "images"
        self.video_dir = self.cache_dir / "videos"
        self.reaction_dir = self.cache_dir / "reactions"
        
        for dir_path in [self.image_dir, self.video_dir, self.reaction_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Cache index file
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from a URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_asset_dir(self, asset_type: str) -> Path:
        """Get the appropriate directory for an asset type."""
        if asset_type == "image":
            return self.image_dir
        elif asset_type == "video":
            return self.video_dir
        elif asset_type == "reaction":
            return self.reaction_dir
        else:
            return self.cache_dir
    
    def _get_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """Determine file extension from URL or content type."""
        # Try to get extension from URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.mp4', '.mov', '.webm']:
            if path.endswith(ext):
                return ext
        
        # Fall back to content type
        if content_type:
            content_type = content_type.lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                return '.jpg'
            elif 'png' in content_type:
                return '.png'
            elif 'webp' in content_type:
                return '.webp'
            elif 'gif' in content_type:
                return '.gif'
            elif 'mp4' in content_type:
                return '.mp4'
            elif 'webm' in content_type:
                return '.webm'
        
        # Default based on asset type
        return '.jpg'  # Safe default
    
    def is_cached(self, url: str) -> bool:
        """Check if an asset is already cached.
        
        Args:
            url: URL of the asset
            
        Returns:
            True if cached and file exists
        """
        cache_key = self._get_cache_key(url)
        if cache_key in self.cache_index:
            cached_path = Path(self.cache_index[cache_key]['local_path'])
            return cached_path.exists()
        return False
    
    def get_cached_path(self, url: str) -> Optional[str]:
        """Get the local path of a cached asset.
        
        Args:
            url: URL of the asset
            
        Returns:
            Local file path or None if not cached
        """
        cache_key = self._get_cache_key(url)
        if cache_key in self.cache_index:
            cached_path = Path(self.cache_index[cache_key]['local_path'])
            if cached_path.exists():
                return str(cached_path)
        return None
    
    async def download_asset(
        self,
        asset: Asset,
        standardize: bool = True,
        timeout: int = 30
    ) -> Optional[str]:
        """Download an asset and cache it locally.
        
        Args:
            asset: Asset to download
            standardize: Whether to standardize images/videos to portrait
            timeout: Download timeout in seconds
            
        Returns:
            Local path to downloaded file or None if failed
        """
        # Check if already cached
        cached_path = self.get_cached_path(asset.url)
        if cached_path:
            self.logger.info(f"Asset already cached: {cached_path}")
            asset.local_path = cached_path
            return cached_path
        
        try:
            self.logger.info(f"Downloading {asset.type} from {asset.source}: {asset.url}")
            
            # Download the file
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'AutoSocialMedia/1.0',
                    'Accept': '*/*'
                }
                
                async with session.get(asset.url, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    
                    # Determine file extension
                    content_type = response.headers.get('Content-Type', '')
                    ext = self._get_file_extension(asset.url, content_type)
                    
                    # Generate filename
                    cache_key = self._get_cache_key(asset.url)
                    asset_dir = self._get_asset_dir(asset.type)
                    filename = f"{asset.source}_{cache_key}{ext}"
                    local_path = asset_dir / filename
                    
                    # Download to temporary file first
                    temp_path = local_path.with_suffix('.tmp')
                    
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    # Move temp file to final location
                    temp_path.rename(local_path)
                    
                    # Standardize if requested
                    if standardize:
                        if asset.type == "image" and ext in ['.jpg', '.jpeg', '.png', '.webp']:
                            if not standardize_image(str(local_path)):
                                self.logger.warning(f"Failed to standardize image: {local_path}")
                        elif asset.type == "video" and ext in ['.mp4', '.mov', '.webm']:
                            if not standardize_video(str(local_path)):
                                self.logger.warning(f"Failed to standardize video: {local_path}")
                    
                    # Update cache index
                    self.cache_index[cache_key] = {
                        'url': asset.url,
                        'local_path': str(local_path),
                        'asset_type': asset.type,
                        'source': asset.source,
                        'downloaded_at': os.path.getmtime(str(local_path))
                    }
                    self._save_cache_index()
                    
                    # Update asset with local path
                    asset.local_path = str(local_path)
                    
                    self.logger.info(f"Successfully downloaded to: {local_path}")
                    return str(local_path)
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout downloading asset: {asset.url}")
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error downloading asset: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error downloading asset: {e}")
        
        return None
    
    def download_asset_sync(
        self,
        asset: Asset,
        standardize: bool = True,
        timeout: int = 30
    ) -> Optional[str]:
        """Synchronous wrapper for download_asset.
        
        Args:
            asset: Asset to download
            standardize: Whether to standardize images/videos
            timeout: Download timeout
            
        Returns:
            Local path or None if failed
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.download_asset(asset, standardize, timeout)
            )
        finally:
            loop.close()
    
    async def download_assets_batch(
        self,
        assets: List[Asset],
        standardize: bool = True,
        max_concurrent: int = 3
    ) -> List[Optional[str]]:
        """Download multiple assets concurrently.
        
        Args:
            assets: List of assets to download
            standardize: Whether to standardize images/videos
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of local paths (None for failed downloads)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(asset):
            async with semaphore:
                return await self.download_asset(asset, standardize)
        
        tasks = [download_with_semaphore(asset) for asset in assets]
        return await asyncio.gather(*tasks)
    
    def copy_to_run_dir(self, asset: Asset, run_dir: str, cue_id: str) -> Optional[str]:
        """Copy a cached asset to the run directory.
        
        Args:
            asset: Asset with local_path set
            run_dir: Target run directory
            cue_id: Visual cue ID for naming
            
        Returns:
            Path in run directory or None if failed
        """
        if not asset.local_path or not os.path.exists(asset.local_path):
            self.logger.error(f"Asset not downloaded: {asset.id}")
            return None
        
        try:
            # Create visuals subdirectory
            visuals_dir = Path(run_dir) / "visuals"
            visuals_dir.mkdir(exist_ok=True)
            
            # Generate target filename
            source_path = Path(asset.local_path)
            ext = source_path.suffix
            target_filename = f"{cue_id}_{asset.source}{ext}"
            target_path = visuals_dir / target_filename
            
            # Copy file
            shutil.copy2(source_path, target_path)
            
            self.logger.info(f"Copied asset to: {target_path}")
            return str(target_path)
            
        except Exception as e:
            self.logger.error(f"Failed to copy asset to run directory: {e}")
            return None
    
    def clean_old_cache(self, max_age_days: int = 30) -> int:
        """Remove cached files older than specified days.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of files removed
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        removed_count = 0
        keys_to_remove = []
        
        for cache_key, info in self.cache_index.items():
            file_path = Path(info['local_path'])
            
            # Check if file exists and is old
            if file_path.exists():
                file_age = current_time - info.get('downloaded_at', 0)
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        keys_to_remove.append(cache_key)
                        removed_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to remove old cache file: {e}")
            else:
                # File missing, remove from index
                keys_to_remove.append(cache_key)
        
        # Update index
        for key in keys_to_remove:
            del self.cache_index[key]
        
        if removed_count > 0:
            self._save_cache_index()
            self.logger.info(f"Removed {removed_count} old cache files")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        file_count = 0
        missing_files = 0
        
        type_stats = {
            'image': {'count': 0, 'size': 0},
            'video': {'count': 0, 'size': 0},
            'reaction': {'count': 0, 'size': 0}
        }
        
        for info in self.cache_index.values():
            file_path = Path(info['local_path'])
            if file_path.exists():
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
                
                asset_type = info.get('asset_type', 'unknown')
                if asset_type in type_stats:
                    type_stats[asset_type]['count'] += 1
                    type_stats[asset_type]['size'] += size
            else:
                missing_files += 1
        
        return {
            'total_files': file_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'missing_files': missing_files,
            'type_breakdown': type_stats,
            'cache_directory': str(self.cache_dir)
        }


# Global cache manager instance
_cache_manager = None


def get_cache_manager(cache_dir: Optional[str] = None) -> AssetCacheManager:
    """Get or create the global cache manager instance.
    
    Args:
        cache_dir: Optional cache directory path
        
    Returns:
        AssetCacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = AssetCacheManager(cache_dir or "cache/visual_assets")
    return _cache_manager
