"""
Base Visual Adapter Interface

This module defines the base interface that all visual adapters must implement
for consistent integration with the enhanced visual director system.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time

from .asset_types import Asset, AssetSearchResult


class VisualAdapter(ABC):
    """
    Base class for all visual adapters.
    
    Defines the interface that all adapters must implement to provide
    consistent search and asset acquisition functionality.
    """
    
    def __init__(self, name: str, timeout: float = 30.0):
        """
        Initialize the base adapter.
        
        Args:
            name: Name of the adapter (e.g., 'pexels', 'searxng')
            timeout: Default timeout for requests in seconds
        """
        self.name = name
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics tracking
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_assets_found': 0,
            'average_response_time': 0.0
        }
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 20,
        media_type: Optional[str] = None,
        **kwargs
    ) -> List[Asset]:
        """
        Search for assets based on a query.
        
        Args:
            query: Search query string
            limit: Maximum number of assets to return
            media_type: Preferred media type ('image', 'video', 'gif')
            **kwargs: Additional adapter-specific parameters
            
        Returns:
            List of found assets
        """
        pass
    
    @abstractmethod
    async def download_asset(self, asset: Asset, output_path: str) -> bool:
        """
        Download an asset to a local path.
        
        Args:
            asset: Asset to download
            output_path: Local path to save the asset
            
        Returns:
            True if download successful, False otherwise
        """
        pass
    
    async def search_with_metadata(
        self,
        query: str,
        segment: Dict[str, Any],
        limit: int = 20,
        **kwargs
    ) -> AssetSearchResult:
        """
        Search for assets and return with metadata.
        
        Args:
            query: Search query string
            segment: Segment data for context
            limit: Maximum number of assets to return
            **kwargs: Additional parameters
            
        Returns:
            AssetSearchResult with assets and metadata
        """
        start_time = time.time()
        
        try:
            assets = await self.search(query, limit, **kwargs)
            search_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_searches'] += 1
            self.stats['successful_searches'] += 1
            self.stats['total_assets_found'] += len(assets)
            self._update_average_response_time(search_time)
            
            return AssetSearchResult(
                query=query,
                segment=segment,
                assets=assets,
                adapter_name=self.name,
                search_time=search_time,
                error=None
            )
            
        except Exception as e:
            search_time = time.time() - start_time
            error_msg = f"Search failed: {str(e)}"
            
            # Update statistics
            self.stats['total_searches'] += 1
            self.stats['failed_searches'] += 1
            self._update_average_response_time(search_time)
            
            self.logger.error(f"Search failed for query '{query}': {e}")
            
            return AssetSearchResult(
                query=query,
                segment=segment,
                assets=[],
                adapter_name=self.name,
                search_time=search_time,
                error=error_msg
            )
    
    def _update_average_response_time(self, new_time: float):
        """Update the average response time statistic."""
        if self.stats['total_searches'] == 1:
            self.stats['average_response_time'] = new_time
        else:
            # Running average
            current_avg = self.stats['average_response_time']
            count = self.stats['total_searches']
            self.stats['average_response_time'] = ((current_avg * (count - 1)) + new_time) / count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset adapter statistics."""
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_assets_found': 0,
            'average_response_time': 0.0
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.stats['total_searches'] == 0:
            return 0.0
        return self.stats['successful_searches'] / self.stats['total_searches']
    
    def supports_media_type(self, media_type: str) -> bool:
        """
        Check if adapter supports a specific media type.
        
        Args:
            media_type: Media type to check ('image', 'video', 'gif')
            
        Returns:
            True if supported, False otherwise
        """
        # Default implementation - subclasses should override
        return media_type in ['image', 'video']
    
    def get_supported_media_types(self) -> List[str]:
        """
        Get list of supported media types.
        
        Returns:
            List of supported media types
        """
        # Default implementation - subclasses should override
        return ['image', 'video']
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate adapter configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation - subclasses should override
        return True
    
    async def test_connection(self) -> bool:
        """
        Test connection to the adapter's service.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Default test - try a simple search
            assets = await self.search("test", limit=1)
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def create_asset_id(self, source_id: str) -> str:
        """
        Create a unique asset ID.
        
        Args:
            source_id: Source-specific asset ID
            
        Returns:
            Unique asset ID
        """
        return f"{self.name}_{source_id}"
    
    def extract_media_info(self, url: str) -> Dict[str, Any]:
        """
        Extract media information from URL.
        
        Args:
            url: Media URL
            
        Returns:
            Dictionary with media information
        """
        # Default implementation - subclasses should override
        return {
            'url': url,
            'type': 'unknown',
            'dimensions': (0, 0),
            'duration': None
        }


class AdapterRegistry:
    """Registry for managing visual adapters."""
    
    def __init__(self):
        self.adapters: Dict[str, VisualAdapter] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, adapter: VisualAdapter):
        """
        Register an adapter.
        
        Args:
            adapter: Adapter instance to register
        """
        self.adapters[adapter.name] = adapter
        self.logger.info(f"Registered adapter: {adapter.name}")
    
    def get(self, name: str) -> Optional[VisualAdapter]:
        """
        Get an adapter by name.
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter instance or None if not found
        """
        return self.adapters.get(name)
    
    def get_all(self) -> List[VisualAdapter]:
        """Get all registered adapters."""
        return list(self.adapters.values())
    
    def get_by_media_type(self, media_type: str) -> List[VisualAdapter]:
        """
        Get adapters that support a specific media type.
        
        Args:
            media_type: Media type to filter by
            
        Returns:
            List of supporting adapters
        """
        return [
            adapter for adapter in self.adapters.values()
            if adapter.supports_media_type(media_type)
        ]
    
    def remove(self, name: str) -> bool:
        """
        Remove an adapter from the registry.
        
        Args:
            name: Adapter name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.adapters:
            del self.adapters[name]
            self.logger.info(f"Removed adapter: {name}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all adapters."""
        return {
            name: adapter.get_stats()
            for name, adapter in self.adapters.items()
        }
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections for all adapters.
        
        Returns:
            Dictionary mapping adapter names to connection status
        """
        results = {}
        
        for name, adapter in self.adapters.items():
            try:
                results[name] = await adapter.test_connection()
            except Exception as e:
                self.logger.error(f"Connection test failed for {name}: {e}")
                results[name] = False
        
        return results


# Global adapter registry
_adapter_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    return _adapter_registry


def register_adapter(adapter: VisualAdapter):
    """Register an adapter with the global registry."""
    _adapter_registry.register(adapter)


def get_adapter(name: str) -> Optional[VisualAdapter]:
    """Get an adapter by name from the global registry."""
    return _adapter_registry.get(name)
