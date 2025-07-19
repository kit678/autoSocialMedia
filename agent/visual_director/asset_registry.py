"""Asset adapter registry and interface definitions.

This module defines the abstract base class for asset adapters and manages
the registry of available adapters for fetching visual content from various sources.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .asset_types import Asset, AssetSearchResult


class AssetAdapter(ABC):
    """Abstract base class for visual asset adapters.
    
    Each adapter implements the interface for searching and fetching assets
    from a specific source (e.g., SearXNG, Pexels, Openverse, etc.).
    """
    
    # Adapter metadata
    name: str = ""
    supported_types: List[str] = ["image"]  # image, video, reaction
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter with optional configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def search(self, query: str, segment: Dict[str, Any], limit: int = 10) -> List[Asset]:
        """Search for assets matching the query and segment context.
        
        Args:
            query: Search query string
            segment: Segment data containing context, intent, emotion, etc.
            limit: Maximum number of results to return
            
        Returns:
            List of Asset objects
        """
        pass
    
    def is_available(self) -> bool:
        """Check if this adapter is available and properly configured.
        
        Returns:
            True if adapter can be used, False otherwise
        """
        if self.requires_api_key and self.api_key_env_var:
            import os
            return bool(os.getenv(self.api_key_env_var))
        return True
    
    def supports_type(self, asset_type: str) -> bool:
        """Check if adapter supports the requested asset type.
        
        Args:
            asset_type: Type of asset (image, video, reaction)
            
        Returns:
            True if supported, False otherwise
        """
        return asset_type in self.supported_types
    
    async def search_with_fallback(self, query: str, segment: Dict[str, Any], limit: int = 10) -> AssetSearchResult:
        """Search with error handling and result wrapping.
        
        Args:
            query: Search query string
            segment: Segment data
            limit: Maximum number of results
            
        Returns:
            AssetSearchResult with assets or error information
        """
        import time
        start_time = time.time()
        
        try:
            assets = await self.search(query, segment, limit)
            search_time = time.time() - start_time
            
            return AssetSearchResult(
                query=query,
                segment=segment,
                assets=assets,
                adapter_name=self.name,
                search_time=search_time
            )
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            search_time = time.time() - start_time
            
            return AssetSearchResult(
                query=query,
                segment=segment,
                assets=[],
                adapter_name=self.name,
                search_time=search_time,
                error=str(e)
            )


class AssetAdapterRegistry:
    """Registry for managing available asset adapters.
    
    This class maintains a registry of all available adapters and provides
    methods for discovering assets from multiple sources.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._adapters: Dict[str, AssetAdapter] = {}
        self._adapter_classes: Dict[str, Type[AssetAdapter]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for parallel searches
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def register_adapter_class(self, name: str, adapter_class: Type[AssetAdapter]) -> None:
        """Register an adapter class.
        
        Args:
            name: Unique name for the adapter
            adapter_class: The adapter class to register
        """
        if not issubclass(adapter_class, AssetAdapter):
            raise ValueError(f"{adapter_class} must be a subclass of AssetAdapter")
        
        self._adapter_classes[name] = adapter_class
        self.logger.info(f"Registered adapter class: {name}")
    
    def initialize_adapter(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[AssetAdapter]:
        """Initialize an adapter instance.
        
        Args:
            name: Name of the adapter to initialize
            config: Optional configuration for the adapter
            
        Returns:
            Initialized adapter instance or None if unavailable
        """
        if name not in self._adapter_classes:
            self.logger.error(f"Unknown adapter: {name}")
            return None
        
        try:
            adapter = self._adapter_classes[name](config)
            if adapter.is_available():
                self._adapters[name] = adapter
                self.logger.info(f"Initialized adapter: {name}")
                return adapter
            else:
                self.logger.warning(f"Adapter {name} is not available (missing API key or configuration)")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize adapter {name}: {e}")
            return None
    
    def get_adapter(self, name: str) -> Optional[AssetAdapter]:
        """Get an initialized adapter by name.
        
        Args:
            name: Name of the adapter
            
        Returns:
            Adapter instance or None if not found
        """
        return self._adapters.get(name)
    
    def get_available_adapters(self, asset_type: Optional[str] = None) -> List[AssetAdapter]:
        """Get all available adapters, optionally filtered by asset type.
        
        Args:
            asset_type: Optional filter by asset type (image, video, reaction)
            
        Returns:
            List of available adapter instances
        """
        adapters = list(self._adapters.values())
        
        if asset_type:
            adapters = [a for a in adapters if a.supports_type(asset_type)]
        
        return adapters
    
    async def search_all_adapters(
        self,
        query: str,
        segment: Dict[str, Any],
        asset_type: Optional[str] = None,
        limit_per_adapter: int = 5
    ) -> List[AssetSearchResult]:
        """Search across all available adapters in parallel.
        
        Args:
            query: Search query
            segment: Segment context
            asset_type: Optional filter by asset type
            limit_per_adapter: Max results per adapter
            
        Returns:
            List of search results from all adapters
        """
        adapters = self.get_available_adapters(asset_type)
        
        if not adapters:
            self.logger.warning(f"No adapters available for type: {asset_type}")
            return []
        
        # Create search tasks for all adapters
        tasks = [
            adapter.search_with_fallback(query, segment, limit_per_adapter)
            for adapter in adapters
        ]
        
        # Execute searches in parallel
        results = await asyncio.gather(*tasks)
        
        # Filter out failed searches
        successful_results = [r for r in results if r.success]
        
        self.logger.info(
            f"Searched {len(adapters)} adapters for '{query}', "
            f"{len(successful_results)} returned results"
        )
        
        return results
    
    def search_priority_adapters(
        self,
        query: str,
        segment: Dict[str, Any],
        priority_order: List[str],
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Asset]:
        """Search adapters in priority order until enough assets are found.
        
        Args:
            query: Search query
            segment: Segment context
            priority_order: List of adapter names in priority order
            asset_type: Optional filter by asset type
            limit: Total number of assets needed
            
        Returns:
            List of assets from the first adapter(s) that provide results
        """
        assets = []
        
        for adapter_name in priority_order:
            if len(assets) >= limit:
                break
            
            adapter = self.get_adapter(adapter_name)
            if not adapter:
                continue
            
            if asset_type and not adapter.supports_type(asset_type):
                continue
            
            try:
                # Use synchronous search for simplicity
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    adapter.search_with_fallback(query, segment, limit - len(assets))
                )
                
                if result.success:
                    assets.extend(result.assets)
                    self.logger.info(
                        f"Found {len(result.assets)} assets from {adapter_name} "
                        f"for query '{query}'"
                    )
                
            except Exception as e:
                self.logger.error(f"Error searching {adapter_name}: {e}")
                continue
            finally:
                loop.close()
        
        return assets[:limit]  # Ensure we don't exceed the limit


# Global registry instance
_registry = AssetAdapterRegistry()


def get_registry() -> AssetAdapterRegistry:
    """Get the global asset adapter registry."""
    return _registry


def register_adapter(name: str, adapter_class: Type[AssetAdapter]) -> None:
    """Register an adapter class with the global registry.
    
    Args:
        name: Unique name for the adapter
        adapter_class: The adapter class to register
    """
    _registry.register_adapter_class(name, adapter_class)


def initialize_adapters(config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Initialize all registered adapters with configuration.
    
    Args:
        config: Optional configuration dict mapping adapter names to configs
    """
    config = config or {}
    
    for name in _registry._adapter_classes:
        adapter_config = config.get(name, {})
        _registry.initialize_adapter(name, adapter_config)
