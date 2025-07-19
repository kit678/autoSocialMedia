"""
Integrated Visual Director

This module provides the full integration of all visual director components,
bridging the enhanced system with the existing pipeline.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from .config_manager import get_config_manager
import yaml
from pathlib import Path

from .asset_types import Asset, AssetScoringConfig
from .asset_registry import AssetAdapterRegistry
from .media_selector import SmartMediaSelector
from .entity_canonicalizer import EntityCanonicalizer
from .advanced_cache import AdvancedCache, get_cache
from .llm_intent_tagger import tag_segment_intent
from .visual_scorer import get_visual_scorer
from .video_processor import VideoProcessor
from .attribution_manager import AttributionManager
from .adapter_router import AdapterRouter
from .reaction_detector import ReactionPointDetector
from .enhanced_segment import EnhancedSegment, SegmentProcessor

# Import all adapters
from .adapters.searxng_adapter import SearXNGAdapter
from .adapters.pexels_adapter import PexelsAdapter
from .adapters.tenor_adapter import TenorAdapter
from .adapters.openverse_adapter import OpenverseAdapter
from .adapters.wikimedia_adapter import WikimediaAdapter
from .adapters.nasa_adapter import NASAAdapter
from .adapters.archive_tv_adapter import ArchiveTVAdapter
from .adapters.gdelt_adapter import GDELTAdapter
from .adapters.coverr_adapter import CoverrAdapter


class IntegratedVisualDirector:
    """
    Full integration of the enhanced visual director system.
    
    This class orchestrates all components to provide a complete
    visual asset selection and processing pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated visual director.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration using the config manager
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.config
        
        # Initialize cache
        cache_dir = self.config.get('cache_dir', './cache')
        self.cache = get_cache(cache_dir)
        
        # Initialize components
        self.registry = AssetAdapterRegistry()
        self.media_selector = SmartMediaSelector()
        self.canonicalizer = EntityCanonicalizer(self.cache)
        self.visual_scorer = get_visual_scorer() if self.config.get('enable_clip', True) else None
        self.video_processor = VideoProcessor()
        self.attribution_manager = AttributionManager(
            self.config.get('project_dir', './projects')
        )
        
        # Initialize smart routing and reaction detection
        self.adapter_router = AdapterRouter(self.registry, self.config)
        self.reaction_detector = ReactionPointDetector(self.config)
        self.segment_processor = SegmentProcessor(self.reaction_detector, self.registry)
        
        # Initialize adapters
        self._init_adapters()
        
        # Log adapter status
        self.config_manager.log_provider_status()
        
        # Scoring configuration
        self.scoring_config = AssetScoringConfig(
            relevance_weight=self.config.get('scoring', {}).get('relevance_weight', 0.5),
            quality_weight=self.config.get('scoring', {}).get('quality_weight', 0.3),
            diversity_weight=self.config.get('scoring', {}).get('diversity_weight', 0.2)
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'visual_director': {
                'allow_paid': False,
                'max_results_per_source': 20,
                'portrait_width': 1080,
                'portrait_height': 1920,
                'min_score': 0.45,
                'reaction_min_conf': 0.6,
                'providers': {}
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                config = default_config.copy()
                config.update(loaded_config)
                return config['visual_director']
        
        return default_config['visual_director']
    
    def _init_adapters(self):
        """Initialize all configured adapters."""

        available_providers = self.config_manager.get_available_providers()
        
        # Register and initialize available adapters
        adapter_classes = {
            'searxng': SearXNGAdapter,
            'pexels': PexelsAdapter,
            'tenor': TenorAdapter,
            'openverse': OpenverseAdapter,
            'wikimedia': WikimediaAdapter,
            'nasa': NASAAdapter,
            'archive_tv': ArchiveTVAdapter,
            'gdelt': GDELTAdapter,
            'coverr': CoverrAdapter,
        }

        for adapter_name in available_providers:
            self.registry.register_adapter_class(adapter_name, adapter_classes[adapter_name])
            provider_config = self.config_manager.get_provider_config(adapter_name)
            self.registry.initialize_adapter(adapter_name, provider_config['config'])
    
    async def process_segments(
        self,
        segments: List[Dict[str, Any]],
        article_context: Dict[str, Any],
        project_id: str
    ) -> List[Dict[str, Any]]:
        """
        Process segments to add visual assets.
        
        Args:
            segments: List of narrative segments
            article_context: Context about the article/story
            project_id: Unique project identifier
            
        Returns:
            Segments with visual assets attached
        """
        processed_segments = []
        
        for i, segment in enumerate(segments):
            try:
                # 1. Analyze segment intent and emotion
                segment_intent = await tag_segment_intent(
                    segment, article_context,
                    segments[i-1] if i > 0 else None,
                    segments[i+1] if i < len(segments)-1 else None
                )
                
                # 2. Determine media type
                media_type, confidence, reasoning = self.media_selector.select_media_type(
                    segment, article_context
                )
                
                # Override with LLM intent if high confidence
                if segment_intent.confidence > 0.8:
                    media_type = segment_intent.visual_type
                
                segment['visual_intent'] = {
                    'type': media_type,
                    'emotion': segment_intent.emotion,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
                
                # 3. Search for assets
                assets = await self._search_assets_for_segment(
                    segment, segment_intent, media_type
                )
                
                # 4. Select best asset
                if assets:
                    best_asset = await self._select_best_asset(
                        assets, segment, segment_intent
                    )
                    
                    # 5. Process video if needed
                    if best_asset and best_asset.type == 'video':
                        best_asset = await self._process_video_asset(
                            best_asset, segment
                        )
                    
                    # 6. Record attribution
                    if best_asset:
                        self.attribution_manager.add_asset_attribution(
                            best_asset, segment
                        )
                        segment['asset'] = best_asset
                        
                        # Cache for future use
                        self.cache.cache_asset(best_asset)
                        self.cache.record_usage(
                            best_asset, segment.get('id', f'seg_{i}'),
                            project_id, 0.0
                        )
                
                processed_segments.append(segment)
                
            except Exception as e:
                self.logger.error(f"Error processing segment {i}: {e}")
                processed_segments.append(segment)
        
        return processed_segments
    
    async def _search_assets_for_segment(
        self,
        segment: Dict[str, Any],
        intent: Any,
        media_type: str
    ) -> List[Asset]:
        """Search for assets across all adapters."""
        # Build search queries
        queries = []
        
        # Add entity-based queries
        for entity in segment.get('entities', []):
            if isinstance(entity, dict):
                queries.append(entity.get('text', ''))
            else:
                queries.append(str(entity))
        
        # Add intent-based queries
        if intent.search_terms:
            queries.extend(intent.search_terms)
        
        # Add topic-based queries
        queries.extend(segment.get('topics', []))
        
        # Remove duplicates and empty
        queries = list(filter(None, set(queries)))
        
        if not queries:
            # Fallback to segment text keywords
            text = segment.get('text', '')
            words = text.split()
            # Extract capitalized words as potential entities
            queries = [w for w in words if w[0].isupper() and len(w) > 3][:3]
        
        # Get smart routing based on content
        available_adapters = [name for name, _ in self.registry._adapters.items()]
        routed_adapters = await self.adapter_router.route_query(
            segment, intent, available_adapters
        )
        
        # Log routing decision
        routing_explanation = await self.adapter_router.get_routing_explanation(
            segment, intent, routed_adapters
        )
        self.logger.info(f"Routing: {routing_explanation}")
        
        # Search only the top routed adapters
        all_assets = []
        adapters_to_search = routed_adapters[:self.config.get('smart_routing', {}).get('max_adapters_per_query', 3)]
        
        for adapter_name in adapters_to_search:
            adapter = self.registry.get_adapter(adapter_name)
            if not adapter:
                continue
                
            # Search this adapter with all queries
            for query in queries[:2]:  # Limit queries per adapter
                try:
                    assets = await adapter.search(
                        query, segment, 
                        limit=self.config.get('max_results_per_source', 20) // max(len(queries), 1)
                    )
                    all_assets.extend(assets)
                except Exception as e:
                    self.logger.warning(f"Search failed for {adapter_name}: {e}")
            
            # Stop early if we have enough good results
            if len(all_assets) >= 10:
                break
        
        # If no assets found and fallback is enabled, try fallback queries
        if not all_assets and self.config_manager.is_fallback_enabled():
            self.logger.info(f"No assets found for segment queries {queries}, trying fallback queries")
            fallback_queries = self.config_manager.get_fallback_queries()
            
            for fallback_query in fallback_queries[:3]:  # Try first 3 fallback queries
                results = await self.registry.search_all_adapters(
                    fallback_query, segment, media_type,
                    limit_per_adapter=5  # Fewer results for fallback
                )
                
                for result in results:
                    all_assets.extend(result.assets)
                
                # Stop if we found some assets
                if all_assets:
                    break
        
        return all_assets
    
    async def _select_best_asset(
        self,
        assets: List[Asset],
        segment: Dict[str, Any],
        intent: Any
    ) -> Optional[Asset]:
        """Select the best asset from candidates."""
        if not assets:
            return None
        
        # Check for canonical entity
        entities = segment.get('entities', [])
        if entities:
            entity = entities[0]
            entity_name = entity.get('text', '') if isinstance(entity, dict) else str(entity)
            entity_type = entity.get('type', 'MISC').lower() if isinstance(entity, dict) else 'misc'
            
            canonical = self.canonicalizer.select_canonical_asset(
                entity_name, entity_type, assets, segment
            )
            if canonical:
                return canonical
        
        # Score all assets
        for asset in assets:
            # Visual semantic scoring if available
            if self.visual_scorer and self.visual_scorer.enabled:
                visual_score = self.visual_scorer.score_visual_similarity(
                    asset.url, segment.get('text', '')
                )
                asset.relevance_score = (asset.relevance_score + visual_score) / 2
            
            # Calculate composite score
            asset.composite_score = self._calculate_composite_score(
                asset, segment, intent
            )
        
        # Sort by composite score
        assets.sort(key=lambda a: a.composite_score, reverse=True)
        
        # Return best if above threshold
        if assets[0].composite_score >= self.config.get('min_score', 0.45):
            return assets[0]
        
        return None
    
    def _calculate_composite_score(
        self,
        asset: Asset,
        segment: Dict[str, Any],
        intent: Any
    ) -> float:
        """Calculate composite score for asset selection."""
        scores = {
            'relevance': asset.relevance_score,
            'quality': asset.quality_score,
            'diversity': asset.diversity_score
        }
        
        # Adjust for media type preference
        if asset.type == segment.get('visual_intent', {}).get('type'):
            scores['relevance'] += 0.1
        
        # Adjust for emotion match (reactions)
        if asset.type == 'reaction' and intent.emotion:
            if intent.emotion in asset.metadata.get('emotion', ''):
                scores['relevance'] += 0.2
        
        # Portrait preference for mobile
        if asset.is_portrait:
            scores['quality'] += self.scoring_config.prefer_portrait
        
        # License preference
        if asset.licence in self.scoring_config.license_penalties:
            penalty = self.scoring_config.license_penalties[asset.licence]
            scores['quality'] -= penalty
        
        # Source preference
        if asset.source in self.scoring_config.source_bonuses:
            bonus = self.scoring_config.source_bonuses[asset.source]
            scores['quality'] += bonus
        
        # Calculate weighted score
        total = (
            scores['relevance'] * self.scoring_config.relevance_weight +
            scores['quality'] * self.scoring_config.quality_weight +
            scores['diversity'] * self.scoring_config.diversity_weight
        )
        
        return min(total, 1.0)
    
    async def _process_video_asset(
        self,
        asset: Asset,
        segment: Dict[str, Any]
    ) -> Asset:
        """Process video asset (trim, reframe, etc)."""
        if not asset.local_path:
            # Download first
            # This would use a downloader component
            pass
        
        # Trim to segment duration if needed
        segment_duration = segment.get('duration', 5.0)
        if asset.duration and asset.duration > segment_duration:
            output_path = f"{asset.local_path}_trimmed.mp4"
            success = self.video_processor.trim_video(
                asset.local_path,
                output_path,
                0,  # Start at beginning
                segment_duration
            )
            if success:
                asset.local_path = output_path
                asset.duration = segment_duration
                asset.metadata['processed'] = True
                asset.metadata['trim_duration'] = segment_duration
        
        # Reframe to portrait if needed
        if asset.is_landscape and self.config.get('portrait_width'):
            output_path = f"{asset.local_path}_portrait.mp4"
            success = self.video_processor.reframe_to_portrait(
                asset.local_path,
                output_path
            )
            if success:
                asset.local_path = output_path
                asset.dimensions = (
                    self.config.get('portrait_width', 1080),
                    self.config.get('portrait_height', 1920)
                )
                asset.metadata['reframed'] = True
        
        return asset
    
    def generate_attribution_files(self, project_id: str) -> Dict[str, str]:
        """Generate attribution files for the project."""
        return self.attribution_manager.generate_all_formats(project_id)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_cache_stats()
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.cache.cleanup_expired()


# Backward compatibility function
async def create_enhanced_visual_director(config_path: Optional[str] = None) -> IntegratedVisualDirector:
    """
    Create an instance of the enhanced visual director.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured IntegratedVisualDirector instance
    """
    director = IntegratedVisualDirector(config_path)
    return director
