"""
Enhanced Visual Director Integration

This module integrates all visual director components into a unified system
that processes narrative segments into visually-rich video sequences.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from collections import defaultdict

from .asset_types import Asset, AssetSource
from .base_adapter import VisualAdapter
from .adapters.searxng_adapter import SearXNGAdapter
from .adapters.pexels_adapter import PexelsAdapter
from .adapters.tenor_adapter import TenorAdapter
from .llm_tagger import LLMIntentTagger, VisualIntent
from .clip_scorer import CLIPScorer
from .video_processor import VideoProcessor
from .attribution_manager import AttributionManager
from .advanced_cache import AdvancedCache, get_cache


class EnhancedVisualDirector:
    """
    Enhanced Visual Director that orchestrates all components
    to transform narrative segments into visual sequences.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        cache_dir: Optional[str] = None,
        enable_clip_scoring: bool = True
    ):
        """
        Initialize the enhanced visual director.
        
        Args:
            config: Configuration dictionary with API keys and settings
            cache_dir: Directory for caching
            enable_clip_scoring: Whether to use CLIP for semantic scoring
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = get_cache(cache_dir)
        
        # Initialize adapters
        self.adapters: Dict[str, VisualAdapter] = {}
        self._init_adapters()
        
        # Initialize components
        self.intent_tagger = LLMIntentTagger(
            model_name=config.get('llm_model', 'gpt-3.5-turbo'),
            api_key=config.get('openai_api_key')
        )
        
        self.clip_scorer = CLIPScorer() if enable_clip_scoring else None
        self.video_processor = VideoProcessor()
        self.attribution_manager = AttributionManager(project_dir=self.cache.cache_dir)
        
        # Asset selection weights
        self.weights = config.get('selection_weights', {
            'relevance': 0.4,
            'quality': 0.3,
            'diversity': 0.2,
            'semantic': 0.1
        })
    
    def _init_adapters(self):
        """Initialize available adapters based on config."""
        # SearXNG adapter
        if self.config.get('searxng_url'):
            self.adapters['searxng'] = SearXNGAdapter(
                base_url=self.config['searxng_url'],
                timeout=self.config.get('searxng_timeout', 30)
            )
        
        # Pexels adapter
        if self.config.get('pexels_api_key'):
            self.adapters['pexels'] = PexelsAdapter(
                api_key=self.config['pexels_api_key']
            )
        
        # Tenor adapter for GIFs
        if self.config.get('tenor_api_key'):
            self.adapters['tenor'] = TenorAdapter(
                api_key=self.config['tenor_api_key'],
                cache_dir=os.path.join(self.cache.cache_dir, 'tenor')
            )
    
    async def process_segment(
        self,
        segment: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single narrative segment into visual assets.
        
        Args:
            segment: Narrative segment with text, entities, etc.
            project_context: Overall project/video context
            
        Returns:
            Processed segment with visual assets and metadata
        """
        segment_id = segment.get('id', f"seg_{datetime.now().timestamp()}")
        
        # Step 1: Analyze intent and emotions
        visual_intent = await self._analyze_intent(segment, project_context)
        
        # Step 2: Generate search queries
        queries = self._generate_queries(segment, visual_intent)
        
        # Step 3: Search for assets
        all_assets = await self._search_assets(queries, visual_intent)
        
        # Step 4: Score and rank assets
        ranked_assets = await self._score_and_rank_assets(
            all_assets, 
            segment, 
            visual_intent
        )
        
        # Step 5: Select final assets
        selected_assets = self._select_final_assets(
            ranked_assets,
            visual_intent,
            segment.get('duration', 5.0)
        )
        
        # Step 6: Process videos if needed
        processed_assets = await self._process_video_assets(
            selected_assets,
            segment,
            visual_intent
        )
        
        # Step 7: Record usage and prepare result
        result = self._prepare_segment_result(
            segment,
            visual_intent,
            processed_assets,
            segment_id
        )
        
        # Cache canonical entities
        self._update_canonical_entities(segment, processed_assets)
        
        return result
    
    async def _analyze_intent(
        self,
        segment: Dict[str, Any],
        project_context: Optional[Dict[str, Any]]
    ) -> VisualIntent:
        """Analyze segment for visual intent and emotions."""
        try:
            # Try LLM analysis first
            intent = await self.intent_tagger.analyze_segment(
                segment,
                project_context
            )
            return intent
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}, using fallback")
            # Use fallback heuristics
            return self.intent_tagger.fallback_analysis(segment)
    
    def _generate_queries(
        self,
        segment: Dict[str, Any],
        intent: VisualIntent
    ) -> List[Tuple[str, str]]:
        """
        Generate search queries based on segment and intent.
        
        Returns:
            List of (query, adapter_name) tuples
        """
        queries = []
        
        # Entity-based queries
        for entity in segment.get('entities', []):
            entity_name = entity.get('text', '')
            entity_type = entity.get('type', 'MISC')
            
            # Check cache for canonical representation
            canonical = self.cache.get_canonical_entity(
                entity_name, 
                entity_type.lower()
            )
            
            if not canonical:
                # Generate queries for different adapters
                if entity_type in ['PERSON', 'ORG', 'LOC']:
                    queries.append((entity_name, 'pexels'))
                    queries.append((entity_name, 'searxng'))
        
        # Intent-based queries
        for visual_need in intent.visual_needs:
            query = visual_need['query']
            
            # Route to appropriate adapter based on need type
            if visual_need['type'] == 'emotion' and 'tenor' in self.adapters:
                queries.append((query, 'tenor'))
            elif visual_need['type'] in ['object', 'scene']:
                queries.append((query, 'pexels'))
                queries.append((query, 'searxng'))
            else:
                # Use all available adapters
                for adapter_name in self.adapters:
                    queries.append((query, adapter_name))
        
        # Topic-based queries from segment
        for topic in segment.get('topics', []):
            queries.append((topic, 'pexels'))
            queries.append((topic, 'searxng'))
        
        return queries
    
    async def _search_assets(
        self,
        queries: List[Tuple[str, str]],
        intent: VisualIntent
    ) -> List[Asset]:
        """Search for assets using adapters with caching."""
        all_assets = []
        seen_urls = set()
        
        for query, adapter_name in queries:
            if adapter_name not in self.adapters:
                continue
            
            # Check cache first
            cached_results = self.cache.get_cached_query_results(
                query, 
                adapter_name
            )
            
            if cached_results:
                self.logger.info(f"Cache hit for {query} on {adapter_name}")
                assets = cached_results
            else:
                # Perform search
                adapter = self.adapters[adapter_name]
                
                try:
                    # Special handling for Tenor emotion queries
                    if adapter_name == 'tenor' and intent.emotions:
                        assets = await adapter.search_by_emotion(
                            intent.emotions[0],
                            query,
                            limit=10
                        )
                    else:
                        assets = await adapter.search(query, limit=15)
                    
                    # Cache results
                    self.cache.cache_query_results(query, adapter_name, assets)
                    
                except Exception as e:
                    self.logger.error(f"Search failed for {query} on {adapter_name}: {e}")
                    assets = []
            
            # Deduplicate by URL
            for asset in assets:
                if asset.url not in seen_urls:
                    seen_urls.add(asset.url)
                    all_assets.append(asset)
                    # Cache individual asset
                    self.cache.cache_asset(asset)
        
        return all_assets
    
    async def _score_and_rank_assets(
        self,
        assets: List[Asset],
        segment: Dict[str, Any],
        intent: VisualIntent
    ) -> List[Asset]:
        """Score and rank assets based on multiple criteria."""
        if not assets:
            return []
        
        # Calculate semantic scores if CLIP is enabled
        if self.clip_scorer:
            # Combine text for CLIP scoring
            text_query = segment.get('text', '')
            if intent.visual_needs:
                text_query += ' ' + ' '.join(
                    vn['query'] for vn in intent.visual_needs
                )
            
            # Score assets with CLIP
            assets = await self.clip_scorer.score_assets(assets, text_query)
        
        # Calculate composite scores
        for asset in assets:
            # Get base scores
            relevance = asset.relevance_score
            quality = asset.quality_score
            diversity = asset.diversity_score
            semantic = getattr(asset, 'semantic_score', 0.5)
            
            # Apply weights
            composite_score = (
                self.weights['relevance'] * relevance +
                self.weights['quality'] * quality +
                self.weights['diversity'] * diversity +
                self.weights['semantic'] * semantic
            )
            
            # Boost score for certain conditions
            if asset.source == 'tenor' and intent.emotions:
                composite_score *= 1.2  # Boost emotion-matched GIFs
            
            if asset.licence in ['CC0', 'Public Domain']:
                composite_score *= 1.1  # Prefer freely usable content
            
            asset.composite_score = composite_score
        
        # Sort by composite score
        ranked_assets = sorted(
            assets,
            key=lambda a: getattr(a, 'composite_score', 0),
            reverse=True
        )
        
        return ranked_assets
    
    def _select_final_assets(
        self,
        ranked_assets: List[Asset],
        intent: VisualIntent,
        segment_duration: float
    ) -> List[Asset]:
        """Select final assets for the segment."""
        selected = []
        total_duration = 0
        used_sources = defaultdict(int)
        
        # Determine target number of assets
        target_count = max(1, int(segment_duration / 3))  # ~3 seconds per asset
        
        for asset in ranked_assets:
            if len(selected) >= target_count:
                break
            
            # Ensure diversity of sources
            if used_sources[asset.source] >= target_count // 2:
                continue
            
            # Check duration for videos
            if asset.type == 'video':
                if asset.duration and asset.duration > segment_duration * 1.5:
                    continue  # Skip videos that are too long
            
            selected.append(asset)
            used_sources[asset.source] += 1
            
            if asset.duration:
                total_duration += min(asset.duration, segment_duration / len(selected))
        
        # If we need emotion emphasis and have a GIF, ensure it's included
        if intent.emotions and not any(a.source == 'tenor' for a in selected):
            # Find best emotion GIF
            emotion_gifs = [a for a in ranked_assets if a.source == 'tenor']
            if emotion_gifs and len(selected) > 1:
                selected[-1] = emotion_gifs[0]  # Replace last asset
        
        return selected
    
    async def _process_video_assets(
        self,
        assets: List[Asset],
        segment: Dict[str, Any],
        intent: VisualIntent
    ) -> List[Asset]:
        """Process video assets (trimming, effects, etc.)."""
        processed = []
        segment_duration = segment.get('duration', 5.0)
        
        for i, asset in enumerate(assets):
            if asset.type == 'video' and asset.local_path:
                try:
                    # Determine processing needs
                    if asset.duration and asset.duration > segment_duration / len(assets):
                        # Trim video
                        start_time = 0
                        duration = segment_duration / len(assets)
                        
                        # Add some variety to start times
                        if asset.duration > duration * 2:
                            start_time = (asset.duration - duration) * (i / len(assets))
                        
                        output_path = f"{asset.local_path}_trimmed.mp4"
                        await self.video_processor.extract_clip(
                            asset.local_path,
                            output_path,
                            start_time,
                            duration
                        )
                        
                        # Update asset
                        asset.local_path = output_path
                        asset.duration = duration
                        asset.metadata['processed'] = True
                        asset.metadata['trim_start'] = start_time
                    
                    # Apply motion effects for emphasis
                    if intent.emphasis_level > 0.7:
                        output_path = f"{asset.local_path}_motion.mp4"
                        await self.video_processor.add_zoom_pan_effect(
                            asset.local_path,
                            output_path,
                            zoom_factor=1.0 + (intent.emphasis_level - 0.7)
                        )
                        asset.local_path = output_path
                        asset.metadata['motion_effect'] = 'zoom_pan'
                
                except Exception as e:
                    self.logger.error(f"Video processing failed for {asset.id}: {e}")
            
            processed.append(asset)
        
        return processed
    
    def _prepare_segment_result(
        self,
        segment: Dict[str, Any],
        intent: VisualIntent,
        assets: List[Asset],
        segment_id: str
    ) -> Dict[str, Any]:
        """Prepare the final segment result with all metadata."""
        # Generate attributions
        attributions = []
        for asset in assets:
            if asset.attribution or asset.licence != 'unknown':
                attr = self.attribution_manager.format_attribution(
                    asset.source,
                    asset.attribution,
                    asset.licence,
                    asset.url
                )
                attributions.append(attr)
        
        # Record usage
        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for asset in assets:
            self.cache.record_usage(
                asset,
                segment_id,
                project_id,
                performance_score=getattr(asset, 'composite_score', 0.5)
            )
        
        return {
            'id': segment_id,
            'text': segment.get('text', ''),
            'duration': segment.get('duration', 5.0),
            'visual_intent': {
                'emotions': intent.emotions,
                'visual_needs': intent.visual_needs,
                'emphasis_level': intent.emphasis_level,
                'scene_type': intent.scene_type
            },
            'assets': [
                {
                    'id': asset.id,
                    'url': asset.url,
                    'local_path': asset.local_path,
                    'type': asset.type,
                    'source': asset.source,
                    'duration': asset.duration,
                    'dimensions': asset.dimensions,
                    'scores': {
                        'relevance': asset.relevance_score,
                        'quality': asset.quality_score,
                        'semantic': getattr(asset, 'semantic_score', 0.0),
                        'composite': getattr(asset, 'composite_score', 0.0)
                    },
                    'metadata': asset.metadata
                }
                for asset in assets
            ],
            'attributions': attributions,
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'adapters_used': list(set(a.source for a in assets)),
                'cache_hits': self.cache.metrics['hits'],
                'processing_time': 0.0  # Would be set by timing decorator
            }
        }
    
    def _update_canonical_entities(
        self,
        segment: Dict[str, Any],
        assets: List[Asset]
    ):
        """Update canonical entity representations based on usage."""
        # Group assets by potential entity matches
        for entity in segment.get('entities', []):
            entity_name = entity.get('text', '')
            entity_type = entity.get('type', 'MISC').lower()
            
            # Find best performing asset for this entity
            entity_assets = [
                a for a in assets 
                if entity_name.lower() in a.metadata.get('query', '').lower()
            ]
            
            if entity_assets:
                best_asset = max(
                    entity_assets,
                    key=lambda a: getattr(a, 'composite_score', 0)
                )
                
                # Update canonical if this performs better
                current = self.cache.get_canonical_entity(entity_name, entity_type)
                if not current or getattr(best_asset, 'composite_score', 0) > 0.8:
                    self.cache.set_canonical_entity(
                        entity_name,
                        entity_type,
                        best_asset,
                        confidence=getattr(best_asset, 'composite_score', 0.5)
                    )
    
    async def process_video_project(
        self,
        segments: List[Dict[str, Any]],
        project_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an entire video project with multiple segments.
        
        Args:
            segments: List of narrative segments
            project_metadata: Overall project information
            
        Returns:
            Complete project result with all segments processed
        """
        processed_segments = []
        all_attributions = set()
        
        # Create project context for intent analysis
        project_context = {
            'title': project_metadata.get('title', ''),
            'topics': project_metadata.get('topics', []),
            'tone': project_metadata.get('tone', 'neutral')
        }
        
        # Process each segment
        for segment in segments:
            result = await self.process_segment(segment, project_context)
            processed_segments.append(result)
            
            # Collect attributions
            for attr in result.get('attributions', []):
                all_attributions.add(attr)
        
        # Generate attribution files
        project_id = project_metadata.get('id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        attribution_dir = os.path.join(self.cache.cache_dir, 'attributions', project_id)
        os.makedirs(attribution_dir, exist_ok=True)
        
        # Create various attribution formats
        attribution_files = {}
        
        # Text file
        attr_list = list(all_attributions)
        if attr_list:
            text_path = os.path.join(attribution_dir, 'attributions.txt')
            self.attribution_manager.create_attribution_file(attr_list, text_path)
            attribution_files['text'] = text_path
            
            # SRT file for video overlay
            srt_path = os.path.join(attribution_dir, 'attributions.srt')
            self.attribution_manager.create_srt_file(
                processed_segments,
                srt_path
            )
            attribution_files['srt'] = srt_path
            
            # JSON file
            json_path = os.path.join(attribution_dir, 'attributions.json')
            self.attribution_manager.create_json_file(
                processed_segments,
                json_path
            )
            attribution_files['json'] = json_path
            
            # HTML file
            html_path = os.path.join(attribution_dir, 'attributions.html')
            self.attribution_manager.create_html_file(
                attr_list,
                html_path,
                project_title=project_metadata.get('title', 'Video Project')
            )
            attribution_files['html'] = html_path
        
        # Get cache statistics
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'project_id': project_id,
            'metadata': project_metadata,
            'segments': processed_segments,
            'attribution_files': attribution_files,
            'statistics': {
                'total_segments': len(segments),
                'total_assets': sum(len(s['assets']) for s in processed_segments),
                'unique_sources': len(set(
                    asset['source'] 
                    for s in processed_segments 
                    for asset in s['assets']
                )),
                'cache_performance': cache_stats['performance'],
                'processing_complete': datetime.now().isoformat()
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        stats = {
            'cache': self.cache.get_cache_stats(),
            'adapters': {
                name: adapter.get_stats() if hasattr(adapter, 'get_stats') else {}
                for name, adapter in self.adapters.items()
            },
            'components': {
                'clip_enabled': self.clip_scorer is not None,
                'adapters_count': len(self.adapters),
                'video_processor_ready': self.video_processor is not None
            }
        }
        
        # Get popular assets
        popular_assets = self.cache.get_popular_assets(limit=10)
        stats['popular_assets'] = [
            {
                'id': asset.id,
                'source': asset.source,
                'usage_count': count,
                'type': asset.type
            }
            for asset, count in popular_assets
        ]
        
        return stats


async def create_visual_director(config_path: str) -> EnhancedVisualDirector:
    """
    Factory function to create a configured Visual Director instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured EnhancedVisualDirector instance
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create director
    director = EnhancedVisualDirector(
        config=config,
        cache_dir=config.get('cache_dir'),
        enable_clip_scoring=config.get('enable_clip_scoring', True)
    )
    
    # Run cleanup on startup
    deleted = director.cache.cleanup_expired()
    if deleted > 0:
        logging.info(f"Cleaned up {deleted} expired cache entries")
    
    return director


def run(run_dir: str, transcript: Dict[str, Any], creative_brief: Dict[str, Any], logger) -> Optional[Dict[str, Any]]:
    """
    Component interface wrapper for the Enhanced Visual Director.
    
    Args:
        run_dir: Directory for the current run
        transcript: Transcript data with word timings
        creative_brief: Creative brief with story information
        logger: Decision logger instance
        
    Returns:
        Dictionary containing visual results or None if failed
    """
    import asyncio
    import os
    import json
    from typing import Optional, Dict, Any
    
    try:
        logging.info("🎬 Starting Enhanced Visual Director process")
        
        # Load visual story plan
        story_plan_path = os.path.join(run_dir, 'visual_story_plan.json')
        if not os.path.exists(story_plan_path):
            logging.error("Visual story plan not found")
            return None
            
        with open(story_plan_path, 'r', encoding='utf-8') as f:
            visual_story_plan = json.load(f)
        
        # Load article text for context
        article_path = os.path.join(run_dir, 'article.txt')
        article_text = ""
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                article_text = f.read()
        
        # Load headline
        headline_path = os.path.join(run_dir, 'headline.json')
        headline = ""
        if os.path.exists(headline_path):
            with open(headline_path, 'r', encoding='utf-8') as f:
                headline_data = json.load(f)
                headline = headline_data.get('title', '')
        
        # Create configuration
        config = {
            'llm_model': 'gemini-2.0-flash-exp',
            'searxng_url': os.getenv('SEARXNG_URL', 'http://localhost:8888'),
            'pexels_api_key': os.getenv('PEXELS_API_KEY'),
            'tenor_api_key': os.getenv('TENOR_API_KEY'),
            'enable_clip_scoring': True,
            'selection_weights': {
                'relevance': 0.4,
                'quality': 0.3,
                'diversity': 0.2,
                'semantic': 0.1
            }
        }
        
        # Create director
        director = EnhancedVisualDirector(
            config=config,
            cache_dir=os.path.join(run_dir, 'cache'),
            enable_clip_scoring=True
        )
        
        # Process segments
        segments = visual_story_plan.get('visual_segments', [])
        
        # Create project metadata
        project_metadata = {
            'title': headline,
            'topics': creative_brief.get('topics', []),
            'tone': creative_brief.get('tone', 'neutral')
        }
        
        # Process project
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                director.process_video_project(segments, project_metadata)
            )
        finally:
            loop.close()
        
        # Convert to expected format
        visual_map = {}
        visual_timeline = []
        visual_timeline_simple = []
        
        for i, segment_result in enumerate(result['segments']):
            cue_id = f"visual_{i:02d}"
            
            # Find best asset for this segment
            if segment_result['assets']:
                best_asset = segment_result['assets'][0]  # Already ranked
                
                # Create visual map entry
                visual_map[cue_id] = best_asset['local_path']
                
                # Create timeline entries
                timeline_entry = {
                    'cue_id': cue_id,
                    'start_time': segment_result.get('start_time', i * 5),
                    'end_time': segment_result.get('end_time', (i + 1) * 5),
                    'trigger_keyword': segment_result.get('text', ''),
                    'visual_type': best_asset['type'],
                    'visual_file': best_asset['local_path'],
                    'asset_info': {
                        'source': best_asset['source'],
                        'type': best_asset['type'],
                        'score': best_asset['scores']['composite']
                    }
                }
                
                visual_timeline.append(timeline_entry)
                visual_timeline_simple.append({
                    'cue_id': cue_id,
                    'start_time': timeline_entry['start_time'],
                    'end_time': timeline_entry['end_time'],
                    'trigger_keyword': timeline_entry['trigger_keyword'],
                    'visual_type': timeline_entry['visual_type'],
                    'visual_file': timeline_entry['visual_file']
                })
                
                # Log decision
                logger.log_decision(
                    step=f"visual_selection_{cue_id}",
                    decision=f"Selected {best_asset['type']} from {best_asset['source']}",
                    reasoning=f"Best match with LLM-based adapter routing, score {best_asset['scores']['composite']:.2f}",
                    confidence=best_asset['scores']['composite'],
                    metadata={
                        'asset_id': best_asset['id'],
                        'asset_type': best_asset['type'],
                        'asset_source': best_asset['source'],
                        'scores': best_asset['scores'],
                        'segment_context': segment_result['visual_intent']
                    }
                )
        
        # Create final result
        final_result = {
            'visual_timeline': visual_timeline,
            'visual_timeline_simple': visual_timeline_simple,
            'visual_map': visual_map,
            'visual_strategy': {
                'opening_strategy': {
                    'screenshot_duration': 3.0
                },
                'total_visuals': len(visual_timeline_simple)
            },
            'segments': visual_timeline_simple,
            'attribution_files': result.get('attribution_files', {}),
            'statistics': result.get('statistics', {})
        }
        
        # Save visual map data
        visual_map_path = os.path.join(run_dir, 'visual_map.json')
        with open(visual_map_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2)
        
        # Summary decision
        logger.log_decision(
            step="visual_director_summary",
            decision="Completed visual acquisition using Enhanced Director",
            reasoning="Used LLM-based adapter routing with intelligent visual selection",
            confidence=0.95,
            metadata={
                'total_segments': len(visual_timeline_simple),
                'sources_used': list(set(t['asset_info']['source'] for t in visual_timeline)),
                'llm_routing_enabled': True,
                'adapters_available': len(director.adapters),
                'cache_performance': result['statistics'].get('cache_performance', {})
            }
        )
        
        logging.info(f"✅ Enhanced Visual Director completed successfully with {len(visual_timeline_simple)} visuals")
        return final_result
        
    except Exception as e:
        logging.error(f"❌ Enhanced Visual Director failed: {e}")
        logger.log_decision(
            step="enhanced_visual_director_error",
            decision="Enhanced Visual Director failed",
            reasoning=f"Error: {e}",
            confidence=0.0
        )
        return None
