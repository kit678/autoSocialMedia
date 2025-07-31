"""
Conventional Visual Director

This module handles the existing visual acquisition workflow using adapters 
like Pexels, SearXNG, and other external sources.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from .visual_director.asset_registry import get_registry, initialize_adapters
from .visual_director.asset_types import Asset, AssetScoringConfig
from .visual_director.asset_scorer import select_best_asset, get_scoring_report
from .visual_director.cache_manager import get_cache_manager
from .visual_director.llm_tagger import tag_segments_with_intent
from .visual_director.adapters import register_all_adapters
from .utils import get_audio_duration
from .media_utils import get_media_dimensions, get_orientation
from .visual_director.visual_timing_alignment import align_segments_with_transcript, ensure_segment_coverage, add_static_shots


def run(run_dir: str, transcript: Dict[str, Any], creative_brief: Dict[str, Any], logger) -> Optional[Dict[str, Any]]:
    """
    Main conventional visual director function using the existing adapter system.
    
    Args:
        run_dir: Directory for the current run
        transcript: Transcript data with word timings
        creative_brief: Creative brief with story information
        logger: Decision logger instance
        
    Returns:
        Dictionary containing visual results or None if failed
    """
    try:
        logging.info("🎬 Starting Conventional Visual Director process")
        
        # Initialize the adapter system
        register_all_adapters()
        initialize_adapters()
        registry = get_registry()
        cache_manager = get_cache_manager()
        
        # Load visual story plan
        story_plan_path = os.path.join(run_dir, 'visual_story_plan.json')
        if not os.path.exists(story_plan_path):
            logging.error("Visual story plan not found")
            return None
            
        with open(story_plan_path, 'r', encoding='utf-8') as f:
            visual_story_plan = json.load(f)
        
        logging.info("📋 Loaded visual story plan")
        
        # Get segments from the plan
        segments = visual_story_plan.get('visual_segments', [])
        
        # Load transcript data if available for timing alignment
        transcript_path = os.path.join(run_dir, 'transcript_data.json')
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Align segments with actual word timings
            logging.info("📍 Aligning visual segments with transcript timings...")
            segments = align_segments_with_transcript(segments, transcript_data)
            ensure_segment_coverage(segments, transcript_data.get('total_duration', 30.0))
        else:
            logging.warning("No transcript data found, using default timing")
        
        # Load article text for LLM tagging
        article_path = os.path.join(run_dir, 'article.txt')
        article_text = ""
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                article_text = f.read()
        
        # Get headline
        headline_path = os.path.join(run_dir, 'headline.json')
        headline = ""
        if os.path.exists(headline_path):
            with open(headline_path, 'r', encoding='utf-8') as f:
                headline_data = json.load(f)
                headline = headline_data.get('title', '')
        
        # Tag segments with intent and emotion if not already done
        segments_need_tagging = any(
            'intent' not in seg for seg in segments
        )
        
        if segments_need_tagging:
            logging.info("🏷️  Tagging segments with intent and emotion...")
            # Run tagging synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                segments = loop.run_until_complete(
                    tag_segments_with_intent(segments, article_text, headline, logger)
                )
            finally:
                loop.close()
        
        # Process each segment to find visuals
        enhanced_timeline = []
        selected_assets = []
        visual_map = {}
        
        # Configure scoring
        scoring_config = AssetScoringConfig(
            prefer_portrait=0.15,  # Strong preference for portrait
            prefer_video=0.1,
            source_bonuses={}  # Will be set per segment based on recommendations
        )
        
        # Get audio duration for timing
        audio_path = os.path.join(run_dir, 'voice.mp3')
        total_audio_duration = get_audio_duration(audio_path) if os.path.exists(audio_path) else 30.0
        
        for i, segment in enumerate(segments):
            cue_id = f"visual_{i:02d}"
            logging.info(f"\n📸 Processing {cue_id}: {segment.get('primary_search_term', '')}")
            
            # Use transcript-based timing if available
            if 'start_time' in segment and 'end_time' in segment:
                start_time = segment['start_time']
                end_time = segment['end_time']
            else:
                # Fallback to equal distribution
                segment_duration = total_audio_duration / len(segments)
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
            
            # Search for assets using the registry
            primary_term = segment.get('primary_search_term', '')
            if not primary_term:
                logging.warning(f"No search term for segment {i}")
                continue
            
            # Use recommended adapters from visual story plan
            recommended_adapters = segment.get('recommended_adapters', [])
            if not recommended_adapters:
                # Fallback to default if no recommendations
                priority_order = ['pexels', 'searxng']
                logging.warning(f"No adapter recommendations for segment {i}, using defaults")
            else:
                # Extract adapter names from recommendations, sorted by score
                priority_order = [adapter['name'] for adapter in recommended_adapters]
                logging.info(f"Using recommended adapters: {priority_order}")
            
            # Set source bonuses based on adapter recommendations
            segment_scoring_config = AssetScoringConfig(
                prefer_portrait=scoring_config.prefer_portrait,
                prefer_video=scoring_config.prefer_video,
                source_bonuses={}
            )
            
            # Give bonuses to recommended adapters based on their scores
            for adapter_info in recommended_adapters:
                adapter_name = adapter_info['name']
                adapter_score = adapter_info.get('score', 0.5)
                # Convert adapter recommendation score to bonus (0.1 to 0.3 range)
                bonus = 0.1 + (adapter_score * 0.2)
                segment_scoring_config.source_bonuses[adapter_name] = bonus
            
            # Search for assets
            all_assets = []
            for adapter_info in (recommended_adapters if recommended_adapters else [{'name': name} for name in priority_order]):
                adapter_name = adapter_info['name'] if isinstance(adapter_info, dict) else adapter_info
                adapter = registry.get_adapter(adapter_name)
                if not adapter:
                    continue
                
                # Check if adapter supports the preferred media type
                preferred_media = segment.get('preferred_media', 'image')
                if preferred_media == 'video' and not adapter.supports_type('video'):
                    continue
                
                # Search using the adapter
                try:
                    assets = registry.search_priority_adapters(
                        query=primary_term,
                        segment=segment,
                        priority_order=[adapter_name],
                        asset_type=preferred_media if preferred_media != 'any' else None,
                        limit=5
                    )
                    all_assets.extend(assets)
                    
                    if len(all_assets) >= 5:
                        break
                        
                except Exception as e:
                    logging.error(f"Search failed with {adapter_name}: {e}")
            
            # Select best asset using scoring
            best_asset = select_best_asset(
                all_assets,
                segment,
                selected_assets,
                segment_scoring_config,  # Use segment-specific config with adapter bonuses
                min_score=0.4
            )
            
            if not best_asset:
                logging.error(f"No suitable asset found for {cue_id}")
                # Try fallback search with secondary keywords
                if segment.get('secondary_keywords'):
                    for keyword in segment['secondary_keywords'][:2]:
                        assets = registry.search_priority_adapters(
                            query=keyword,
                            segment=segment,
                            priority_order=priority_order,
                            limit=3
                        )
                        best_asset = select_best_asset(
                            assets,
                            segment,
                            selected_assets,
                            segment_scoring_config,  # Use segment-specific config
                            min_score=0.3
                        )
                        if best_asset:
                            break
            
            if not best_asset:
                error_msg = f"CRITICAL: No visual found for segment {cue_id}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Download the asset
            local_path = cache_manager.download_asset_sync(best_asset)
            if not local_path:
                logging.error(f"Failed to download asset for {cue_id}")
                continue
            
            # Copy to run directory
            run_visual_path = cache_manager.copy_to_run_dir(best_asset, run_dir, cue_id)
            if not run_visual_path:
                run_visual_path = local_path  # Fallback to cache path
            
            # Add to selected assets
            selected_assets.append(best_asset)
            visual_map[cue_id] = run_visual_path
            
            # Log the decision
            scoring_report = get_scoring_report(
                best_asset,
                segment,
                selected_assets[:-1],  # Exclude current
                segment_scoring_config  # Use segment-specific config
            )
            
            logger.log_decision(
                step=f"visual_selection_{cue_id}",
                decision=f"Selected {best_asset.type} from {best_asset.source}",
                reasoning=f"Best match for '{primary_term}' with score {best_asset.composite_score:.2f}",
                confidence=best_asset.composite_score,
                metadata=scoring_report
            )
            
            # Create enhanced timeline entry
            enhanced_entry = {
                'cue_id': cue_id,
                'start_time': start_time,
                'end_time': end_time,
                'trigger_keyword': primary_term,
                'visual_type': segment.get('visual_type', 'concept'),
                'visual_file': run_visual_path,
                'asset_info': {
                    'source': best_asset.source,
                    'type': best_asset.type,
                    'licence': best_asset.licence,
                    'attribution': best_asset.attribution,
                    'dimensions': best_asset.dimensions,
                    'duration': best_asset.duration,
                    'score': best_asset.composite_score
                },
                'segment_metadata': {
                    'intent': segment.get('intent'),
                    'emotion': segment.get('emotion'),
                    'entities': segment.get('entities', []),
                    'preferred_media': segment.get('preferred_media')
                },
                'validation': {
                    'file_exists': os.path.exists(run_visual_path),
                    'narrative_text': segment.get('narrative_context', ''),
                    'search_terms_used': [primary_term] + segment.get('secondary_keywords', [])[:2],
                    'source_used': best_asset.source
                }
            }
            
            enhanced_timeline.append(enhanced_entry)
            
            logging.info(
                f"✅ {cue_id}: {best_asset.type} from {best_asset.source} "
                f"({os.path.basename(run_visual_path)}) - Score: {best_asset.composite_score:.2f}"
            )
        
        # Convert to simple timeline for video assembly compatibility
        simple_timeline = []
        for entry in enhanced_timeline:
            simple_entry = {
                'cue_id': entry['cue_id'],
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'trigger_keyword': entry['trigger_keyword'],
                'visual_type': entry['visual_type'],
                'visual_file': entry['visual_file']
            }
            simple_timeline.append(simple_entry)
        
        # Add opening webpage video and closing logo to timeline
        webpage_video_path = os.path.join(run_dir, 'webpage_capture.mp4')
        logo_path = "E:\\Dev\\AutoSocialMedia\\assets\\company_logo_closing.mp4"
        
        # Check if webpage video exists
        if not os.path.exists(webpage_video_path):
            logging.error(f"Webpage capture video not found: {webpage_video_path}")
            return None
            
        # Add static shots to timeline
        enhanced_timeline_with_static, visual_map = add_static_shots(
            enhanced_timeline,
            visual_map,
            webpage_video_path,
            logo_path,
            total_audio_duration,
            opening_duration=3.0,
            closing_duration=3.0
        )
        
        # Update the enhanced timeline
        enhanced_timeline = enhanced_timeline_with_static
        
        # Convert to simple timeline for video assembly compatibility (updated with static shots)
        simple_timeline = []
        for entry in enhanced_timeline:
            simple_entry = {
                'cue_id': entry['cue_id'],
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'trigger_keyword': entry['trigger_keyword'],
                'visual_type': entry['visual_type'],
                'visual_file': entry['visual_file']
            }
            simple_timeline.append(simple_entry)
        
        # Create output data structure
        result = {
            'visual_timeline': enhanced_timeline,
            'visual_timeline_simple': simple_timeline,
            'visual_map': visual_map,
            'visual_strategy': {
                'opening_strategy': {
                    'screenshot_duration': 3.0
                },
                'total_visuals': len(simple_timeline)
            },
            'segments': simple_timeline  # Add segments for compatibility with the slideshow component
        }
        
        # Save visual map data
        visual_map_path = os.path.join(run_dir, 'visual_map.json')
        with open(visual_map_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"✅ Conventional visual director completed successfully with {len(simple_timeline)} visuals")
        return result
        
    except Exception as e:
        logging.error(f"❌ Conventional visual director failed: {e}")
        logger.log_decision(
            step="conventional_visual_director_error",
            decision="Conventional visual director failed",
            reasoning=f"Error: {e}",
            confidence=0.0
        )
        return None
