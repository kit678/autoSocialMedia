"""
Enhanced Segment Processing

Extended processing pipeline with reaction detection and smart routing.
"""

import logging
from typing import List, Dict, Any, Optional

from .integrated_director import IntegratedVisualDirector
from .enhanced_segment import EnhancedSegment
from .reaction_detector import ReactionPoint
from .llm_intent_tagger import tag_segment_intent


async def process_segments_with_reactions(
    director: IntegratedVisualDirector,
    segments: List[Dict[str, Any]],
    article_context: Dict[str, Any],
    project_id: str
) -> List[EnhancedSegment]:
    """
    Process segments with full enhancement including reaction detection.
    
    Args:
        director: The integrated visual director instance
        segments: List of narrative segments
        article_context: Context about the article/story
        project_id: Unique project identifier
        
    Returns:
        List of EnhancedSegment objects with visuals and reactions
    """
    logger = logging.getLogger(__name__)
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        try:
            # 1. Create enhanced segment
            enhanced = EnhancedSegment.from_segment(segment)
            
            # 2. Analyze segment intent and emotion
            segment_intent = await tag_segment_intent(
                segment, article_context,
                segments[i-1] if i > 0 else None,
                segments[i+1] if i < len(segments)-1 else None
            )
            
            # 3. Determine media type
            media_type, confidence, reasoning = director.media_selector.select_media_type(
                segment, article_context
            )
            
            # Override with LLM intent if high confidence
            if segment_intent.confidence > 0.8:
                media_type = segment_intent.visual_type
            
            enhanced.visual_type = media_type
            enhanced.intent = segment_intent.intent
            enhanced.emotion = segment_intent.emotion
            
            # 4. Detect reaction points
            reaction_points = await director.reaction_detector.detect_reaction_points(segment)
            enhanced.reaction_points = reaction_points
            
            # 5. Search for primary visual asset with smart routing
            primary_assets = await director._search_assets_for_segment(
                segment, segment_intent, media_type
            )
            
            if primary_assets:
                best_asset = await director._select_best_asset(
                    primary_assets, segment, segment_intent
                )
                
                if best_asset:
                    # Process video if needed
                    if best_asset.type == 'video':
                        best_asset = await director._process_video_asset(best_asset, segment)
                    
                    enhanced.primary_visual = best_asset
                    
                    # Record attribution
                    director.attribution_manager.add_asset_attribution(best_asset, segment)
                    
                    # Cache for future use
                    director.cache.cache_asset(best_asset)
                    director.cache.record_usage(
                        best_asset, enhanced.id or f'seg_{i}',
                        project_id, 0.0
                    )
            
            # 6. Search for reaction assets for each reaction point
            for reaction_point in enhanced.reaction_points:
                await _find_reaction_asset(director, enhanced, reaction_point, project_id)
            
            enhanced_segments.append(enhanced)
            
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            # Create minimal enhanced segment on error
            enhanced = EnhancedSegment.from_segment(segment)
            enhanced_segments.append(enhanced)
    
    return enhanced_segments


async def _find_reaction_asset(
    director: IntegratedVisualDirector,
    enhanced: EnhancedSegment,
    reaction_point: ReactionPoint,
    project_id: str
):
    """Find and attach reaction asset for a specific reaction point."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get search terms for the emotion
        search_terms = director.reaction_detector.get_search_terms_for_emotion(
            reaction_point.emotion
        )
        
        if not search_terms:
            return
        
        # Create reaction-specific segment data
        reaction_segment = {
            'text': reaction_point.phrase,
            'visual_type': 'reaction',
            'emotion': reaction_point.emotion,
            'intent': 'reaction'
        }
        
        # Route to reaction-specialized adapters (Tenor first)
        available_adapters = list(director.registry._adapters.keys())
        
        # Manually prioritize Tenor for reactions
        if 'tenor' in available_adapters:
            available_adapters.remove('tenor')
            available_adapters.insert(0, 'tenor')
        
        # Search for reaction assets
        for search_term in search_terms[:2]:  # Try first 2 search terms
            adapter = director.registry.get_adapter('tenor')
            if adapter:
                try:
                    assets = await adapter.search(
                        search_term,
                        reaction_segment,
                        limit=5
                    )
                    
                    if assets:
                        # Pick the best reaction asset
                        best_reaction = assets[0]  # Tenor results are pre-sorted by relevance
                        
                        # Add reaction overlay
                        overlay = enhanced.add_reaction_overlay(reaction_point, best_reaction)
                        
                        # Record attribution
                        director.attribution_manager.add_asset_attribution(
                            best_reaction, reaction_segment
                        )
                        
                        # Cache the reaction asset
                        director.cache.cache_asset(best_reaction)
                        
                        logger.info(
                            f"Added {reaction_point.emotion} reaction for "
                            f"'{reaction_point.phrase}' at {reaction_point.start_time:.2f}s"
                        )
                        
                        break  # Found a reaction, stop searching
                        
                except Exception as e:
                    logger.warning(f"Failed to search Tenor for reaction: {e}")
        
        # If Tenor failed, try other adapters
        if not enhanced.reaction_overlays:
            # Fall back to searching other adapters for GIF/short video
            pass  # This could search Pexels or other sources for short videos
            
    except Exception as e:
        logger.error(f"Error finding reaction asset: {e}")


def convert_to_timeline_format(enhanced_segments: List[EnhancedSegment]) -> Dict[str, Any]:
    """
    Convert enhanced segments to timeline format for video generation.
    
    Returns a timeline with primary visuals and reaction overlays.
    """
    timeline = {
        'segments': [],
        'total_duration': 0.0,
        'reaction_overlays': []
    }
    
    for enhanced in enhanced_segments:
        segment_data = {
            'id': enhanced.id,
            'text': enhanced.text,
            'start_time': enhanced.start_time,
            'end_time': enhanced.end_time,
            'duration': enhanced.duration,
            'primary_visual': None,
            'has_reactions': enhanced.has_reactions()
        }
        
        # Add primary visual info
        if enhanced.primary_visual:
            segment_data['primary_visual'] = {
                'asset_id': enhanced.primary_visual.id,
                'type': enhanced.primary_visual.type,
                'url': enhanced.primary_visual.url,
                'local_path': enhanced.primary_visual.local_path,
                'duration': enhanced.primary_visual.duration
            }
        
        timeline['segments'].append(segment_data)
        
        # Add reaction overlays to global timeline
        for overlay in enhanced.reaction_overlays:
            overlay_data = {
                'segment_id': enhanced.id,
                'asset_id': overlay.asset.id,
                'type': 'reaction',
                'url': overlay.asset.url,
                'local_path': overlay.asset.local_path,
                'start_time': overlay.start_time,
                'end_time': overlay.end_time,
                'position': overlay.position,
                'scale': overlay.scale,
                'fade_in': overlay.fade_in,
                'fade_out': overlay.fade_out,
                'emotion': overlay.reaction_point.emotion
            }
            timeline['reaction_overlays'].append(overlay_data)
    
    # Calculate total duration
    if enhanced_segments:
        timeline['total_duration'] = enhanced_segments[-1].end_time
    
    return timeline
