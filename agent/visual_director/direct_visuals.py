import os
import logging
import json
from typing import Dict, Any, Optional, List

from ..images.search_google import search_with_searxng
from ..images.search_stock import search_pexels_primary  
from ..images.generate_ai import generate_ai_images
from ..video_sources.pexels_video import search_and_download_pexels_video
from ..visual_validation.gemini_validator import validate_visual_relevance
from ..utils import get_audio_duration

def run(run_dir: str, transcript: Dict[str, Any], creative_brief: Dict[str, Any], logger) -> Optional[Dict[str, Any]]:
    """
    Main visual director function that orchestrates the complete visual acquisition process.
    
    Args:
        run_dir: Directory for the current run
        transcript: Transcript data with word timings
        creative_brief: Creative brief with story information
        logger: Decision logger instance
        
    Returns:
        Dictionary containing visual results or None if failed
    """
    try:
        logging.info("🎬 Starting Visual Director process")
        
        # Step 1: Load existing visual story plan (created by script component)
        story_plan_path = os.path.join(run_dir, 'visual_story_plan.json')
        if not os.path.exists(story_plan_path):
            logging.error("Visual story plan not found - script component should have created it")
            return None
            
        with open(story_plan_path, 'r', encoding='utf-8') as f:
            visual_story_plan = json.load(f)
        
        logging.info("📋 Loaded existing visual story plan")
        
        # Step 2: Create enhanced visual timeline from the plan
        logging.info("⏱️  Creating enhanced visual timeline")
        enhanced_timeline = create_enhanced_visual_timeline(visual_story_plan, run_dir, logger)
        
        # Step 3: Convert to simple timeline for video assembly compatibility
        simple_timeline = _convert_enhanced_to_simple_timeline(enhanced_timeline)
        
        # Step 4: Create visual map from timeline
        visual_map = {}
        for entry in simple_timeline:
            cue_id = entry.get('cue_id')
            visual_file = entry.get('visual_file')
            if cue_id and visual_file:
                visual_map[cue_id] = visual_file
        
        # Step 5: Create output data structure
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
        
        # Step 6: Save visual map data
        visual_map_path = os.path.join(run_dir, 'visual_map.json')
        with open(visual_map_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"✅ Visual director completed successfully with {len(simple_timeline)} visuals")
        return result
        
    except Exception as e:
        logging.error(f"❌ Visual director failed: {e}")
        logger.log_decision(
            step="visual_director_error",
            decision="Visual director failed",
            reasoning=f"Error: {e}",
            confidence=0.0
        )
        return None

def create_enhanced_visual_timeline(visual_story_plan: Dict[str, Any], run_dir: str, logger) -> List[Dict[str, Any]]:
    """
    Creates enhanced visual timeline with embedded validation data.
    """
    segments = visual_story_plan.get('visual_segments', [])
    enhanced_timeline = []
    
    # Load transcript data to get proper timing
    transcript_path = os.path.join(run_dir, 'transcript_data.json')
    transcript_data = None
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            logging.info("📋 Loaded transcript data for visual-audio synchronization")
        except Exception as e:
            logging.warning(f"Failed to load transcript data: {e}")
    
    # Get total audio duration for fallback timing
    audio_path = os.path.join(run_dir, 'voice.mp3')
    total_audio_duration = get_audio_duration(audio_path) if os.path.exists(audio_path) else 30.0
    
    for i, segment in enumerate(segments):
        cue_id = f"visual_{i:02d}"
        
        # Calculate timing based on transcript data or fallback to even distribution
        if transcript_data and 'words' in transcript_data:
            # Find when this segment's keywords appear in the transcript
            primary_term = segment.get('primary_search_term', '').lower()
            secondary_terms = [term.lower() for term in segment.get('secondary_keywords', [])]
            all_terms = [primary_term] + secondary_terms
            
            # Find first occurrence of any term in the transcript
            start_time = None
            for word_data in transcript_data['words']:
                word_text = word_data.get('word', '').lower()
                if any(term in word_text or word_text in term for term in all_terms if term):
                    start_time = word_data.get('start', 0)
                    break
            
            if start_time is not None:
                # Calculate segment duration based on narrative length and speaking rate
                narrative_words = len(segment.get('narrative_context', '').split())
                words_per_second = 2.5  # Average speaking rate
                segment_duration = max(narrative_words / words_per_second, 2.0)  # Minimum 2 seconds
                end_time = min(start_time + segment_duration, total_audio_duration)
            else:
                # Fallback: distribute evenly across remaining time
                segment_duration = total_audio_duration / len(segments)
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
        else:
            # Fallback: even distribution when no transcript available
            segment_duration = total_audio_duration / len(segments)
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
        
        # Create planned search from segment data
        planned_search = {
            'search_terms': [segment.get('primary_search_term', '')] + segment.get('secondary_keywords', []),
            'source_priority': ['SearXNG', 'Pexels Photos', 'AI Generation', 'Pexels Video'],
            'narrative_context': segment.get('narrative_context', ''),
            'visual_type': segment.get('visual_type', 'concept')
        }
        
        # Execute search and get validated visual
        visual_file = _execute_planned_search(planned_search, run_dir, cue_id, logger)
        
        # Check if visual file exists
        file_exists = visual_file and os.path.exists(visual_file)
        if not file_exists:
            error_msg = f"CRITICAL: No visual found for segment {cue_id}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg + " Pipeline configured to crash on missing visuals.")
        
        # Create enhanced timeline entry with proper timing
        enhanced_entry = {
            'cue_id': cue_id,
            'start_time': start_time,
            'end_time': end_time,
            'trigger_keyword': segment.get('primary_search_term', ''),
            'visual_type': segment.get('visual_type', 'concept'),
            'visual_file': visual_file,
            'validation': {
                'file_exists': file_exists,
                'narrative_text': segment.get('narrative_context', ''),
                'word_count': len(segment.get('narrative_context', '').split()),
                'search_terms_used': planned_search['search_terms'][:3],  # First 3 terms
                'source_used': _get_source_from_path(visual_file),
                'timing_method': 'transcript_sync' if transcript_data else 'even_distribution'
            }
        }
        
        enhanced_timeline.append(enhanced_entry)
        
        duration = end_time - start_time
        logging.info(f"✅ Enhanced timeline entry {cue_id}: {os.path.basename(visual_file) if visual_file else 'MISSING'} ({start_time:.1f}s - {end_time:.1f}s, {duration:.1f}s)")
    
    return enhanced_timeline

def _convert_enhanced_to_simple_timeline(enhanced_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts enhanced timeline to simple format for video assembly compatibility.
    """
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
    
    return simple_timeline

def _get_source_from_path(file_path: str) -> str:
    """
    Determines the source of a visual file based on its path.
    """
    if not file_path:
        return 'failed'
    
    filename = os.path.basename(file_path).lower()
    
    if 'pexels_video' in filename or filename.endswith('.mp4'):
        return 'pexels_video'
    elif 'pexels' in filename:
        return 'pexels_photos'
    elif 'searxng' in filename or 'google' in filename:
        return 'searxng'
    elif 'ai_generated' in filename:
        return 'ai_generated'
    else:
        return 'unknown'

def _map_visual_type_to_pexels_category(visual_type: str) -> Optional[str]:
    """Maps the internal visual type to a Pexels-compatible category."""
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
    
    # These are the valid categories from Pexels API documentation
    valid_categories = [
        "backgrounds", "fashion", "nature", "science", "education", 
        "feelings", "health", "people", "religion", "places", "animals", 
        "industry", "computer", "food", "sports", "transportation", 
        "travel", "buildings", "business", "music"
    ]
    
    category = mapping.get(visual_type)
    return category if category in valid_categories else None

def _execute_planned_search(planned_search: Dict[str, Any], run_dir: str, cue_id: str, logger: 'DecisionLogger') -> Optional[str]:
    """
    Executes a planned search for a single visual segment, fetching and validating one visual.
    
    Returns:
        A single validated visual file path, or None if no suitable visual found.
    """
    search_terms = planned_search.get('search_terms', [])
    source_priority = planned_search.get('source_priority', [])
    narrative_context = planned_search.get('narrative_context', "")
    visual_type = planned_search.get('visual_type', 'concept')
    
    # Get Pexels category from visual type
    pexels_category = _map_visual_type_to_pexels_category(visual_type)
    
    visuals_dir = os.path.join(run_dir, 'visuals')
    validated_visual = None
    
    if not narrative_context:
        logging.warning(f"    > Skipping validation for cue {cue_id} due to missing narrative context.")
        return None
    
    # Define common English stop words to filter from the narrative context
    stop_words = {"a", "an", "the", "and", "but", "or", "in", "on", "at", "to", "of", "for", "by", "with", "is", "was", "are", "were", "it", "this", "that", "he", "she", "they"}
    
    # Iterate through each search term (primary, secondary, etc.)
    for term in search_terms:
        if validated_visual:
            break
            
        # Iterate through each search source (Pexels, SearXNG, etc.)
        for source in source_priority:
            if validated_visual:
                break

            logging.info(f"    > Searching {source} for: '{term}'" + (f" in category '{pexels_category}'" if pexels_category and source == "Pexels Photos" else ""))
            
            # Fetch up to 3 candidates from the source
            candidates = []
            if source == "SearXNG":
                # Enrich the query with the first 4 meaningful words from the narrative
                all_words = narrative_context.lower().split()
                meaningful_words = [word for word in all_words if word not in stop_words]
                context_keywords = meaningful_words[:4]
                candidates = search_with_searxng([term], 3, visuals_dir, context_keywords=context_keywords)
            elif source == "Pexels Photos":
                candidates = search_pexels_primary([term], 3, visuals_dir, category=pexels_category)
            elif source == "AI Generation":
                candidates = generate_ai_images([term], 1, visuals_dir) # AI gen is slow, only 1
            elif source == "Pexels Video":
                # Video search returns one validated path for now (no category support in video API)
                video_path = search_and_download_pexels_video(term, visuals_dir, f"{cue_id}")
                if video_path:
                    candidates.append(video_path)
            
            if not candidates:
                continue

            # Validate each candidate until we find one good one
            for candidate_path in candidates:
                if validated_visual:
                    break
                
                # Videos are not validated by Gemini Vision for now
                if candidate_path.endswith(('.mp4', '.mov')):
                    validated_visual = candidate_path
                    logging.info(f"      > SUCCESS (Video): {os.path.basename(candidate_path)}")
                    break

                logging.info(f"      > Validating: {os.path.basename(candidate_path)}")
                
                # Corrected call to the validator
                validation_result = validate_visual_relevance(
                    image_path=candidate_path,
                    expected_concept=narrative_context,
                    search_terms=planned_search.get('search_terms', []),
                    visual_type=planned_search.get('visual_type', 'concept'),
                    context=narrative_context
                )
                
                if validation_result.get('is_relevant', False):
                    validated_visual = candidate_path
                    score = validation_result.get('confidence', 0.0)
                    reason = validation_result.get('description', 'No reason provided')
                    # Clean reason text to avoid Unicode encoding issues
                    reason = reason.encode('ascii', 'ignore').decode('ascii') if isinstance(reason, str) else str(reason)
                    logging.info(f"      > SUCCESS (Score: {score:.2f}): {os.path.basename(candidate_path)} - {reason}")
                    break
                else:
                    score = validation_result.get('confidence', 0.0)
                    issues = validation_result.get('issues', ['No reason provided'])
                    reason = ", ".join(str(issue) for issue in issues)
                    # Clean reason text to avoid Unicode encoding issues
                    reason = reason.encode('ascii', 'ignore').decode('ascii') if isinstance(reason, str) else str(reason)
                    logging.warning(f"      > REJECTED (Score: {score:.2f}): {os.path.basename(candidate_path)} - {reason}")
                    # Optional: Delete irrelevant image to save space
                    try:
                        os.remove(candidate_path)
                    except OSError as e:
                        logging.error(f"Could not delete rejected image: {e}")

    return validated_visual 