"""
Smart Visual Collector - Gathers visuals from multiple sources based on AI direction
Enhanced with Gemini Vision validation for relevance checking
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from agent.decision_logger import log_decision
from agent.visual_validation.gemini_validator import (
    validate_visual_relevance,
    generate_search_variations,
    select_best_visual
)

def _clean_up_image_and_metadata(image_path: str):
    """Clean up both image file and its URL metadata file"""
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
        url_metadata_path = image_path + ".url"
        if os.path.exists(url_metadata_path):
            os.remove(url_metadata_path)
    except Exception as e:
        logging.debug(f"Failed to clean up {image_path}: {e}")

def _safe_str_lower(value: Any, default: str = '') -> str:
    """Safely convert value to lowercase string with fallback."""
    if value is None:
        return default.lower()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value).lower()
    if isinstance(value, str):
        return value.lower()
    # Fallback for other types
    return str(value).lower()

def _safe_bool_conversion(value: Union[bool, str, Any], default: bool = False) -> bool:
    """Safely convert various input types to boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    if isinstance(value, (int, float)):
        return bool(value)
    # Fallback for other types
    try:
        return bool(value)
    except:
        return default

def _safe_get_string(data: Dict[str, Any], key: str, default: str = '') -> str:
    """Safely get string value from dictionary with type conversion."""
    value = data.get(key, default)
    if isinstance(value, str):
        return value
    if value is None:
        return default
    # Convert other types to string
    return str(value)

def _validate_and_fix_cue(cue: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix a cue object to ensure all required fields are present and properly typed."""
    fixed_cue = cue.copy()
    
    # Ensure required string fields exist
    required_string_fields = {
        'trigger_keyword': 'unknown_concept',
        'visual_type': 'concept',
        'priority': 'medium',
        'context': '',
        'effect': 'fade'
    }
    
    for field, default_value in required_string_fields.items():
        fixed_cue[field] = _safe_get_string(fixed_cue, field, default_value)
    
    # Ensure timing fields exist and are numeric
    if 'start_time' not in fixed_cue or not isinstance(fixed_cue['start_time'], (int, float)):
        fixed_cue['start_time'] = 0.0
    if 'end_time' not in fixed_cue or not isinstance(fixed_cue['end_time'], (int, float)):
        fixed_cue['end_time'] = 3.0
    
    # Ensure source_recommendation exists and is a dict
    if 'source_recommendation' not in fixed_cue or not isinstance(fixed_cue['source_recommendation'], dict):
        fixed_cue['source_recommendation'] = {}
    
    return fixed_cue

# Confidence thresholds for different sources
CONFIDENCE_THRESHOLDS = {
    "google": 0.8,      # High standard for Google images (specific entities)
    "stock_photo": 0.7,  # Medium standard for stock photos  
    "stock_video": 0.7,  # Medium standard for stock videos
    "ai_generated": 0.5  # Lower standard for AI generation (fallback)
}

# Maximum attempts per source before moving to next fallback
MAX_ATTEMPTS_PER_SOURCE = {
    "google": 5,         # 5 search variations for Google
    "stock_photo": 5,    # 5 search variations for stock
    "stock_video": 3,    # 3 search variations for video
    "ai_generated": 3    # 3 generation attempts
}

def gather_visuals_for_timeline(visual_timeline: List[Dict[str, Any]], 
                              visual_strategy: Dict[str, Any],
                              article_images: List[str],
                              screenshot_path: str) -> Dict[str, Any]:
    """
    Gather visuals for each timeline cue using comprehensive fallback system
    
    Args:
        visual_timeline: List of visual cues with timing
        visual_strategy: AI-determined visual strategy  
        article_images: Pre-downloaded article images
        screenshot_path: Website screenshot path
        
    Returns:
        Dict mapping visual concepts to file paths
    """
    try:
        logging.info(f"🎯 Starting comprehensive visual gathering for {len(visual_timeline)} timeline cues")
        
        # Create visuals directory
        visuals_dir = os.path.join("runs", "current", "visuals")
        os.makedirs(visuals_dir, exist_ok=True)
        
        visual_map = {}
        success_count = 0
        failure_count = 0
        
        # Process each visual cue with comprehensive fallback
        for i, cue in enumerate(visual_timeline):
            cue_id = f"visual_{i:02d}"
            
            # Validate and fix cue data to prevent type errors
            fixed_cue = _validate_and_fix_cue(cue)
            trigger_keyword = fixed_cue['trigger_keyword']
            visual_type = fixed_cue['visual_type']
            
            logging.info(f"\n🔍 Processing {cue_id}: '{trigger_keyword}' ({visual_type})")
            
            # Try comprehensive fallback system
            visual_path = gather_visual_with_fallback(fixed_cue, visuals_dir, cue_id)
            
            if visual_path:
                visual_map[cue_id] = {
                    'path': visual_path,
                    'concept': trigger_keyword,
                    'start_time': fixed_cue['start_time'],
                    'end_time': fixed_cue['end_time'],
                    'effect': fixed_cue['effect'],
                    'visual_type': fixed_cue['visual_type'],
                    'source': determine_source_from_path(visual_path)
                }
                success_count += 1
                logging.info(f"✅ {cue_id}: SUCCESS - {os.path.basename(visual_path)}")
            else:
                # Even with comprehensive fallback, this visual failed completely
                visual_map[cue_id] = {
                    'path': None,
                    'concept': trigger_keyword,
                    'start_time': fixed_cue['start_time'],
                    'end_time': fixed_cue['end_time'],
                    'effect': fixed_cue['effect'],
                    'visual_type': fixed_cue['visual_type'],
                    'source': 'failed'
                }
                failure_count += 1
                logging.error(f"❌ {cue_id}: COMPLETE FAILURE - No visual found despite comprehensive fallback")
        
        # Calculate success metrics
        total_cues = len(visual_timeline)
        success_rate = (success_count / total_cues) * 100 if total_cues > 0 else 0
        
        logging.info(f"\n📊 VISUAL GATHERING SUMMARY:")
        logging.info(f"   Total cues: {total_cues}")
        logging.info(f"   Successful: {success_count} ({success_rate:.1f}%)")
        logging.info(f"   Failed: {failure_count}")
        
        # Log comprehensive gathering results
        log_decision(
            step="comprehensive_visual_gathering",
            decision=f"Completed visual gathering with {success_rate:.1f}% success rate",
            reasoning=f"Used comprehensive fallback system to gather {success_count}/{total_cues} visuals",
            confidence=success_rate / 100,
            alternatives=[f"Failed visuals: {failure_count}"] if failure_count > 0 else [],
            metadata={
                "total_cues": total_cues,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "visual_map": {k: v['source'] for k, v in visual_map.items()}
            }
        )
        
        return visual_map
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR: Visual gathering system failed: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

def determine_source_from_path(visual_path: str) -> str:
    """Determine which source was used based on file path"""
    if not visual_path:
        return 'failed'
    
    filename = os.path.basename(visual_path).lower()
    
    if 'google' in filename:
        return 'google'
    elif 'stock_photo' in filename or 'pexels' in filename:
        return 'stock_photo'
    elif 'stock_video' in filename or '.mp4' in filename or '.webm' in filename:
        return 'stock_video'
    elif 'ai_generated' in filename:
        return 'ai_generated'
    else:
        return 'unknown'

def determine_source_priority(cue: Dict[str, Any]) -> List[str]:
    """Determine fallback order based on visual type and content assessment recommendations"""
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    
    # Check if content assessment specifically recommended stock_video
    source_recommendation = cue.get('source_recommendation', {})
    primary_source = _safe_get_string(source_recommendation, 'primary', '')
    requires_motion = _safe_bool_conversion(source_recommendation.get('requires_motion', False))
    
    # Honor content assessment's stock_video recommendation
    if primary_source == 'stock_video' or requires_motion:
        return ['stock_video', 'stock_photo', 'ai_generated']
    elif primary_source == 'stock_photo':
        return ['stock_photo', 'google', 'ai_generated']
    elif primary_source == 'google':
        return ['google', 'stock_photo', 'ai_generated']
    elif primary_source == 'ai_generation':
        return ['ai_generated', 'stock_photo', 'google']
    
    # Fallback to original visual_type logic if no specific recommendation
    if visual_type in ['company', 'person', 'location', 'product']:
        # Specific entities: Google → Stock → AI
        return ['google', 'stock_photo', 'ai_generated']
    elif visual_type == 'action':
        # Actions: Stock Video → Stock Photo → AI
        return ['stock_video', 'stock_photo', 'ai_generated']
    else:
        # Concepts: Stock Photo → Google → AI
        return ['stock_photo', 'google', 'ai_generated']

def try_article_images(cue: Dict[str, Any], article_images: List[str], 
                      visuals_dir: str, cue_id: str) -> Optional[str]:
    """
    Try to find relevant image from article images - REMOVED FALLBACK BEHAVIOR
    """
    if not article_images:
        return None
    
    # REMOVED: The problematic "always use first image" fallback
    # This was causing the same image to be repeated for every visual cue
    # Instead, we now require specific visual gathering to work properly
    
    logging.info(f"Article images available but not using fallback for {cue_id}")
    logging.info(f"Pipeline configured to crash instead of using repetitive fallback visuals")
    
    # Return None to force proper visual gathering
    return None

def try_stock_footage(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """
    Search for stock footage/video clips with validation and retry logic
    """
    from ..video_sources.pexels_video import search_and_download_pexels_video
    
    # Generate improved search variations
    search_variations = generate_search_variations(cue)
    candidates = []
    max_attempts_per_term = 2  # Limit video downloads per term
    
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    priority = _safe_get_string(cue, 'priority', 'medium')
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    logging.info(f"🎬 Stock footage search with validation for '{trigger_keyword}'")
    logging.info(f"Search variations: {search_variations}")
    
    # Determine duration preference based on visual context
    context_str = _safe_get_string(cue, 'context', '')
    if visual_type == "action" or "action" in context_str.lower():
        duration_pref = "short"  # Action clips are usually short and punchy
    elif priority == "high":
        duration_pref = "medium"  # Important content gets longer clips
    else:
        duration_pref = "short"  # Default to short clips for fast pacing
    
    try:
        # Try each search variation
        for search_term in search_variations[:3]:  # Limit to 3 variations
            for attempt in range(max_attempts_per_term):
                try:
                    # Create unique filename for this candidate
                    temp_filename = f"{cue_id}_footage_candidate_{len(candidates):02d}"
                    
                    video_path = search_and_download_pexels_video(
                        query=search_term,
                        output_dir=visuals_dir,
                        filename=temp_filename,
                        duration_preference=duration_pref
                    )
                    
                    if video_path:
                        # Validate video relevance
                        validation = validate_visual_relevance(
                            video_path,
                            trigger_keyword,
                            [search_term],
                            visual_type,
                            _safe_get_string(cue, 'context', '')
                        )
                        
                        if validation['is_relevant'] and validation['confidence'] > 0.6:
                            candidates.append(video_path)
                            logging.info(f"✅ Found relevant stock footage: {search_term} (confidence: {validation['confidence']:.2f})")
                            
                            # Early return if we have a high-confidence match
                            if validation['confidence'] > 0.85:
                                break
                        else:
                            # Remove irrelevant candidate
                            try:
                                os.remove(video_path)
                            except:
                                pass
                            logging.debug(f"❌ Rejected stock footage: {search_term} (confidence: {validation['confidence']:.2f})")
                    else:
                        logging.debug(f"No footage found for '{search_term}' (attempt {attempt + 1})")
                        
                except Exception as e:
                    logging.warning(f"Stock footage search failed for '{search_term}' (attempt {attempt + 1}): {e}")
                    continue
                    
                # Break out of attempts loop if we found a good candidate
                if candidates and validation['confidence'] > 0.85:
                    break
        
        # Select best candidate if any found
        if candidates:
            best_path = select_best_visual(candidates, cue)
            if best_path:
                # Rename to final filename
                final_path = os.path.join(visuals_dir, f"{cue_id}_footage.mp4")
                import shutil
                shutil.move(best_path, final_path)
                logging.info(f"🎯 Selected best stock footage for '{trigger_keyword}'")
                return final_path
        
        logging.info(f"No relevant stock footage found for '{trigger_keyword}'")
        return None
            
    except Exception as e:
        logging.warning(f"Stock footage search with validation failed: {e}")
        return None

def try_stock_photos(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """
    Search for stock photos with validation and retry logic
    """
    from agent.images.search_stock import search_stock_images
    import shutil
    
    # Generate improved search variations
    search_variations = generate_search_variations(cue)
    candidates = []
    max_attempts_per_term = 3
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    
    logging.info(f"🔍 Stock photo search with validation for '{trigger_keyword}'")
    logging.info(f"Search variations: {search_variations}")
    
    try:
        # Try each search variation
        for search_term in search_variations[:3]:  # Limit to 3 variations
            try:
                # Get multiple candidates per search term
                image_paths = search_stock_images([search_term], max_attempts_per_term, visuals_dir)
                
                if image_paths:
                    for i, original_path in enumerate(image_paths):
                        # Create unique filename for this candidate
                        temp_filename = f"{cue_id}_stock_candidate_{len(candidates):02d}.jpg"
                        candidate_path = os.path.join(visuals_dir, temp_filename)
                        
                        # Move to standardized location
                        if original_path != candidate_path:
                            shutil.move(original_path, candidate_path)
                        
                        # Validate relevance
                        validation = validate_visual_relevance(
                            candidate_path,
                            trigger_keyword,
                            [search_term],
                            visual_type,
                            _safe_get_string(cue, 'context', '')
                        )
                        
                        if validation['is_relevant'] and validation['confidence'] > 0.6:
                            candidates.append(candidate_path)
                            logging.info(f"✅ Found relevant stock photo: {search_term} (confidence: {validation['confidence']:.2f})")
                            
                            # Early return if we have a high-confidence match
                            if validation['confidence'] > 0.9:
                                break
                        else:
                            # Remove irrelevant candidate and its metadata
                            _clean_up_image_and_metadata(candidate_path)
                            logging.debug(f"❌ Rejected stock photo: {search_term} (confidence: {validation['confidence']:.2f})")
                            
            except Exception as e:
                logging.warning(f"Stock photo search failed for '{search_term}': {e}")
                continue
        
        # Select best candidate if any found
        if candidates:
            best_path = select_best_visual(candidates, cue)
            if best_path:
                # Rename to final filename
                final_path = os.path.join(visuals_dir, f"{cue_id}_stock.jpg")
                shutil.move(best_path, final_path)
                logging.info(f"🎯 Selected best stock photo for '{trigger_keyword}'")
                return final_path
        
        logging.info(f"No relevant stock photos found for '{trigger_keyword}'")
        return None
            
    except Exception as e:
        logging.warning(f"Stock photo search with validation failed: {e}")
        return None

def try_google_images(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """
    Search for Google images with validation and retry logic
    """
    from agent.images.search_google import search_google_images
    import shutil
    
    # Generate improved search variations
    search_variations = generate_search_variations(cue)
    candidates = []
    max_attempts_per_term = 3
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    
    logging.info(f"🔍 Google image search with validation for '{trigger_keyword}'")
    logging.info(f"Search variations: {search_variations}")
    
    try:
        # Try each search variation
        for search_term in search_variations[:3]:  # Limit to 3 variations
            try:
                # Get multiple candidates per search term
                image_paths = search_google_images([search_term], max_attempts_per_term, visuals_dir)
                
                if image_paths:
                    for i, original_path in enumerate(image_paths):
                        # Create unique filename for this candidate
                        temp_filename = f"{cue_id}_google_candidate_{len(candidates):02d}.jpg"
                        candidate_path = os.path.join(visuals_dir, temp_filename)
                        
                        # Move to standardized location
                        if original_path != candidate_path:
                            shutil.move(original_path, candidate_path)
                        
                        # Validate relevance
                        validation = validate_visual_relevance(
                            candidate_path,
                            trigger_keyword,
                            [search_term],
                            visual_type,
                            _safe_get_string(cue, 'context', '')
                        )
                        
                        if validation['is_relevant'] and validation['confidence'] > 0.6:
                            candidates.append(candidate_path)
                            logging.info(f"✅ Found relevant Google image: {search_term} (confidence: {validation['confidence']:.2f})")
                            
                            # Early return if we have a high-confidence match
                            if validation['confidence'] > 0.9:
                                break
                        else:
                            # Remove irrelevant candidate and its metadata
                            _clean_up_image_and_metadata(candidate_path)
                            logging.debug(f"❌ Rejected Google image: {search_term} (confidence: {validation['confidence']:.2f})")
                            
            except Exception as e:
                logging.warning(f"Google image search failed for '{search_term}': {e}")
                continue
        
        # Select best candidate if any found
        if candidates:
            best_path = select_best_visual(candidates, cue)
            if best_path:
                # Rename to final filename
                final_path = os.path.join(visuals_dir, f"{cue_id}_google.jpg")
                shutil.move(best_path, final_path)
                logging.info(f"🎯 Selected best Google image for '{trigger_keyword}'")
                return final_path
        
        logging.info(f"No relevant Google images found for '{trigger_keyword}'")
        return None
            
    except Exception as e:
        logging.warning(f"Google image search with validation failed: {e}")
        return None

def create_text_visual(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> str:
    """
    Create a text-based visual as fallback
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    
    # Create a simple text image
    width, height = 1080, 1920  # 9:16 aspect ratio
    
    # Create image with dark background
    img = Image.new('RGB', (width, height), color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    # Prepare text
    text = _safe_get_string(cue, 'trigger_keyword', 'UNKNOWN').upper()
    
    try:
        # Try to load a nice font
        font = ImageFont.truetype("arial.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with outline
    outline_width = 3
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill='black')
    
    draw.text((x, y), text, font=font, fill='white')
    
    # Save image
    output_path = os.path.join(visuals_dir, f"{cue_id}_text.png")
    img.save(output_path)
    
    return output_path

def process_opening_screenshot(screenshot_path: str, opening_strategy: Dict[str, Any], 
                             output_dir: str) -> str:
    """
    Process the opening screenshot according to AI strategy
    """
    from PIL import Image
    import os
    
    try:
        # Load screenshot
        img = Image.open(screenshot_path)
        
        # For now, just resize to proper aspect ratio
        # Could be enhanced with AI-powered smart cropping based on zoom_focus
        target_width, target_height = 1080, 1920  # 9:16
        
        # Resize maintaining aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Image is wider, crop sides
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller, crop top/bottom
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        # Resize to target dimensions
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Save processed screenshot
        processed_path = os.path.join(output_dir, 'opening_screenshot.jpg')
        img.save(processed_path, 'JPEG', quality=95)
        
        zoom_focus = opening_strategy.get('zoom_focus', 'center')
        logging.info(f"Processed opening screenshot: {zoom_focus}")
        
        return processed_path
        
    except Exception as e:
        logging.error(f"Error processing opening screenshot: {e}")
        # Return original if processing fails
        return screenshot_path 

def gather_visual_with_fallback(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """
    Comprehensive fallback: Primary → Secondary → AI Generation
    
    Args:
        cue: Visual cue with trigger_keyword, visual_type, etc.
        visuals_dir: Directory to save visuals
        cue_id: Unique identifier for this visual
        
    Returns:
        Path to successful visual or None if all sources fail
    """
    sources = determine_source_priority(cue)
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    
    logging.info(f"🎯 Visual gathering for '{trigger_keyword}' using fallback chain: {sources}")
    
    for source in sources:
        logging.info(f"🔄 Trying {source} for '{trigger_keyword}'")
        
        result = None
        try:
            if source == "google":
                result = try_google_with_persistence(cue, visuals_dir, cue_id)
            elif source == "stock_photo":
                result = try_stock_photos_with_persistence(cue, visuals_dir, cue_id)
            elif source == "stock_video":
                result = try_stock_video_with_persistence(cue, visuals_dir, cue_id)
            elif source == "ai_generated":
                result = try_ai_generation(cue, visuals_dir, cue_id)
            
            if result:
                logging.info(f"✅ Found visual using {source}: {os.path.basename(result)}")
                
                # Log successful gathering
                log_decision(
                    step="visual_gathering_success",
                    decision=f"Successfully found visual using {source}",
                    reasoning=f"Source {source} provided relevant visual for {trigger_keyword}",
                    metadata={
                        "source_used": source,
                        "trigger_keyword": trigger_keyword,
                        "visual_type": _safe_get_string(cue, 'visual_type', 'concept'),
                        "visual_path": result
                    }
                )
                
                return result
            else:
                logging.warning(f"❌ {source} failed for '{trigger_keyword}'")
        
        except Exception as e:
            logging.error(f"❌ {source} failed with error: {e}")
    
    # CRITICAL: Log detailed failure analysis
    log_gathering_failure(cue, sources)
    return None 

def try_google_with_persistence(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """Keep trying Google until high confidence found or max attempts reached"""
    attempts = 0
    max_attempts = MAX_ATTEMPTS_PER_SOURCE["google"]
    min_confidence = CONFIDENCE_THRESHOLDS["google"]
    
    search_variations = generate_search_variations(cue)
    
    # Expand search variations if needed
    if len(search_variations) < max_attempts:
        search_variations.extend(generate_broader_search_terms(cue))
    
    for search_term in search_variations[:max_attempts]:
        attempts += 1
        logging.info(f"🔍 Google attempt {attempts}/{max_attempts}: '{search_term}'")
        
        result = try_google_images_single_term(search_term, cue, visuals_dir, cue_id, min_confidence)
        if result:
            return result
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    logging.warning(f"Google exhausted after {attempts} attempts for '{trigger_keyword}'")
    return None

def try_stock_photos_with_persistence(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """Keep trying stock photos until good confidence found or max attempts reached"""
    attempts = 0
    max_attempts = MAX_ATTEMPTS_PER_SOURCE["stock_photo"]
    min_confidence = CONFIDENCE_THRESHOLDS["stock_photo"]
    
    search_variations = generate_search_variations(cue)
    
    # Expand search variations if needed
    if len(search_variations) < max_attempts:
        search_variations.extend(generate_broader_search_terms(cue))
    
    for search_term in search_variations[:max_attempts]:
        attempts += 1
        logging.info(f"📸 Stock photo attempt {attempts}/{max_attempts}: '{search_term}'")
        
        result = try_stock_photos_single_term(search_term, cue, visuals_dir, cue_id, min_confidence)
        if result:
            return result
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    logging.warning(f"Stock photos exhausted after {attempts} attempts for '{trigger_keyword}'")
    return None

def try_stock_video_with_persistence(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """Keep trying stock videos until good confidence found or max attempts reached"""
    attempts = 0
    max_attempts = MAX_ATTEMPTS_PER_SOURCE["stock_video"]
    min_confidence = CONFIDENCE_THRESHOLDS["stock_video"]
    
    search_variations = generate_search_variations(cue)
    
    # Expand search variations if needed
    if len(search_variations) < max_attempts:
        search_variations.extend(generate_broader_search_terms(cue))
    
    for search_term in search_variations[:max_attempts]:
        attempts += 1
        logging.info(f"🎬 Stock video attempt {attempts}/{max_attempts}: '{search_term}'")
        
        result = try_stock_footage_single_term(search_term, cue, visuals_dir, cue_id, min_confidence)
        if result:
            return result
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    logging.warning(f"Stock video exhausted after {attempts} attempts for '{trigger_keyword}'")
    return None

def try_ai_generation(cue: Dict[str, Any], visuals_dir: str, cue_id: str) -> Optional[str]:
    """AI generation as final fallback"""
    from agent.images.generate_ai import generate_ai_images
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    logging.info(f"🤖 Trying AI generation for '{trigger_keyword}'")
    
    # Create AI-friendly prompts
    ai_prompts = create_ai_generation_prompts(cue)
    max_attempts = MAX_ATTEMPTS_PER_SOURCE["ai_generated"]
    
    for attempt in range(max_attempts):
        try:
            logging.info(f"🎨 AI generation attempt {attempt + 1}/{max_attempts}")
            
            generated_files = generate_ai_images(ai_prompts, 1, visuals_dir)
            if generated_files and os.path.exists(generated_files[0]):
                # AI images always accepted (no validation needed - they're generated for purpose)
                ai_file = generated_files[0]
                
                # Rename to match expected pattern
                final_name = f"{cue_id}_ai_generated.png"
                final_path = os.path.join(visuals_dir, final_name)
                
                if ai_file != final_path:
                    os.rename(ai_file, final_path)
                
                logging.info(f"✅ AI generation successful: {final_name}")
                return final_path
            
        except Exception as e:
            logging.warning(f"AI generation attempt {attempt + 1} failed: {e}")
    
    logging.error(f"AI generation failed after {max_attempts} attempts")
    return None

def generate_broader_search_terms(cue: Dict[str, Any]) -> List[str]:
    """Generate broader search terms for fallback searches"""
    trigger = _safe_get_string(cue, 'trigger_keyword', 'concept')
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    
    broader_terms = []
    
    if visual_type == 'company':
        broader_terms.extend([
            f"{trigger} business",
            f"{trigger} corporate",
            f"company {trigger}",
            f"business logo",
            f"corporate branding"
        ])
    elif visual_type == 'person':
        broader_terms.extend([
            f"business person",
            f"professional portrait",
            f"executive photo",
            f"business leader",
            f"professional headshot"
        ])
    elif visual_type == 'action':
        broader_terms.extend([
            f"business activity",
            f"office work",
            f"professional activity",
            f"workplace",
            f"business process"
        ])
    else:  # concept
        broader_terms.extend([
            f"business concept",
            f"professional illustration",
            f"corporate graphics",
            f"business visual",
            f"infographic"
        ])
    
    return broader_terms

def create_ai_generation_prompts(cue: Dict[str, Any]) -> List[str]:
    """Create AI-optimized prompts for visual generation"""
    
    trigger = _safe_get_string(cue, 'trigger_keyword', 'concept')
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    context = _safe_get_string(cue, 'context', '')
    
    base_style = "professional, high-quality, clean background, suitable for video content, modern design, vertical composition, portrait orientation, 9:16 aspect ratio"
    
    if visual_type == 'company':
        return [
            f"Clean, professional logo design for {trigger}, minimalist style, {base_style}",
            f"Corporate branding elements for {trigger}, modern design, {base_style}",
            f"Business logo representing {trigger}, professional appearance, {base_style}"
        ]
    elif visual_type == 'person':
        return [
            f"Professional portrait of business person related to {trigger}, {base_style}",
            f"Business professional in context of {trigger}, corporate setting, {base_style}",
            f"Expert or specialist in {trigger} field, professional headshot, {base_style}"
        ]
    elif visual_type == 'action':
        return [
            f"People performing {trigger} activity, professional setting, {base_style}",
            f"Business process showing {trigger}, professional environment, {base_style}",
            f"Workflow demonstrating {trigger}, corporate context, {base_style}"
        ]
    else:  # concept
        return [
            f"Visual representation of {trigger} concept, professional illustration, {base_style}",
            f"Infographic elements showing {trigger}, modern design, {base_style}",
            f"Abstract visualization of {trigger}, professional graphics, {base_style}"
        ] 

def try_google_images_single_term(search_term: str, cue: Dict[str, Any], visuals_dir: str, cue_id: str, min_confidence: float) -> Optional[str]:
    """Try a single Google search term with validation"""
    try:
        from agent.images.search_google import search_google_images
        
        # Get multiple candidates
        candidates = search_google_images([search_term], 3, visuals_dir)
        if not candidates:
            return None
        
        # Validate each candidate
        best_candidate = None
        best_confidence = 0.0
        
        for candidate in candidates:
            if os.path.exists(candidate):
                validation = validate_visual_relevance(
                    candidate,
                    cue['trigger_keyword'],
                    [search_term],
                    cue.get('visual_type', 'concept'),
                    cue.get('context', '')
                )
                
                if validation['is_relevant'] and validation['confidence'] >= min_confidence:
                    if validation['confidence'] > best_confidence:
                        # Clean up previous best candidate
                        if best_candidate and best_candidate != candidate:
                            _clean_up_image_and_metadata(best_candidate)
                        
                        best_candidate = candidate
                        best_confidence = validation['confidence']
                else:
                    # Remove rejected candidate and its metadata
                    _clean_up_image_and_metadata(candidate)
        
        if best_candidate:
            # Rename to standard format
            final_name = f"{cue_id}_google.jpg"
            final_path = os.path.join(visuals_dir, final_name)
            
            if best_candidate != final_path:
                os.rename(best_candidate, final_path)
            
            logging.info(f"✅ Google success: {search_term} (confidence: {best_confidence:.2f})")
            return final_path
        
        return None
        
    except Exception as e:
        logging.warning(f"Google search failed for '{search_term}': {e}")
        return None

def try_stock_photos_single_term(search_term: str, cue: Dict[str, Any], visuals_dir: str, cue_id: str, min_confidence: float) -> Optional[str]:
    """Try a single stock photo search term with validation"""
    try:
        from agent.images.search_stock import search_stock_images
        
        # Get multiple candidates
        candidates = search_stock_images([search_term], 3, visuals_dir)
        if not candidates:
            return None
        
        # Validate each candidate
        best_candidate = None
        best_confidence = 0.0
        
        for candidate in candidates:
            if os.path.exists(candidate):
                validation = validate_visual_relevance(
                    candidate,
                    cue['trigger_keyword'],
                    [search_term],
                    cue.get('visual_type', 'concept'),
                    cue.get('context', '')
                )
                
                if validation['is_relevant'] and validation['confidence'] >= min_confidence:
                    if validation['confidence'] > best_confidence:
                        # Clean up previous best candidate
                        if best_candidate and best_candidate != candidate:
                            _clean_up_image_and_metadata(best_candidate)
                        
                        best_candidate = candidate
                        best_confidence = validation['confidence']
                else:
                    # Remove rejected candidate and its metadata
                    _clean_up_image_and_metadata(candidate)
        
        if best_candidate:
            # Rename to standard format
            final_name = f"{cue_id}_stock_photo.jpg"
            final_path = os.path.join(visuals_dir, final_name)
            
            if best_candidate != final_path:
                os.rename(best_candidate, final_path)
            
            logging.info(f"✅ Stock photo success: {search_term} (confidence: {best_confidence:.2f})")
            return final_path
        
        return None
        
    except Exception as e:
        logging.warning(f"Stock photo search failed for '{search_term}': {e}")
        return None

def try_stock_footage_single_term(search_term: str, cue: Dict[str, Any], visuals_dir: str, cue_id: str, min_confidence: float) -> Optional[str]:
    """Try a single stock footage search term with validation"""
    try:
        from agent.video_sources.pexels_video import search_and_download_pexels_video
        
        # Download video with duration preference
        duration_pref = "short" if cue.get('visual_type') == 'action' else "medium"
        
        video_path = search_and_download_pexels_video(
            query=search_term,
            output_dir=visuals_dir,
            filename=f"{cue_id}_stock_video",
            duration_preference=duration_pref
        )
        
        if video_path and os.path.exists(video_path):
            # Videos are automatically accepted (validation disabled)
            logging.info(f"✅ Stock video success: {search_term}")
            return video_path
        
        return None
        
    except Exception as e:
        logging.warning(f"Stock video search failed for '{search_term}': {e}")
        return None

def log_gathering_failure(cue: Dict[str, Any], attempted_sources: List[str]):
    """Log detailed failure analysis for debugging"""
    
    trigger_keyword = _safe_get_string(cue, 'trigger_keyword', 'unknown')
    visual_type = _safe_get_string(cue, 'visual_type', 'concept')
    
    failure_data = {
        "trigger_keyword": trigger_keyword,
        "visual_type": visual_type,
        "search_terms": cue.get('search_terms', []),
        "attempted_sources": attempted_sources,
        "failure_reasons": {},
        "suggestions": []
    }
    
    # Add specific failure analysis
    if 'google' in attempted_sources:
        failure_data["failure_reasons"]["google"] = "No high-confidence matches found above 0.8 threshold"
        failure_data["suggestions"].append("Try broader search terms")
    
    if 'stock_photo' in attempted_sources:
        failure_data["failure_reasons"]["stock_photo"] = "No relevant stock images found above 0.7 threshold"
        failure_data["suggestions"].append("Consider different visual type classification")
    
    if 'stock_video' in attempted_sources:
        failure_data["failure_reasons"]["stock_video"] = "No stock videos found for search terms"
        failure_data["suggestions"].append("Use stock photos instead")
    
    if 'ai_generated' in attempted_sources:
        failure_data["failure_reasons"]["ai_generated"] = "AI generation failed"
        failure_data["suggestions"].append("Check AI generation API credentials")
    
    # Log to decision system
    log_decision(
        step="visual_gathering_failure",
        decision=f"CRITICAL: Failed to find visual for {trigger_keyword}",
        reasoning="All visual sources exhausted without finding suitable content above confidence thresholds",
        confidence=0.0,
        alternatives=failure_data["suggestions"],
        metadata=failure_data
    )
    
    logging.error(f"🔥 VISUAL GATHERING COMPLETE FAILURE for '{trigger_keyword}'")
    logging.error(f"   Sources tried: {attempted_sources}")
    logging.error(f"   Visual type: {visual_type}")
    logging.error(f"   Original search terms: {cue.get('search_terms', [])}")
    logging.error(f"   Suggestions: {failure_data['suggestions']}") 