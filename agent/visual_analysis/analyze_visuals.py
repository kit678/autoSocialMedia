import os
import json
import logging
import base64
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from agent.utils import http_retry_session, rate_limit_gemini

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def run(script_text: str, all_image_paths: Dict[str, str], primary_visual_path: str, audio_duration: float) -> Dict[str, Any]:
    """
    Analyzes script for visual cues and determines timeline using video when available.
    
    Args:
        script_text: The generated video script with [Visual: ...] cues
        all_image_paths: Dictionary mapping image/video keys to their file paths
        primary_visual_path: Path to the primary visual (video or screenshot)
        audio_duration: Duration of audio in seconds
        
    Returns:
        Dictionary containing visual timeline and effects
    """
    # Create a list of visual paths to process, excluding webpage video/screenshot
    visual_paths = [path for key, path in all_image_paths.items() 
                   if key not in ['url_screenshot', 'webpage_video']]
    
    # First, parse visual cues from the script
    visual_timeline = parse_script_visual_cues(script_text, all_image_paths, audio_duration)
    
    # Then, determine Ken Burns effects (skip for video files)
    if GEMINI_API_KEY and len(visual_paths) > 0 and not primary_visual_path.endswith('.mp4'):
        effects_analysis = analyze_visual_effects(visual_paths, primary_visual_path)
        if effects_analysis:
            # Merge effects with timeline
            visual_timeline = merge_effects_with_timeline(visual_timeline, effects_analysis)
            return visual_timeline
    
    # Return basic timeline if analysis fails
    return visual_timeline

def parse_script_visual_cues(script_text: str, all_image_paths: Dict[str, str], audio_duration: float) -> Dict[str, Any]:
    """
    Parse [Visual: ...] cues from script and create timeline.
    """
    # Extract visual cues with regex
    visual_pattern = r'\[Visual:\s*([^\]]+)\]'
    
    # Split script by visual cues
    parts = re.split(visual_pattern, script_text)
    
    segments = []
    visual_concepts = []
    text_segments = []
    
    # Create a list of available keys, prioritizing webpage video over screenshot
    available_keys = []
    
    # Add webpage video first if available
    if 'webpage_video' in all_image_paths:
        available_keys.append('webpage_video')
    elif 'url_screenshot' in all_image_paths:
        available_keys.append('url_screenshot')
    
    # Add other visual assets
    other_keys = [key for key in all_image_paths.keys() 
                 if key not in ['url_screenshot', 'webpage_video']]
    available_keys.extend(other_keys)
    
    # Process parts (alternating text and visual cues)
    current_visual = None
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Text part
            text = part.strip()
            if text:
                text_segments.append({
                    'visual': current_visual,
                    'text': text
                })
        else:  # Visual cue
            current_visual = part.strip()
            visual_concepts.append(current_visual)
    
    # Calculate timing based on text length
    total_text_length = sum(len(seg['text']) for seg in text_segments)
    
    current_time = 0.0
    visual_timeline = []
    
    for i, segment in enumerate(text_segments):
        # Calculate duration based on text proportion
        text_proportion = len(segment['text']) / total_text_length if total_text_length > 0 else 1.0 / len(text_segments)
        duration = audio_duration * text_proportion
        
        # Determine which visual to use from the available keys
        # This assigns visuals sequentially to cues in the script
        if i < len(available_keys):
            image_source = available_keys[i]
        else:
            # Fallback to webpage video if available, otherwise screenshot
            image_source = "webpage_video" if 'webpage_video' in all_image_paths else "url_screenshot"
        
        visual_timeline.append({
            "start_time": current_time,
            "end_time": min(current_time + duration, audio_duration),
            "image_source": image_source,
            "visual_concept": segment.get('visual'),
            "transition_in": "fade",
            "transition_out": "fade"
        })
        
        current_time = min(current_time + duration, audio_duration)
    
    # If no visual cues found, create basic timeline
    if not visual_timeline:
        visual_timeline = create_basic_timeline(available_keys, audio_duration)
    
    # Create appropriate assessment based on what we're using
    primary_key = "webpage_video" if 'webpage_video' in all_image_paths else "url_screenshot"
    
    return {
        "visual_timeline": visual_timeline,
        "ken_burns_effects": {},
        "image_assessment": {
            primary_key: {"suitable": True, "needs_scroll": False, "is_video": primary_key == "webpage_video"}
        },
        "metadata": {
            "timeline_method": "script_parsing",
            "total_segments": len(visual_timeline),
            "primary_visual_type": "video" if primary_key == "webpage_video" else "image"
        }
    }

@rate_limit_gemini
def analyze_visual_effects(visual_paths: List[str], screenshot_path: str) -> Dict[str, Any]:
    """
    Use Gemini Vision to analyze images for Ken Burns effects and scroll decisions only.
    """
    if not GEMINI_API_KEY:
        return None
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        content_parts = []
        content_parts.append(types.Part.from_text(text="""
Analyze these images for video production effects:

For each image, determine:
1. Ken Burns effect parameters (zoom and pan to highlight key areas)
2. For the URL screenshot: whether scrolling would add value

Return JSON with Ken Burns parameters and scroll recommendation.
"""))
        
        # Add images
        for i, image_path in enumerate(visual_paths[:5]):  # Limit to 5 images
            if os.path.exists(image_path):
                image_base64 = _encode_image_to_base64(image_path)
                if image_base64:
                    content_parts.append(types.Part.from_text(text=f"\nImage {i}:"))
                    content_parts.append(types.Part.from_bytes(
                        data=base64.b64decode(image_base64),
                        mime_type='image/jpeg'
                    ))
        
        # Add screenshot
        if os.path.exists(screenshot_path):
            screenshot_base64 = _encode_image_to_base64(screenshot_path)
            if screenshot_base64:
                content_parts.append(types.Part.from_text(text="\nURL Screenshot:"))
                content_parts.append(types.Part.from_bytes(
                    data=base64.b64decode(screenshot_base64),
                    mime_type='image/png'
                ))
        
        content_parts.append(types.Part.from_text(text="""
Return JSON format:
{
    "ken_burns_effects": {
        "image_0": {"start_zoom": 1.0, "end_zoom": 1.3, "focus_area": "center"},
        "image_1": {"start_zoom": 1.1, "end_zoom": 1.4, "focus_area": "top-left"},
        "url_screenshot": {"start_zoom": 1.0, "end_zoom": 1.2, "focus_area": "headline"}
    },
    "scroll_recommendation": {
        "needs_scroll": true/false,
        "reason": "why or why not"
    }
}
"""))
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            ),
        )
        
        response_text = response.text.strip()
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        
        return json.loads(response_text)
            
    except Exception as e:
        logging.warning(f"Visual effects analysis failed: {e}")
        return None

def merge_effects_with_timeline(timeline_data: Dict, effects_data: Dict) -> Dict:
    """
    Merge Ken Burns effects and scroll decision with timeline.
    """
    if effects_data:
        # Update Ken Burns effects
        timeline_data['ken_burns_effects'] = effects_data.get('ken_burns_effects', {})
        
        # Update scroll recommendation
        scroll_rec = effects_data.get('scroll_recommendation', {})
        timeline_data['image_assessment']['url_screenshot']['needs_scroll'] = scroll_rec.get('needs_scroll', False)
    
    return timeline_data

def create_basic_timeline(available_keys: List[str], audio_duration: float) -> List[Dict]:
    """
    Create a basic timeline when parsing fails.
    """
    time_per_key = audio_duration / max(len(available_keys), 1)
    timeline = []
    
    for i, key in enumerate(available_keys):
        timeline.append({
            "start_time": i * time_per_key,
            "end_time": min((i + 1) * time_per_key, audio_duration),
            "image_source": key,
            "transition_in": "fade",
            "transition_out": "fade"
        })
    
    return timeline

def _encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None 