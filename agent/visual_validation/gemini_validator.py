"""
Gemini Vision-based Visual Validator
Validates downloaded visuals for relevance using multimodal AI
"""

import os
import base64
import logging
import mimetypes
from typing import Dict, List, Any, Optional
from agent.utils import get_gemini_client, rate_limit_gemini, VALIDATION_FUNCTION_SCHEMA

@rate_limit_gemini
def validate_visual_relevance(
    image_path: str,
    expected_concept: str,
    search_terms: List[str],
    visual_type: str,
    context: str = "",
    original_url: str = None
) -> Dict[str, Any]:
    """
    Validate if downloaded visual is relevant to the expected concept using function calling
    
    Args:
        image_path: Path to downloaded image/video
        expected_concept: What we expected to find (trigger keyword)
        search_terms: Original search terms used
        visual_type: Type of visual (person, company, concept, action, etc.)
        context: Additional context about the video content
        
    Returns:
        Dict with validation results
    """
    try:
        # Skip validation for videos (as decided in earlier fixes)
        if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.wmv')):
            return _validate_video_relevance(image_path, expected_concept, search_terms, visual_type, context)
        
        # Validate image exists and get proper MIME type
        if not os.path.exists(image_path):
            return _create_validation_result(False, 0.0, "Image file not found", ["File does not exist"])
        
        # Get original URL for logging if not provided
        if original_url is None:
            original_url = _get_original_url(image_path)
        
        # Detect proper MIME type
        mime_type = _detect_image_mime_type(image_path)
        if not mime_type:
            return _create_validation_result(False, 0.0, "Unsupported image format", ["Could not determine image type"])
        
        # Encode image for Gemini Vision
        image_data = _encode_image_for_gemini(image_path)
        if not image_data:
            return _create_validation_result(False, 0.0, "Failed to encode image", ["Image encoding failed"])
        
        # Try function calling validation with proper error handling
        try:
            return _validate_with_function_calling(image_path, image_data, mime_type, expected_concept, search_terms, visual_type, context, original_url)
        except Exception as fc_error:
            logging.warning(f"Function calling failed: {fc_error}, trying text-based validation")
            # Fallback to text-based validation
            return _validate_with_text_response(image_path, image_data, mime_type, expected_concept, search_terms, visual_type, context, original_url)
        
    except Exception as e:
        logging.error(f"Visual validation failed with error: {str(e)}")
        # Return neutral result on error (don't block visual gathering)
        return _create_validation_result(True, 0.5, f"Validation error: {str(e)}", ["Validation API failed"])

def _detect_image_mime_type(image_path: str) -> Optional[str]:
    """Detect proper MIME type for image file"""
    try:
        # First try mimetypes library
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type.startswith('image/'):
            return mime_type
        
        # Fallback: check file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        
        return mime_map.get(ext, 'image/jpeg')  # Default to JPEG if unknown
        
    except Exception as e:
        logging.warning(f"MIME type detection failed: {e}")
        return 'image/jpeg'  # Safe default

def _validate_with_function_calling(
    image_path: str, 
    image_data: str, 
    mime_type: str,
    expected_concept: str, 
    search_terms: List[str], 
    visual_type: str, 
    context: str,
    original_url: str = None
) -> Dict[str, Any]:
    """Attempt validation using function calling"""
    
    # Create validation prompt for function calling
    prompt = _create_function_calling_prompt(expected_concept, search_terms, visual_type, context)
    
    # Call Gemini Vision API with function calling
    client = get_gemini_client()
    
    from google.genai import types
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type=mime_type
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent evaluation
                max_output_tokens=500,
                tools=[types.Tool(function_declarations=[VALIDATION_FUNCTION_SCHEMA])]
            ),
        )
        
        # Parse function call response
        validation_result = _parse_function_call_response(response)
        
        logging.info(f"📊 Function calling validation result: {validation_result['is_relevant']} (confidence: {validation_result['confidence']:.2f})")
        if not validation_result['is_relevant']:
            url_info = f" | URL: {original_url}" if original_url else ""
            logging.warning(f"❌ Visual rejected: {validation_result['issues']}{url_info}")
        elif original_url:
            logging.info(f"✅ Visual accepted | URL: {original_url}")
        
        return validation_result
        
    except Exception as api_error:
        # Log the specific API error for debugging
        error_msg = str(api_error)
        logging.error(f"Function calling API error: {error_msg}")
        
        # Re-raise to trigger fallback
        raise api_error

def _validate_with_text_response(
    image_path: str, 
    image_data: str, 
    mime_type: str,
    expected_concept: str, 
    search_terms: List[str], 
    visual_type: str, 
    context: str,
    original_url: str = None
) -> Dict[str, Any]:
    """Fallback validation using text response (no function calling)"""
    
    # Create simpler prompt for text response
    prompt = _create_text_validation_prompt(expected_concept, search_terms, visual_type, context)
    
    client = get_gemini_client()
    
    from google.genai import types
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type=mime_type
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
                # No function calling tools - just text response
            ),
        )
        
        # Parse text response
        validation_result = _parse_text_response_fallback(response.text)
        
        logging.info(f"📊 Text validation result: {validation_result['is_relevant']} (confidence: {validation_result['confidence']:.2f})")
        if not validation_result['is_relevant']:
            url_info = f" | URL: {original_url}" if original_url else ""
            logging.warning(f"❌ Visual rejected (text validation): {validation_result['issues']}{url_info}")
        elif original_url:
            logging.info(f"✅ Visual accepted (text validation) | URL: {original_url}")
        
        return validation_result
        
    except Exception as text_error:
        logging.error(f"Text validation also failed: {text_error}")
        # Return permissive result to avoid blocking pipeline
        return _create_validation_result(True, 0.6, "Validation fallback", ["Both validation methods failed"])

def _create_text_validation_prompt(expected_concept: str, search_terms: List[str], visual_type: str, context: str) -> str:
    """Create simple text validation prompt"""
    
    return f"""
    Analyze this image and determine if it's relevant for a video about: "{expected_concept}"
    
    CONTEXT: {context}
    SEARCH TERMS USED: {search_terms}
    VISUAL TYPE: {visual_type}
    
    Respond in this exact format:
    RELEVANT: YES or NO
    CONFIDENCE: 0.0 to 1.0
    DESCRIPTION: Brief description of what you see
    ISSUES: Any problems (or NONE)
    
    Be reasonably permissive - reject only if clearly irrelevant or inappropriate.
    """

@rate_limit_gemini
def generate_search_variations(cue: Dict[str, Any]) -> List[str]:
    """Generate comprehensive search variations with multiple fallback strategies"""
    
    # Start with original terms
    variations = list(cue.get('search_terms', []))
    
    # Add trigger keyword variations
    trigger = cue['trigger_keyword']
    if trigger:
        variations.extend([
            trigger,
            trigger.replace(' ', '+'),  # URL-friendly version
            f'"{trigger}"',  # Exact phrase search
        ])
    
    # Add type-specific variations
    visual_type = cue.get('visual_type', 'concept')
    
    if visual_type == 'company':
        variations.extend([
            f"{trigger} logo",
            f"{trigger} brand",
            f"{trigger} company logo",
            f"{trigger} official logo",
            f"{trigger} corporate branding",
            f"{trigger} business logo"
        ])
    elif visual_type == 'person':
        variations.extend([
            f"{trigger} photo",
            f"{trigger} professional",
            f"{trigger} headshot",
            f"{trigger} portrait",
            f"{trigger} business photo",
            f"{trigger} executive"
        ])
    elif visual_type == 'action':
        variations.extend([
            f"{trigger} action",
            f"{trigger} activity",
            f"people {trigger}",
            f"{trigger} process",
            f"{trigger} workflow",
            f"business {trigger}"
        ])
    elif visual_type == 'concept':
        variations.extend([
            f"{trigger} concept",
            f"{trigger} illustration",
            f"{trigger} infographic",
            f"{trigger} visual",
            f"business {trigger}",
            f"professional {trigger}"
        ])
    elif visual_type == 'location':
        variations.extend([
            f"{trigger} building",
            f"{trigger} office",
            f"{trigger} headquarters",
            f"{trigger} location",
            f"{trigger} facility"
        ])
    elif visual_type == 'product':
        variations.extend([
            f"{trigger} product",
            f"{trigger} device",
            f"{trigger} technology",
            f"{trigger} software",
            f"{trigger} platform"
        ])
    
    # Use AI for additional creative variations (rate-limited)
    try:
        ai_variations = generate_ai_search_variations(cue)
        if ai_variations:
            variations.extend(ai_variations)
    except Exception as e:
        logging.warning(f"AI search variation generation failed: {e}")
    
    # Add broader fallback terms for difficult searches
    broader_terms = generate_contextual_fallbacks(cue)
    variations.extend(broader_terms)
    
    # Deduplicate, filter, and return
    unique_variations = []
    seen = set()
    
    for var in variations:
        if var and len(var.strip()) > 2:  # Filter out empty/too short terms
            clean_var = var.strip().lower()
            if clean_var not in seen:
                seen.add(clean_var)
                unique_variations.append(var.strip())
    
    # Limit to reasonable number and prioritize original terms
    return unique_variations[:8]  # Max 8 variations per search

def generate_ai_search_variations(cue: Dict[str, Any]) -> List[str]:
    """Generate AI-powered search variations (rate-limited)"""
    try:
        logging.info(f"🔄 Generating AI search variations for: {cue['trigger_keyword']}")
        
        client = get_gemini_client()
        
        prompt = f"""
        I need to find highly relevant visuals for a video. Generate 3 improved search queries.
        
        TARGET CONCEPT: {cue['trigger_keyword']}
        VISUAL TYPE: {cue['visual_type']}
        CONTEXT: {cue.get('context', 'General video content')}
        ORIGINAL SEARCH TERMS: {cue.get('search_terms', [])}
        
        Create 3 search queries that would find the most relevant and high-quality visuals:
        1. One focused on the exact concept with descriptive terms
        2. One using synonyms or related industry terms  
        3. One with more specific visual descriptors
        
        Requirements:
        - Each query should be 2-4 words maximum
        - Focus on visual clarity and professional appearance
        - Consider what would actually appear in stock photos/videos/Google images
        - Avoid overly abstract or complex terms
        - Prioritize terms that photographers/designers would use
        
        Return only the 3 search queries, one per line, no numbering or explanation.
        """
        
        from google.genai import types
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200,
            ),
        )
        
        # Parse search variations
        variations = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*') and len(line) > 3:
                # Clean up any numbering or bullets
                cleaned_line = line
                if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                    cleaned_line = line[2:].strip()
                elif line.startswith('- '):
                    cleaned_line = line[2:].strip()
                
                if cleaned_line:
                    variations.append(cleaned_line)
        
        logging.info(f"✨ Generated {len(variations)} AI search variations: {variations}")
        return variations[:3]  # Limit to 3 AI variations
        
    except Exception as e:
        logging.warning(f"AI search variation generation failed: {e}")
        return []

def generate_contextual_fallbacks(cue: Dict[str, Any]) -> List[str]:
    """Generate broader contextual fallback terms"""
    visual_type = cue.get('visual_type', 'concept')
    
    # Generic professional terms as ultimate fallbacks
    fallbacks = {
        'company': ['corporate logo', 'business branding', 'company symbol', 'brand identity'],
        'person': ['business professional', 'executive portrait', 'professional headshot', 'business leader'],
        'action': ['business activity', 'office work', 'professional process', 'workplace action'],
        'concept': ['business concept', 'professional illustration', 'corporate visual', 'business graphic'],
        'location': ['office building', 'corporate headquarters', 'business facility', 'modern office'],
        'product': ['technology product', 'business software', 'digital platform', 'tech device']
    }
    
    return fallbacks.get(visual_type, ['business illustration', 'professional graphic', 'corporate visual'])[:3]

def select_best_visual(
    candidate_paths: List[str],
    cue: Dict[str, Any]
) -> Optional[str]:
    """
    Compare multiple visual candidates and select the best one
    
    Args:
        candidate_paths: List of paths to candidate images
        cue: Visual cue information
        
    Returns:
        Path to best visual, or None if all are poor
    """
    try:
        if not candidate_paths:
            return None
        
        if len(candidate_paths) == 1:
            return candidate_paths[0]
        
        logging.info(f"🏆 Comparing {len(candidate_paths)} visual candidates for '{cue['trigger_keyword']}'")
        
        # Validate each candidate first
        scored_candidates = []
        
        for path in candidate_paths:
            if os.path.exists(path):
                validation = validate_visual_relevance(
                    path,
                    cue['trigger_keyword'],
                    cue.get('search_terms', []),
                    cue['visual_type'],
                    cue.get('context', '')
                )
                
                if validation['is_relevant']:
                    scored_candidates.append({
                        'path': path,
                        'confidence': validation['confidence'],
                        'description': validation['description']
                    })
        
        if not scored_candidates:
            logging.warning("No relevant candidates found")
            return None
        
        # Sort by confidence score
        scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        best_candidate = scored_candidates[0]
        
        logging.info(f"🎯 Selected best visual: {os.path.basename(best_candidate['path'])} (confidence: {best_candidate['confidence']:.2f})")
        
        # Clean up non-selected candidates
        for candidate in scored_candidates[1:]:
            try:
                os.remove(candidate['path'])
                logging.debug(f"Removed non-selected candidate: {os.path.basename(candidate['path'])}")
            except:
                pass
        
        return best_candidate['path']
        
    except Exception as e:
        logging.error(f"Visual selection failed: {e}")
        # Return first candidate as fallback
        return candidate_paths[0] if candidate_paths else None

def _encode_image_for_gemini(image_path: str) -> Optional[str]:
    """Encode image file as base64 for Gemini Vision API"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding failed: {e}")
        return None

def _get_original_url(image_path: str) -> Optional[str]:
    """Get original URL from metadata file if available"""
    try:
        url_metadata_path = image_path + ".url"
        if os.path.exists(url_metadata_path):
            with open(url_metadata_path, 'r') as url_file:
                return url_file.read().strip()
    except Exception as e:
        logging.debug(f"Could not read URL metadata for {image_path}: {e}")
    return None

def _create_function_calling_prompt(expected_concept: str, search_terms: List[str], visual_type: str, context: str) -> str:
    """Create appropriate validation prompt for function calling"""
    
    base_prompt = f"""
    Analyze this image and determine if it's relevant for a video about: "{expected_concept}"
    
    CONTEXT: {context}
    SEARCH TERMS USED: {search_terms}
    VISUAL TYPE: {visual_type}
    """
    
    if visual_type == "person":
        specific_prompt = f"""
        Does this image show a person who appears to be related to: {expected_concept}?
        Consider: facial expression, setting, profession, activity.
        Is this person clearly visible and appropriate for the context?
        """
    elif visual_type == "company":
        specific_prompt = f"""
        Is this a clear, recognizable logo or representation of {expected_concept}?
        Consider: brand clarity, logo quality, official appearance.
        Would viewers immediately recognize this as {expected_concept}?
        """
    elif visual_type == "action":
        specific_prompt = f"""
        Does this image clearly depict the action or activity: {expected_concept}?
        Consider: clarity of action, relevance to context, visual impact.
        Is the action/concept easily understandable from the image?
        """
    else:  # concept, product, location
        specific_prompt = f"""
        Does this image effectively represent or illustrate: {expected_concept}?
        Consider: visual metaphor strength, conceptual clarity, appropriateness.
        Would this help viewers understand {expected_concept}?
        """
    
    return base_prompt + "\n" + specific_prompt + """
    
    Use the validate_visual_relevance function to provide your assessment with:
    - is_relevant: boolean indicating if the image matches the concept
    - confidence: score from 0.0 to 1.0 indicating your confidence
    - description: brief description of what you see in the image
    - issues: list of any problems with the visual (empty array if none)
    - suggestions: list of suggestions for better search terms (empty array if none)
    """

def _parse_function_call_response(response) -> Dict[str, Any]:
    """Parse Gemini's function call response into structured data"""
    try:
        result = {
            "is_relevant": False,
            "confidence": 0.0,
            "description": "",
            "issues": [],
            "suggestions": []
        }
        
        # Check if response has function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Extract function call arguments
                            args = part.function_call.args
                            result['is_relevant'] = args.get('is_relevant', False)
                            result['confidence'] = float(args.get('confidence', 0.0))
                            result['description'] = args.get('description', '')
                            result['issues'] = args.get('issues', [])
                            result['suggestions'] = args.get('suggestions', [])
                            return result
        
        # If no function call found, try to parse as text (fallback)
        if hasattr(response, 'text') and response.text:
            return _parse_text_response_fallback(response.text)
        
        # Default fallback
        logging.warning("No function call or text response found")
        return _create_validation_result(True, 0.5, "No response content", ["Response parsing failed"])
        
    except Exception as e:
        logging.warning(f"Failed to parse function call response: {e}")
        return _create_validation_result(True, 0.5, "Parse error", ["Could not parse function call response"])

def _parse_text_response_fallback(response_text: str) -> Dict[str, Any]:
    """Fallback text parsing when function calling fails"""
    try:
        result = {
            "is_relevant": False,
            "confidence": 0.0,
            "description": "",
            "issues": [],
            "suggestions": []
        }
        
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('RELEVANT:'):
                result['is_relevant'] = 'YES' in line.upper()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(confidence_str)
                except:
                    result['confidence'] = 0.5
            elif line.startswith('DESCRIPTION:'):
                result['description'] = line.split(':', 1)[1].strip()
            elif line.startswith('ISSUES:'):
                issues_text = line.split(':', 1)[1].strip()
                if issues_text and issues_text.lower() != 'none':
                    result['issues'] = [issues_text]
            elif line.startswith('SUGGESTIONS:'):
                suggestions_text = line.split(':', 1)[1].strip()
                if suggestions_text and suggestions_text.lower() != 'none':
                    result['suggestions'] = [suggestions_text]
        
        # Clean and validate the response text to avoid encoding issues
        if isinstance(result['description'], str):
            # Remove or replace problematic Unicode characters
            result['description'] = result['description'].encode('ascii', 'ignore').decode('ascii')
            if not result['description'].strip():
                result['description'] = "Visual validation completed"
        
        # Process validation logic
        result['is_relevant'] = False
        result['confidence'] = 0.0
        result['issues'] = []
        
        # Extract validation decision from description
        description_lower = result['description'].lower()
        
        if any(word in description_lower for word in ['suitable', 'relevant', 'good', 'appropriate', 'valid']):
            result['is_relevant'] = True
            result['confidence'] = 0.8
        elif any(word in description_lower for word in ['reject', 'unsuitable', 'irrelevant', 'poor', 'inappropriate']):
            result['is_relevant'] = False
            result['confidence'] = 0.2
            result['issues'].append("Rejected by validator")
        else:
            # Fallback based on length and content
            result['is_relevant'] = len(result['description'].strip()) > 10
            result['confidence'] = 0.5 if result['is_relevant'] else 0.1
            if not result['is_relevant']:
                result['issues'].append("Insufficient validation response")
        
        return result
        
    except Exception as e:
        logging.warning(f"Failed to parse text response fallback: {e}")
        return _create_validation_result(True, 0.5, "Parse error", ["Could not parse validation response"])

def _validate_video_relevance(
    video_path: str,
    expected_concept: str,
    search_terms: List[str],
    visual_type: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    Skip video validation to avoid API issues - return success for all videos
    
    Args:
        video_path: Path to video file
        expected_concept: What we expected to find
        search_terms: Original search terms used
        visual_type: Type of visual (action, concept, etc.)
        context: Additional context
        
    Returns:
        Dict with validation results
    """
    try:
        logging.info(f"🎬 Skipping video validation for: {os.path.basename(video_path)} (avoiding API issues)")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            return _create_validation_result(False, 0.0, "Video file not found", ["File does not exist"])
        
        # Get video file size for logging
        file_size = os.path.getsize(video_path)
        logging.info(f"Video file is {file_size / (1024*1024):.1f}MB - accepting without validation")
        
        # Return success without actual validation to avoid API issues
        return _create_validation_result(
            True, 
            0.75,  # Reasonable confidence for Pexels videos
            f"Video accepted (validation skipped): {expected_concept}", 
            []
        )
            
    except Exception as e:
        logging.error(f"Video validation failed: {e}")
        return _create_validation_result(True, 0.5, f"Video validation error: {str(e)}", ["Video validation API failed"])

def _validate_video_inline(
    video_path: str,
    expected_concept: str,
    search_terms: List[str],
    visual_type: str,
    context: str
) -> Dict[str, Any]:
    """Validate video using inline data (for files < 20MB)"""
    try:
        # Read video file as bytes
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        
        # Detect MIME type based on file extension
        mime_type = _get_video_mime_type(video_path)
        
        # Create validation prompt
        prompt = _create_video_validation_prompt(expected_concept, search_terms, visual_type, context)
        
        # Call Gemini Vision API with video
        client = get_gemini_client()
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',  # 2.5+ supports video
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(video_bytes).decode('utf-8')
                            }
                        }
                    ]
                }
            ],
            config={
                'temperature': 0.1,
                'max_output_tokens': 500,
            }
        )
        
        # Parse response
        response_text = response.text.strip()
        validation_result = _parse_validation_response(response_text)
        
        logging.info(f"🎬 Video validation result: {validation_result['is_relevant']} (confidence: {validation_result['confidence']:.2f})")
        
        return validation_result
        
    except Exception as e:
        logging.error(f"Inline video validation failed: {e}")
        return _create_validation_result(True, 0.5, f"Video validation error: {str(e)}", ["Inline video validation failed"])

def _validate_video_with_files_api(
    video_path: str,
    expected_concept: str,
    search_terms: List[str],
    visual_type: str,
    context: str
) -> Dict[str, Any]:
    """Validate video using Files API (for larger files)"""
    try:
        logging.info("Using Files API for large video validation")
        
        # Upload video file
        client = get_gemini_client()
        uploaded_file = client.files.upload(file=video_path)
        
        # Wait a moment for processing
        import time
        time.sleep(2)
        
        # Create validation prompt
        prompt = _create_video_validation_prompt(expected_concept, search_terms, visual_type, context)
        
        # Generate content using uploaded file
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[uploaded_file, prompt],
            config={
                'temperature': 0.1,
                'max_output_tokens': 500,
            }
        )
        
        # Parse response
        response_text = response.text.strip()
        validation_result = _parse_validation_response(response_text)
        
        logging.info(f"🎬 Video validation result: {validation_result['is_relevant']} (confidence: {validation_result['confidence']:.2f})")
        
        # Clean up uploaded file
        try:
            client.files.delete(uploaded_file.name)
        except:
            pass  # Don't fail if cleanup fails
        
        return validation_result
        
    except Exception as e:
        logging.error(f"Files API video validation failed: {e}")
        return _create_validation_result(True, 0.5, f"Video validation error: {str(e)}", ["Files API video validation failed"])

def _get_video_mime_type(video_path: str) -> str:
    """Get MIME type based on video file extension"""
    ext = os.path.splitext(video_path)[1].lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/avi',
        '.mov': 'video/mov',
        '.webm': 'video/webm',
        '.wmv': 'video/wmv',
        '.3gpp': 'video/3gpp',
        '.mpg': 'video/mpg',
        '.mpeg': 'video/mpeg'
    }
    return mime_types.get(ext, 'video/mp4')  # Default to mp4

def _create_video_validation_prompt(expected_concept: str, search_terms: List[str], visual_type: str, context: str) -> str:
    """Create appropriate validation prompt for video content"""
    
    base_prompt = f"""
    Analyze this video and determine if it's relevant for a video about: "{expected_concept}"
    
    CONTEXT: {context}
    SEARCH TERMS USED: {search_terms}
    VISUAL TYPE: {visual_type}
    
    Please analyze the video content including:
    - Visual elements throughout the video
    - Any actions or movements shown
    - Objects, people, or scenes depicted
    - Overall theme and message
    """
    
    if visual_type == "action":
        specific_prompt = f"""
        Does this video clearly show the action or activity: {expected_concept}?
        Consider: clarity of the action, relevance to context, visual quality.
        Is the action/concept the main focus of the video?
        """
    elif visual_type == "person":
        specific_prompt = f"""
        Does this video show a person related to: {expected_concept}?
        Consider: person's activity, setting, professional context.
        Is the person clearly visible and appropriately portrayed?
        """
    elif visual_type == "company":
        specific_prompt = f"""
        Does this video represent or relate to the company: {expected_concept}?
        Consider: branding, logos, company activities, or related content.
        Would viewers associate this with {expected_concept}?
        """
    else:  # concept, product, location
        specific_prompt = f"""
        Does this video effectively illustrate or represent: {expected_concept}?
        Consider: visual metaphors, conceptual clarity, thematic relevance.
        Would this help viewers understand {expected_concept}?
        """
    
    return base_prompt + "\n" + specific_prompt + """
    
    Respond in this exact format:
    RELEVANT: [YES/NO]
    CONFIDENCE: [0.0-1.0]
    DESCRIPTION: [Brief description of what you see in the video]
    ISSUES: [List any problems or why it might not be perfect]
    SUGGESTIONS: [Any suggestions for better search terms]
    """

def _create_validation_result(is_relevant: bool, confidence: float, description: str, issues: List[str]) -> Dict[str, Any]:
    """Helper to create consistent validation result structure"""
    return {
        "is_relevant": is_relevant,
        "confidence": confidence,
        "description": description,
        "issues": issues,
        "suggestions": []
    } 