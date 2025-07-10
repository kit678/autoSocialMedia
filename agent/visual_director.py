"""
AI Visual Director - Makes intelligent decisions about video visual strategy
"""

import os
import json
import logging
import re
from typing import Dict, List, Tuple, Any
from agent.decision_logger import log_decision
from agent.utils import rate_limit_gemini

@rate_limit_gemini
def analyze_visual_requirements(script_text: str, screenshot_path: str, audio_duration: float) -> Dict[str, Any]:
    """
    Use AI to analyze script and determine comprehensive visual strategy with deterministic source selection
    
    Args:
        script_text: The narration script (ONLY source for visual cues)
        screenshot_path: Path to website screenshot
        audio_duration: Duration of audio narration in seconds
        
    Returns:
        Dict containing visual direction decisions
    """
    from agent.utils import get_gemini_client
    
    try:
        # First: Extract entities deterministically
        script_entities = analyze_script_entities(script_text)
        
        client = get_gemini_client()
        
        prompt = f"""
        You are an expert video director specializing in fast-paced, engaging news content for social media.
        
        SCRIPT TO ANALYZE FOR VISUALS:
        {script_text}
        
        AUDIO DURATION: {audio_duration} seconds
        SCREENSHOT AVAILABLE: Yes (website screenshot of the headline)
        
        CRITICAL: Generate visual cues based ONLY on what appears in the SCRIPT TEXT above.
        Do NOT create visual cues for elements that don't appear in the script.
        Only reference people, companies, concepts, and actions explicitly mentioned in the script.
        
        Create a comprehensive visual strategy for this video. Return a JSON with these decisions:
        
        {{
            "opening_strategy": {{
                "screenshot_duration": <float: seconds to show screenshot (2-4 seconds)>,
                "zoom_focus": "<string: describe what area to zoom into on screenshot>",
                "transition_out": "<string: how to transition from screenshot to main content>"
            }},
            "pacing_strategy": {{
                "target_visuals_count": <int: total number of visual cuts needed>,
                "avg_visual_duration": <float: average seconds per visual>,
                "rhythm": "<string: fast/medium/varied - how to pace cuts>",
                "high_energy_moments": [<list of keywords FROM SCRIPT where cuts should be faster>]
            }},
            "visual_requirements": {{
                "named_entities": [<list of people, companies, products mentioned IN THE SCRIPT>],
                "key_concepts": [<list of important concepts mentioned IN THE SCRIPT>],
                "action_words": [<list of action moments mentioned IN THE SCRIPT>],
                "emotion_tone": "<string: serious/exciting/informative/urgent>"
            }},
            "transition_strategy": {{
                "primary_transition": "<string: main transition type to use>",
                "accent_transitions": [<list of special transitions for emphasis>],
                "fade_points": [<list of moments that need softer transitions>]
            }}
        }}
        
        GUIDELINES:
        - For {audio_duration} seconds, aim for engaging pacing (new visual every 2-4 seconds typically)
        - Consider the news tone - serious tech news vs exciting announcements
        - ONLY include entities, concepts, and actions that are explicitly mentioned in the script
        - Prioritize showing people ONLY when they're mentioned in the script
        - Use action footage for dynamic moments mentioned in the script
        - Balance information density with visual engagement
        - Screenshot should highlight the main headline/topic
        
        CRITICAL: Only reference elements that actually appear in the provided script text.
        
        Return ONLY the JSON, no additional text.
        """
        
        from google.genai import types
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Lower temperature for more deterministic results
                max_output_tokens=2000,
            ),
        )
        
        # Parse the JSON response
        response_text = response.text.strip()
        
        # Strip markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()
        
        try:
            visual_strategy = json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from AI visual director: {e}")
            logging.error(f"Raw response: {response_text[:500]}...")
            raise Exception(f"AI visual strategy JSON parsing failed: {e}")
        
        # Apply deterministic source selection instead of AI guessing
        visual_strategy["priority_sources"] = determine_priority_sources(
            visual_strategy.get("visual_requirements", {}), 
            script_entities
        )
        
        # Log the decision
        log_decision(
            step="visual_direction",
            decision="Created comprehensive visual strategy with deterministic source selection",
            reasoning="AI analyzed script content and duration, deterministic logic selected sources based on extracted entities",
            metadata={
                "script_length": len(script_text),
                "audio_duration": audio_duration,
                "extracted_entities": script_entities,
                "visual_strategy": visual_strategy
            }
        )
        
        logging.info(f"AI Visual Director created strategy: {visual_strategy['pacing_strategy']['target_visuals_count']} visuals over {audio_duration}s")
        logging.info(f"📊 Deterministic sources: {visual_strategy['priority_sources']}")
        
        return visual_strategy
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR: Visual requirements analysis failed: {e}"
        logging.error(error_msg)
        
        # CRASH instead of using fallback
        raise Exception(error_msg + " Pipeline configured to crash instead of using fallback visual strategies.")

@rate_limit_gemini
def extract_visual_timeline(script_text: str, visual_strategy: Dict[str, Any], audio_path: str = None) -> List[Dict[str, Any]]:
    """
    Use LLM to extract keywords with ACTUAL timestamps using whisper-timestamped
    
    Args:
        script_text: The narration script
        visual_strategy: Visual strategy from analyze_visual_requirements
        audio_path: Path to audio file for timestamp extraction (REQUIRED)
        
    Returns:
        List of visual cues with REAL timestamps and requirements
    """
    from agent.utils import get_gemini_client
    
    try:
        # REQUIRE audio file - no fallback to estimated timing
        if not audio_path or not os.path.exists(audio_path):
            error_msg = f"CRITICAL: Audio file required for whisper-timestamped integration: {audio_path}"
            logging.error(error_msg)
            logging.error("The audio file is essential for extracting word-level timestamps")
            logging.error("Without it, visual timing would be pure guesswork")
            raise Exception(error_msg + " Pipeline configured to crash instead of using fallback timing.")
        
        client = get_gemini_client()
        
        target_count = visual_strategy['pacing_strategy']['target_visuals_count']
        named_entities = visual_strategy['visual_requirements']['named_entities']
        key_concepts = visual_strategy['visual_requirements']['key_concepts']
        action_words = visual_strategy['visual_requirements']['action_words']
        
        prompt = f"""
        You are an expert at creating visual timelines for video content. Analyze this script and create a timeline of visual cues.
        
        SCRIPT:
        {script_text}
        
        VISUAL STRATEGY CONTEXT:
        - Target {target_count} visuals total
        - Key entities to show: {named_entities}
        - Important concepts: {key_concepts}  
        - Action moments: {action_words}
        
        Create a timeline of visual cues. For each cue, identify:
        1. SPECIFIC TRIGGER WORD/PHRASE that should appear in the spoken narration
        2. What type of visual is needed
        3. How long it should be shown (2-4 seconds typically)
        
        CRITICAL: Choose trigger words that will actually be SPOKEN in the narration.
        Only select words/phrases that explicitly appear in the provided script text.
        The timing will be determined by when these words are actually spoken in the audio.
        
        SCRIPT TEXT FOR REFERENCE:
        "{script_text}"
        
        Return a JSON array:
        [
            {{
                "trigger_keyword": "<EXACT word/phrase from script that triggers this visual>",
                "visual_type": "<string: person|company|concept|action|product|location>",
                "search_terms": [<list of search terms for finding this visual>],
                "duration": <float: seconds to show this visual (2.0-4.0)>,
                "priority": "<string: high|medium|low>",
                "visual_style": "<string: photo|logo|illustration|footage>",
                "context": "<string: why this visual is needed>"
            }}
        ]
        
        GUIDELINES:
        - Choose trigger words that DEFINITELY appear in the script text
        - Prioritize showing people when their names are mentioned
        - Show company logos when companies are mentioned
        - Use concept visuals for abstract ideas
        - Action footage for dynamic moments
        - High priority for named entities
        - Duration should be 2-4 seconds per visual
        
        Return ONLY the JSON array, no additional text.
        """
        
        # Extract trigger keywords using AI first
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=[prompt],
            config={
                'temperature': 0.1,
                'max_output_tokens': 3000,
            },
        )
        
        response_text = response.text.strip()
        json_text = _clean_json(response_text)
        
        try:
            ai_timeline = json.loads(json_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from AI timeline extractor: {e}")
            logging.error(f"Cleaned JSON: {json_text[:500]}...")
            logging.error(f"Original response: {response_text[:500]}...")
            raise Exception(f"AI timeline JSON parsing failed: {e}")
        
        # NOW use whisper-timestamped to get ACTUAL timing for keywords
        logging.info("=== USING WHISPER-TIMESTAMPED FOR PRECISE TIMING ===")
        
        from agent.audio.timestamp_extractor import extract_word_timestamps, find_keyword_timing, get_word_timing_summary
        
        # Extract word-level timestamps from audio
        word_timings = extract_word_timestamps(audio_path, script_text)
        
        # Debug: Show what words we found
        logging.info(get_word_timing_summary(word_timings))
        
        # Validate that all trigger keywords exist in the script before processing
        trigger_keywords = [entry.get('trigger_keyword', '') for entry in ai_timeline]
        script_lower = script_text.lower()
        invalid_keywords = []
        
        for keyword in trigger_keywords:
            # Check if keyword appears in script (case-insensitive)
            if keyword.lower() not in script_lower:
                invalid_keywords.append(keyword)
        
        if invalid_keywords:
            logging.warning(f"Removing {len(invalid_keywords)} keywords not found in script: {invalid_keywords}")
            # Filter out invalid keywords
            ai_timeline = [entry for entry in ai_timeline 
                          if entry.get('trigger_keyword', '').lower() in script_lower]
            trigger_keywords = [entry.get('trigger_keyword', '') for entry in ai_timeline]
            logging.info(f"Proceeding with {len(trigger_keywords)} valid keywords")
        
        # Process each visual entry and CRASH IMMEDIATELY if any keyword fails
        for i, entry in enumerate(ai_timeline):
            trigger_keyword = entry.get('trigger_keyword', '')
            
            # Find actual timestamp when this keyword is spoken
            actual_timestamp = find_keyword_timing(trigger_keyword, word_timings)
            
            if actual_timestamp is not None:
                # REAL timestamp from audio
                entry['start_time'] = actual_timestamp
                entry['end_time'] = actual_timestamp + entry.get('duration', 3.0)
                
                # Fix timeline_position calculation to avoid Infinity
                total_duration = word_timings.get('total_duration', 0)
                if total_duration <= 0:
                    error_msg = f"CRITICAL: Audio duration is {total_duration}s - invalid audio file or whisper processing failed"
                    logging.error(error_msg)
                    logging.error("Cannot calculate visual timing without valid audio duration")
                    logging.error("Pipeline configured to crash instead of using fallback duration")
                    raise Exception(error_msg + " Pipeline configured to crash instead of using fallback timing.")
                
                entry['timeline_position'] = actual_timestamp / total_duration
                
                entry['timing_source'] = 'whisper_actual'
                logging.info(f"✅ {trigger_keyword}: FOUND at {actual_timestamp:.1f}s")
                
                # Add required fields for compatibility
                entry.setdefault('concept', f'visual_{i}')
                entry.setdefault('effect', 'fade')
            else:
                # CRASH IMMEDIATELY - no fallback processing
                error_msg = f"CRITICAL: Keyword '{trigger_keyword}' not found in audio transcription"
                logging.error(error_msg)
                logging.error("Available words in transcription:")
                word_data = word_timings.get("word_timings", {})
                available_words = list(word_data.keys())[:20]  # Show first 20 words
                logging.error(f"  {available_words}")
                logging.error("This means the keyword doesn't appear in the actual spoken audio")
                logging.error("Pipeline configured to crash instead of using estimated timing")
                raise Exception(error_msg + " Pipeline configured to crash instead of using fallback timing.")
        
        # Sort by actual start time
        ai_timeline.sort(key=lambda x: x.get('start_time', 0))
        
        logging.info(f"✅ Successfully created timeline with ACTUAL timestamps from whisper-timestamped")
        logging.info(f"✅ ALL {len(ai_timeline)} keywords mapped to real speech timing")
        
        return ai_timeline
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR: Visual timeline extraction failed: {e}"
        logging.error(error_msg)
        
        # CRASH instead of using fallback
        raise Exception(error_msg + " Pipeline configured to crash instead of using fallback visual timeline.")

def _clean_json(raw: str) -> str:
    """Clean Gemini response for JSON parsing."""
    try:
        import re
        
        # Strip markdown fences
        cleaned = raw.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        # Remove comments and fix common issues
        # Remove // comments
        cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
        
        # Remove /* */ comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Fix trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix single quotes to double quotes (be more careful)
        cleaned = re.sub(r"'([^']*)'(\s*):", r'"\1"\2:', cleaned)
        
        # Remove any non-printable characters but preserve newlines and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        return cleaned.strip()
        
    except Exception as e:
        logging.warning(f"JSON cleaning failed: {e}")
        return raw.strip()

def analyze_script_entities(script_text: str) -> Dict[str, List[str]]:
    """Extract named entities from script for deterministic source selection"""
    try:
        script_lower = script_text.lower()
        
        # Common company indicators
        company_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corporation|Corp\.|Company|Co\.|LLC|Ltd\.)',
            r'\b([A-Z][a-z]+)\s+(?:announced|launched|released|reported|said)',
            r'\bCEO\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(Apple|Google|Microsoft|Meta|Amazon|Netflix|Tesla|SpaceX|OpenAI|Anthropic|Block|Square|PayPal|Stripe|Coinbase|Robinhood|Uber|Lyft|Airbnb|Zoom|Slack|Salesforce|Adobe|Oracle|IBM|Intel|NVIDIA|AMD|HP|Dell|Sony|Samsung|LG|Huawei|Xiaomi|Twitter|Facebook|Instagram|LinkedIn|TikTok|YouTube|WhatsApp)\b',
        ]
        
        # Common person name patterns
        person_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:said|announced|stated|told|reported)',
            r'\bCEO\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'\b(?:Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        # Common location patterns
        location_patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s+[A-Z]{2})?)',
            r'\b(Silicon Valley|Wall Street|New York|California|Texas|London|Tokyo|Seoul|Beijing|Shanghai|Berlin|Paris|Amsterdam)\b',
        ]
        
        # Common product patterns
        product_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:app|software|platform|service|device|product)',
            r'\b(iPhone|iPad|MacBook|Android|Windows|Chrome|Safari|Edge|Firefox)\b',
        ]
        
        companies = []
        people = []
        locations = []
        products = []
        
        # Extract companies
        for pattern in company_patterns:
            matches = re.findall(pattern, script_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match and len(match) > 2:
                    companies.append(match.strip())
        
        # Extract people
        for pattern in person_patterns:
            matches = re.findall(pattern, script_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match and len(match.split()) >= 2:
                    people.append(match.strip())
        
        # Extract locations
        for pattern in location_patterns:
            matches = re.findall(pattern, script_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match and len(match) > 2:
                    locations.append(match.strip())
        
        # Extract products
        for pattern in product_patterns:
            matches = re.findall(pattern, script_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match and len(match) > 1:
                    products.append(match.strip())
        
        # Remove duplicates and clean up
        entities = {
            "companies": list(set([c for c in companies if c])),
            "people": list(set([p for p in people if p])), 
            "locations": list(set([l for l in locations if l])),
            "products": list(set([pr for pr in products if pr]))
        }
        
        logging.info(f"📋 Extracted entities: {len(entities['companies'])} companies, {len(entities['people'])} people, {len(entities['locations'])} locations, {len(entities['products'])} products")
        
        return entities
        
    except Exception as e:
        logging.warning(f"Entity extraction failed: {e}")
        return {"companies": [], "people": [], "locations": [], "products": []}

def determine_priority_sources(visual_requirements: Dict, script_entities: Dict) -> Dict[str, str]:
    """Deterministic source selection based on script content"""
    try:
        sources = {}
        
        # Explicit rules instead of AI guessing
        if script_entities["people"]:
            sources["people_images"] = "google"  # Always Google for specific people
            logging.info(f"🔍 Using Google for people (found: {script_entities['people'][:3]})")
        else:
            sources["people_images"] = "stock_photo"
        
        if script_entities["companies"]:
            sources["company_logos"] = "google"  # Always Google for company logos
            logging.info(f"🔍 Using Google for companies (found: {script_entities['companies'][:3]})")
        else:
            sources["company_logos"] = "stock_photo"
        
        if script_entities["locations"]:
            sources["location_images"] = "google"  # Always Google for real places
            logging.info(f"🔍 Using Google for locations (found: {script_entities['locations'][:3]})")
        else:
            sources["location_images"] = "stock_photo"
        
        if script_entities["products"]:
            sources["product_images"] = "google"  # Always Google for branded products
            logging.info(f"🔍 Using Google for products (found: {script_entities['products'][:3]})")
        else:
            sources["product_images"] = "stock_photo"
        
        # Action footage always stock video
        sources["action_footage"] = "stock_video"
        
        # Concepts are abstract - always stock
        sources["concept_visuals"] = "stock_photo"
        
        logging.info(f"📋 Deterministic source selection: {sources}")
        return sources
        
    except Exception as e:
        error_msg = f"CRITICAL: Source determination failed: {e}"
        logging.error(error_msg)
        logging.error("Cannot proceed without valid source selection")
        logging.error("Pipeline configured to crash instead of using fallback source selection")
        raise Exception(error_msg + " Pipeline configured to crash instead of using fallback sources.") 