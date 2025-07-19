"""
LLM Intent & Emotion Tagging for Visual Director

This module implements the LLM-based intent and emotion tagging system
that determines visual requirements for each narrative segment.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field

from agent.utils import get_gemini_client


# Reaction emotion mapping as specified in the revamp plan
REACTION_TAG_MAP = {
    "laugh": ["lol", "laugh", "haha", "funny", "hilarious"],
    "facepalm": ["facepalm", "smh", "stupid", "dumb", "ridiculous"],
    "mind_blown": ["mind blown", "explosion", "whoa", "amazing", "incredible", "breakthrough"],
    "shrug": ["shrug", "idk", "uncertain", "maybe", "unclear"],
    "applause": ["applause", "clap", "congratulations", "well done", "success"],
    "shock": ["shock", "shocked", "omg", "surprising", "unexpected", "stunned"],
    "thinking": ["thinking", "hmm", "consider", "ponder", "wonder"],
    "celebration": ["celebrate", "party", "victory", "win", "achievement"],
    "confused": ["confused", "what", "huh", "puzzled", "perplexed"],
    "angry": ["angry", "mad", "furious", "outrage", "upset"]
}


@dataclass
class SegmentIntent:
    """Represents the visual intent for a narrative segment."""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    needs_visual: bool = True
    visual_type: Literal["image", "video", "reaction", "none"] = "image"
    search_terms: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    emotion: Optional[str] = None
    intensity: float = 0.5  # 0-1 scale for reaction intensity
    confidence: float = 0.8


def tag_segment_intent(
    segment: Dict[str, Any],
    context: Dict[str, Any],
    previous_segment: Optional[Dict[str, Any]] = None,
    next_segment: Optional[Dict[str, Any]] = None
) -> SegmentIntent:
    """
    Use LLM to determine visual intent and emotion for a segment.
    
    Args:
        segment: Current segment with text and timing
        context: Story context (angle, tone, etc.)
        previous_segment: Previous segment for context
        next_segment: Next segment for context
        
    Returns:
        SegmentIntent with visual requirements
    """
    client = get_gemini_client()
    
    # Build context for the LLM
    segment_text = segment.get('text', '')
    story_angle = context.get('story_angle', {})
    
    # Extract any pre-identified entities (from NER or previous analysis)
    known_entities = segment.get('entities', [])
    
    prompt = f"""
You are a visual director for short-form social media videos. Analyze this narration segment and determine visual requirements.

CURRENT SEGMENT: "{segment_text}"
STORY CONTEXT: {story_angle.get('angle_name', 'general news')} - {story_angle.get('tone', 'informative')}
KNOWN ENTITIES: {', '.join(known_entities) if known_entities else 'none identified'}

{f'PREVIOUS: "{previous_segment.get("text", "")}"' if previous_segment else ''}
{f'NEXT: "{next_segment.get("text", "")}"' if next_segment else ''}

Determine:
1. Should we show a visual for this segment? (usually yes, unless it's a transition phrase)
2. What type of visual:
   - "image": for people, companies, products, places (static visuals)
   - "video": for actions, demonstrations, events (things happening)
   - "reaction": for emphasis, humor, emotional punctuation (reaction GIFs)
   - "none": keep narrator/previous visual on screen
3. Search terms for finding the visual
4. Any emotional tone that suggests a reaction (if visual_type is "reaction")

Consider:
- Named entities should usually get an image (logo, headshot, product photo)
- Action verbs ("launched", "demonstrated", "crashed") suggest video
- Emotional or emphatic statements might benefit from reaction GIFs
- Technical explanations might need infographics (search for "diagram", "infographic")

Return ONLY a JSON object in this format:
{{
    "needs_visual": true,
    "visual_type": "image|video|reaction|none",
    "search_terms": ["primary term", "secondary term", "fallback term"],
    "entities": ["Entity Name"],
    "emotion": "mind_blown|shock|laugh|facepalm|applause|thinking|none",
    "reasoning": "Brief explanation of choice"
}}
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[prompt],
            config={
                'temperature': 0.3,  # Lower temperature for consistency
                'max_output_tokens': 500,
            }
        )
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        
        # Create SegmentIntent from LLM response
        return SegmentIntent(
            segment_id=segment.get('id', f"seg_{segment.get('index', 0):02d}"),
            text=segment_text,
            start_time=segment.get('start_time', 0.0),
            end_time=segment.get('end_time', 0.0),
            needs_visual=result.get('needs_visual', True),
            visual_type=result.get('visual_type', 'image'),
            search_terms=result.get('search_terms', []),
            entities=result.get('entities', known_entities),
            emotion=result.get('emotion', None),
            confidence=0.9 if result.get('reasoning') else 0.7
        )
        
    except Exception as e:
        logging.warning(f"LLM intent tagging failed: {e}, using fallback heuristics")
        return fallback_intent_detection(segment, known_entities)


def fallback_intent_detection(segment: Dict[str, Any], entities: List[str]) -> SegmentIntent:
    """
    Fallback heuristic-based intent detection when LLM fails.
    """
    text = segment.get('text', '').lower()
    segment_id = segment.get('id', f"seg_{segment.get('index', 0):02d}")
    
    # Detect visual type based on keywords
    visual_type = "image"  # default
    emotion = None
    search_terms = entities.copy() if entities else []
    
    # Check for action words suggesting video
    action_words = ['demonstrate', 'show', 'reveal', 'launch', 'crash', 'explode', 
                    'move', 'run', 'fly', 'drive', 'operate', 'perform']
    if any(word in text for word in action_words):
        visual_type = "video"
        search_terms.extend([word for word in action_words if word in text])
    
    # Check for emotional/reaction triggers
    for emotion_key, triggers in REACTION_TAG_MAP.items():
        if any(trigger in text for trigger in triggers):
            visual_type = "reaction"
            emotion = emotion_key
            break
    
    # Add generic search terms if none found
    if not search_terms:
        if 'ai' in text or 'artificial intelligence' in text:
            search_terms = ['artificial intelligence', 'AI technology', 'machine learning']
        elif 'company' in text or 'startup' in text:
            search_terms = ['tech company', 'startup office', 'business']
        else:
            # Extract nouns as potential search terms
            import re
            nouns = re.findall(r'\b[A-Z][a-z]+\b', segment.get('text', ''))
            search_terms = nouns[:3] if nouns else ['technology']
    
    return SegmentIntent(
        segment_id=segment_id,
        text=segment.get('text', ''),
        start_time=segment.get('start_time', 0.0),
        end_time=segment.get('end_time', 0.0),
        needs_visual=True,
        visual_type=visual_type,
        search_terms=search_terms,
        entities=entities,
        emotion=emotion,
        confidence=0.5  # Lower confidence for fallback
    )


def batch_tag_segments(
    segments: List[Dict[str, Any]], 
    context: Dict[str, Any],
    use_llm: bool = True
) -> List[SegmentIntent]:
    """
    Process multiple segments and determine visual intents.
    
    Args:
        segments: List of narrative segments
        context: Story context
        use_llm: Whether to use LLM (True) or fallback heuristics
        
    Returns:
        List of SegmentIntent objects
    """
    results = []
    
    for i, segment in enumerate(segments):
        previous = segments[i-1] if i > 0 else None
        next_seg = segments[i+1] if i < len(segments)-1 else None
        
        if use_llm:
            intent = tag_segment_intent(segment, context, previous, next_seg)
        else:
            # Use fallback for testing or when LLM is unavailable
            intent = fallback_intent_detection(segment, segment.get('entities', []))
        
        results.append(intent)
        
        logging.info(f"Segment {i}: type={intent.visual_type}, "
                    f"emotion={intent.emotion}, terms={intent.search_terms[:2]}")
    
    return results


def determine_media_preference(visual_type: str, segment_text: str) -> str:
    """
    Determine preferred media type based on visual type and context.
    
    Args:
        visual_type: The visual type (image/video/reaction)
        segment_text: The segment text for additional context
        
    Returns:
        Preferred media type for adapter queries
    """
    if visual_type == "reaction":
        return "gif"  # Reactions are typically GIFs or short MP4s
    elif visual_type == "video":
        return "video"
    else:
        return "image"


def get_reaction_search_terms(emotion: str) -> List[str]:
    """
    Get search terms for a specific reaction emotion.
    
    Args:
        emotion: The emotion tag (e.g., "mind_blown", "facepalm")
        
    Returns:
        List of search terms for finding appropriate reactions
    """
    if emotion in REACTION_TAG_MAP:
        return REACTION_TAG_MAP[emotion]
    
    # Fallback to emotion itself
    return [emotion, "reaction", "gif"]


def requires_reaction_overlay(intent: SegmentIntent) -> bool:
    """
    Determine if this segment should have a reaction overlay.
    
    Args:
        intent: The segment intent
        
    Returns:
        True if a reaction overlay should be added
    """
    return (
        intent.visual_type == "reaction" and 
        intent.emotion is not None and
        intent.intensity > 0.3
    )
