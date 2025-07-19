"""
LLM Intent Tagger for Enhanced Visual Director

This module implements the LLM-based intent and emotion analysis system
for the enhanced visual director integration.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
import asyncio

from agent.utils import get_gemini_client


@dataclass
class VisualIntent:
    """Represents the visual intent for a narrative segment."""
    segment_id: str
    text: str
    emotions: List[str] = field(default_factory=list)
    visual_needs: List[Dict[str, Any]] = field(default_factory=list)
    emphasis_level: float = 0.5  # 0-1 scale
    scene_type: Optional[str] = None
    confidence: float = 0.8


class LLMIntentTagger:
    """
    LLM-based intent and emotion analysis for visual segments.
    
    Uses Gemini to analyze narrative segments and determine visual requirements.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize the LLM intent tagger.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the model (optional if using default client)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.client = get_gemini_client()
        self.logger = logging.getLogger(__name__)
        
        # Emotion to search term mapping
        self.emotion_map = {
            "excitement": ["excited", "celebration", "party", "joy", "happy"],
            "shock": ["shocked", "surprised", "stunned", "amazed"],
            "confusion": ["confused", "puzzled", "thinking", "wondering"],
            "concern": ["worried", "concerned", "anxious", "trouble"],
            "anger": ["angry", "frustrated", "mad", "upset"],
            "sadness": ["sad", "disappointed", "down", "depressed"],
            "laughter": ["laughing", "funny", "hilarious", "comedy"],
            "applause": ["clapping", "applause", "celebration", "success"]
        }
    
    async def analyze_segment(
        self,
        segment: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> VisualIntent:
        """
        Analyze a segment to determine visual intent and emotional requirements.
        
        Args:
            segment: Segment data with text, entities, etc.
            project_context: Overall project context
            
        Returns:
            VisualIntent object with analysis results
        """
        segment_text = segment.get('text', '')
        entities = segment.get('entities', [])
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(segment_text, entities, project_context)
        
        try:
            response = await self._call_llm(prompt)
            intent = self._parse_response(response, segment)
            return intent
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}, using fallback")
            return self.fallback_analysis(segment)
    
    def _build_analysis_prompt(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        project_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build the analysis prompt for the LLM.
        """
        entity_names = [e.get('text', '') for e in entities]
        context_info = ""
        
        if project_context:
            context_info = f"""
Project Context:
- Title: {project_context.get('title', 'Unknown')}
- Topics: {', '.join(project_context.get('topics', []))}
- Tone: {project_context.get('tone', 'neutral')}
"""
        
        prompt = f"""
You are a visual director for short-form social media videos. Analyze this narration segment and determine visual requirements.

{context_info}

Narration Segment: "{text}"
Identified Entities: {', '.join(entity_names) if entity_names else 'none'}

Analyze and determine:
1. Emotional tone and intensity (0-1 scale)
2. Visual needs (what types of visuals would enhance this segment)
3. Emphasis level (how much visual impact this segment needs)
4. Scene type (talking head, b-roll, graphics, etc.)

For each visual need, specify:
- Type: "entity" (for people/places/things), "concept" (for abstract ideas), "emotion" (for emotional reactions)
- Query: search terms to find the visual
- Priority: 1-5 (5 = critical, 1 = nice to have)

Return ONLY a JSON object in this format:
{{
    "emotions": ["emotion1", "emotion2"],
    "emphasis_level": 0.7,
    "scene_type": "b-roll",
    "visual_needs": [
        {{
            "type": "entity",
            "query": "search terms",
            "priority": 4
        }}
    ],
    "confidence": 0.9
}}
"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the analysis prompt.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config={
                    'temperature': 0.3,
                    'max_output_tokens': 1000,
                }
            )
            
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str, segment: Dict[str, Any]) -> VisualIntent:
        """
        Parse the LLM response into a VisualIntent object.
        """
        try:
            # Clean up response
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            data = json.loads(response)
            
            return VisualIntent(
                segment_id=segment.get('id', 'unknown'),
                text=segment.get('text', ''),
                emotions=data.get('emotions', []),
                visual_needs=data.get('visual_needs', []),
                emphasis_level=data.get('emphasis_level', 0.5),
                scene_type=data.get('scene_type'),
                confidence=data.get('confidence', 0.8)
            )
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return self.fallback_analysis(segment)
    
    def fallback_analysis(self, segment: Dict[str, Any]) -> VisualIntent:
        """
        Fallback analysis when LLM fails.
        """
        text = segment.get('text', '').lower()
        entities = segment.get('entities', [])
        
        # Detect emotions from keywords
        emotions = []
        for emotion, keywords in self.emotion_map.items():
            if any(keyword in text for keyword in keywords):
                emotions.append(emotion)
        
        # Generate visual needs from entities
        visual_needs = []
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('type', 'MISC')
            
            if entity_type in ['PERSON', 'ORG', 'PRODUCT']:
                visual_needs.append({
                    'type': 'entity',
                    'query': entity_text,
                    'priority': 4
                })
            elif entity_type in ['LOC', 'GPE']:
                visual_needs.append({
                    'type': 'entity',
                    'query': f"{entity_text} location",
                    'priority': 3
                })
        
        # Add concept-based needs
        if 'technology' in text or 'ai' in text:
            visual_needs.append({
                'type': 'concept',
                'query': 'artificial intelligence technology',
                'priority': 3
            })
        
        # Determine emphasis level
        emphasis_words = ['breakthrough', 'revolutionary', 'amazing', 'incredible', 'shocking']
        emphasis_level = 0.8 if any(word in text for word in emphasis_words) else 0.5
        
        return VisualIntent(
            segment_id=segment.get('id', 'unknown'),
            text=segment.get('text', ''),
            emotions=emotions,
            visual_needs=visual_needs,
            emphasis_level=emphasis_level,
            scene_type='b-roll',
            confidence=0.6  # Lower confidence for fallback
        )


# Backward compatibility function
async def tag_segments_with_intent(
    segments: List[Dict[str, Any]], 
    article_text: str, 
    headline: str,
    logger: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Tag segments with intent and emotion using LLM.
    
    Backward compatibility function for existing code.
    
    Args:
        segments: List of visual segments from story planning
        article_text: Full article text for context
        headline: Article headline
        logger: Optional decision logger
        
    Returns:
        Enriched segments with additional metadata
    """
    tagger = LLMIntentTagger()
    
    # Create project context
    project_context = {
        'title': headline,
        'topics': [],
        'tone': 'neutral'
    }
    
    enriched_segments = []
    
    for segment in segments:
        try:
            # Use the new tagger
            intent = await tagger.analyze_segment(segment, project_context)
            
            # Convert to old format for backward compatibility
            enriched_segment = segment.copy()
            enriched_segment.update({
                'intent': 'inform',  # Default intent
                'emotion': intent.emotions[0] if intent.emotions else 'neutral',
                'entities': [{'text': vn['query'], 'type': 'ENTITY'} for vn in intent.visual_needs if vn['type'] == 'entity'],
                'preferred_media': 'reaction' if intent.emotions else 'image',
                'motion_hint': 'dynamic' if intent.emphasis_level > 0.7 else 'static',
                'licence_requirement': 'any'
            })
            
            enriched_segments.append(enriched_segment)
            
        except Exception as e:
            logging.error(f"Error tagging segment: {e}")
            # Add defaults if tagging fails
            enriched_segment = segment.copy()
            enriched_segment.update({
                'intent': 'inform',
                'emotion': 'neutral',
                'entities': [],
                'preferred_media': 'image',
                'motion_hint': 'static',
                'licence_requirement': 'any'
            })
            enriched_segments.append(enriched_segment)
    
    return enriched_segments
