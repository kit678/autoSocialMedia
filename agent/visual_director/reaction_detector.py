"""
Reaction Point Detector

Identifies specific moments within segments that warrant reaction GIF overlays,
enabling precise emotional punctuation in videos.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from .llm_intent_tagger import REACTION_TAG_MAP


@dataclass
class ReactionPoint:
    """Represents a specific moment for a reaction overlay."""
    phrase: str
    start_word_index: int
    end_word_index: int
    start_time: float = 0.0
    end_time: float = 0.0
    emotion: str = "surprise"
    intensity: float = 0.5
    duration: float = 2.0
    position: str = "bottom-right"
    scale: float = 0.3
    reasoning: str = ""


@dataclass
class ReactionOverlay:
    """Reaction overlay with asset and timing information."""
    asset: Any  # Asset type from asset_types.py
    reaction_point: ReactionPoint
    start_time: float
    end_time: float
    position: str = "bottom-right"
    scale: float = 0.3
    fade_in: float = 0.2
    fade_out: float = 0.2
    z_index: int = 10


class ReactionPointDetector:
    """Identifies specific moments within segments for reaction overlays."""
    
    # Reaction trigger patterns
    SURPRISE_PATTERNS = [
        r"it turns out (that)?",
        r"surprisingly",
        r"unexpectedly", 
        r"who would have thought",
        r"believe it or not",
        r"plot twist",
        r"shocking(ly)?",
        r"mind[\-\s]?blown",
        r"incredible(ly)?",
        r"unbelievable"
    ]
    
    EXCITEMENT_PATTERNS = [
        r"breaking(\s+news)?",
        r"just in",
        r"this changes everything",
        r"game[\-\s]?changer",
        r"revolutionary",
        r"finally",
        r"at last",
        r"amazing(ly)?",
        r"fantastic",
        r"phenomenal"
    ]
    
    CONCERN_PATTERNS = [
        r"concerning(ly)?",
        r"worrying(ly)?",
        r"alarming(ly)?",
        r"troubling",
        r"disturbing(ly)?",
        r"unfortunately",
        r"bad news",
        r"the problem is",
        r"dangerous(ly)?",
        r"at risk"
    ]
    
    IRONY_PATTERNS = [
        r"but actually",
        r"ironically",
        r"in reality",
        r"however",
        r"on the contrary",
        r"despite (this|that)",
        r"nevertheless",
        r"turns out",
        r"supposedly",
        r"so[\-\s]?called"
    ]
    
    THINKING_PATTERNS = [
        r"let('s| us) think",
        r"consider(ing)?",
        r"imagine (if|that)",
        r"what if",
        r"suppose (that)?",
        r"theoretically",
        r"hypothetically",
        r"ponder(ing)?",
        r"reflect(ing)?",
        r"contemplate"
    ]
    
    CELEBRATION_PATTERNS = [
        r"congrat(ulations|s)",
        r"celebrate",
        r"victory",
        r"success(ful(ly)?)?",
        r"achievement",
        r"milestone",
        r"record[\-\s]?breaking",
        r"historic(al)?",
        r"triumph",
        r"win(ning)?"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reaction detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.min_intensity = config.get('reaction_detection', {}).get('min_intensity', 0.4)
        self.max_overlays_per_segment = config.get('reaction_detection', {}).get('max_overlays_per_segment', 2)
        
        # Compile patterns for efficiency
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns for efficiency."""
        return {
            'surprise': [re.compile(p, re.IGNORECASE) for p in self.SURPRISE_PATTERNS],
            'excitement': [re.compile(p, re.IGNORECASE) for p in self.EXCITEMENT_PATTERNS],
            'concern': [re.compile(p, re.IGNORECASE) for p in self.CONCERN_PATTERNS],
            'facepalm': [re.compile(p, re.IGNORECASE) for p in self.IRONY_PATTERNS],
            'thinking': [re.compile(p, re.IGNORECASE) for p in self.THINKING_PATTERNS],
            'applause': [re.compile(p, re.IGNORECASE) for p in self.CELEBRATION_PATTERNS]
        }
    
    async def detect_reaction_points(
        self,
        segment: Dict[str, Any],
        use_llm: bool = True
    ) -> List[ReactionPoint]:
        """
        Find specific phrases/timestamps that warrant reactions.
        
        Args:
            segment: The segment to analyze
            use_llm: Whether to use LLM analysis (if available)
            
        Returns:
            List of reaction points
        """
        reaction_points = []
        
        # First, try pattern-based detection
        pattern_points = self._detect_with_patterns(segment)
        reaction_points.extend(pattern_points)
        
        # Then, if LLM is available and enabled, get LLM suggestions
        if use_llm and self._is_llm_available():
            try:
                llm_points = await self._detect_with_llm(segment)
                # Merge LLM points, avoiding duplicates
                for llm_point in llm_points:
                    if not self._is_duplicate_point(llm_point, reaction_points):
                        reaction_points.append(llm_point)
            except Exception as e:
                self.logger.warning(f"LLM reaction detection failed: {e}")
        
        # Sort by position and limit to max overlays
        reaction_points.sort(key=lambda p: p.start_word_index)
        
        # Filter by intensity threshold
        reaction_points = [
            p for p in reaction_points
            if p.intensity >= self.min_intensity
        ]
        
        # Limit to max overlays, prioritizing by intensity
        if len(reaction_points) > self.max_overlays_per_segment:
            reaction_points.sort(key=lambda p: p.intensity, reverse=True)
            reaction_points = reaction_points[:self.max_overlays_per_segment]
            reaction_points.sort(key=lambda p: p.start_word_index)
        
        # Calculate timing if word timing is available
        if 'word_timings' in segment:
            self._add_timing_to_points(reaction_points, segment['word_timings'])
        
        return reaction_points
    
    def _detect_with_patterns(self, segment: Dict[str, Any]) -> List[ReactionPoint]:
        """Detect reaction points using regex patterns."""
        text = segment.get('text', '')
        words = text.split()
        reaction_points = []
        
        for emotion, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Find word indices
                    start_char = match.start()
                    end_char = match.end()
                    
                    # Convert character positions to word indices
                    start_word_idx = len(text[:start_char].split()) - 1
                    end_word_idx = len(text[:end_char].split()) - 1
                    
                    # Skip if indices are invalid
                    if start_word_idx < 0 or end_word_idx >= len(words):
                        continue
                    
                    # Determine intensity based on pattern and context
                    intensity = self._calculate_pattern_intensity(
                        match.group(), emotion, text
                    )
                    
                    reaction_point = ReactionPoint(
                        phrase=match.group(),
                        start_word_index=max(0, start_word_idx),
                        end_word_index=min(end_word_idx, len(words) - 1),
                        emotion=emotion,
                        intensity=intensity,
                        duration=self._get_duration_for_emotion(emotion),
                        position=self._get_position_for_emotion(emotion),
                        scale=self._get_scale_for_emotion(emotion),
                        reasoning=f"Pattern match: {pattern.pattern}"
                    )
                    
                    reaction_points.append(reaction_point)
        
        return reaction_points
    
    async def _detect_with_llm(self, segment: Dict[str, Any]) -> List[ReactionPoint]:
        """Use LLM to detect reaction-worthy moments."""
        # This is a placeholder for LLM integration
        # In production, this would call DeepSeek or another LLM
        # For now, return empty list
        return []
    
    def _calculate_pattern_intensity(
        self,
        matched_text: str,
        emotion: str,
        full_text: str
    ) -> float:
        """Calculate intensity based on pattern and context."""
        base_intensity = 0.6
        
        # Boost intensity for certain conditions
        if matched_text.isupper():  # All caps
            base_intensity += 0.2
        
        if '!' in full_text[full_text.find(matched_text):full_text.find(matched_text) + 20]:
            base_intensity += 0.1
        
        # Emotion-specific adjustments
        if emotion == 'surprise' and 'mind' in matched_text.lower():
            base_intensity += 0.15
        elif emotion == 'excitement' and 'breaking' in matched_text.lower():
            base_intensity += 0.2
        elif emotion == 'concern' and 'dangerous' in matched_text.lower():
            base_intensity += 0.15
        
        return min(base_intensity, 1.0)
    
    def _get_duration_for_emotion(self, emotion: str) -> float:
        """Get appropriate duration for an emotion."""
        durations = {
            'surprise': 2.5,
            'excitement': 2.0,
            'concern': 2.0,
            'facepalm': 1.5,
            'thinking': 2.5,
            'applause': 3.0
        }
        return durations.get(emotion, 2.0)
    
    def _get_position_for_emotion(self, emotion: str) -> str:
        """Get appropriate position for an emotion overlay."""
        positions = {
            'surprise': 'center',
            'excitement': 'center',
            'concern': 'bottom-right',
            'facepalm': 'bottom-right',
            'thinking': 'top-right',
            'applause': 'center'
        }
        return positions.get(emotion, 'bottom-right')
    
    def _get_scale_for_emotion(self, emotion: str) -> float:
        """Get appropriate scale for an emotion overlay."""
        scales = {
            'surprise': 0.4,
            'excitement': 0.35,
            'concern': 0.3,
            'facepalm': 0.25,
            'thinking': 0.25,
            'applause': 0.4
        }
        return scales.get(emotion, 0.3)
    
    def _is_duplicate_point(
        self,
        point: ReactionPoint,
        existing_points: List[ReactionPoint]
    ) -> bool:
        """Check if a point overlaps with existing points."""
        for existing in existing_points:
            # Check word index overlap
            if (point.start_word_index <= existing.end_word_index and
                point.end_word_index >= existing.start_word_index):
                return True
        return False
    
    def _add_timing_to_points(
        self,
        points: List[ReactionPoint],
        word_timings: List[Dict[str, Any]]
    ):
        """Add precise timing information to reaction points."""
        for point in points:
            if point.start_word_index < len(word_timings):
                point.start_time = word_timings[point.start_word_index].get('start', 0.0)
            
            if point.end_word_index < len(word_timings):
                point.end_time = word_timings[point.end_word_index].get('end', point.start_time + 0.5)
    
    def _is_llm_available(self) -> bool:
        """Check if LLM is configured and available."""
        return bool(self.config.get('llm', {}).get('enabled', False))
    
    def create_reaction_overlay(
        self,
        reaction_point: ReactionPoint,
        asset: Any
    ) -> ReactionOverlay:
        """
        Create a reaction overlay from a reaction point and asset.
        
        Args:
            reaction_point: The detected reaction point
            asset: The reaction asset (GIF/video)
            
        Returns:
            ReactionOverlay object
        """
        return ReactionOverlay(
            asset=asset,
            reaction_point=reaction_point,
            start_time=reaction_point.start_time,
            end_time=reaction_point.start_time + reaction_point.duration,
            position=reaction_point.position,
            scale=reaction_point.scale,
            fade_in=0.2,
            fade_out=0.2,
            z_index=10
        )
    
    def get_search_terms_for_emotion(self, emotion: str) -> List[str]:
        """Get search terms for finding appropriate reaction GIFs."""
        if emotion in REACTION_TAG_MAP:
            return REACTION_TAG_MAP[emotion]
        
        # Fallback search terms
        return [emotion, "reaction", "gif", "meme"]
