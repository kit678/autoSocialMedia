"""
Smart Media Selection Logic

This module implements intelligent media type selection based on linguistic
analysis of segment text, including verb analysis and contextual cues.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import spacy

# Try to load spaCy model for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available for advanced linguistic analysis")


class SmartMediaSelector:
    """
    Intelligently selects media type (image/video/reaction) based on
    linguistic analysis of segment text.
    """
    
    # Action verbs that strongly suggest video content
    ACTION_VERBS = {
        'movement': ['run', 'jump', 'walk', 'fly', 'drive', 'race', 'chase', 'escape', 'dance', 'swim'],
        'demonstration': ['show', 'demonstrate', 'reveal', 'display', 'exhibit', 'present', 'perform'],
        'change': ['transform', 'change', 'evolve', 'develop', 'grow', 'build', 'create', 'destroy'],
        'event': ['launch', 'announce', 'release', 'unveil', 'introduce', 'debut', 'premiere'],
        'interaction': ['talk', 'speak', 'discuss', 'debate', 'argue', 'negotiate', 'interview'],
        'process': ['assemble', 'construct', 'operate', 'function', 'process', 'manufacture'],
        'collision': ['crash', 'collide', 'hit', 'strike', 'impact', 'explode', 'burst']
    }
    
    # Stative verbs that suggest static images
    STATIVE_VERBS = ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'seem', 'appear', 
                     'look', 'remain', 'stay', 'exist', 'belong', 'contain', 'consist']
    
    # Emotion triggers for reaction GIFs
    EMOTION_TRIGGERS = {
        'surprise': ['shocked', 'surprising', 'unexpected', 'unbelievable', 'astonishing', 'incredible'],
        'excitement': ['exciting', 'amazing', 'awesome', 'fantastic', 'wonderful', 'breakthrough'],
        'humor': ['funny', 'hilarious', 'ridiculous', 'absurd', 'laughable', 'comical'],
        'frustration': ['frustrating', 'annoying', 'ridiculous', 'stupid', 'dumb', 'terrible'],
        'celebration': ['celebrate', 'victory', 'success', 'achievement', 'milestone', 'win']
    }
    
    # Context patterns that suggest specific media types
    CONTEXT_PATTERNS = {
        'video': [
            r'\b(watch|see|observe|witness)\s+(this|how|as)',
            r'\b(in\s+action|in\s+motion|moving|dynamic)',
            r'\b(demonstration|tutorial|example|footage)',
            r'\b(live|real-time|happening|ongoing)'
        ],
        'image': [
            r'\b(portrait|photo|picture|image)\s+of',
            r'\b(logo|brand|symbol|icon)',
            r'\b(chart|graph|diagram|infographic)',
            r'\b(headquarters|building|location|place)'
        ],
        'reaction': [
            r'\b(can\'t\s+believe|mind\s+blown|face\s+palm)',
            r'\b(lol|omg|wtf|seriously)',
            r'\b(reaction|response|feeling)',
            r'[!?]{2,}',  # Multiple exclamation/question marks
            r'😂|😱|🤯|🤦|🎉'  # Common reaction emojis
        ]
    }
    
    def __init__(self):
        """Initialize the smart media selector."""
        self.logger = logging.getLogger(__name__)
    
    def select_media_type(
        self, 
        segment: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, str]:
        """
        Select the most appropriate media type for a segment.
        
        Args:
            segment: Segment containing text and metadata
            context: Optional context about the overall story
            
        Returns:
            Tuple of (media_type, confidence, reasoning)
        """
        text = segment.get('text', '')
        
        # Check for explicit media preferences if already tagged
        if 'preferred_media' in segment:
            return segment['preferred_media'], 1.0, "Pre-tagged preference"
        
        # Analyze the text
        scores = {
            'image': 0.3,  # Default bias toward images
            'video': 0.0,
            'reaction': 0.0
        }
        reasoning = []
        
        # 1. Check emotion triggers for reactions
        emotion_score, emotion_reason = self._check_emotion_triggers(text)
        if emotion_score > 0:
            scores['reaction'] += emotion_score
            reasoning.append(emotion_reason)
        
        # 2. Analyze verbs
        verb_scores, verb_reasons = self._analyze_verbs(text)
        for media_type, score in verb_scores.items():
            scores[media_type] += score
        reasoning.extend(verb_reasons)
        
        # 3. Check context patterns
        pattern_scores, pattern_reasons = self._check_patterns(text)
        for media_type, score in pattern_scores.items():
            scores[media_type] += score
        reasoning.extend(pattern_reasons)
        
        # 4. Consider entities and visual type hints
        entity_score, entity_reason = self._analyze_entities(segment)
        scores['image'] += entity_score
        if entity_reason:
            reasoning.append(entity_reason)
        
        # 5. Adjust based on segment metadata
        if segment.get('visual_type') == 'Proper Noun':
            scores['image'] += 0.3
            reasoning.append("Proper noun suggests static image")
        elif segment.get('visual_type') == 'Action':
            scores['video'] += 0.3
            reasoning.append("Action type suggests video")
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total
        
        # Select highest scoring type
        selected_type = max(scores, key=scores.get)
        confidence = scores[selected_type]
        
        # Compile reasoning
        final_reasoning = f"Selected {selected_type}: " + "; ".join(reasoning[:3])
        
        return selected_type, confidence, final_reasoning
    
    def _check_emotion_triggers(self, text: str) -> Tuple[float, str]:
        """Check for emotion triggers that suggest reaction GIFs."""
        text_lower = text.lower()
        
        for emotion, triggers in self.EMOTION_TRIGGERS.items():
            for trigger in triggers:
                if trigger in text_lower:
                    return 0.7, f"Emotion trigger '{trigger}' suggests reaction"
        
        # Check for emoji patterns
        if re.search(r'[😂😱🤯🤦🎉💀🙄😮]', text):
            return 0.6, "Emoji suggests emotional reaction"
        
        # Check for exclamation patterns
        if re.search(r'[!?]{2,}', text):
            return 0.4, "Multiple punctuation suggests emphasis"
        
        return 0.0, ""
    
    def _analyze_verbs(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """Analyze verbs to determine if content is static or dynamic."""
        scores = {'image': 0.0, 'video': 0.0, 'reaction': 0.0}
        reasons = []
        
        text_lower = text.lower()
        
        # Check for action verbs
        for category, verbs in self.ACTION_VERBS.items():
            for verb in verbs:
                if re.search(rf'\b{verb}(?:s|ed|ing)?\b', text_lower):
                    scores['video'] += 0.5
                    reasons.append(f"Action verb '{verb}' ({category})")
                    break
        
        # Check for stative verbs
        for verb in self.STATIVE_VERBS:
            if re.search(rf'\b{verb}\b', text_lower):
                scores['image'] += 0.3
                if not reasons:  # Only add if no action verbs found
                    reasons.append(f"Stative verb '{verb}' suggests static content")
                break
        
        # Use spaCy for more advanced analysis if available
        if SPACY_AVAILABLE:
            doc = nlp(text)
            verb_count = sum(1 for token in doc if token.pos_ == "VERB")
            noun_count = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"])
            
            if verb_count > noun_count:
                scores['video'] += 0.2
                reasons.append("High verb density")
            elif noun_count > verb_count * 2:
                scores['image'] += 0.2
                reasons.append("High noun density")
        
        return scores, reasons[:2]  # Limit reasons
    
    def _check_patterns(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """Check for context patterns that suggest media types."""
        scores = {'image': 0.0, 'video': 0.0, 'reaction': 0.0}
        reasons = []
        
        for media_type, patterns in self.CONTEXT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[media_type] += 0.4
                    match = re.search(pattern, text, re.IGNORECASE)
                    reasons.append(f"Pattern '{match.group()}' suggests {media_type}")
                    break
        
        return scores, reasons[:2]
    
    def _analyze_entities(self, segment: Dict[str, Any]) -> Tuple[float, str]:
        """Analyze entities to determine if they need static representation."""
        entities = segment.get('entities', [])
        
        if not entities:
            return 0.0, ""
        
        # Companies and people usually need static images (logos, headshots)
        entity_types = [e.get('type', '') for e in entities] if isinstance(entities[0], dict) else []
        
        if any(t in ['ORG', 'PERSON'] for t in entity_types):
            return 0.4, "Organization/person entities suggest static image"
        elif any(t in ['LOC', 'GPE'] for t in entity_types):
            return 0.3, "Location entities suggest static image"
        
        return 0.2, "Entities present"


# Convenience function for backward compatibility
def select_media_type_smart(
    segment: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Select media type using smart analysis.
    
    Args:
        segment: Segment to analyze
        context: Optional story context
        
    Returns:
        Media type: 'image', 'video', or 'reaction'
    """
    selector = SmartMediaSelector()
    media_type, confidence, reasoning = selector.select_media_type(segment, context)
    
    # Log the decision
    logging.info(f"Media selection: {media_type} (confidence: {confidence:.2f}) - {reasoning}")
    
    return media_type
