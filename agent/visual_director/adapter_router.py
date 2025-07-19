"""
Intelligent Adapter Router

Routes visual search queries to the most appropriate adapters based on
content analysis, reducing API calls and improving relevance.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from .asset_registry import AssetAdapterRegistry
from .llm_intent_tagger import SegmentIntent


@dataclass
class AdapterProfile:
    """Defines an adapter's strengths and ideal use cases."""
    name: str
    topics: List[str] = None
    entities: List[str] = None
    entity_types: List[str] = None  # PERSON, ORG, LOC, etc.
    visual_types: List[str] = None  # image, video, reaction
    temporal_preference: str = None  # recent, historical, any
    emotion_triggers: bool = False
    reaction_specialist: bool = False
    confidence_boost: float = 0.0
    quality_score: float = 0.8
    free_tier: bool = True
    rate_limited: bool = False


class AdapterRouter:
    """Routes queries to most appropriate adapters based on content analysis."""
    
    def __init__(self, registry: AssetAdapterRegistry, config: Dict[str, Any]):
        """
        Initialize the adapter router.
        
        Args:
            registry: The adapter registry
            config: Configuration dictionary
        """
        self.registry = registry
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.adapter_profiles = self._build_adapter_profiles()
        
        # Cache for routing decisions
        self._routing_cache = {}
    
    def _build_adapter_profiles(self) -> Dict[str, AdapterProfile]:
        """Define each adapter's strengths and ideal use cases."""
        return {
            "nasa": AdapterProfile(
                name="nasa",
                topics=["space", "astronomy", "rocket", "planet", "satellite", "cosmos",
                       "astronaut", "spacecraft", "nebula", "galaxy", "mars", "moon"],
                entities=["NASA", "SpaceX", "ISS", "Mars", "Moon", "Jupiter", "Saturn",
                         "Hubble", "James Webb", "Apollo", "Artemis"],
                entity_types=["ORG", "LOC"],
                visual_types=["image", "video"],
                temporal_preference="any",
                confidence_boost=0.4,
                quality_score=0.95,
                free_tier=True
            ),
            
            "gdelt": AdapterProfile(
                name="gdelt",
                topics=["news", "politics", "breaking", "current events", "election",
                       "president", "minister", "parliament", "congress", "policy"],
                entities=["president", "minister", "government", "congress", "senate"],
                entity_types=["PERSON", "ORG", "GPE"],
                visual_types=["video"],
                temporal_preference="recent",
                confidence_boost=0.3,
                quality_score=0.85,
                free_tier=True
            ),
            
            "archive_tv": AdapterProfile(
                name="archive_tv",
                topics=["history", "historical", "archive", "past", "decade", "century",
                       "war", "vintage", "classic", "documentary"],
                entities=[],
                entity_types=["EVENT", "DATE"],
                visual_types=["video"],
                temporal_preference="historical",
                confidence_boost=0.25,
                quality_score=0.75,
                free_tier=True
            ),
            
            "tenor": AdapterProfile(
                name="tenor",
                topics=["reaction", "emotion", "feeling", "mood"],
                entities=[],
                visual_types=["reaction", "gif"],
                emotion_triggers=True,
                reaction_specialist=True,
                confidence_boost=0.5,
                quality_score=0.8,
                free_tier=True
            ),
            
            "pexels": AdapterProfile(
                name="pexels",
                topics=["professional", "business", "lifestyle", "nature", "technology",
                       "people", "abstract", "background"],
                entities=[],
                visual_types=["image", "video"],
                temporal_preference="any",
                confidence_boost=0.1,
                quality_score=0.9,
                free_tier=True,
                rate_limited=True
            ),
            
            "wikimedia": AdapterProfile(
                name="wikimedia",
                topics=["education", "encyclopedia", "academic", "research", "science",
                       "history", "geography", "biology"],
                entities=[],
                entity_types=["LOC", "ORG"],
                visual_types=["image"],
                temporal_preference="any",
                confidence_boost=0.15,
                quality_score=0.7,
                free_tier=True
            ),
            
            "openverse": AdapterProfile(
                name="openverse",
                topics=["art", "creative", "culture", "museum", "artistic", "design",
                       "illustration", "painting"],
                entities=[],
                visual_types=["image"],
                temporal_preference="any",
                confidence_boost=0.2,
                quality_score=0.8,
                free_tier=True
            ),
            
            "searxng": AdapterProfile(
                name="searxng",
                topics=[],  # General purpose
                entities=[],
                visual_types=["image"],
                temporal_preference="any",
                confidence_boost=0.0,  # No boost, it's a fallback
                quality_score=0.7,
                free_tier=True
            ),
            
            "coverr": AdapterProfile(
                name="coverr",
                topics=["cinematic", "drone", "aerial", "landscape", "urban", "nature"],
                entities=[],
                visual_types=["video"],
                temporal_preference="any",
                confidence_boost=0.2,
                quality_score=0.85,
                free_tier=False,
                rate_limited=True
            )
        }
    
    async def route_query(
        self,
        segment: Dict[str, Any],
        intent: SegmentIntent,
        available_adapters: List[str]
    ) -> List[str]:
        """
        Returns ordered list of adapter names based on relevance.
        
        Args:
            segment: The segment to find visuals for
            intent: The analyzed segment intent
            available_adapters: List of currently available adapter names
            
        Returns:
            Ordered list of adapter names (most relevant first)
        """
        # Quick routing for reactions
        if intent.visual_type == "reaction":
            reaction_adapters = ["tenor", "pexels", "searxng"]
            return [a for a in reaction_adapters if a in available_adapters]
        
        # Create cache key
        cache_key = self._create_cache_key(segment, intent)
        if cache_key in self._routing_cache:
            cached = self._routing_cache[cache_key]
            return [a for a in cached if a in available_adapters]
        
        # Score adapters based on content match
        adapter_scores = await self._score_adapters(segment, intent, available_adapters)
        
        # Sort by score and return adapter names
        sorted_adapters = [
            name for name, _ in sorted(
                adapter_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        # Cache the decision
        self._routing_cache[cache_key] = sorted_adapters
        
        # Log routing decision
        self.logger.info(
            f"Routed query for segment to: {sorted_adapters[:3]} "
            f"(visual_type: {intent.visual_type})"
        )
        
        return sorted_adapters
    
    async def _score_adapters(
        self,
        segment: Dict[str, Any],
        intent: SegmentIntent,
        available_adapters: List[str]
    ) -> Dict[str, float]:
        """
        Score each adapter based on relevance to the content.
        
        Returns:
            Dictionary mapping adapter names to scores
        """
        scores = {}
        
        for adapter_name in available_adapters:
            if adapter_name not in self.adapter_profiles:
                scores[adapter_name] = 0.5  # Default score for unknown adapters
                continue
            
            profile = self.adapter_profiles[adapter_name]
            score = 0.5  # Base score
            
            # Visual type match
            if intent.visual_type in (profile.visual_types or []):
                score += 0.3
            
            # Topic match
            segment_text = segment.get('text', '').lower()
            topic_matches = sum(
                1 for topic in (profile.topics or [])
                if topic in segment_text
            )
            if topic_matches:
                score += min(0.2, topic_matches * 0.05)
            
            # Entity match
            segment_entities = intent.entities or []
            for entity in segment_entities:
                entity_text = entity.get('text', '') if isinstance(entity, dict) else str(entity)
                entity_type = entity.get('type', '') if isinstance(entity, dict) else ''
                
                # Direct entity match
                if entity_text in (profile.entities or []):
                    score += 0.15
                
                # Entity type match
                if entity_type in (profile.entity_types or []):
                    score += 0.1
            
            # Temporal preference
            if profile.temporal_preference:
                if profile.temporal_preference == "recent" and "breaking" in segment_text:
                    score += 0.1
                elif profile.temporal_preference == "historical" and any(
                    word in segment_text for word in ["history", "past", "ago", "decade"]
                ):
                    score += 0.1
            
            # Emotion/reaction specialist
            if profile.emotion_triggers and intent.emotion:
                score += 0.2
            
            # Apply confidence boost
            score += profile.confidence_boost
            
            # Apply quality factor
            score *= profile.quality_score
            
            # Penalize rate-limited adapters if we have alternatives
            if profile.rate_limited and len(available_adapters) > 3:
                score *= 0.8
            
            scores[adapter_name] = min(score, 1.0)
        
        return scores
    
    def _create_cache_key(self, segment: Dict[str, Any], intent: SegmentIntent) -> str:
        """Create a cache key for routing decisions."""
        # Use visual type, first few entities, and first 50 chars of text
        entities_str = "_".join([
            str(e.get('text', '') if isinstance(e, dict) else e)
            for e in (intent.entities or [])[:3]
        ])
        text_preview = segment.get('text', '')[:50].replace(' ', '_')
        
        return f"{intent.visual_type}_{entities_str}_{text_preview}"
    
    async def get_routing_explanation(
        self,
        segment: Dict[str, Any],
        intent: SegmentIntent,
        routing: List[str]
    ) -> str:
        """
        Get a human-readable explanation of the routing decision.
        
        Args:
            segment: The segment
            intent: The segment intent
            routing: The routing decision
            
        Returns:
            Explanation string
        """
        if not routing:
            return "No adapters available for routing"
        
        primary = routing[0]
        profile = self.adapter_profiles.get(primary)
        
        if not profile:
            return f"Routed to {primary} (no profile available)"
        
        reasons = []
        
        if intent.visual_type in (profile.visual_types or []):
            reasons.append(f"supports {intent.visual_type} content")
        
        if profile.reaction_specialist and intent.emotion:
            reasons.append(f"specializes in {intent.emotion} reactions")
        
        if profile.topics:
            matched_topics = [
                t for t in profile.topics
                if t in segment.get('text', '').lower()
            ]
            if matched_topics:
                reasons.append(f"matches topics: {', '.join(matched_topics[:3])}")
        
        if profile.entities:
            matched_entities = [
                e for e in (intent.entities or [])
                if str(e) in profile.entities
            ]
            if matched_entities:
                reasons.append(f"has content for: {matched_entities[0]}")
        
        reason_str = " and ".join(reasons) if reasons else "general purpose adapter"
        
        return f"Primary: {primary} ({reason_str})"
    
    def get_adapter_profile(self, adapter_name: str) -> Optional[AdapterProfile]:
        """Get the profile for a specific adapter."""
        return self.adapter_profiles.get(adapter_name)
    
    def suggest_missing_adapters(
        self,
        segment: Dict[str, Any],
        intent: SegmentIntent
    ) -> List[str]:
        """
        Suggest adapters that would be ideal but aren't available.
        
        Returns:
            List of adapter names that would improve results
        """
        suggestions = []
        
        # Check which ideal adapters are missing
        ideal_adapters = self._get_ideal_adapters(segment, intent)
        available = set(self.registry.get_available_adapters())
        
        for adapter in ideal_adapters:
            if adapter not in available:
                suggestions.append(adapter)
        
        return suggestions
    
    def _get_ideal_adapters(
        self,
        segment: Dict[str, Any],
        intent: SegmentIntent
    ) -> List[str]:
        """Get the ideal adapters for this content type."""
        ideal = []
        
        # Space content -> NASA
        if any(word in segment.get('text', '').lower() 
               for word in ["space", "nasa", "rocket", "planet"]):
            ideal.append("nasa")
        
        # News content -> GDELT
        if any(word in segment.get('text', '').lower()
               for word in ["breaking", "news", "president", "election"]):
            ideal.append("gdelt")
        
        # Reactions -> Tenor
        if intent.visual_type == "reaction" or intent.emotion:
            ideal.append("tenor")
        
        # Professional content -> Pexels
        if "business" in segment.get('text', '').lower():
            ideal.append("pexels")
        
        return ideal
