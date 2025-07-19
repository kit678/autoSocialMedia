"""
Enhanced Segment

Segment data structure with support for reaction points and precise timing.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .asset_types import Asset
from .reaction_detector import ReactionPoint, ReactionOverlay


@dataclass
class EnhancedSegment:
    """Segment with reaction points and precise timing."""
    
    # Base segment data
    id: str
    text: str
    start_time: float
    end_time: float
    duration: float
    
    # Visual intent
    visual_type: str = "image"
    intent: str = "inform"
    emotion: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # Primary visual asset
    primary_visual: Optional[Asset] = None
    
    # Reaction points and overlays
    reaction_points: List[ReactionPoint] = field(default_factory=list)
    reaction_overlays: List[ReactionOverlay] = field(default_factory=list)
    
    # Word timings for precise sync
    word_timings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_segment(cls, segment: Dict[str, Any]) -> 'EnhancedSegment':
        """Create an EnhancedSegment from a regular segment dict."""
        return cls(
            id=segment.get('id', ''),
            text=segment.get('text', ''),
            start_time=segment.get('start_time', 0.0),
            end_time=segment.get('end_time', 0.0),
            duration=segment.get('duration', segment.get('end_time', 0.0) - segment.get('start_time', 0.0)),
            visual_type=segment.get('visual_type', 'image'),
            intent=segment.get('intent', 'inform'),
            emotion=segment.get('emotion'),
            entities=segment.get('entities', []),
            topics=segment.get('topics', []),
            word_timings=segment.get('word_timings', []),
            metadata=segment.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format."""
        return {
            'id': self.id,
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'visual_type': self.visual_type,
            'intent': self.intent,
            'emotion': self.emotion,
            'entities': self.entities,
            'topics': self.topics,
            'primary_visual': self.primary_visual.to_dict() if self.primary_visual else None,
            'reaction_points': [self._reaction_point_to_dict(rp) for rp in self.reaction_points],
            'reaction_overlays': [self._reaction_overlay_to_dict(ro) for ro in self.reaction_overlays],
            'word_timings': self.word_timings,
            'metadata': self.metadata
        }
    
    def add_reaction_overlay(self, point: ReactionPoint, asset: Asset) -> ReactionOverlay:
        """Add a reaction overlay at specific timestamp."""
        overlay = ReactionOverlay(
            asset=asset,
            reaction_point=point,
            start_time=self._calculate_overlay_start_time(point),
            end_time=self._calculate_overlay_end_time(point),
            position=point.position,
            scale=point.scale,
            fade_in=0.2,
            fade_out=0.2
        )
        self.reaction_overlays.append(overlay)
        return overlay
    
    def _calculate_overlay_start_time(self, point: ReactionPoint) -> float:
        """Calculate precise start time for overlay."""
        if point.start_time > 0:
            return point.start_time
        
        # Calculate based on word index
        if self.word_timings and point.start_word_index < len(self.word_timings):
            return self.word_timings[point.start_word_index].get('start', self.start_time)
        
        # Fallback: estimate based on position in text
        word_position_ratio = point.start_word_index / max(len(self.text.split()), 1)
        return self.start_time + (self.duration * word_position_ratio)
    
    def _calculate_overlay_end_time(self, point: ReactionPoint) -> float:
        """Calculate precise end time for overlay."""
        start_time = self._calculate_overlay_start_time(point)
        return start_time + point.duration
    
    def get_active_overlays_at_time(self, timestamp: float) -> List[ReactionOverlay]:
        """Get all overlays active at a specific timestamp."""
        active = []
        for overlay in self.reaction_overlays:
            if overlay.start_time <= timestamp <= overlay.end_time:
                active.append(overlay)
        return active
    
    def has_reactions(self) -> bool:
        """Check if this segment has any reaction overlays."""
        return len(self.reaction_overlays) > 0
    
    def get_reaction_timeline(self) -> List[Tuple[float, float, str]]:
        """Get timeline of reactions as (start, end, emotion) tuples."""
        timeline = []
        for overlay in self.reaction_overlays:
            timeline.append((
                overlay.start_time,
                overlay.end_time,
                overlay.reaction_point.emotion
            ))
        return sorted(timeline, key=lambda x: x[0])
    
    def _reaction_point_to_dict(self, point: ReactionPoint) -> Dict[str, Any]:
        """Convert ReactionPoint to dict."""
        return {
            'phrase': point.phrase,
            'start_word_index': point.start_word_index,
            'end_word_index': point.end_word_index,
            'start_time': point.start_time,
            'end_time': point.end_time,
            'emotion': point.emotion,
            'intensity': point.intensity,
            'duration': point.duration,
            'position': point.position,
            'scale': point.scale,
            'reasoning': point.reasoning
        }
    
    def _reaction_overlay_to_dict(self, overlay: ReactionOverlay) -> Dict[str, Any]:
        """Convert ReactionOverlay to dict."""
        return {
            'asset_id': overlay.asset.id if overlay.asset else None,
            'start_time': overlay.start_time,
            'end_time': overlay.end_time,
            'position': overlay.position,
            'scale': overlay.scale,
            'fade_in': overlay.fade_in,
            'fade_out': overlay.fade_out,
            'z_index': overlay.z_index,
            'emotion': overlay.reaction_point.emotion
        }


class SegmentProcessor:
    """Process segments with reaction detection and asset assignment."""
    
    def __init__(self, reaction_detector, asset_registry):
        """
        Initialize the segment processor.
        
        Args:
            reaction_detector: ReactionPointDetector instance
            asset_registry: AssetAdapterRegistry instance
        """
        self.reaction_detector = reaction_detector
        self.asset_registry = asset_registry
    
    async def process_segment(
        self,
        segment: Dict[str, Any],
        primary_asset: Optional[Asset] = None
    ) -> EnhancedSegment:
        """
        Process a segment to create an enhanced version with reactions.
        
        Args:
            segment: Base segment dictionary
            primary_asset: Optional primary visual asset
            
        Returns:
            EnhancedSegment with reaction points and overlays
        """
        # Create enhanced segment
        enhanced = EnhancedSegment.from_segment(segment)
        
        # Set primary visual if provided
        if primary_asset:
            enhanced.primary_visual = primary_asset
        
        # Detect reaction points
        reaction_points = await self.reaction_detector.detect_reaction_points(segment)
        enhanced.reaction_points = reaction_points
        
        # Search for reaction assets for each point
        for point in reaction_points:
            search_terms = self.reaction_detector.get_search_terms_for_emotion(point.emotion)
            
            # Search specifically for reaction GIFs
            reaction_segment = {
                **segment,
                'visual_type': 'reaction',
                'emotion': point.emotion
            }
            
            # Use Tenor adapter if available, fallback to others
            results = await self.asset_registry.search_all_adapters(
                search_terms[0],  # Use primary search term
                reaction_segment,
                asset_type='reaction',
                limit_per_adapter=5
            )
            
            # Select best reaction asset
            if results and results[0].assets:
                best_reaction = results[0].assets[0]
                enhanced.add_reaction_overlay(point, best_reaction)
        
        return enhanced
