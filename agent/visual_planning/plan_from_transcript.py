"""
Visual Planning from Audio Transcripts

This module creates visual plans based on actual spoken words from audio transcripts,
replacing the disconnected pre-generated visual concepts approach.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from agent.decision_logger import get_decision_logger

def plan_visuals_from_transcript(transcript_data: Dict[str, Any], story_angle: Dict[str, Any], logger=None) -> List[Dict[str, Any]]:
    """
    Create visual timeline based on actual audio transcript with word-level timestamps.
    
    Args:
        transcript_data: Whisper transcript with word-level timing
        story_angle: Selected story angle from content assessment
        logger: Decision logger instance (optional)
        
    Returns:
        Visual timeline with exact timing based on spoken words
    """
    if logger is None:
        logger = get_decision_logger()
    
    try:
        audio_duration = transcript_data.get('total_duration', 30.0)
        segments = transcript_data.get('segments', [])
        
        logging.info(f"  > Planning visuals for {audio_duration:.1f}s audio with {len(segments)} segments")
        
        visual_timeline = []
        
        # Always start with webpage capture (0-3s)
        visual_timeline.append({
            'start_time': 0.0,
            'end_time': min(3.0, audio_duration * 0.15),  # 15% of audio or 3s, whichever is smaller
            'spoken_text': '[Opening]',
            'visual_concept': 'webpage_capture',
            'visual_type': 'webpage',
            'timing_source': 'transcript_based',
            'search_query': 'webpage capture with intelligent zoom',
            'priority': 'high'
        })
        
        # Extract visual concepts from transcript segments
        for i, segment in enumerate(segments):
            words = segment.get('words', [])
            text = segment.get('text', '').strip()
            
            if not words or not text:
                continue
                
            # Skip very short segments (less than 1 second)
            segment_duration = words[-1].get('end', 0) - words[0].get('start', 0)
            if segment_duration < 1.0:
                continue
            
            # Extract visual concepts from spoken text
            visual_concepts = extract_visual_concepts_from_text(text, story_angle)
            
            if visual_concepts:
                for concept in visual_concepts:
                    visual_timeline.append({
                        'start_time': words[0].get('start', 0),
                        'end_time': words[-1].get('end', 0),
                        'spoken_text': text,
                        'visual_concept': concept['concept'],
                        'visual_type': concept['type'],
                        'timing_source': 'transcript_based',
                        'search_query': concept['search_query'],
                        'priority': concept['priority'],
                        'reasoning': concept['reasoning']
                    })
        
        # If we have very few visuals, create additional ones based on key phrases
        if len(visual_timeline) < 3:
            additional_visuals = create_additional_visuals_from_keywords(transcript_data, story_angle)
            visual_timeline.extend(additional_visuals)
        
        # Sort by start time and merge overlapping segments
        visual_timeline = merge_overlapping_visuals(visual_timeline)
        
        logger.log_decision(
            step="visual_planning_completion",
            decision=f"Created visual timeline with {len(visual_timeline)} segments",
            reasoning=f"Based on transcript analysis of {audio_duration:.1f}s audio",
            confidence=0.9,
            metadata={
                "audio_duration": audio_duration,
                "transcript_segments": len(segments),
                "visual_segments": len(visual_timeline),
                "planning_method": "transcript_based"
            }
        )
        
        logging.info(f"  > Created {len(visual_timeline)} visual segments from transcript analysis")
        return visual_timeline
        
    except Exception as e:
        logging.error(f"  > Visual planning failed: {e}")
        logger.log_decision(
            step="visual_planning_error",
            decision="Visual planning failed - using fallback",
            reasoning=f"Error: {e}",
            confidence=0.1
        )
        
        # Return minimal fallback timeline
        return [{
            'start_time': 0.0,
            'end_time': transcript_data.get('total_duration', 30.0),
            'spoken_text': '[Fallback]',
            'visual_concept': 'webpage_capture',
            'visual_type': 'webpage',
            'timing_source': 'fallback',
            'search_query': 'webpage capture',
            'priority': 'high'
        }]

def extract_visual_concepts_from_text(text: str, story_angle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract visual concepts from spoken text based on keywords and context.
    
    Args:
        text: Spoken text from transcript segment
        story_angle: Story angle context
        
    Returns:
        List of visual concepts with search queries
    """
    concepts = []
    text_lower = text.lower()
    
    # Technology/AI related keywords
    if any(keyword in text_lower for keyword in ['ai', 'artificial intelligence', 'technology', 'algorithm']):
        concepts.append({
            'concept': 'AI technology representation',
            'type': 'concept',
            'search_query': 'artificial intelligence technology digital',
            'priority': 'high',
            'reasoning': 'Text mentions AI/technology concepts'
        })
    
    # People/human related keywords  
    if any(keyword in text_lower for keyword in ['people', 'person', 'human', 'user', 'individual']):
        concepts.append({
            'concept': 'People using technology',
            'type': 'concept', 
            'search_query': 'people using computers technology workplace',
            'priority': 'medium',
            'reasoning': 'Text refers to people/humans'
        })
    
    # Problem/concern related keywords
    if any(keyword in text_lower for keyword in ['problem', 'issue', 'concern', 'worry', 'hesitate', 'skeptical', 'doubt']):
        concepts.append({
            'concept': 'Technology concerns',
            'type': 'concept',
            'search_query': 'person concerned worried technology laptop',
            'priority': 'high',
            'reasoning': 'Text expresses concerns or hesitation'
        })
    
    # Quality/output related keywords
    if any(keyword in text_lower for keyword in ['quality', 'output', 'result', 'accuracy', 'performance']):
        concepts.append({
            'concept': 'Quality assessment',
            'type': 'concept',
            'search_query': 'quality control assessment performance metrics',
            'priority': 'medium',
            'reasoning': 'Text discusses quality or output'
        })
    
    # Privacy/security related keywords
    if any(keyword in text_lower for keyword in ['privacy', 'security', 'data', 'information', 'protection']):
        concepts.append({
            'concept': 'Data privacy security',
            'type': 'concept',
            'search_query': 'data privacy security lock protection digital',
            'priority': 'medium',
            'reasoning': 'Text mentions privacy or security'
        })
    
    # Learning/education related keywords
    if any(keyword in text_lower for keyword in ['learn', 'education', 'student', 'knowledge', 'understand']):
        concepts.append({
            'concept': 'Learning and education',
            'type': 'concept',
            'search_query': 'student learning education knowledge books',
            'priority': 'medium',
            'reasoning': 'Text relates to learning or education'
        })
    
    # Work/business related keywords
    if any(keyword in text_lower for keyword in ['work', 'business', 'office', 'professional', 'company']):
        concepts.append({
            'concept': 'Professional workplace',
            'type': 'concept',
            'search_query': 'professional office workplace business technology',
            'priority': 'medium',
            'reasoning': 'Text mentions work or business context'
        })
    
    # If no specific concepts found, create a generic one based on story angle
    if not concepts:
        angle_name = story_angle.get('story_angle', {}).get('angle_name', '')
        if 'hesitate' in angle_name.lower() or 'reluctant' in angle_name.lower():
            concepts.append({
                'concept': 'Technology hesitation',
                'type': 'concept',
                'search_query': 'person uncertain about technology computer',
                'priority': 'medium',
                'reasoning': 'Generic concept based on story angle'
            })
        else:
            concepts.append({
                'concept': 'Modern technology',
                'type': 'concept',
                'search_query': 'modern technology digital innovation',
                'priority': 'medium',
                'reasoning': 'Generic technology concept'
            })
    
    return concepts

def create_additional_visuals_from_keywords(transcript_data: Dict[str, Any], story_angle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create additional visual segments if transcript-based extraction yields too few visuals.
    """
    additional_visuals = []
    audio_duration = transcript_data.get('total_duration', 30.0)
    
    # Create time-based segments for key story elements
    if audio_duration > 10:
        # Middle segment - main content
        additional_visuals.append({
            'start_time': audio_duration * 0.3,
            'end_time': audio_duration * 0.7,
            'spoken_text': '[Middle content]',
            'visual_concept': 'Technology in daily life',
            'visual_type': 'concept',
            'timing_source': 'time_based',
            'search_query': 'people using technology daily life modern',
            'priority': 'medium',
            'reasoning': 'Time-based middle content segment'
        })
    
    if audio_duration > 15:
        # Conclusion segment
        additional_visuals.append({
            'start_time': audio_duration * 0.8,
            'end_time': audio_duration,
            'spoken_text': '[Conclusion]',
            'visual_concept': 'Future of technology',
            'visual_type': 'concept',
            'timing_source': 'time_based',
            'search_query': 'future technology innovation bright',
            'priority': 'medium',
            'reasoning': 'Time-based conclusion segment'
        })
    
    return additional_visuals

def merge_overlapping_visuals(visual_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge overlapping visual segments and ensure proper spacing.
    """
    if not visual_timeline:
        return []
    
    # Sort by start time
    sorted_timeline = sorted(visual_timeline, key=lambda x: x['start_time'])
    merged = []
    
    for visual in sorted_timeline:
        if not merged:
            merged.append(visual)
            continue
        
        last_visual = merged[-1]
        
        # If segments overlap significantly, merge them
        if visual['start_time'] < last_visual['end_time'] - 0.5:
            # Extend the last visual or create a combined one
            if visual['priority'] == 'high' or last_visual['priority'] != 'high':
                # Replace with higher priority or new visual
                last_visual['end_time'] = visual['end_time']
                last_visual['spoken_text'] += ' ' + visual['spoken_text']
                # Keep higher priority concept
                if visual['priority'] == 'high':
                    last_visual['visual_concept'] = visual['visual_concept']
                    last_visual['search_query'] = visual['search_query']
            # Otherwise keep the existing one
        else:
            merged.append(visual)
    
    return merged