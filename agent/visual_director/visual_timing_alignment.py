"""
Visual Timing Alignment Module

Aligns visual segments with transcript word timings to ensure proper synchronization.
Also handles opening webpage video and closing logo shots.
"""

import logging
from typing import List, Dict, Any, Optional


def align_segments_with_transcript(
    segments: List[Dict[str, Any]], 
    transcript_data: Dict[str, Any],
    opening_duration: float = 3.0,  # Duration of opening webpage video
    closing_duration: float = 3.0   # Duration of closing logo
) -> List[Dict[str, Any]]:
    """
    Aligns visual segments with actual word timings from transcript.
    
    Args:
        segments: List of visual segments from visual story plan
        transcript_data: Transcript data with word timings
        opening_duration: Duration of opening webpage video in seconds
        closing_duration: Duration of closing logo in seconds
        
    Returns:
        Updated segments with proper start/end times
    """
    if not segments:
        return segments
        
    # Get total audio duration
    total_duration = transcript_data.get('total_duration', 30.0)
    
    # Account for opening and closing shots
    content_start_time = opening_duration
    content_end_time = total_duration - closing_duration
    content_duration = content_end_time - content_start_time
    
    if content_duration <= 0:
        logging.warning(f"Content duration too short ({content_duration}s) after accounting for opening/closing shots")
        content_duration = total_duration * 0.8  # Use 80% of total duration as fallback
        content_start_time = total_duration * 0.1
        content_end_time = total_duration * 0.9
    
    # Get all words in chronological order
    all_words = transcript_data.get('all_words', [])
    
    # Process each segment
    aligned_segments = []
    segment_duration = content_duration / len(segments)
    
    for i, segment in enumerate(segments):
        # Try to find the primary search term in the transcript
        primary_term = segment.get('primary_search_term', '').lower()
        
        # Calculate default timing based on equal distribution
        default_start = content_start_time + (i * segment_duration)
        default_end = content_start_time + ((i + 1) * segment_duration)
        
        # Try to find actual timing from transcript
        actual_start = None
        actual_end = None
        
        if primary_term and all_words:
            # Search for the term in the chronological word list
            term_words = primary_term.split()
            
            for j, word_info in enumerate(all_words):
                # Check if this word matches the start of our search term
                if word_info['word'] == term_words[0]:
                    # Check if this could be the start of our phrase
                    if len(term_words) == 1:
                        # Single word match
                        actual_start = word_info['start_time']
                        # Find a reasonable end time (next segment start or a pause)
                        if j + 1 < len(all_words):
                            # Look ahead for a good breaking point
                            for k in range(j + 1, min(j + 10, len(all_words))):
                                next_word = all_words[k]
                                # If there's a significant pause, use it as end time
                                if next_word['start_time'] - word_info['end_time'] > 0.5:
                                    actual_end = next_word['start_time']
                                    break
                            if not actual_end:
                                actual_end = all_words[min(j + 5, len(all_words) - 1)]['end_time']
                        break
                    else:
                        # Multi-word phrase - check if the full phrase matches
                        match = True
                        for k, term_word in enumerate(term_words[1:], 1):
                            if j + k >= len(all_words) or all_words[j + k]['word'] != term_word:
                                match = False
                                break
                        if match:
                            actual_start = word_info['start_time']
                            if j + len(term_words) < len(all_words):
                                actual_end = all_words[j + len(term_words) - 1]['end_time']
                            break
        
        # Use actual timing if found, otherwise use default
        if actual_start is not None:
            segment['start_time'] = actual_start
            segment['end_time'] = actual_end if actual_end is not None else min(actual_start + segment_duration, content_end_time)
            logging.info(f"Segment {i}: '{primary_term}' aligned to {segment['start_time']:.2f}-{segment['end_time']:.2f}s")
        else:
            segment['start_time'] = default_start
            segment['end_time'] = default_end
            logging.info(f"Segment {i}: '{primary_term}' using default timing {segment['start_time']:.2f}-{segment['end_time']:.2f}s")
        
        aligned_segments.append(segment)
    
    return aligned_segments


def ensure_segment_coverage(
    segments: List[Dict[str, Any]], 
    total_duration: float,
    opening_duration: float = 3.0,
    closing_duration: float = 3.0
) -> None:
    """
    Ensures visual segments properly cover the content duration without gaps.
    
    Args:
        segments: List of aligned segments
        total_duration: Total audio duration
        opening_duration: Duration of opening webpage video
        closing_duration: Duration of closing logo
    """
    if not segments:
        return
    
    content_start = opening_duration
    content_end = total_duration - closing_duration
    
    # Ensure first segment doesn't start before content area
    if segments[0]['start_time'] < content_start:
        segments[0]['start_time'] = content_start
    
    # Ensure last segment doesn't extend past content area
    if segments[-1]['end_time'] > content_end:
        segments[-1]['end_time'] = content_end
    
    # Fill any gaps between segments
    for i in range(len(segments) - 1):
        current_end = segments[i]['end_time']
        next_start = segments[i + 1]['start_time']
        
        if current_end < next_start - 0.1:  # Gap detected
            # Extend current segment to meet next segment
            gap_duration = next_start - current_end
            segments[i]['end_time'] = current_end + (gap_duration * 0.5)
            segments[i + 1]['start_time'] = segments[i]['end_time']
        elif current_end > next_start + 0.1:  # Overlap detected
            # Adjust to remove overlap
            segments[i]['end_time'] = next_start


def add_static_shots(
    visual_timeline: List[Dict[str, Any]],
    visual_map: Dict[str, str],
    opening_video_path: str,
    closing_logo_path: str,
    total_duration: float,
    opening_duration: float = 3.0,
    closing_duration: float = 3.0
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Adds opening webpage video and closing logo to the visual timeline.
    
    Args:
        visual_timeline: Existing visual timeline
        visual_map: Mapping of visual IDs to file paths
        opening_video_path: Path to webpage capture video
        closing_logo_path: Path to company logo
        total_duration: Total audio duration
        opening_duration: Duration of opening shot
        closing_duration: Duration of closing shot
        
    Returns:
        Updated (visual_timeline, visual_map) with static shots added
    """
    # Add opening webpage video
    opening_segment = {
        'cue_id': 'opening_webpage',
        'start_time': 0.0,
        'end_time': opening_duration,
        'visual_type': 'video',
        'trigger_keyword': 'opening_shot',
        'visual_file': opening_video_path,
        'segment_metadata': {
            'intent': 'establish',
            'emotion': 'neutral',
            'preferred_media': 'video'
        }
    }
    
    # Add closing logo
    closing_segment = {
        'cue_id': 'closing_logo',
        'start_time': total_duration - closing_duration,
        'end_time': total_duration,
        'visual_type': 'image',
        'trigger_keyword': 'closing_shot',
        'visual_file': closing_logo_path,
        'segment_metadata': {
            'intent': 'brand',
            'emotion': 'professional',
            'preferred_media': 'image'
        }
    }
    
    # Update visual map
    updated_map = visual_map.copy()
    updated_map['opening_webpage'] = opening_video_path
    updated_map['closing_logo'] = closing_logo_path
    
    # Create new timeline with static shots
    new_timeline = [opening_segment] + visual_timeline + [closing_segment]
    
    return new_timeline, updated_map
