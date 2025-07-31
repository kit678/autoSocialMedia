"""
Media Duration Utilities

This module provides utilities for detecting actual durations of visual assets
and handling duration mismatches in the visual timeline.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from agent.utils import run_command


def get_media_duration(file_path: str) -> Optional[float]:
    """
    Get the actual duration of a media file (video or image).
    
    Args:
        file_path: Path to the media file
        
    Returns:
        Duration in seconds, or None if it's an image or if detection fails
    """
    if not os.path.exists(file_path):
        logging.warning(f"Media file not found: {file_path}")
        return None
    
    # Check if it's a video file
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in video_extensions:
        # It's an image, return None (images are static)
        return None
    
    try:
        # Use ffprobe to get video duration
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', file_path
        ]
        
        success, output, error = run_command(cmd)
        
        if success and output.strip():
            duration = float(output.strip())
            logging.debug(f"Detected duration for {os.path.basename(file_path)}: {duration:.2f}s")
            return duration
        else:
            logging.warning(f"Could not detect duration for {file_path}: {error}")
            return None
            
    except Exception as e:
        logging.error(f"Error detecting duration for {file_path}: {e}")
        return None


def analyze_visual_timeline_durations(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze the visual timeline and identify duration mismatches.
    
    Args:
        timeline: List of visual timeline segments
        
    Returns:
        List of segments with duration analysis added
    """
    analyzed_timeline = []
    
    for segment in timeline:
        analyzed_segment = segment.copy()
        
        # Check if segment has timing information
        if 'start_time' not in segment or 'end_time' not in segment:
            # Skip analysis for segments without timing info (e.g., during planning phase)
            logging.debug(f"Skipping duration analysis for segment without timing: {segment.get('segment_number', 'unknown')}")
            analyzed_timeline.append(analyzed_segment)
            continue
        
        # Get the expected slot duration
        slot_duration = segment['end_time'] - segment['start_time']
        
        # Get the actual media duration
        visual_file = segment.get('visual_file')
        if visual_file:
            actual_duration = get_media_duration(visual_file)
            
            # Add duration analysis
            analyzed_segment['duration_analysis'] = {
                'slot_duration': slot_duration,
                'actual_duration': actual_duration,
                'is_video': actual_duration is not None,
                'has_mismatch': False,
                'mismatch_type': None,
                'gap_duration': 0.0
            }
            
            if actual_duration is not None:
                # It's a video, check for mismatch
                if actual_duration < slot_duration:
                    analyzed_segment['duration_analysis']['has_mismatch'] = True
                    analyzed_segment['duration_analysis']['mismatch_type'] = 'too_short'
                    analyzed_segment['duration_analysis']['gap_duration'] = slot_duration - actual_duration
                    
                    logging.warning(
                        f"Duration mismatch in {segment['cue_id']}: "
                        f"video is {actual_duration:.2f}s but slot is {slot_duration:.2f}s "
                        f"(gap: {slot_duration - actual_duration:.2f}s)"
                    )
                    
                elif actual_duration > slot_duration:
                    analyzed_segment['duration_analysis']['has_mismatch'] = True
                    analyzed_segment['duration_analysis']['mismatch_type'] = 'too_long'
                    analyzed_segment['duration_analysis']['gap_duration'] = actual_duration - slot_duration
                    
                    logging.info(
                        f"Video {segment['cue_id']} is longer than slot, will be trimmed: "
                        f"video is {actual_duration:.2f}s but slot is {slot_duration:.2f}s"
                    )
            else:
                # It's an image, no mismatch possible
                logging.debug(f"Segment {segment['cue_id']} uses static image, no duration issues")
        
        analyzed_timeline.append(analyzed_segment)
    
    return analyzed_timeline


def find_duration_gaps(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find segments that have duration gaps that need to be filled.
    
    Args:
        timeline: Analyzed visual timeline
        
    Returns:
        List of segments that need additional visuals
    """
    segments_needing_fill = []
    
    for segment in timeline:
        duration_analysis = segment.get('duration_analysis', {})
        
        if (duration_analysis.get('has_mismatch') and 
            duration_analysis.get('mismatch_type') == 'too_short' and
            duration_analysis.get('gap_duration', 0) > 0.5):  # Only if gap is > 0.5 seconds
            
            segments_needing_fill.append({
                'segment': segment,
                'gap_duration': duration_analysis['gap_duration'],
                'fill_start_time': segment['start_time'] + duration_analysis['actual_duration'],
                'fill_end_time': segment['end_time']
            })
    
    return segments_needing_fill


def create_gap_fill_segments(gap_info: Dict[str, Any], segment_index: int) -> List[Dict[str, Any]]:
    """
    Create additional visual segments to fill duration gaps.
    
    Args:
        gap_info: Information about the gap to fill
        segment_index: Index of the original segment
        
    Returns:
        List of new segments to fill the gap
    """
    original_segment = gap_info['segment']
    gap_duration = gap_info['gap_duration']
    
    # Create a complementary segment with related search terms
    trigger_keyword = original_segment.get('trigger_keyword', '')
    visual_type = original_segment.get('visual_type', 'Abstract Concept')
    
    # Generate complementary search terms based on the original
    complementary_terms = generate_complementary_search_terms(
        trigger_keyword, 
        visual_type,
        original_segment.get('segment_metadata', {})
    )
    
    fill_segments = []
    
    # If gap is large (>3 seconds), split it into multiple segments
    if gap_duration > 3.0:
        num_segments = min(3, int(gap_duration / 2))  # Max 3 segments, ~2s each
        segment_duration = gap_duration / num_segments
        
        for i in range(num_segments):
            fill_segment = {
                'cue_id': f"{original_segment['cue_id']}_fill_{i+1}",
                'start_time': gap_info['fill_start_time'] + (i * segment_duration),
                'end_time': gap_info['fill_start_time'] + ((i + 1) * segment_duration),
                'trigger_keyword': complementary_terms[i % len(complementary_terms)],
                'visual_type': visual_type,
                'visual_file': None,  # To be filled by visual director
                'is_gap_fill': True,
                'original_segment_id': original_segment['cue_id'],
                'segment_metadata': original_segment.get('segment_metadata', {}).copy()
            }
            fill_segments.append(fill_segment)
    else:
        # Single fill segment for smaller gaps
        fill_segment = {
            'cue_id': f"{original_segment['cue_id']}_fill",
            'start_time': gap_info['fill_start_time'],
            'end_time': gap_info['fill_end_time'],
            'trigger_keyword': complementary_terms[0],
            'visual_type': visual_type,
            'visual_file': None,  # To be filled by visual director
            'is_gap_fill': True,
            'original_segment_id': original_segment['cue_id'],
            'segment_metadata': original_segment.get('segment_metadata', {}).copy()
        }
        fill_segments.append(fill_segment)
    
    return fill_segments


def generate_complementary_search_terms(original_term: str, visual_type: str, metadata: Dict[str, Any]) -> List[str]:
    """
    Generate complementary search terms for gap-filling visuals.
    
    Args:
        original_term: The original search term
        visual_type: Type of visual (Abstract Concept, Proper Noun, etc.)
        metadata: Segment metadata for context
        
    Returns:
        List of complementary search terms
    """
    terms = []
    
    # Get entities and emotion for context
    entities = metadata.get('entities', [])
    emotion = metadata.get('emotion', 'neutral')
    intent = metadata.get('intent', 'inform')
    
    # Generate terms based on visual type and context
    if visual_type == "Abstract Concept":
        if 'AI' in entities:
            terms.extend([
                "artificial intelligence technology",
                "machine learning visualization",
                "AI innovation",
                "technology future"
            ])
        elif 'work' in original_term.lower() or 'job' in original_term.lower():
            terms.extend([
                "workplace technology",
                "professional environment",
                "office innovation",
                "career development"
            ])
        else:
            terms.extend([
                "technology concept",
                "digital innovation",
                "modern technology",
                "tech visualization"
            ])
    
    elif visual_type == "Proper Noun":
        # For proper nouns, use related company/product terms
        if 'ChatGPT' in original_term:
            terms.extend([
                "OpenAI technology",
                "AI chatbot interface",
                "conversational AI",
                "language model"
            ])
        else:
            terms.extend([
                "technology company",
                "tech innovation",
                "software interface",
                "digital platform"
            ])
    
    elif visual_type == "Concrete Object/Action":
        if emotion == 'concerned' or intent == 'warn':
            terms.extend([
                "worried professionals",
                "workplace concern",
                "professional anxiety",
                "job uncertainty"
            ])
        elif emotion == 'happy' or intent == 'celebrate':
            terms.extend([
                "successful collaboration",
                "workplace success",
                "professional achievement",
                "team productivity"
            ])
        else:
            terms.extend([
                "professional work",
                "workplace activity",
                "business process",
                "office environment"
            ])
    
    # Fallback terms
    if not terms:
        terms = [
            "technology",
            "innovation",
            "digital transformation",
            "modern workplace"
        ]
    
    return terms[:4]  # Return up to 4 terms


def log_duration_analysis(timeline: List[Dict[str, Any]], logger) -> None:
    """
    Log detailed duration analysis for debugging.
    
    Args:
        timeline: Analyzed timeline
        logger: Decision logger instance
    """
    mismatches = []
    total_gaps = 0.0
    
    for segment in timeline:
        duration_analysis = segment.get('duration_analysis', {})
        if duration_analysis.get('has_mismatch'):
            mismatch_info = {
                'cue_id': segment['cue_id'],
                'mismatch_type': duration_analysis['mismatch_type'],
                'slot_duration': duration_analysis['slot_duration'],
                'actual_duration': duration_analysis['actual_duration'],
                'gap_duration': duration_analysis['gap_duration']
            }
            mismatches.append(mismatch_info)
            
            if duration_analysis['mismatch_type'] == 'too_short':
                total_gaps += duration_analysis['gap_duration']
    
    if mismatches:
        logger.log_decision(
            step="duration_analysis_complete",
            decision=f"Found {len(mismatches)} duration mismatches",
            reasoning=f"Videos shorter than slots need gap filling, total gap time: {total_gaps:.2f}s",
            confidence=1.0,
            metadata={
                "mismatches": mismatches,
                "total_gap_duration": total_gaps,
                "segments_analyzed": len(timeline)
            }
        )
        
        for mismatch in mismatches:
            logging.warning(
                f"Duration mismatch: {mismatch['cue_id']} - "
                f"{mismatch['mismatch_type']} by {mismatch['gap_duration']:.2f}s"
            )
    else:
        logger.log_decision(
            step="duration_analysis_complete",
            decision="No duration mismatches found",
            reasoning="All video durations match their allocated time slots",
            confidence=1.0,
            metadata={"segments_analyzed": len(timeline)}
        )
