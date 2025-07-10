"""
Word-Level Timestamp Extractor
Extracts precise timing for when each word is spoken in audio
"""

import os
import logging
import sys
from typing import Dict, List, Any, Optional

def install_whisper_timestamped() -> bool:
    """Install whisper-timestamped if not available"""
    try:
        import whisper_timestamped
        return True
    except ImportError:
        logging.info("Installing whisper-timestamped...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "whisper-timestamped>=1.14.0"])
            import whisper_timestamped
            return True
        except Exception as e:
            logging.error(f"Failed to install whisper-timestamped: {e}")
            return False

def extract_word_timestamps(audio_path: str, script_text: str = None) -> Dict[str, Any]:
    """
    Extract word-level timestamps from audio file
    
    Args:
        audio_path: Path to audio file
        script_text: Optional script text for alignment validation
        
    Returns:
        Dict containing word timing data
    """
    if not install_whisper_timestamped():
        raise Exception("Could not install whisper-timestamped dependency")
    
    try:
        import whisper_timestamped as whisper
        
        logging.info("Loading Whisper model for word-level transcription...")
        model = whisper.load_model("small", device="cpu")  # Use small model that user already has
        
        logging.info("Extracting word-level timestamps...")
        audio = whisper.load_audio(audio_path)
        result = whisper.transcribe(
            model, 
            audio, 
            language="en",
            verbose=False
        )
        
        # Process segments to extract word timing
        word_timings = {}
        all_words = []
        max_end_time = 0.0
        
        for segment in result.get("segments", []):
            # Track the latest end time to calculate total duration
            segment_end = segment.get("end", 0.0)
            if segment_end > max_end_time:
                max_end_time = segment_end
                
            for word_data in segment.get("words", []):
                word = word_data.get("text", "").strip().lower()
                start_time = word_data.get("start", 0.0)
                end_time = word_data.get("end", 0.0)
                confidence = word_data.get("confidence", 0.0)
                
                # Track the latest word end time as well
                if end_time > max_end_time:
                    max_end_time = end_time
                
                # Clean word (remove punctuation)
                import re
                clean_word = re.sub(r'[^\w\s]', '', word).strip()
                
                if clean_word and len(clean_word) > 1:  # Skip very short words
                    word_timings[clean_word] = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "confidence": confidence,
                        "original_text": word_data.get("text", "")
                    }
                    
                    all_words.append({
                        "word": clean_word,
                        "start_time": start_time,
                        "end_time": end_time,
                        "confidence": confidence
                    })
        
        # Calculate total duration from actual content
        calculated_duration = max_end_time
        
        logging.info(f"Extracted timestamps for {len(word_timings)} unique words")
        logging.info(f"Calculated audio duration: {calculated_duration:.1f}s")
        
        return {
            "word_timings": word_timings,
            "all_words": all_words,
            "segments": result.get("segments", []),
            "total_duration": calculated_duration,
            "text": result.get("text", "")
        }
        
    except Exception as e:
        logging.error(f"Error extracting word timestamps: {e}")
        raise

def find_keyword_timing(keyword: str, word_timings: Dict[str, Any]) -> Optional[float]:
    """
    Find the timestamp when a keyword is spoken
    
    Args:
        keyword: The word/phrase to find
        word_timings: Word timing data from extract_word_timestamps
        
    Returns:
        Start time in seconds when keyword is spoken, or None if not found
    """
    word_data = word_timings.get("word_timings", {})
    
    # Try exact match first
    clean_keyword = keyword.lower().strip()
    import re
    clean_keyword = re.sub(r'[^\w\s]', '', clean_keyword).strip()
    
    # For single words
    if " " not in clean_keyword:
        if clean_keyword in word_data:
            return word_data[clean_keyword]["start_time"]
    
    # For phrases, find the first word
    else:
        first_word = clean_keyword.split()[0]
        if first_word in word_data:
            return word_data[first_word]["start_time"]
    
    # Fuzzy matching for partial matches
    for word, timing in word_data.items():
        if clean_keyword in word or word in clean_keyword:
            return timing["start_time"]
    
    logging.warning(f"Could not find timing for keyword: {keyword}")
    return None

def create_keyword_timeline(trigger_keywords: List[str], word_timings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create timeline mapping keywords to their actual spoken times
    
    Args:
        trigger_keywords: List of keywords that trigger visuals
        word_timings: Word timing data from extract_word_timestamps
        
    Returns:
        List of timeline entries with actual timestamps
    """
    timeline = []
    found_keywords = []
    missing_keywords = []
    
    for keyword in trigger_keywords:
        timestamp = find_keyword_timing(keyword, word_timings)
        
        if timestamp is not None:
            timeline.append({
                "keyword": keyword,
                "start_time": timestamp,
                "found": True
            })
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # CRASH if any keywords are missing - no fallback timing
    if missing_keywords:
        error_msg = f"CRITICAL: Keywords not found in audio transcription: {missing_keywords}"
        logging.error(error_msg)
        logging.error("Available words in transcription:")
        word_data = word_timings.get("word_timings", {})
        available_words = list(word_data.keys())[:20]  # Show first 20 words
        logging.error(f"  {available_words}")
        logging.error("Possible solutions:")
        logging.error("- Check if keywords actually appear in the script")
        logging.error("- Verify audio quality and whisper transcription accuracy")
        logging.error("- Refine trigger keywords to match spoken words exactly")
        raise Exception(error_msg + " Pipeline configured to crash instead of using fallback timing.")
    
    # Sort by start time
    timeline.sort(key=lambda x: x["start_time"])
    
    logging.info(f"✅ Successfully mapped ALL {len(found_keywords)} keywords to actual spoken times")
    
    return timeline

def get_word_timing_summary(word_timings: Dict[str, Any]) -> str:
    """Generate a summary of available word timings for debugging"""
    word_data = word_timings.get("word_timings", {})
    all_words = word_timings.get("all_words", [])
    
    summary = f"Total words with timestamps: {len(word_data)}\n"
    summary += f"Total audio duration: {word_timings.get('total_duration', 0):.1f}s\n"
    summary += f"Transcribed text: {word_timings.get('text', '')[:100]}...\n"
    
    # Show first 10 words with timing
    summary += "\nFirst 10 words with timestamps:\n"
    for i, word_info in enumerate(all_words[:10]):
        summary += f"  {word_info['word']}: {word_info['start_time']:.1f}s\n"
    
    return summary 