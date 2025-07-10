"""
Word-Level Caption System - Modern TikTok/YouTube Shorts style captions
"""

import os
import logging
import subprocess
import sys
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from agent.utils import run_command

def _sanitize_path_for_ffmpeg_filter(path: str) -> str:
    """
    Prepares a path for use in an FFmpeg filter graph on Windows.
    - Converts backslashes to forward slashes 
    - Properly escapes the colon and backslashes for FFmpeg filter syntax
    """
    if sys.platform == "win32":
        # Convert to forward slashes first
        path = path.replace('\\', '/')
        # Escape the colon after the drive letter for FFmpeg filter syntax
        if ':' in path and len(path) > 2 and path[1] == ':':
            # For Windows paths like C:/path, escape as C\\:/path  
            path = path.replace(':', '\\:')
    return path

def add_word_captions(video_path: str, transcript_data: dict, output_path: str) -> str:
    """
    Add modern word-by-word captions to video using pre-existing transcript data.
    
    Args:
        video_path: Input video file
        transcript_data: Word-level timestamp data from timing_extraction
        output_path: Output video with captions
        
    Returns:
        Path to captioned video
    """
    try:
        logging.info("=== ADDING MODERN WORD-LEVEL CAPTIONS (from pre-existing data) ===")
        
        if not transcript_data or not transcript_data.get("segments"):
            raise Exception("No valid transcript data provided")
        
        # Create ASS subtitle file from the provided transcript data
        ass_path = video_path.replace('.mp4', '_words.ass')
        _create_word_ass_file(transcript_data, ass_path)
        
        # Use FFmpeg to burn captions
        logging.info("Burning modern word-level captions with FFmpeg...")
        
        # Use absolute paths for reliability, especially on Windows
        abs_video_path = os.path.abspath(video_path)
        abs_output_path = os.path.abspath(output_path)
        abs_ass_path = os.path.abspath(ass_path)

        # Check if ASS file was created successfully
        if not os.path.exists(abs_ass_path):
            raise Exception(f"ASS file was not created: {abs_ass_path}")
        
        # Get ASS file size to ensure it's not empty
        ass_size = os.path.getsize(abs_ass_path)
        if ass_size == 0:
            raise Exception(f"ASS file is empty: {abs_ass_path}")
        
        logging.info(f"ASS file validated: {ass_size} bytes")
        
        # ULTIMATE WINDOWS FIX: Use drawtext filter which avoids path issues
        logging.info("Using drawtext-based captions (Windows compatible)...")
        
        # Extract text data from ASS file for drawtext
        caption_data = _extract_captions_from_ass(abs_ass_path)
        
        # Use drawtext approach
        success = _burn_captions_with_drawtext(abs_video_path, caption_data, abs_output_path)
        
        # Clean up the temporary .ass file
        if os.path.exists(abs_ass_path):
            try:
                os.remove(abs_ass_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary file {abs_ass_path}: {e}")

        if not success:
            raise Exception("Drawtext caption burning failed")
        
        logging.info("Modern word-level captions successfully added!")
        return output_path
            
    except Exception as e:
        error_msg = f"Caption creation failed: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

def _create_word_ass_file(result: dict, ass_path: str) -> None:
    """Create ASS subtitle file with modern, karaoke-style highlighting."""
    
    # Define ASS styles for karaoke effect
    # Default: White text, transparent outline/shadow for clean look
    # Highlight: White text with a yellow outline
    ass_content = """[Script Info]
Title: Modern Karaoke Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,20,20,50,1
Style: Highlight,Arial,38,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,20,20,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Process words from transcript data
    for segment in result["segments"]:
        if 'words' not in segment:
            continue
            
        # Group words into lines for better display
        line_words = []
        line_start_time = None
        max_words_per_line = 5
        
        for word_data in segment['words']:
            if line_start_time is None:
                line_start_time = word_data['start']

            line_words.append(word_data)
            
            # Flush line when it's full or at the end of a segment
            if len(line_words) >= max_words_per_line:
                ass_content += _write_ass_line(line_words, line_start_time)
                line_words = []
                line_start_time = None
        
        # Write any remaining words in the last line
        if line_words:
            ass_content += _write_ass_line(line_words, line_start_time)

    # Write ASS file
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    logging.info(f"Created ASS file with modern captions: {ass_path}")

def _write_ass_line(words: list, start_time: float) -> str:
    """Writes a single line of karaoke-style ASS subtitles."""
    
    line_text_parts = []
    full_line = " ".join([w['text'].strip().upper() for w in words])
    end_time = words[-1]['end']
    
    # Calculate per-word karaoke timing
    for i, word_data in enumerate(words):
        duration_ms = int((word_data['end'] - word_data['start']) * 100)
        
        # Build the karaoke-timed text
        line_text_parts.append(f"{{\\k{duration_ms}}}{word_data['text'].strip().upper()}")
    
    karaoke_line = " ".join(line_text_parts)
    
    # Create a dialogue line for the whole phrase
    start_ass = _seconds_to_ass_time(start_time)
    end_ass = _seconds_to_ass_time(end_time)
    
    return f"Dialogue: 0,{start_ass},{end_ass},Highlight,,0,0,0,,{karaoke_line}\n"

def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def _convert_ass_to_srt(ass_path: str, srt_path: str) -> None:
    """Convert ASS file to SRT format for better Windows compatibility."""
    try:
        with open(ass_path, 'r', encoding='utf-8') as f:
            ass_content = f.read()
        
        # Extract dialogue lines
        dialogue_lines = []
        for line in ass_content.split('\n'):
            if line.startswith('Dialogue:'):
                dialogue_lines.append(line)
        
        # Convert to SRT format
        srt_content = ""
        for i, dialogue in enumerate(dialogue_lines):
            if not dialogue.strip():
                continue
                
            parts = dialogue.split(',', 9)
            if len(parts) < 10:
                continue
                
            start_time = parts[1]
            end_time = parts[2]
            text = parts[9]
            
            # Clean text - remove karaoke timing and formatting
            import re
            text = re.sub(r'\{\\k\d+\}', '', text)  # Remove karaoke timing
            text = re.sub(r'\{[^}]*\}', '', text)   # Remove other formatting
            text = text.strip()
            
            if not text:
                continue
            
            # Convert ASS time to SRT time
            srt_start = _ass_time_to_srt_time(start_time)
            srt_end = _ass_time_to_srt_time(end_time)
            
            srt_content += f"{i + 1}\n"
            srt_content += f"{srt_start} --> {srt_end}\n"
            srt_content += f"{text}\n\n"
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
            
        logging.info(f"Converted ASS to SRT: {len(dialogue_lines)} captions")
        
    except Exception as e:
        logging.error(f"Error converting ASS to SRT: {e}")
        raise

def _ass_time_to_srt_time(ass_time: str) -> str:
    """Convert ASS time format to SRT time format."""
    # ASS: H:MM:SS.CC -> SRT: HH:MM:SS,mmm
    parts = ass_time.split(':')
    if len(parts) != 3:
        return "00:00:00,000"
    
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split('.')
    seconds = int(sec_parts[0])
    centiseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    milliseconds = centiseconds * 10
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def _extract_captions_from_ass(ass_path: str) -> list:
    """Extract caption data from ASS file for use with MoviePy."""
    captions = []
    
    try:
        with open(ass_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.split('\n'):
            if line.startswith('Dialogue:'):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    start_time = parts[1]
                    end_time = parts[2]
                    text = parts[9]
                    
                    # Clean text - remove karaoke timing and formatting
                    import re
                    text = re.sub(r'\{\\k\d+\}', '', text)  # Remove karaoke timing
                    text = re.sub(r'\{[^}]*\}', '', text)   # Remove other formatting
                    text = text.strip()
                    
                    if text:
                        # Convert ASS time to seconds
                        start_seconds = _ass_time_to_seconds(start_time)
                        end_seconds = _ass_time_to_seconds(end_time)
                        
                        captions.append({
                            'start': start_seconds,
                            'end': end_seconds,
                            'text': text
                        })
        
        logging.info(f"Extracted {len(captions)} captions from ASS file")
        return captions
        
    except Exception as e:
        logging.error(f"Error extracting captions: {e}")
        return []

def _ass_time_to_seconds(ass_time: str) -> float:
    """Convert ASS time format to seconds."""
    parts = ass_time.split(':')
    if len(parts) != 3:
        return 0.0
    
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split('.')
    seconds = int(sec_parts[0])
    centiseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0

def _burn_captions_with_drawtext(video_path: str, caption_data: list, output_path: str) -> bool:
    """Burn captions onto video using FFmpeg drawtext filter - Windows-compatible approach."""
    try:
        # Build drawtext filter chain for all captions
        filter_parts = []
        
        for caption in caption_data:
            # Escape text for FFmpeg
            safe_text = caption['text'].replace("'", "'\\''").replace(":", "\\:")
            
            # Create drawtext filter for this caption
            drawtext_filter = (
                f"drawtext=text='{safe_text}'"
                f":fontfile='C\\:/Windows/Fonts/arial.ttf'"
                f":fontsize=38"
                f":fontcolor=white"
                f":borderw=2"
                f":bordercolor=black"
                f":x=(w-text_w)/2"
                f":y=h-text_h-50"
                f":enable='between(t,{caption['start']},{caption['end']})'"
            )
            filter_parts.append(drawtext_filter)
        
        # Combine all drawtext filters
        if not filter_parts:
            logging.warning("No captions to burn")
            return False
        
        # Create complex filter
        video_filter = ",".join(filter_parts)
        
        logging.info(f"Building FFmpeg command with {len(filter_parts)} caption overlays...")
        
        # Build FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', video_filter,
            '-c:a', 'copy',
            output_path
        ]
        
        # Execute FFmpeg
        success, _, stderr = run_command(ffmpeg_cmd)
        
        if success:
            logging.info("Caption burning completed successfully with drawtext")
            return True
        else:
            logging.error(f"Drawtext FFmpeg failed: {stderr}")
            return False
        
    except Exception as e:
        logging.error(f"Drawtext caption burning failed: {e}")
        return False

def _create_script_captions(video_path: str, script_text: str, output_path: str) -> str:
    """Create script-based captions using FFmpeg."""
    try:
        logging.info("Creating script-based captions...")
        
        # Create SRT file
        srt_path = video_path.replace('.mp4', '.srt')
        
        if not _create_srt_from_script(script_text, srt_path, video_path):
            raise Exception("Failed to create SRT file from script")
        
        # Burn captions with FFmpeg
        logging.info("Burning captions with FFmpeg...")
        
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        output_filename = os.path.basename(output_path)
        srt_filename = os.path.basename(srt_path)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_filename,
            '-vf', f"subtitles={srt_filename}:force_style='FontSize=32,Bold=1,PrimaryColour=&H00FFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=3'",
            '-c:a', 'copy',
            '-y',
            output_filename
        ]
        
        # Run FFmpeg in video directory
        original_cwd = os.getcwd()
        try:
            os.chdir(video_dir)
            success, _, stderr = run_command(ffmpeg_cmd)
            
            if not success:
                raise Exception(f"FFmpeg failed: {stderr}")
                
        finally:
            os.chdir(original_cwd)
            
            # Clean up SRT file
            if os.path.exists(srt_path):
                os.remove(srt_path)
        
        logging.info("Script-based captions successfully added!")
        return output_path
        
    except Exception as e:
        error_msg = f"Script caption creation failed: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

def _create_srt_from_script(script_text: str, srt_path: str, video_path: str) -> bool:
    """Create SRT subtitle file from script text."""
    try:
        # Get video duration
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                     'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        success, stdout, _ = run_command(probe_cmd)
        
        if not success or not stdout:
            logging.error("Could not determine video duration")
            return False
            
        video_duration = float(stdout.strip())
        
        # Clean script and split into sentences
        import re
        clean_script = re.sub(r'\[Visual:.*?\]', '', script_text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', clean_script)
        
        # Calculate timing
        words_per_second = 2.5
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            current_time = 0.0
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                word_count = len(sentence.split())
                duration = max(word_count / words_per_second, 1.5)
                
                if current_time >= video_duration:
                    break
                    
                end_time = min(current_time + duration, video_duration)
                
                # Write SRT entry
                f.write(f"{i + 1}\n")
                f.write(f"{_format_srt_time(current_time)} --> {_format_srt_time(end_time)}\n")
                f.write(f"{sentence.strip()}\n\n")
                
                current_time = end_time
        
        logging.info(f"Created SRT file with {len(sentences)} captions")
        return True
        
    except Exception as e:
        logging.error(f"Error creating SRT from script: {e}")
        return False

def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}" 