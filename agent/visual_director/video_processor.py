"""
Advanced Video Processing and Trimming

This module implements video trimming, clip extraction, and intelligent
video processing features for the visual director.
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from agent.utils import run_command


class VideoProcessor:
    """
    Handles advanced video processing including trimming, reframing,
    and intelligent clip extraction.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            temp_dir: Directory for temporary files. Uses system temp if None.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.ffmpeg_path = self._find_ffmpeg()
        
        if not self.ffmpeg_path:
            logging.warning("FFmpeg not found. Video processing features will be limited.")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return "ffmpeg"
        except:
            pass
        
        # Try common locations
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def trim_video(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        copy_codec: bool = True
    ) -> bool:
        """
        Trim a video to a specific time range.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            start_time: Start time in seconds
            end_time: End time in seconds
            copy_codec: Whether to copy codec (fast) or re-encode
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            logging.error("FFmpeg not available for video trimming")
            return False
        
        duration = end_time - start_time
        if duration <= 0:
            logging.error(f"Invalid trim duration: {duration}")
            return False
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", input_path
        ]
        
        if copy_codec:
            # Fast copy without re-encoding
            cmd.extend(["-c", "copy"])
        else:
            # Re-encode for better compatibility
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac"
            ])
        
        cmd.append(output_path)
        
        try:
            logging.info(f"Trimming video: {start_time}s to {end_time}s")
            success, stdout, stderr = run_command(cmd)
            
            if success and os.path.exists(output_path):
                return True
            else:
                logging.error(f"Video trim failed: {stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Video trim error: {e}")
            return False
    
    def extract_clip_with_motion(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float,
        zoom_start: float = 1.0,
        zoom_end: float = 1.2,
        pan_x: float = 0.0,
        pan_y: float = 0.0
    ) -> bool:
        """
        Extract a clip with zoom and pan motion effects.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            start_time: Start time in seconds
            duration: Duration in seconds
            zoom_start: Initial zoom level (1.0 = normal)
            zoom_end: Final zoom level
            pan_x: Horizontal pan (-1 to 1)
            pan_y: Vertical pan (-1 to 1)
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            return False
        
        # Build zoompan filter
        fps = 30  # Assume 30fps, could detect from input
        total_frames = int(duration * fps)
        
        # Calculate zoom interpolation
        zoom_expr = f"zoom='linear(on/{total_frames},{zoom_start},{zoom_end})'"
        
        # Calculate pan based on zoom to keep it smooth
        x_expr = f"x='(iw-iw/zoom)/2+{pan_x}*iw/zoom/4'"
        y_expr = f"y='(ih-ih/zoom)/2+{pan_y}*ih/zoom/4'"
        
        filter_str = f"zoompan={zoom_expr}:{x_expr}:{y_expr}:d={total_frames}:s=1920x1080:fps={fps}"
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", input_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ]
        
        try:
            success, _, stderr = run_command(cmd)
            return success and os.path.exists(output_path)
        except Exception as e:
            logging.error(f"Clip extraction error: {e}")
            return False
    
    def reframe_to_portrait(
        self,
        input_path: str,
        output_path: str,
        focus_area: Optional[str] = "center"
    ) -> bool:
        """
        Reframe a video to portrait orientation (9:16).
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            focus_area: Where to focus ("center", "left", "right", "top", "bottom")
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            return False
        
        # Portrait dimensions
        target_width = 1080
        target_height = 1920
        target_ratio = target_width / target_height
        
        # Get input video info
        info = self.get_video_info(input_path)
        if not info:
            return False
        
        src_width = info['width']
        src_height = info['height']
        src_ratio = src_width / src_height
        
        # Calculate crop parameters
        if src_ratio > target_ratio:
            # Source is wider - crop sides
            new_width = int(src_height * target_ratio)
            new_height = src_height
            
            if focus_area == "left":
                x_offset = 0
            elif focus_area == "right":
                x_offset = src_width - new_width
            else:  # center
                x_offset = (src_width - new_width) // 2
            
            y_offset = 0
        else:
            # Source is taller - crop top/bottom
            new_width = src_width
            new_height = int(src_width / target_ratio)
            
            x_offset = 0
            
            if focus_area == "top":
                y_offset = 0
            elif focus_area == "bottom":
                y_offset = src_height - new_height
            else:  # center
                y_offset = (src_height - new_height) // 2
        
        # Build filter
        filter_str = (
            f"crop={new_width}:{new_height}:{x_offset}:{y_offset},"
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", input_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            success, _, _ = run_command(cmd)
            return success and os.path.exists(output_path)
        except Exception as e:
            logging.error(f"Reframe error: {e}")
            return False
    
    def create_video_montage(
        self,
        clips: List[Dict[str, Any]],
        output_path: str,
        transition_duration: float = 0.5
    ) -> bool:
        """
        Create a montage from multiple video clips with transitions.
        
        Args:
            clips: List of clip info dicts with 'path', 'start', 'duration'
            output_path: Path for output video
            transition_duration: Duration of transitions in seconds
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path or not clips:
            return False
        
        # Build complex filter for transitions
        inputs = []
        filter_parts = []
        
        # Add inputs
        for i, clip in enumerate(clips):
            if 'start' in clip and 'duration' in clip:
                inputs.extend([
                    "-ss", str(clip['start']),
                    "-t", str(clip['duration'])
                ])
            inputs.extend(["-i", clip['path']])
        
        # Build xfade filter chain
        if len(clips) == 1:
            # Single clip, just copy
            filter_str = "[0:v]copy[vout]"
        else:
            # Multiple clips with transitions
            prev_tag = "[0:v]"
            offset = 0.0
            
            for i in range(1, len(clips)):
                clip_duration = clips[i-1].get('duration', 3.0)
                offset += clip_duration - transition_duration
                
                if i == len(clips) - 1:
                    output_tag = "[vout]"
                else:
                    output_tag = f"[v{i}]"
                
                filter_parts.append(
                    f"{prev_tag}[{i}:v]xfade=transition=fade:"
                    f"duration={transition_duration}:offset={offset}{output_tag}"
                )
                
                prev_tag = output_tag
            
            filter_str = ";".join(filter_parts)
        
        cmd = [
            self.ffmpeg_path,
            "-y"
        ]
        cmd.extend(inputs)
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[vout]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ])
        
        try:
            success, _, _ = run_command(cmd)
            return success and os.path.exists(output_path)
        except Exception as e:
            logging.error(f"Montage creation error: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video info or None
        """
        if not self.ffmpeg_path:
            return None
        
        ffprobe = self.ffmpeg_path.replace("ffmpeg", "ffprobe")
        if not os.path.exists(ffprobe):
            ffprobe = "ffprobe"  # Try system path
        
        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = None
                for stream in data.get('streams', []):
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'duration': float(data.get('format', {}).get('duration', 0)),
                        'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                        'codec': video_stream.get('codec_name', ''),
                        'bitrate': int(data.get('format', {}).get('bit_rate', 0))
                    }
        except Exception as e:
            logging.warning(f"Failed to get video info: {e}")
        
        return None
    
    def extract_interesting_clips(
        self,
        video_path: str,
        num_clips: int = 3,
        clip_duration: float = 3.0,
        method: str = "uniform"
    ) -> List[Dict[str, float]]:
        """
        Extract interesting clips from a video.
        
        Args:
            video_path: Path to video
            num_clips: Number of clips to extract
            clip_duration: Duration of each clip
            method: "uniform", "motion", or "scene"
            
        Returns:
            List of clip info with start times
        """
        info = self.get_video_info(video_path)
        if not info:
            return []
        
        video_duration = info['duration']
        clips = []
        
        if method == "uniform":
            # Evenly distributed clips
            if video_duration <= clip_duration:
                # Video too short, use whole video
                clips.append({
                    'start': 0.0,
                    'duration': video_duration
                })
            else:
                # Calculate spacing
                available_duration = video_duration - clip_duration
                if num_clips == 1:
                    # Single clip from middle
                    start = available_duration / 2
                    clips.append({
                        'start': start,
                        'duration': clip_duration
                    })
                else:
                    # Multiple clips evenly spaced
                    spacing = available_duration / (num_clips - 1)
                    for i in range(num_clips):
                        clips.append({
                            'start': i * spacing,
                            'duration': clip_duration
                        })
        
        elif method == "motion":
            # TODO: Implement motion-based clip detection
            # For now, fall back to uniform
            return self.extract_interesting_clips(
                video_path, num_clips, clip_duration, "uniform"
            )
        
        elif method == "scene":
            # TODO: Implement scene-based clip detection
            # For now, fall back to uniform
            return self.extract_interesting_clips(
                video_path, num_clips, clip_duration, "uniform"
            )
        
        return clips
    
    def add_ken_burns_to_image(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        zoom_start: float = 1.0,
        zoom_end: float = 1.3,
        pan_start: Tuple[float, float] = (0, 0),
        pan_end: Tuple[float, float] = (0, 0)
    ) -> bool:
        """
        Add Ken Burns effect to a still image.
        
        Args:
            image_path: Path to input image
            output_path: Path for output video
            duration: Duration in seconds
            zoom_start: Initial zoom (1.0 = no zoom)
            zoom_end: Final zoom
            pan_start: Initial pan position (x, y) from -1 to 1
            pan_end: Final pan position
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            return False
        
        fps = 30
        total_frames = int(duration * fps)
        
        # Build zoompan expression
        zoom_expr = f"zoom='linear(on/{total_frames},{zoom_start},{zoom_end})'"
        
        # Pan expressions with smooth interpolation
        x_start, y_start = pan_start
        x_end, y_end = pan_end
        
        x_expr = f"x='(iw-iw/zoom)/2+lerp({x_start},{x_end},on/{total_frames})*iw/zoom/4'"
        y_expr = f"y='(ih-ih/zoom)/2+lerp({y_start},{y_end},on/{total_frames})*ih/zoom/4'"
        
        filter_str = (
            f"scale=3840:2160,"  # Upscale for quality
            f"zoompan={zoom_expr}:{x_expr}:{y_expr}:d={total_frames}:s=1080x1920:fps={fps},"
            f"format=yuv420p"
        )
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", filter_str,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ]
        
        try:
            success, _, _ = run_command(cmd)
            return success and os.path.exists(output_path)
        except Exception as e:
            logging.error(f"Ken Burns error: {e}")
            return False


# Global processor instance
_video_processor = None


def get_video_processor() -> VideoProcessor:
    """Get or create global video processor instance."""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


def process_video_asset(
    asset,
    segment: Dict[str, Any],
    output_dir: str
) -> Optional[str]:
    """
    Process a video asset for use in a segment.
    
    Args:
        asset: Video asset object
        segment: Segment requiring the video
        output_dir: Directory for processed output
        
    Returns:
        Path to processed video or None
    """
    processor = get_video_processor()
    
    if not asset.local_path or not os.path.exists(asset.local_path):
        logging.error(f"Video asset not downloaded: {asset.id}")
        return None
    
    # Calculate required duration
    segment_duration = segment['end_time'] - segment['start_time']
    
    # Determine output path
    output_filename = f"{asset.id}_processed.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # If asset has specific start/end times, use them
    if asset.metadata.get('start_s') is not None:
        start_time = asset.metadata['start_s']
        end_time = asset.metadata.get('end_s', start_time + segment_duration)
    else:
        # Extract interesting clip
        clips = processor.extract_interesting_clips(
            asset.local_path,
            num_clips=1,
            clip_duration=segment_duration
        )
        if clips:
            start_time = clips[0]['start']
            end_time = start_time + segment_duration
        else:
            start_time = 0
            end_time = segment_duration
    
    # Trim and reframe
    success = processor.trim_video(
        asset.local_path,
        output_path,
        start_time,
        end_time,
        copy_codec=False  # Re-encode for portrait
    )
    
    if success:
        # Reframe to portrait if needed
        info = processor.get_video_info(output_path)
        if info and info['width'] / info['height'] > 1.0:
            # Landscape video, reframe to portrait
            portrait_path = output_path.replace('.mp4', '_portrait.mp4')
            if processor.reframe_to_portrait(output_path, portrait_path):
                os.remove(output_path)
                output_path = portrait_path
    
    return output_path if success else None
