import os
import logging
import json
from typing import List, Dict, Any
from agent.utils import run_command
import random
from agent.video_config import VideoConfig, get_default_config
from agent.video.ffmpeg_utils import scale_crop_str, scale_pad_str, ken_burns_filter
from agent.slideshow.validation import validate_slideshow_inputs, validate_mixed_media_inputs
from agent.slideshow.exceptions import SlideshowError

# List of visually appealing and non-jarring FFmpeg xfade transitions
XFADE_TRANSITIONS = [
    'fade', 'fadeblack', 'fadewhite', 'dissolve', 'distance',
    'wipeleft', 'wiperight', 'wipeup', 'wipedown',
    'slideleft', 'slideright', 'slideup', 'slidedown',
    'smoothleft', 'smoothright', 'smoothup', 'smoothdown',
    'circleopen', 'circleclose',
    'rectcrop',
    'diagtl', 'diagtr', 'diagbl', 'diagbr',
    'hlslice', 'hrslice', 'vuslice', 'vdslice',
    'radial', 'zoomin'
]

def build_transition_chain(stream_tags: list, config: VideoConfig, timeline: list, rng_seed: int = None) -> str:
    if not stream_tags:
        return ""
    if rng_seed is not None:
        random.seed(rng_seed)
    num_streams = len(stream_tags)
    transition_duration = config.transition_duration
    if num_streams == 1:
        return f"{stream_tags[0]}copy[vout];"
    filter_parts = []
    last_output = stream_tags[0]
    cumulative_offset = 0.0
    for i in range(num_streams - 1):
        segment_duration = (timeline[i]['end_time'] - timeline[i]['start_time'])
        offset = cumulative_offset + segment_duration - transition_duration
        current_output = f"[chain{i}]" if i < num_streams - 2 else "[vout]"
        transition = random.choice(XFADE_TRANSITIONS)
        filter_parts.append(f"{last_output}{stream_tags[i+1]}xfade=transition={transition}:duration={transition_duration}:offset={offset}{current_output};")
        last_output = current_output
        cumulative_offset += segment_duration
    return "".join(filter_parts)

def run(visual_analysis: Dict[str, Any], all_image_paths: Dict[str, str], audio_path: str, audio_duration: float, output_path: str, fps: int = 30, rng_seed: int = None, config: VideoConfig | None = None, orientation: str | None = None):
    """
    Main entry point for the smart video creation process using FFmpeg.
    
    Args:
        visual_analysis: Visual analysis data with timeline
        all_image_paths: Dictionary mapping image IDs to file paths
        audio_path: Path to audio file
        audio_duration: Duration of audio in seconds
        output_path: Output video file path
        fps: Frames per second
        rng_seed: Optional RNG seed for reproducible transitions
    """
    logging.info("--- Starting Smart Video Assembly with FFmpeg ---")

    try:
        timeline, actual_audio_duration = validate_slideshow_inputs(
            visual_analysis, all_image_paths, audio_path, audio_duration
        )

        logging.info(f"Validation passed: {len(timeline)} segments, "
                     f"audio duration {actual_audio_duration:.2f}s")

        # Calculate total timeline duration and cap if necessary
        total_timeline_duration = sum(seg['end_time'] - seg['start_time'] for seg in timeline)
        logging.info(f"Timeline total: {total_timeline_duration:.2f}s vs audio: {actual_audio_duration:.2f}s")
        if total_timeline_duration > actual_audio_duration:
            excess = total_timeline_duration - actual_audio_duration
            timeline[-1]['end_time'] -= excess
            logging.info(f"Capped last segment by {excess:.2f}s to match audio duration")

        # Determine configuration to use
        if config is None:
            # If caller supplied an orientation string, use it; otherwise default to 'portrait'
            if orientation is None:
                orientation = "portrait"
            config = get_default_config(orientation)
        else:
            # Ensure basic sanity (e.g. even dimensions)
            try:
                config.validate()
            except Exception as e:
                logging.warning(f"Provided VideoConfig is invalid – falling back to defaults: {e}")
                config = get_default_config(orientation or "portrait")

        # Build the filter graph for transitions and effects
        filter_chains, inputs_args = _build_smart_filter(timeline, all_image_paths, config)

        if not inputs_args:
            raise SlideshowError("Could not build any valid video inputs. Aborting video creation.")

        # Assemble the final command including audio
        command = _assemble_ffmpeg_command(inputs_args, filter_chains, output_path, audio_path, config, timeline, rng_seed)

        if not command:
            raise SlideshowError("Failed to assemble FFmpeg command.")
    
        logging.info("--- Running FFmpeg Command for Smart Video ---")
        logging.info(" ".join(f'\"{c}\"' if " " in c else c for c in command))
        
        success, stdout, stderr = run_command(command, timeout=300)

        if success:
            logging.info(f"SUCCESS: Smart video generated at '{output_path}'")
            return output_path
        else:
            raise SlideshowError("Smart video generation failed.")

    except SlideshowError as e:
        logging.error(e)
        return None

def _build_smart_filter(timeline: list, all_image_paths: Dict[str, str], config: VideoConfig) -> tuple:
    """
    Build the filter graph for transitions and effects.
    Returns: (filter_chains, inputs_args)
    """
    filter_chains: list[str] = []
    inputs_args: list[str] = []
    fps = config.fps

    # Maintain the *real* ffmpeg input index as we append to inputs_args.
    input_stream_idx = 0

    for seg_idx, segment in enumerate(timeline):
        # Determine the source identifier – timeline may use either 'cue_id' (image) or
        # 'image_source' (mixed-media pipelines).
        src_id = segment.get('cue_id') or segment.get('image_source')
        if not src_id:
            logging.warning(f"Segment {seg_idx} missing cue_id/image_source field – skipped")
            continue

        # Video segments inserted by upstream stages can specify an explicit path.
        if segment.get('is_video') and segment.get('video_path'):
            src_path = segment['video_path']
        else:
            if src_id not in all_image_paths:
                logging.warning(f"Segment {seg_idx}: source '{src_id}' not present in asset map – skipped")
                continue
            src_path = all_image_paths[src_id]

        if not os.path.exists(src_path):
            logging.warning(f"File not found, skipping: {src_path}")
            continue

        duration_frames = segment['duration_frames']
        if duration_frames <= 0:
            logging.warning(f"Skipping segment for '{src_id}' due to non-positive duration.")
            continue

        duration_sec = duration_frames / fps

        # Determine if the source is a video clip (mp4/mov/mkv/webm/avi) or a still image.
        video_exts = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}
        ext = os.path.splitext(src_path)[1].lower()
        is_video = ext in video_exts

        # --- Add input arguments ---
        if is_video:
            # Videos must be limited to segment duration to prevent overrun
            inputs_args.extend(['-t', str(duration_sec), '-i', src_path])
        else:
            # Still images must be looped so FFmpeg generates a stream.
            inputs_args.extend(['-loop', '1', '-t', str(duration_sec), '-i', src_path])

        # --- Build per-segment filter ---
        if is_video:
            scale_crop = scale_pad_str(config)
            filter_chains.append(
                f"[{input_stream_idx}:v]trim=duration={duration_sec},setpts=PTS-STARTPTS,{scale_crop},fps={fps}[v{seg_idx}];"
            )
        else:
            kb_filter = _get_ken_burns_params_simple(src_path, duration_frames, config)
            filter_chains.append(f"[{input_stream_idx}:v]{kb_filter}[v{seg_idx}];")

        # Increment stream index for the next loop iteration.
        input_stream_idx += 1

    return filter_chains, inputs_args

def _get_ken_burns_params_simple(image_path: str, duration_frames: int, config: VideoConfig) -> str:
    """Get Ken Burns parameters for a single image using the helper function."""
    # Generate random Ken Burns parameters for variety
    start_zoom = random.uniform(1.0, 1.1)
    end_zoom = random.uniform(1.2, 1.4)
    
    # Random pan direction
    pan_directions = [
        ('left', 'right'), ('right', 'left'), 
        ('top', 'bottom'), ('bottom', 'top'),
        ('center', 'center')
    ]
    start_pos, end_pos = random.choice(pan_directions)
    
    # Convert to coordinates
    positions = {
        'left': '-50', 'right': '50', 'top': '-30', 
        'bottom': '30', 'center': '0'
    }
    
    start_x = positions.get(start_pos, '0')
    start_y = positions.get(start_pos, '0') if start_pos in ['top', 'bottom'] else '0'
    end_x = positions.get(end_pos, '0')
    end_y = positions.get(end_pos, '0') if end_pos in ['top', 'bottom'] else '0'
    
    return ken_burns_filter(
        config=config,
        duration_frames=duration_frames,
        start_zoom=start_zoom,
        end_zoom=end_zoom,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y
    )

def _assemble_ffmpeg_command(inputs_args: list, filter_chains: list, output_path: str, audio_path: str, config: VideoConfig, timeline: list, rng_seed: int = None) -> list:
    """Assembles the final FFmpeg command."""
    
    # Check if there are any video chains to process
    if not filter_chains:
        return []

    # Build the complete filter graph
    filter_graph = "".join(filter_chains)
    
    num_streams = len(filter_chains)
    
    # Build transition chain using the new function
    stream_tags = [f"[v{i}]" for i in range(num_streams)]
    transition_filter = build_transition_chain(stream_tags, config, timeline, rng_seed)
    filter_graph += transition_filter
    
    # --- COMMAND ASSEMBLY ---
    command = ['ffmpeg', '-y']  # Start with ffmpeg and overwrite flag
    
    # Add all image inputs
    command.extend(inputs_args) 
    
    # Add the audio input
    command.extend(['-i', audio_path])
    
    # Add filter complex and output mapping
    command.extend([
        '-filter_complex', filter_graph,
        '-map', '[vout]',      # Map the final video stream
        '-map', f'{num_streams}:a', # Map the audio stream (audio is the last input)
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-r', str(config.fps),
        '-pix_fmt', 'yuv420p',
        output_path
    ])
    
    return command

def _calculate_segment_durations(segments: list, total_duration: float, fps: int) -> list:
    """
    Calculates the duration in frames for each video segment, ensuring the total
    duration matches the audio length.
    """
    total_frames = int(total_duration * fps)
    num_segments = len(segments)
    
    if num_segments == 0:
        return []

    # Distribute frames as evenly as possible
    base_frames = total_frames // num_segments
    remainder = total_frames % num_segments

    timeline = []
    for i, segment in enumerate(segments):
        duration_frames = base_frames
        if i < remainder:
            duration_frames += 1
        
        # Override with specific duration if provided in the visual analysis
        if 'duration_seconds' in segment:
            override_frames = int(segment['duration_seconds'] * fps)
            if override_frames > 0:
                duration_frames = override_frames

        # Create timeline entry with required fields
        timeline_entry = {
            'cue_id': segment.get('cue_id', f'visual_{i:02d}'),
            'duration_frames': duration_frames,
            'start_time': segment.get('start_time', i * (total_duration / num_segments)),
            'end_time': segment.get('end_time', (i + 1) * (total_duration / num_segments))
        }
        
        timeline.append(timeline_entry)

    # Adjust the total duration to match the audio by modifying the last segment
    if timeline:
        final_total_frames = sum(s['duration_frames'] for s in timeline)
        adjustment = total_frames - final_total_frames
        timeline[-1]['duration_frames'] += adjustment
        # Ensure duration is not negative
        timeline[-1]['duration_frames'] = max(1, timeline[-1]['duration_frames'])

    return timeline


def create_with_scroll_video(visual_analysis: Dict, all_paths: Dict, scroll_video_path: str, 
                           audio_duration: float, output_path: str, fps: int = 30) -> bool:
    """
    Creates slideshow incorporating scroll video based on analysis.
    
    Args:
        visual_analysis: Analysis results
        all_paths: All image paths
        scroll_video_path: Path to scroll video if generated
        audio_duration: Audio duration
        output_path: Output path
        fps: Frames per second
    Returns:
        bool: Success status
    """
    # Check if scroll video should be used
    url_screenshot_analysis = visual_analysis.get('image_assessment', {}).get('url_screenshot', {})
    
    if url_screenshot_analysis.get('needs_scroll') and os.path.exists(scroll_video_path):
        logging.info("  > Incorporating scroll video into slideshow")
        
        # Update timeline to use scroll video instead of static screenshot
        timeline = visual_analysis.get('visual_timeline', [])
        for segment in timeline:
            if segment.get('image_source') == 'url_screenshot':
                segment['is_video'] = True
                segment['video_path'] = scroll_video_path
        
        # Build special filter for video incorporation
        config = get_default_config("portrait")  # Mixed media typically uses portrait
        return _create_with_video_segments(timeline, all_paths, audio_duration, output_path, config)
    else:
        # Regular image slideshow
        return run(visual_analysis, all_paths, "", audio_duration, output_path, fps)

def _create_with_video_segments(timeline: List[Dict], all_paths: Dict, 
                               audio_duration: float, output_path: str, config: VideoConfig, rng_seed: int = None) -> bool:
    """Create slideshow mixing static images and video segments with timebase normalization."""
    
    try:
        # Validate mixed media inputs
        validate_mixed_media_inputs(timeline, all_paths)
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        input_idx = 0
        fps = config.fps
        
        # Add inputs and create filter for each segment with consistent timebase handling
        for i, segment in enumerate(timeline):
            if segment.get('is_video'):
                # Add video input
                cmd.extend(['-i', segment['video_path']])
                # Trim video to segment duration with deinterlacing, scaling, and fps normalization
                duration = segment['end_time'] - segment['start_time']
                scale_crop_filter = scale_pad_str(config)
                filter_parts.append(
                    f"[{input_idx}:v]trim=duration={duration},setpts=PTS-STARTPTS,"
                    f"yadif=mode=0:parity=-1:deint=0,"  # Deinterlace if needed
                    f"{scale_crop_filter},"
                    f"fps={fps}[v{i}]"  # Normalize timebase for concat compatibility
                )
            else:
                # Add image input
                image_path = all_paths[segment['image_source']]
                duration = segment['end_time'] - segment['start_time']
                cmd.extend(['-loop', '1', '-t', str(duration), '-i', image_path])
                # Apply Ken Burns to image with high-quality scaling and consistent fps
                scale_crop_filter = scale_crop_str(config)
                filter_parts.append(
                    f"[{input_idx}:v]{scale_crop_filter},zoompan=z='1.0+0.2*on/{fps}':d={int(duration*fps)}:s={config.width}x{config.height}:fps={fps}[v{i}]"
                )
            input_idx += 1
        
        # Use transition chain for consistency (even though we're using concat here)
        # Future enhancement: could switch to xfade transitions for mixed media too
        concat_inputs = ''.join([f'[v{i}]' for i in range(len(timeline))])
        filter_parts.append(f"{concat_inputs}concat=n={len(timeline)}:v=1:a=0[final]")
        
        # Complete command
        cmd.extend([
            '-filter_complex', ';'.join(filter_parts),
            '-map', '[final]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level:v', '4.0',
            '-movflags', '+faststart',
            '-r', str(fps),
            '-t', str(audio_duration),
            output_path
        ])
        
        success, _, stderr = run_command(cmd)
        
        if success:
            logging.info(f"  > Mixed media slideshow created: {output_path}")
            return True
        else:
            logging.error(f"Failed to create mixed slideshow: {stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error in mixed media slideshow: {e}")
        return False

def create_with_webpage_video(visual_analysis: Dict, all_paths: Dict, webpage_video_path: str,
                            audio_duration: float, output_path: str, config: VideoConfig) -> bool:
    """
    Creates slideshow incorporating webpage video clip.
    
    Args:
        visual_analysis: Analysis results
        all_paths: All image paths
        webpage_video_path: Path to webpage capture video
        audio_duration: Audio duration
        output_path: Output path
        fps: Frames per second
    Returns:
        bool: Success status
    """
    if not os.path.exists(webpage_video_path):
        logging.warning("Webpage video not found, falling back to regular slideshow")
        return run(visual_analysis, all_paths, "", audio_duration, output_path, config.fps)
    
    logging.info("  > Creating slideshow with webpage video integration")
    
    try:
        # Update timeline to use webpage video at beginning
        timeline = visual_analysis.get('visual_timeline', [])
        
        # Find good spots for webpage video (usually beginning and/or when discussing the source)
        webpage_segments = []
        
        # Always start with webpage video to establish source credibility
        if timeline:
            webpage_duration = min(3.0, audio_duration * 0.15)  # Max 15% of total or 3 seconds
            webpage_segments.append({
                "start_time": 0.0,
                "end_time": webpage_duration,
                "is_video": True,
                "video_path": webpage_video_path,
                "video_start": 0.0,  # Start from beginning of capture
                "purpose": "establish_source"
            })
            
            # Adjust other segments
            for segment in timeline:
                if segment['start_time'] < webpage_duration:
                    segment['start_time'] = webpage_duration
                if segment['end_time'] < webpage_duration:
                    segment['end_time'] = webpage_duration
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        input_idx = 0
        
        # Add webpage video input
        cmd.extend(['-i', webpage_video_path])
        webpage_input_idx = input_idx
        input_idx += 1
        
        # Add image inputs
        image_inputs = {}
        for segment in timeline:
            if not segment.get('is_video') and segment['image_source'] not in image_inputs:
                image_path = all_paths.get(segment['image_source'])
                if image_path and os.path.exists(image_path):
                    cmd.extend(['-loop', '1', '-t', str(audio_duration), '-i', image_path])
                    image_inputs[segment['image_source']] = input_idx
                    input_idx += 1
        
        # Build filter for webpage video segments
        segment_outputs = []
        
        for i, seg in enumerate(webpage_segments):
            duration = seg['end_time'] - seg['start_time']
            scale_crop_filter = scale_pad_str(config)
            filter_parts.append(
                f"[{webpage_input_idx}:v]trim=start={seg['video_start']}:duration={duration},"
                f"setpts=PTS-STARTPTS,"
                f"yadif=mode=0:parity=-1:deint=0,"  # Deinterlace if needed
                f"{scale_crop_filter},"
                f"fps={config.fps}[webpage{i}]"  # Normalize timebase for consistent concat
            )
            segment_outputs.append(f"[webpage{i}]")
        
        # Build filters for image segments
        for i, segment in enumerate(timeline):
            if segment.get('is_video'):
                continue
                
            input_idx = image_inputs.get(segment['image_source'])
            if input_idx is None:
                continue
                
            duration = segment['end_time'] - segment['start_time']
            
            # Use simplified Ken Burns parameters
            ken_burns_filter = _get_ken_burns_params_simple(segment['image_source'], int(duration*config.fps), config)
            
            filter_parts.append(
                f"[{input_idx}:v]{ken_burns_filter}[img{i}]"
            )
            segment_outputs.append(f"[img{i}]")
        
        # Concatenate all segments
        if len(segment_outputs) > 1:
            concat_filter = ''.join(segment_outputs) + f"concat=n={len(segment_outputs)}:v=1:a=0[final]"
            filter_parts.append(concat_filter)
        else:
            filter_parts.append(f"{segment_outputs[0]}copy[final]")
        
        # Complete command
        cmd.extend([
            '-filter_complex', ';'.join(filter_parts),
            '-map', '[final]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level:v', '4.0',
            '-movflags', '+faststart',
            '-r', str(config.fps),
            '-t', str(audio_duration),
            output_path
        ])
        
        success, _, stderr = run_command(cmd)
        
        if success:
            logging.info(f"  > Slideshow with webpage video created: {output_path}")
            return True
        else:
            logging.error(f"Failed to create slideshow with webpage video: {stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error in webpage video integration: {e}")
        return False 

def test_slideshow_component(runs_dir: str = "runs/current") -> bool:
    """
    Test the slideshow component in isolation using existing pipeline outputs.
    
    Args:
        runs_dir: Directory containing pipeline run data
    Returns:
        bool: True if test passes, False otherwise
    """
    import json
    import os
    
    print("=== TESTING SLIDESHOW COMPONENT IN ISOLATION ===")
    
    # Check required files exist
    required_files = [
        os.path.join(runs_dir, "visual_map.json"),
        os.path.join(runs_dir, "voice.mp3")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    try:
        # Load visual analysis data
        with open(os.path.join(runs_dir, "visual_map.json"), 'r') as f:
            visual_data = json.load(f)
        
        print(f"✅ Loaded visual timeline with {len(visual_data['visual_timeline'])} segments")
        
        # Get audio duration
        audio_path = os.path.join(runs_dir, "voice.mp3")
        from agent.utils import get_audio_duration
        audio_duration = get_audio_duration(audio_path)
        print(f"✅ Audio duration: {audio_duration:.2f}s")
        
        # Build visual analysis structure expected by slideshow component
        visual_analysis = {
            'visual_timeline': [
                {
                    'image_source': seg['cue_id'],
                    'start_time': seg['start_time'], 
                    'end_time': seg['end_time']
                } for seg in visual_data['visual_timeline']
            ],
            'ken_burns_effects': {}  # Using defaults
        }
        
        # Build image paths mapping
        all_image_paths = visual_data['visual_map'].copy()
        
        # Add webpage video if available  
        webpage_video_path = os.path.join(runs_dir, "webpage_capture.mp4")
        if os.path.exists(webpage_video_path):
            all_image_paths['visual_opening'] = webpage_video_path
            print("✅ Found webpage capture video")
            
            # Add opening segment to timeline
            visual_analysis['visual_timeline'].insert(0, {
                'image_source': 'visual_opening',
                'start_time': 0.0,
                'end_time': 3.0
            })
            
            # Adjust other segments
            for seg in visual_analysis['visual_timeline'][1:]:
                seg['start_time'] += 3.0
                seg['end_time'] += 3.0
        
        print(f"✅ Prepared timeline with {len(visual_analysis['visual_timeline'])} segments")
        print(f"✅ Visual assets mapping: {len(all_image_paths)} files")
        
        # Verify all image files exist
        missing_images = []
        for key, path in all_image_paths.items():
            if not os.path.exists(path):
                missing_images.append(f"{key}: {path}")
        
        if missing_images:
            print(f"⚠️  Missing visual files: {missing_images[:3]}{'...' if len(missing_images) > 3 else ''}")
            print(f"   Total missing: {len(missing_images)}")
        else:
            print("✅ All visual assets verified")
        
        # Test output path
        output_path = os.path.join(runs_dir, "test_slideshow_output.mp4")
        
        print("\n--- TESTING SLIDESHOW CREATION ---")
        print(f"Input timeline: {len(visual_analysis['visual_timeline'])} segments")
        print(f"Input assets: {len(all_image_paths)} files")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Output path: {output_path}")
        print("\nCalling slideshow creation...")
        
        # Call the slideshow creation function
        success = run(
            visual_analysis=visual_analysis,
            all_image_paths=all_image_paths,
            audio_path=audio_path, # Pass audio_path
            audio_duration=audio_duration,
            output_path=output_path,
            fps=30
        )
        
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print(f"✅ SUCCESS: Slideshow created at {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            
            # Test video properties
            try:
                duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', output_path]
                from agent.utils import run_command
                success_probe, duration_out, _ = run_command(duration_cmd)
                if success_probe:
                    actual_duration = float(duration_out.strip())
                    print(f"   Video duration: {actual_duration:.2f}s (expected: {audio_duration:.2f}s)")
                    
                # Test video properties  
                info_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', output_path]
                success_info, info_out, _ = run_command(info_cmd)
                if success_info:
                    lines = info_out.strip().split('\n')
                    for line in lines:
                        if line:
                            props = line.split(',')
                            if len(props) >= 3:
                                width, height, fps_frac = props[0], props[1], props[2]
                                print(f"   Video properties: {width}x{height} @ {fps_frac} fps")
                                break
            except Exception as e:
                print(f"   ⚠️ Could not probe video properties: {e}")
            
            return True
        else:
            print("❌ FAILED: Slideshow creation failed")
            return False
            
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_mixed_media():
    """Test slideshow with mixed video and image content - simulates the actual failure scenario."""
    print("\n=== TESTING MIXED MEDIA SCENARIO (VIDEO + IMAGES) ===")
    
    runs_dir = "runs/current"
    
    # Simulate the exact scenario that was failing
    visual_analysis = {
        'visual_timeline': [
            # Opening video segment
            {
                'image_source': 'visual_opening',
                'start_time': 0.0,
                'end_time': 3.0
            },
            # Image segments  
            {
                'image_source': 'visual_00',
                'start_time': 3.0,
                'end_time': 8.0
            },
            {
                'image_source': 'visual_01', 
                'start_time': 8.0,
                'end_time': 13.0
            }
        ],
        'ken_burns_effects': {}
    }
    
    # Mixed assets - video + images
    all_image_paths = {
        'visual_opening': os.path.join(runs_dir, "webpage_capture.mp4"),  # Video
        'visual_00': os.path.join(runs_dir, "visuals/pexels_China_tech_hub_13692064.jpg"),  # Image
        'visual_01': os.path.join(runs_dir, "visuals/pexels_young_Chinese_coders_7595899.jpg")  # Image
    }
    
    # Verify files exist
    missing = [k for k, v in all_image_paths.items() if not os.path.exists(v)]
    if missing:
        print(f"❌ Missing files for mixed media test: {missing}")
        return False
    
    print("✅ Mixed media files verified")
    print("   - Video: webpage_capture.mp4")
    print("   - Image 1: pexels_China_tech_hub_13692064.jpg") 
    print("   - Image 2: pexels_young_Chinese_coders_7595899.jpg")
    
    output_path = os.path.join(runs_dir, "test_mixed_media_output.mp4")
    
    print(f"\nTesting slideshow creation with mixed video+image content...")
    print(f"This tests the exact scenario that was failing due to timebase mismatch")
    
    success = run(
        visual_analysis=visual_analysis,
        all_image_paths=all_image_paths,
        audio_path=os.path.join(runs_dir, "voice.mp3"), # Pass audio_path
        audio_duration=15.0,  # 15 second test 
        output_path=output_path,
        fps=30
    )
    
    if success and os.path.exists(output_path):
        print(f"✅ SUCCESS: Mixed media slideshow created!")
        print(f"   Output: {output_path}")
        return True
    else:
        print(f"❌ FAILED: Mixed media slideshow creation failed")
        return False

if __name__ == "__main__":
    """Run tests when called directly."""
    print("🎬 SLIDESHOW COMPONENT ISOLATION TESTING")
    print("=" * 50)
    
    # Test 1: Full slideshow with existing data
    test1_success = test_slideshow_component()
    
    print("\n" + "=" * 50)
    
    # Test 2: Mixed media scenario (the failing case)
    test2_success = test_with_mixed_media()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print(f"✅ Full slideshow test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Mixed media test: {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        print("🎉 ALL TESTS PASSED - Slideshow component is working correctly!")
        exit(0)
    else:
        print("💥 SOME TESTS FAILED - Issues remain in slideshow component")
        exit(1)