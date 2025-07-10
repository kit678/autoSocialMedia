import os
import logging
import json
from typing import List, Dict, Any
from agent.utils import run_command
import random

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

def run(visual_analysis: Dict[str, Any], all_image_paths: Dict[str, str], audio_path: str, audio_duration: float, output_path: str, fps: int = 30):
    """
    Main entry point for the smart video creation process using FFmpeg.
    """
    logging.info("--- Starting Smart Video Assembly with FFmpeg ---")

    if not visual_analysis.get('segments'):
        logging.error("Visual analysis returned no segments. Cannot create video.")
        return None

    # Check for FFmpeg availability
    success, _, _ = run_command(['ffmpeg', '-version'], timeout=10)
    if not success:
        logging.error("FFmpeg not found. Smart video assembly requires FFmpeg.")
        return None

    logging.info("FFmpeg detected, using FFmpeg-based video assembly...")

    # Calculate segment durations to match audio length
    timeline = _calculate_segment_durations(visual_analysis['segments'], audio_duration, fps)
    
    if not timeline:
        logging.error("Timeline calculation resulted in no segments. Cannot create video.")
        return None

    # Build the filter graph for transitions and effects
    filter_chains, inputs_args = _build_smart_filter(timeline, all_image_paths, fps)

    if not inputs_args:
        logging.error("Could not build any valid video inputs. Aborting video creation.")
        return None

    # Assemble the final command including audio
    command = _assemble_ffmpeg_command(inputs_args, filter_chains, output_path, audio_path, fps)
    
    if not command:
        logging.error("Failed to assemble FFmpeg command.")
        return None
    
    logging.info("--- Running FFmpeg Command for Smart Video ---")
    logging.info(" ".join(f'\"{c}\"' if " " in c else c for c in command))
    
    success, stdout, stderr = run_command(command, timeout=300)

    if success:
        logging.info(f"SUCCESS: Smart video generated at '{output_path}'")
        return output_path
    else:
        logging.error("Smart video generation failed.")
        logging.error(f"FFmpeg STDOUT:\n{stdout}")
        logging.error(f"FFmpeg STDERR:\n{stderr}")
        return None

def _build_smart_filter(timeline: list, all_image_paths: Dict[str, str], fps: int) -> tuple:
    """
    Build the filter graph for transitions and effects.
    Returns: (filter_chains, inputs_args)
    """
    filter_chains = []
    inputs_args = []

    for i, segment in enumerate(timeline):
        image_id = segment.get('cue_id', f'visual_{i:02d}')
        if not image_id or image_id not in all_image_paths:
            logging.warning(f"Skipping segment {i} due to missing image_id or path.")
            continue

        image_path = all_image_paths[image_id]
        duration_frames = segment['duration_frames']
        
        # Ensure duration is at least 1 frame to avoid FFmpeg errors
        if duration_frames <= 0:
            logging.warning(f"Skipping segment for '{image_id}' due to zero or negative duration.")
            continue

        # Ensure path exists before adding to command
        if not os.path.exists(image_path):
            logging.warning(f"File not found, skipping: {image_path}")
            continue

        # Add input arguments for this image
        duration_seconds = duration_frames / fps
        inputs_args.extend(['-loop', '1', '-t', str(duration_seconds), '-i', image_path])

        # Get Ken Burns effect parameters (simplified random generation)
        ken_burns_filter = _get_ken_burns_params_simple(image_path, duration_frames, fps)

        # Build the filter chain for this segment
        input_index = i  # Each image gets its own input index
        filter_chains.append(f"[{input_index}:v]{ken_burns_filter}[v{i}];")

    return filter_chains, inputs_args

def _get_ken_burns_params_simple(image_path: str, duration_frames: int, fps: int) -> str:
    """Get Ken Burns parameters for a single image."""
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
    
    # Build the zoompan filter
    return (
        f"scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop=1080:1920,"
        f"zoompan=z='if(lte(on,1),{start_zoom},{start_zoom}+({end_zoom}-{start_zoom})*on/{duration_frames})'"
        f":x='if(lte(on,1),{start_x},{start_x}+({end_x}-{start_x})*on/{duration_frames})'"
        f":y='if(lte(on,1),{start_y},{start_y}+({end_y}-{start_y})*on/{duration_frames})'"
        f":d={duration_frames}:s=1080x1920:fps={fps}"
    )

def _assemble_ffmpeg_command(inputs_args: list, filter_chains: list, output_path: str, audio_path: str, fps: int) -> list:
    """Assembles the final FFmpeg command."""
    
    # Check if there are any video chains to process
    if not filter_chains:
        return []

    # Build the complete filter graph
    filter_graph = "".join(filter_chains)
    
    num_streams = len(filter_chains)
    
    if num_streams > 1:
        # Build the transition chain
        transition_duration = 0.7  # seconds for each fade
        
        # Start with the first video stream
        last_output = "[v0]"
        for i in range(num_streams - 1):
            # Define the output of the current fade
            current_output = f"[chain{i}]"
            # The last one should be the final output stream
            if i == num_streams - 2:
                current_output = "[vout]"

            # Calculate the offset for the fade
            offset = (i + 1) * 4.0 - transition_duration  # Approximate timing
            
            transition = random.choice(XFADE_TRANSITIONS)
            
            filter_graph += (
                f"{last_output}[v{i+1}]"
                f"xfade=transition={transition}:duration={transition_duration}:offset={offset}"
                f"{current_output};"
            )
            last_output = current_output
    else:
        # If there's only one video stream, just label it for output
        filter_graph += "[v0]copy[vout];"
    
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
        '-r', str(fps),
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

def _get_ken_burns_params(source_key: str, ken_burns_effects: Dict) -> Dict:
    """Get Ken Burns parameters or generate random ones as a fallback."""
    
    # Use existing effect if provided by visual analysis
    if source_key in ken_burns_effects:
        effect = ken_burns_effects[source_key]
        if all(k in effect for k in ['start_zoom', 'end_zoom', 'start_x', 'start_y', 'end_x', 'end_y']):
            return effect

    # Fallback to randomized Ken Burns effect
    start_zoom = random.uniform(1.0, 1.25)
    end_zoom = random.uniform(start_zoom + 0.1, 1.5)

    # Random pan direction
    start_x = random.uniform(0, 200)
    start_y = random.uniform(0, 100)
    end_x = start_x + random.uniform(-100, 100)
    end_y = start_y + random.uniform(-50, 50)
    
    return {
        'start_zoom': start_zoom,
        'end_zoom': end_zoom,
        'start_x': str(start_x),
        'start_y': str(start_y),
        'end_x': str(end_x),
        'end_y': str(end_y)
    }

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
        return _create_with_video_segments(timeline, all_paths, audio_duration, output_path, fps)
    else:
        # Regular image slideshow
        return run(visual_analysis, all_paths, audio_duration, output_path, fps)

def _create_with_video_segments(timeline: List[Dict], all_paths: Dict, 
                               audio_duration: float, output_path: str, fps: int) -> bool:
    """Create slideshow mixing static images and video segments with timebase normalization."""
    
    try:
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        input_idx = 0
        
        # Add inputs and create filter for each segment with consistent timebase handling
        for i, segment in enumerate(timeline):
            if segment.get('is_video'):
                # Add video input
                cmd.extend(['-i', segment['video_path']])
                # Trim video to segment duration with deinterlacing, scaling, and fps normalization
                duration = segment['end_time'] - segment['start_time']
                filter_parts.append(
                    f"[{input_idx}:v]trim=duration={duration},setpts=PTS-STARTPTS,"
                    f"yadif=mode=0:parity=-1:deint=0,"  # Deinterlace if needed
                    f"scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop=1080:1920,"
                    f"fps={fps}[v{i}]"  # Normalize timebase for concat compatibility
                )
            else:
                # Add image input
                image_path = all_paths[segment['image_source']]
                duration = segment['end_time'] - segment['start_time']
                cmd.extend(['-loop', '1', '-t', str(duration), '-i', image_path])
                # Apply Ken Burns to image with high-quality scaling and consistent fps
                filter_parts.append(
                    f"[{input_idx}:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop=1080:1920,zoompan=z='1.0+0.2*on/{fps}':d={int(duration*fps)}:s=1080x1920:fps={fps}[v{i}]"
                )
            input_idx += 1
        
        # Concatenate all segments
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
                            audio_duration: float, output_path: str, fps: int = 30) -> bool:
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
        return run(visual_analysis, all_paths, audio_duration, output_path, fps)
    
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
            filter_parts.append(
                f"[{webpage_input_idx}:v]trim=start={seg['video_start']}:duration={duration},"
                f"setpts=PTS-STARTPTS,"
                f"yadif=mode=0:parity=-1:deint=0,"  # Deinterlace if needed
                f"scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop=1080:1920,"
                f"fps={fps}[webpage{i}]"  # Normalize timebase for consistent concat
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
            
            # Apply Ken Burns if available
            kb_params = _get_ken_burns_params(segment['image_source'], 
                                            visual_analysis.get('ken_burns_effects', {}))
            
            filter_parts.append(
                f"[{input_idx}:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop=1080:1920,"
                f"zoompan=z='{kb_params['start_zoom']}+({kb_params['end_zoom']}-{kb_params['start_zoom']})*on/{int(duration*fps)}'"
                f":d={int(duration*fps)}:s=1080x1920:fps={fps}[img{i}]"
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
            '-r', str(fps),
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