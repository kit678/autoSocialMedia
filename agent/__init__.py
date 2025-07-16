"""
AutoSocialMedia Agent - Automated Social Media Content Generation Pipeline

This module provides an end-to-end pipeline for creating social media videos from news articles.
The pipeline features intelligent visual direction, audio synchronization, and multi-orientation
support for various social media platforms.

Orientation Handling:
    - Portrait Mode (1080x1920): Optimized for TikTok, Instagram Reels, YouTube Shorts
    - Landscape Mode (1920x1080): Optimized for YouTube, Facebook, LinkedIn
    - Square Mode (1080x1080): Optimized for Instagram posts, Twitter videos
    
    Video orientation is configured via the video_config module. Default is landscape,
    but can be adjusted for specific platform requirements.

Timeline Expectations:
    - Average video duration: 30-60 seconds based on script length
    - Visual segments: 2-8 seconds each with AI-synchronized timing
    - Audio synchronization: Word-level timestamp alignment using Whisper
    - Opening sequence: 3-second webpage video capture
    - Transition duration: 1.0 second between visual segments
    
    Timeline calculation is based on natural speech patterns (2.5 words/second)
    and keyword synchronization from the transcript.

Configuration:
    - config.ini: Main configuration file for TTS provider, API keys
    - video_config.py: Video parameters (resolution, FPS, transitions)
    - Component registry: Modular pipeline components with dependency validation
    - Decision logging: Comprehensive logging of AI decisions and pipeline state
    
    The pipeline is designed to be configurable and extensible, with clear
    separation between visual acquisition, audio processing, and video assembly.

Example Usage:
    >>> from agent import create_smart_video
    >>> video_path = create_smart_video()
    >>> print(f"Generated video: {video_path}")

Note:
    This pipeline requires FFmpeg, proper API keys for image/video sources,
    and sufficient disk space for temporary files during processing.
"""

import os
import logging
from typing import Optional

def create_smart_video(max_images: int = 10) -> Optional[str]:
    """
    Main pipeline function with AI-driven visual direction and fast-paced video creation.
    
    Executes the complete AutoSocialMedia pipeline including content discovery,
    article scraping, visual direction, audio generation, and video assembly.
    Creates a professionally edited video with synchronized visuals and captions.
    
    Args:
        max_images (int, optional): Maximum number of images to download. This parameter
            is kept for backward compatibility but is not actively used in the current
            implementation. The visual director determines optimal image count based on
            content length and timing. Defaults to 10.
    
    Returns:
        Optional[str]: Path to the final generated video file (.mp4) if successful,
            None if the pipeline fails at any stage. The video includes:
            - Opening webpage capture sequence (3 seconds)
            - Synchronized visual segments with AI-driven timing
            - Professional transitions between segments
            - Word-level synchronized captions
            - High-quality audio narration
    
    Raises:
        FileNotFoundError: If required configuration files or dependencies are missing
        OSError: If FFmpeg is not installed or accessible
        RuntimeError: If critical pipeline components fail (visual direction, audio generation)
        
    Example:
        >>> from agent import create_smart_video
        >>> video_path = create_smart_video()
        >>> if video_path:
        ...     print(f"Video created successfully: {video_path}")
        ... else:
        ...     print("Pipeline failed - check logs for details")
    
    Note:
        This function requires:
        - FFmpeg installed and in PATH
        - Valid API keys in config.ini
        - Internet connection for content discovery and visual assets
        - Sufficient disk space (~500MB for temporary files)
        
        The pipeline creates a 'runs/current' directory structure with intermediate
        files that can be useful for debugging or manual inspection.
    """
    from .component_runner import ComponentRunner
    from .decision_logger import init_decision_logger, get_decision_logger

    from .video.smart_assembler import create_smart_video as assemble_smart_video

    
    run_dir = 'runs/current'
    
    try:
        # Initialize decision logger
        init_decision_logger(run_dir, debug_mode=True)
        logger = get_decision_logger()
        
        runner = ComponentRunner(run_dir, logger=logger)
        
        # Step 1: Discover headline
        logging.info("=== DISCOVERING HEADLINE ===")
        if not runner.run_component('discover'):
            logging.error("Failed to discover headline")
            return None
        
        # Step 2: Scrape article
        logging.info("=== SCRAPING ARTICLE ===")
        if not runner.run_component('scrape'):
            logging.error("Failed to scrape article")
            return None
            
        # Step 3: Take screenshot  
        logging.info("=== TAKING SCREENSHOT ===")
        if not runner.run_component('screenshot'):
            logging.error("Failed to take screenshot")
            return None
        
        # Step 4: Generate script
        logging.info("=== GENERATING SCRIPT ===")
        if not runner.run_component('script'):
            logging.error("Failed to generate script")
            return None
        
        # Step 5: Generate audio
        logging.info("=== GENERATING AUDIO ===")
        if not runner.run_component('audio'):
            logging.error("Failed to generate audio")
            return None
            
        # Step 6: Extract timing using whisper-timestamped
        logging.info("=== TIMING EXTRACTION ===")
        if not runner.run_component('timing_extraction'):
            logging.error("Failed to extract timing - continuing with estimated timing")
            # Don't fail the pipeline, timing extraction is non-critical
            pass
        
        # Step 8: Visual Direction
        logging.info("=== VISUAL DIRECTION ===")
        if not runner.run_component('visual_director'):
            logging.error("Visual direction component failed")
            return None
        
        # Load AI-generated visual data from component
        visual_map_data = runner._load_json("visual_map.json")
        visual_map = visual_map_data.get('visual_map', {})  # Simple cue_id -> path mapping for video assembly
        visual_timeline = visual_map_data.get('visual_timeline_simple', visual_map_data.get('visual_timeline', []))  # Use simple timeline for video assembly
        visual_strategy = visual_map_data.get('visual_strategy', {})
        
        # Process opening screenshot OR use webpage video if available
        screenshot_path = os.path.join(run_dir, 'url_screenshot.png')
        webpage_video_path = os.path.join(run_dir, 'webpage_capture.mp4')
        
        if os.path.exists(webpage_video_path):
            # Use the webpage video for opening sequence instead of static screenshot
            processed_opening = webpage_video_path
            logging.info(f"Using webpage video for opening sequence: {webpage_video_path}")
        else:
            # This fallback is now dead code since the pipeline crashes if webpage_capture.mp4 is not created.
            logging.error(f"Critical asset missing: webpage_capture.mp4 was not found.")
            return None
        
        logging.info(f"AI created visual strategy with {len(visual_timeline)} visual cues")
        logging.info(f"Gathered {len(visual_map)} visuals for timeline")
        
        # Step 9: Smart Video Assembly
        logging.info("=== SMART VIDEO ASSEMBLY ===")
        
        # Create fast-paced video with synchronized visuals
        video_output_path = os.path.join(run_dir, 'smart_video.mp4')
        audio_path = os.path.join(run_dir, 'voice.mp3')
        
        final_video_path = assemble_smart_video(
            visual_timeline=visual_timeline,
            visual_strategy=visual_strategy, 
            visual_map=visual_map,
            opening_screenshot=processed_opening,  # This can now be a video file
            audio_path=audio_path,
            output_path=video_output_path
        )
        
        # Step 10: Add modern word-level captions
        logging.info("=== ADDING MODERN CAPTIONS ===")
        
        # Use new word-level caption system
        from .video.word_captions import add_word_captions
        
        script_text = runner._load_text("script.txt")
        final_with_captions = os.path.join(run_dir, 'final.mp4')
        add_word_captions(
            video_path=final_video_path,
            audio_path=audio_path,
            script_text=script_text,
            output_path=final_with_captions
        )
        
        logging.info(f"=== PIPELINE COMPLETE ===")
        logging.info(f"Final video: {final_with_captions}")
        
        return final_with_captions
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# Keep the original function for backwards compatibility
def create_video(max_images: int = 10) -> Optional[str]:
    """
    Legacy video creation function that wraps the smart pipeline.
    
    This function is maintained for backward compatibility with existing code
    that uses the original API. It internally calls create_smart_video() with
    the same parameters and behavior.
    
    Args:
        max_images (int, optional): Maximum number of images to download.
            This parameter is kept for backward compatibility but is not
            actively used in the current implementation. Defaults to 10.
    
    Returns:
        Optional[str]: Path to the final generated video file (.mp4) if successful,
            None if the pipeline fails at any stage.
    
    Raises:
        FileNotFoundError: If required configuration files or dependencies are missing
        OSError: If FFmpeg is not installed or accessible
        RuntimeError: If critical pipeline components fail
        
    Example:
        >>> from agent import create_video  # Legacy import
        >>> video_path = create_video()
        >>> if video_path:
        ...     print(f"Video created: {video_path}")
    
    Note:
        For new code, prefer using create_smart_video() directly as it provides
        the same functionality with more explicit naming.
    """
    return create_smart_video(max_images)
