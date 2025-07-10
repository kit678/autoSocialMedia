"""
AutoSocialMedia Agent - Main Pipeline Orchestration with Smart Visual Direction
"""

import os
import logging
from typing import Optional

def create_smart_video(max_images: int = 10) -> Optional[str]:
    """
    Main pipeline function with AI-driven visual direction and fast-paced video creation.
    
    Args:
        max_images: Maximum number of images to download (kept for compatibility)
        
    Returns:
        str: Path to final video if successful, None if failed
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
    Legacy video creation function - now uses smart pipeline
    """
    return create_smart_video(max_images)
