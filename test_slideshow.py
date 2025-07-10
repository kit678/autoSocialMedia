#!/usr/bin/env python3
"""
Standalone test script for the slideshow component.

This script tests the slideshow component in isolation using the existing 
pipeline outputs from the runs directory.

Usage:
    python test_slideshow.py [--run-dir runs/current] [--verbose]
"""

import os
import json
import logging
import sys

# Set up basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_slideshow_test():
    """
    Tests the slideshow component in isolation using existing run data.
    """
    # Add the project root to the Python path to allow for imports
    sys.path.append(os.getcwd())
    
    from agent.slideshow.create_smart_video import run as create_smart_video
    from agent.utils import get_audio_duration

    run_dir = 'runs/current'
    logging.info(f"--- Running Slideshow Test using data from '{run_dir}' ---")

    # --- Load Inputs ---
    visual_map_path = os.path.join(run_dir, 'visual_map.json')
    audio_path = os.path.join(run_dir, 'voice.mp3')
    output_path = os.path.join(run_dir, 'test_slideshow_output.mp4')

    if not os.path.exists(visual_map_path):
        logging.error(f"FATAL: visual_map.json not found at '{visual_map_path}'")
        return

    if not os.path.exists(audio_path):
        logging.error(f"FATAL: voice.mp3 not found at '{audio_path}'")
        return

    # Load the visual map data
    with open(visual_map_path, 'r', encoding='utf-8') as f:
        visual_analysis = json.load(f)

    # --- MANUAL FIX ---
    # The visual_director is failing to add the 'segments' key. We add it here
    # manually to allow the slideshow component to be tested.
    if 'segments' not in visual_analysis:
        logging.warning("'segments' key not found in visual_map.json. Manually adding it for this test.")
        visual_analysis['segments'] = visual_analysis.get('visual_timeline', [])
    # --- END MANUAL FIX ---
    
    all_image_paths = visual_analysis.get('visual_map', {})
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    if not audio_duration:
        logging.error("Could not determine audio duration. Aborting test.")
        return

    logging.info(f"Inputs loaded. Audio duration: {audio_duration:.2f}s. Starting video creation...")

    # --- Execute Slideshow Component ---
    try:
        final_video_path = create_smart_video(
            visual_analysis=visual_analysis,
            all_image_paths=all_image_paths,
            audio_path=audio_path,
            audio_duration=audio_duration,
            output_path=output_path,
            fps=30
        )

        if final_video_path and os.path.exists(final_video_path):
            logging.info(f"✅ SUCCESS: Slideshow component test completed.")
            logging.info(f"✅ Test video created at: {final_video_path}")
        else:
            logging.error("❌ FAILED: Slideshow component execution failed. Check logs for errors.")

    except Exception as e:
        logging.error(f"❌ An exception occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    run_slideshow_test() 