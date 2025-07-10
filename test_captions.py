import os
import json
import logging
import sys

# Set up basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_captions_test():
    """
    Tests the captions component in isolation using existing run data.
    """
    # Add the project root to the Python path to allow for imports
    sys.path.append(os.getcwd())
    
    from agent.video.word_captions import add_word_captions

    run_dir = 'runs/current'
    logging.info(f"--- Running Captions Test using data from '{run_dir}' ---")

    # --- Check Input Files ---
    video_path = os.path.join(run_dir, 'test_slideshow_output.mp4')
    transcript_path = os.path.join(run_dir, 'transcript_data.json')
    output_path = os.path.join(run_dir, 'test_captions_output.mp4')

    missing_files = []
    if not os.path.exists(video_path):
        missing_files.append(f"Video: {video_path}")
    if not os.path.exists(transcript_path):
        missing_files.append(f"Transcript: {transcript_path}")

    if missing_files:
        logging.error(f"FATAL: Missing required files: {missing_files}")
        return False

    logging.info("✅ All required files found")

    # --- Load and Transform Transcript Data ---
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Convert the transcript data to the format expected by add_word_captions
        # The function expects 'segments' with 'words'
        if 'all_words' in transcript_data:
            words = transcript_data['all_words']
            logging.info(f"✅ Loaded transcript with {len(words)} words from 'all_words'")
        else:
            logging.error("❌ No 'all_words' found in transcript")
            return False
            
        if not words:
            logging.error("❌ No word timing data found")
            return False
            
        # Transform data to expected format
        # The caption function expects each word to have 'start', 'end', and 'text' fields
        # But our data has 'start_time', 'end_time', and 'word'
        transformed_words = []
        for word in words:
            transformed_word = {
                'start': word.get('start_time', 0),
                'end': word.get('end_time', 0),
                'text': word.get('word', '')
            }
            transformed_words.append(transformed_word)
        
        formatted_transcript = {
            "segments": [
                {
                    "words": transformed_words
                }
            ]
        }
        
        # Show sample of word timings
        sample_words = words[:3]
        for word in sample_words:
            logging.info(f"   Sample: '{word.get('word', '')}' @ {word.get('start_time', 0):.2f}s")
            
    except Exception as e:
        logging.error(f"❌ Failed to load transcript data: {e}")
        return False

    # --- Test Caption Generation ---
    logging.info("\n--- TESTING CAPTION GENERATION ---")
    logging.info(f"Input video: {video_path}")
    logging.info(f"Output video: {output_path}")
    logging.info(f"Words to caption: {len(words)}")
    
    try:
        # Call the caption generation function
        result = add_word_captions(
            video_path=video_path,
            transcript_data=formatted_transcript,
            output_path=output_path
        )
        
        if result and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            logging.info(f"✅ SUCCESS: Captions added successfully!")
            logging.info(f"   Output file: {output_path}")
            logging.info(f"   File size: {file_size:.2f} MB")
            
            # Test video properties
            try:
                from agent.utils import run_command
                
                # Get video duration
                duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', output_path]
                success_probe, duration_out, _ = run_command(duration_cmd)
                if success_probe:
                    actual_duration = float(duration_out.strip())
                    logging.info(f"   Video duration: {actual_duration:.2f}s")
                
                # Get video properties  
                info_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', output_path]
                success_info, info_out, _ = run_command(info_cmd)
                if success_info:
                    lines = info_out.strip().split('\n')
                    for line in lines:
                        if line:
                            props = line.split(',')
                            if len(props) >= 3:
                                width, height, fps_frac = props[0], props[1], props[2]
                                logging.info(f"   Video properties: {width}x{height} @ {fps_frac} fps")
                                break
                
                # Check if captions are actually embedded
                subtitle_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=codec_name', '-select_streams', 's', '-of', 'csv=p=0', output_path]
                success_sub, sub_out, _ = run_command(subtitle_cmd)
                if success_sub and sub_out.strip():
                    logging.info(f"   Subtitle streams detected: {sub_out.strip()}")
                else:
                    logging.info("   No separate subtitle streams (captions burned into video)")
                    
            except Exception as e:
                logging.warning(f"   ⚠️ Could not probe video properties: {e}")
            
            return True
        else:
            logging.error("❌ FAILED: Caption generation failed")
            if not os.path.exists(output_path):
                logging.error("   Output file was not created")
            return False
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """Run tests when called directly."""
    print("📝 CAPTIONS COMPONENT ISOLATION TESTING")
    print("=" * 50)
    
    # Test caption burning 
    test_success = run_captions_test()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print(f"✅ Caption burning test: {'PASSED' if test_success else 'FAILED'}")
    
    if test_success:
        print("🎉 TEST PASSED - Captions component is working correctly!")
        exit(0)
    else:
        print("💥 TEST FAILED - Issues remain in captions component")
        exit(1) 