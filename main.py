import os
import logging
import argparse
import shutil
import json
from datetime import datetime
import sys
import codecs

# Force unbuffered output for immediate log visibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# --- Fix for UnicodeEncodeError on Windows ---
# When running in a Windows console, default encoding might not be UTF-8.
# This configures the stream handler to use UTF-8, preventing errors when
# logging characters like emojis or special symbols.
if sys.platform == "win32":
    if sys.stdout.encoding != 'utf-8':
        try:
            # Attempt to reopen stdout with UTF-8 encoding
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except TypeError:
            # In some environments (like standard cmd.exe), reconfigure is not available.
            # This is a fallback for those cases.
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
# --- End of Fix ---

from agent.component_runner import ComponentRunner
from agent.decision_logger import DecisionLogger
import configparser

def clear_current_run_directory(run_dir):
    """
    Clears the current run directory to ensure fresh state.
    Handles locked files gracefully and preserves directory structure.
    """
    if not os.path.exists(run_dir):
        return
    
    print(f"Clearing previous run data from: {run_dir}")
    
    try:
        # Remove all contents but preserve the main directory
        for item in os.listdir(run_dir):
            item_path = os.path.join(run_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
                print(f"  ✓ Removed: {item}")
            except Exception as e:
                print(f"  ⚠ Could not remove {item}: {e}")
                # Continue with other files even if one fails
                continue
        
        print("✓ Current run directory cleared successfully")
        
    except Exception as e:
        print(f"⚠ Warning: Could not fully clear run directory: {e}")
        # Don't fail the pipeline if clearing fails

def setup_logging(run_dir):
    log_file = os.path.join(run_dir, 'run.log')
    
    # Create a custom StreamHandler with immediate flushing
    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Set to DEBUG level for more verbose logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                                  FlushingStreamHandler(sys.stdout)])
    # Set specific loggers to appropriate levels to reduce noise
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('moviepy').setLevel(logging.ERROR)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('undetected_chromedriver').setLevel(logging.ERROR)  # Suppress Windows handle warnings
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Suppress Chrome WebDriver cleanup warnings on Windows
    if sys.platform == "win32":
        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning, message=".*Chrome.*")
        warnings.filterwarnings("ignore", category=OSError, message=".*WinError 6.*")
    
    # Log that verbose logging is enabled
    logging.info("=== VERBOSE LOGGING ENABLED (DEBUG LEVEL) ===")

def load_pipeline_config():
    """Load pipeline configuration including visual director settings."""
    config_data = {}
    
    # Load traditional config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Load AI visual director config if it exists
    ai_config_path = 'config/ai_visual_config.json'
    if os.path.exists(ai_config_path):
        with open(ai_config_path, 'r') as f:
            config_data.update(json.load(f))
        logging.info(f"Loaded AI visual director configuration from {ai_config_path}")
    else:
        logging.info("No AI visual config found, using conventional visual director")
    
    return config, config_data

def main():
    # --- Configuration ---
    config_ini, pipeline_config = load_pipeline_config()
    tts_provider = config_ini.get('audio', 'tts_provider', fallback='kokoro').lower()

    # --- Setup ---
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "current")
    
    # Clear previous run data for fresh state
    clear_current_run_directory(run_dir)
    
    os.makedirs(run_dir, exist_ok=True)
    
    decisions_dir = os.path.join(run_dir, "decisions")
    os.makedirs(decisions_dir, exist_ok=True)
    
    setup_logging(run_dir)
    logging.info("=== STARTING AUTOSOCIALMEDIA PIPELINE ===")
    
    # Log which visual director will be used
    director_type = pipeline_config.get('visual_director', {}).get('type', 'conventional')
    logging.info(f"Using {director_type} visual director")
    
    # --- Initialization ---
    logger = DecisionLogger(run_dir=run_dir)
    runner = ComponentRunner(run_dir=run_dir, logger=logger, tts_provider=tts_provider, config=pipeline_config)
    
    # --- Pipeline Execution ---
    try:
        runner.run_pipeline()
        logging.info("=== AUTOSOCIALMEDIA PIPELINE COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        logging.error(f"PIPELINE FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    main() 